"""
Audio capture module for the SO-101 voice pipeline.

Provides three audio sources:
  1. LiveAudioSource  — real-time mic input via sounddevice (for Jetson + ReSpeaker)
  2. FileAudioSource  — reads .wav files for offline testing
  3. SimulatedAudioSource — generates synthetic audio events for logic testing

All sources yield audio chunks as numpy arrays (int16, 16kHz mono).
"""

import logging
import queue
import struct
import threading
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from ..config.settings import AudioConfig

logger = logging.getLogger(__name__)


class AudioSource(ABC):
    """Base class for audio sources."""

    def __init__(self, config: AudioConfig):
        self.config = config

    @abstractmethod
    def stream_chunks(self) -> Generator[np.ndarray, None, None]:
        """Yield audio chunks as numpy int16 arrays."""
        ...

    @abstractmethod
    def record_utterance(self, max_seconds: Optional[float] = None) -> np.ndarray:
        """Record a complete utterance (after wake word trigger) and return as a single array."""
        ...

    def close(self):
        """Clean up resources."""
        pass


class LiveAudioSource(AudioSource):
    """
    Real-time microphone input via sounddevice.

    Uses a callback-based approach with a queue to decouple audio capture
    from processing. This prevents buffer overflows when processing is slow.

    Designed for ReSpeaker USB Mic Array v2.0, but works with any ALSA mic.
    On Jetson, the ReSpeaker appears as a standard USB audio device.
    """

    # Max frames to buffer before dropping old ones (prevents unbounded memory)
    MAX_QUEUE_SIZE = 100  # ~3 seconds at 30ms chunks

    def __init__(self, config: AudioConfig, device_index: Optional[int] = None):
        super().__init__(config)
        self.device_index = device_index
        self._stream = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._stop_event = threading.Event()

        try:
            import sounddevice as sd
            self.sd = sd

            # List available devices for debugging
            devices = sd.query_devices()
            logger.info(f"Available audio devices:\n{devices}")

            if device_index is None:
                # Try to auto-detect ReSpeaker
                for i, dev in enumerate(devices):
                    if "respeaker" in str(dev.get("name", "")).lower():
                        self.device_index = i
                        logger.info(f"Auto-detected ReSpeaker at device index {i}")
                        break

            if self.device_index is not None:
                sd.default.device = self.device_index

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialise audio device: {e}\n"
                "Make sure a microphone is connected and sounddevice is installed."
            )

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback invoked by sounddevice in a separate thread."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        try:
            # Non-blocking put - if queue is full, drop the oldest frame
            self._audio_queue.put_nowait(indata.copy().flatten())
        except queue.Full:
            # Drop oldest frame and add new one
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(indata.copy().flatten())
                logger.debug("Dropped oldest audio frame (processing behind)")
            except queue.Empty:
                pass

    def stream_chunks(self) -> Generator[np.ndarray, None, None]:
        """Yield audio chunks from the microphone in real time."""
        chunk_samples = self.config.chunk_samples
        self._stop_event.clear()

        with self.sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=chunk_samples,
            device=self.device_index,
            callback=self._audio_callback,
        ) as stream:
            self._stream = stream
            logger.info("Microphone stream started (callback mode)")
            while not self._stop_event.is_set():
                try:
                    # Block with timeout so we can check stop_event periodically
                    chunk = self._audio_queue.get(timeout=0.1)
                    yield chunk
                except queue.Empty:
                    continue

    def record_utterance(self, max_seconds: Optional[float] = None) -> np.ndarray:
        """
        Record audio until silence is detected or max duration reached.
        Returns the complete utterance as a numpy array.
        """
        max_seconds = max_seconds or self.config.max_record_seconds
        max_samples = int(self.config.sample_rate * max_seconds)
        chunk_samples = self.config.chunk_samples
        silence_threshold = 500  # RMS threshold for "silence" (tune for your mic)
        silence_chunks_needed = int(
            self.config.silence_timeout_seconds / (self.config.chunk_duration_ms / 1000)
        )

        recorded = []
        silence_count = 0
        record_queue: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Record callback status: {status}")
            record_queue.put(indata.copy().flatten())

        with self.sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=chunk_samples,
            device=self.device_index,
            callback=callback,
        ) as stream:
            total_samples = 0
            while total_samples < max_samples:
                try:
                    chunk = record_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                recorded.append(chunk)
                total_samples += len(chunk)

                # Simple energy-based silence detection
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                if rms < silence_threshold:
                    silence_count += 1
                else:
                    silence_count = 0

                if silence_count >= silence_chunks_needed:
                    logger.debug(f"Silence detected after {total_samples / self.config.sample_rate:.1f}s")
                    break

        audio = np.concatenate(recorded)
        logger.info(f"Recorded utterance: {len(audio) / self.config.sample_rate:.1f}s")
        return audio

    def close(self):
        self._stop_event.set()
        if self._stream is not None:
            self._stream.close()


class FileAudioSource(AudioSource):
    """
    Read audio from .wav files for offline testing.
    
    Usage:
        source = FileAudioSource(config, "test_command.wav")
        audio = source.record_utterance()
    """

    def __init__(self, config: AudioConfig, filepath: str):
        super().__init__(config)
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {self.filepath}")

    def _load_wav(self) -> np.ndarray:
        """Load a WAV file and resample to expected format."""
        with wave.open(str(self.filepath), "rb") as wf:
            assert wf.getsampwidth() == 2, "Expected 16-bit WAV"
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

            # Convert stereo to mono if needed
            if wf.getnchannels() == 2:
                audio = audio[::2]

            # Simple resample if sample rate differs
            if wf.getframerate() != self.config.sample_rate:
                ratio = self.config.sample_rate / wf.getframerate()
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
                audio = audio[indices]

            logger.info(
                f"Loaded {self.filepath.name}: {len(audio)/self.config.sample_rate:.1f}s"
            )
            return audio

    def stream_chunks(self) -> Generator[np.ndarray, None, None]:
        """Yield chunks from the loaded audio file."""
        audio = self._load_wav()
        chunk_size = self.config.chunk_samples
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            yield chunk
            # Simulate real-time playback speed
            time.sleep(self.config.chunk_duration_ms / 1000)

    def record_utterance(self, max_seconds: Optional[float] = None) -> np.ndarray:
        """Return the entire file as an utterance."""
        audio = self._load_wav()
        if max_seconds:
            max_samples = int(self.config.sample_rate * max_seconds)
            audio = audio[:max_samples]
        return audio


class SimulatedAudioSource(AudioSource):
    """
    Generates synthetic audio events for testing pipeline logic
    without any real audio hardware or files.
    
    Produces sequences like:
      [silence] → [wake word trigger] → [simulated speech] → [silence]
    """

    def __init__(self, config: AudioConfig, commands: Optional[list[str]] = None):
        super().__init__(config)
        self.commands = commands or [
            "pick up my phone",
            "put it down",
            "get my cup",
            "stop",
        ]
        self._command_index = 0

    def _generate_tone(self, duration_s: float, freq: float = 440.0) -> np.ndarray:
        """Generate a simple sine wave tone (simulates speech energy)."""
        t = np.arange(int(self.config.sample_rate * duration_s))
        tone = (np.sin(2 * np.pi * freq * t / self.config.sample_rate) * 10000).astype(np.int16)
        return tone

    def _generate_silence(self, duration_s: float) -> np.ndarray:
        """Generate silence."""
        return np.zeros(int(self.config.sample_rate * duration_s), dtype=np.int16)

    def stream_chunks(self) -> Generator[np.ndarray, None, None]:
        """Yield alternating silence and 'speech' tones."""
        chunk_size = self.config.chunk_samples
        while True:
            # Silence gap
            silence = self._generate_silence(2.0)
            for i in range(0, len(silence), chunk_size):
                yield silence[i : i + chunk_size]

            # Simulated wake word + command as tone burst
            speech = self._generate_tone(2.5, freq=300.0)
            for i in range(0, len(speech), chunk_size):
                chunk = speech[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                yield chunk

    def record_utterance(self, max_seconds: Optional[float] = None) -> np.ndarray:
        """Return a tone burst simulating a spoken command."""
        duration = min(max_seconds or 3.0, 3.0)
        return self._generate_tone(duration, freq=350.0)

    def get_simulated_transcript(self) -> str:
        """Get the 'expected' transcript for the current simulated command."""
        cmd = self.commands[self._command_index % len(self.commands)]
        self._command_index += 1
        return cmd


class AudioCapture:
    """
    Factory that selects the appropriate audio source based on environment.
    
    Usage:
        capture = AudioCapture(config)
        source = capture.get_source()
    """

    def __init__(self, config: AudioConfig, test_mode: bool = False):
        self.config = config
        self.test_mode = test_mode

    def get_source(
        self,
        filepath: Optional[str] = None,
        simulated_commands: Optional[list[str]] = None,
    ) -> AudioSource:
        """
        Get the best available audio source:
          1. If filepath is given → FileAudioSource
          2. If test_mode → SimulatedAudioSource
          3. If mic available → LiveAudioSource
          4. Else → SimulatedAudioSource with warning
        """
        if filepath:
            logger.info(f"Using file audio source: {filepath}")
            return FileAudioSource(self.config, filepath)

        if self.test_mode:
            logger.info("Using simulated audio source (test mode)")
            return SimulatedAudioSource(self.config, simulated_commands)

        # Try live mic
        try:
            source = LiveAudioSource(self.config)
            logger.info("Using live microphone")
            return source
        except Exception as e:
            logger.warning(f"No microphone available ({e}), falling back to simulated source")
            return SimulatedAudioSource(self.config, simulated_commands)
