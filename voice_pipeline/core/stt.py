"""
Speech-to-Text module for the SO-101 assistive arm.

Uses OpenAI Whisper for transcription. Supports multiple backends:
  1. openai-whisper (PyTorch-based, easiest to install)
  2. whisper.cpp via Python bindings (faster on Jetson with CUDA)
  3. Simulated STT for testing (returns predefined transcripts)

All backends produce the same output: a transcript string.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config.settings import STTConfig, AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    """Result from speech-to-text."""
    text: str                    # The transcribed text
    language: str = "en"         # Detected language
    confidence: float = 0.0      # Average log probability (higher = more confident)
    duration_s: float = 0.0      # Audio duration processed
    processing_time_s: float = 0.0  # Time taken to transcribe
    no_speech_prob: float = 0.0  # Probability that segment contains no speech

    @property
    def is_valid(self) -> bool:
        """Check if the transcript likely contains real speech."""
        return (
            len(self.text.strip()) > 0
            and self.no_speech_prob < 0.7
        )

    def __str__(self) -> str:
        return (
            f"TranscriptResult(text='{self.text}', confidence={self.confidence:.2f}, "
            f"no_speech={self.no_speech_prob:.2f}, time={self.processing_time_s:.2f}s)"
        )


class STTEngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """Transcribe an audio array (int16, 16kHz mono) to text."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...


class WhisperSTT(STTEngine):
    """
    OpenAI Whisper speech-to-text.

    On Jetson Orin Nano with CUDA:
      - "tiny":  ~1.0s for 5s audio
      - "base":  ~1.5s for 5s audio
      - "small": ~2.5s for 5s audio (recommended for accuracy)

    On CPU (dev machine):
      - "tiny":  ~2-3s for 5s audio
      - "small": ~8-12s for 5s audio
    """

    def __init__(self, config: STTConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self._model = None
        self._loaded = False

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return

        try:
            import whisper

            logger.info(f"Loading Whisper '{self.config.model_size}' on {self.config.device}...")
            start = time.time()
            self._model = whisper.load_model(
                self.config.model_size,
                device=self.config.device,
            )
            elapsed = time.time() - start
            logger.info(f"Whisper '{self.config.model_size}' loaded in {elapsed:.1f}s")
            self._loaded = True

        except ImportError:
            raise ImportError(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )

    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """Transcribe audio to text using Whisper."""
        self._load_model()

        # Convert int16 to float32 normalised [-1.0, 1.0]
        audio_float = audio.astype(np.float32) / 32768.0
        duration_s = len(audio) / self.audio_config.sample_rate

        logger.debug(f"Transcribing {duration_s:.1f}s of audio...")
        start = time.time()

        result = self._model.transcribe(
            audio_float,
            language=self.config.language,
            fp16=self.config.fp16,
            temperature=self.config.temperature,
            no_speech_threshold=self.config.no_speech_threshold,
            initial_prompt=self.config.initial_prompt,
        )

        processing_time = time.time() - start

        # Extract results
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        # Compute average confidence and no-speech probability
        avg_confidence = 0.0
        avg_no_speech = 0.0
        if segments:
            avg_confidence = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
            avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)

        transcript = TranscriptResult(
            text=text,
            language=result.get("language", "en"),
            confidence=avg_confidence,
            duration_s=duration_s,
            processing_time_s=processing_time,
            no_speech_prob=avg_no_speech,
        )

        logger.info(f"STT result: {transcript}")
        return transcript

    def is_loaded(self) -> bool:
        return self._loaded


class SimulatedSTT(STTEngine):
    """
    Simulated STT for testing without Whisper installed.
    
    Returns predefined transcripts from a command queue.
    Useful for testing the intent parser and downstream logic.
    """

    def __init__(self, config: STTConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self._commands = [
            "hey arm pick up my phone",
            "put it down",
            "hey arm get my cup",
            "bring it to my mouth",
            "put it back",
            "hey arm press the button",
            "stop",
            "hey arm scratch my chin",
            "hey arm move that out of the way",
            "the red one",
        ]
        self._index = 0
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """Return the next simulated command."""
        duration_s = len(audio) / self.audio_config.sample_rate

        # Check if audio has energy (simulated speech) or is silence
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        if rms < 500:
            return TranscriptResult(
                text="",
                duration_s=duration_s,
                processing_time_s=0.01,
                no_speech_prob=0.95,
            )

        text = self._commands[self._index % len(self._commands)]
        self._index += 1

        logger.info(f"Simulated STT: '{text}'")
        return TranscriptResult(
            text=text,
            confidence=-0.3,  # Typical good confidence
            duration_s=duration_s,
            processing_time_s=0.05,
            no_speech_prob=0.05,
        )

    def set_next_command(self, command: str):
        """Override the next command to be returned (for targeted testing)."""
        self._commands.insert(self._index, command)

    def is_loaded(self) -> bool:
        return self._loaded


class SpeechToText:
    """
    Factory + wrapper for STT engines.
    
    Auto-selects the best available backend:
      1. WhisperSTT (if openai-whisper is installed)
      2. SimulatedSTT (test mode or no Whisper available)
    """

    def __init__(
        self,
        stt_config: STTConfig,
        audio_config: AudioConfig,
        test_mode: bool = False,
    ):
        self.engine: STTEngine
        self._engine_name: str

        if test_mode:
            self.engine = SimulatedSTT(stt_config, audio_config)
            self._engine_name = "simulated"
            logger.info("STT: using simulated engine (test mode)")
            return

        # Try Whisper
        try:
            self.engine = WhisperSTT(stt_config, audio_config)
            self._engine_name = "whisper"
            logger.info(f"STT: using Whisper '{stt_config.model_size}'")
            return
        except ImportError as e:
            logger.warning(f"Whisper not available ({e}), using simulated STT")

        self.engine = SimulatedSTT(stt_config, audio_config)
        self._engine_name = "simulated"

    def transcribe(self, audio: np.ndarray) -> TranscriptResult:
        return self.engine.transcribe(audio)

    @property
    def engine_name(self) -> str:
        return self._engine_name

    def is_loaded(self) -> bool:
        return self.engine.is_loaded()
