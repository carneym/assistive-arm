"""
Voice Activity Detection (VAD) for the SO-101 assistive arm.

Determines when the user is speaking vs silent, used to:
  1. Know when an utterance (post-wake-word) has finished
  2. Trim silence from audio before sending to Whisper
  3. Avoid sending pure silence to STT (saves compute)

Two implementations:
  - SileroVAD: Neural network-based, very accurate (recommended)
  - EnergyVAD: Simple RMS energy threshold (fallback, no dependencies)
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..config.settings import VADConfig, AudioConfig

logger = logging.getLogger(__name__)


class VADEngine(ABC):
    @abstractmethod
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Returns True if the audio frame contains speech."""
        ...

    @abstractmethod
    def reset(self):
        """Reset internal state between utterances."""
        ...


class SileroVAD(VADEngine):
    """
    Silero VAD — lightweight neural VAD model.
    
    Runs on CPU with minimal overhead. Very accurate at distinguishing
    speech from background noise, music, etc.
    
    Requires: torch, torchaudio
    """

    def __init__(self, config: VADConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self._model = None
        self._sample_rate = audio_config.sample_rate

        try:
            import torch
            self._torch = torch

            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            (
                self._get_speech_timestamps,
                self._save_audio,
                self._read_audio,
                self._VADIterator,
                self._collect_chunks,
            ) = utils

            logger.info("Silero VAD loaded successfully")

        except Exception as e:
            raise ImportError(f"Failed to load Silero VAD: {e}")

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Check if a single frame contains speech."""
        audio_float = audio_frame.astype(np.float32) / 32768.0
        tensor = self._torch.from_numpy(audio_float)

        # Silero VAD expects specific frame sizes (typically 512 samples at 16kHz)
        confidence = self._model(tensor, self._sample_rate).item()
        return confidence > self.config.threshold

    def reset(self):
        """Reset model state between utterances."""
        if self._model is not None:
            self._model.reset_states()


class EnergyVAD(VADEngine):
    """
    Simple energy-based Voice Activity Detection.

    Uses RMS energy + zero-crossing rate as a basic speech indicator.
    No external dependencies. Less accurate than Silero but works anywhere.

    Supports child voice mode with adjusted thresholds for higher-pitched,
    often quieter voices.
    """

    def __init__(self, config: VADConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self._child_mode = getattr(audio_config, "child_voice_mode", False)

        # Adaptive threshold — starts at a default, adjusts based on ambient noise
        # Children's voices are often quieter, so use lower defaults
        self._energy_threshold = 400.0 if self._child_mode else 800.0
        self._ambient_energy = 0.0
        self._calibration_frames = 0
        self._calibration_target = 30  # ~1 second of calibration at 30ms frames
        self._calibrated = False

        if self._child_mode:
            logger.info("VAD: child voice mode enabled (lower thresholds)")

    def _rms_energy(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    def _zero_crossing_rate(self, frame: np.ndarray) -> float:
        signs = np.sign(frame.astype(np.float32))
        crossings = np.abs(np.diff(signs))
        return float(np.mean(crossings > 0))

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        energy = self._rms_energy(audio_frame)

        # Calibrate ambient noise level from first ~1 second
        if not self._calibrated:
            self._ambient_energy += energy
            self._calibration_frames += 1
            if self._calibration_frames >= self._calibration_target:
                avg_ambient = self._ambient_energy / self._calibration_frames
                # Lower multiplier and minimum for children's quieter voices
                if self._child_mode:
                    self._energy_threshold = max(avg_ambient * 2.0, 250.0)
                else:
                    self._energy_threshold = max(avg_ambient * 3.0, 500.0)
                self._calibrated = True
                logger.info(
                    f"VAD calibrated: ambient={avg_ambient:.0f}, "
                    f"threshold={self._energy_threshold:.0f}"
                    f"{' (child mode)' if self._child_mode else ''}"
                )
            return False

        # Speech = energy above threshold + sufficient zero-crossings
        zcr = self._zero_crossing_rate(audio_frame)
        is_energetic = energy > self._energy_threshold
        # Children have higher-pitched voices = higher ZCR
        zcr_threshold = 0.08 if self._child_mode else 0.05
        has_zcr = zcr > zcr_threshold

        return is_energetic and has_zcr

    def reset(self):
        """Reset for next utterance (keep calibration)."""
        pass


class VoiceActivityDetector:
    """
    Factory + wrapper for VAD engines.
    
    Also provides utterance segmentation: collects audio frames after
    wake word trigger until speech ends, then returns the full utterance.
    """

    def __init__(
        self,
        vad_config: VADConfig,
        audio_config: AudioConfig,
        test_mode: bool = False,
    ):
        self.config = vad_config
        self.audio_config = audio_config
        self.engine: VADEngine
        self._engine_name: str

        if test_mode:
            # Use energy VAD in test mode (no torch dependency)
            self.engine = EnergyVAD(vad_config, audio_config)
            self._engine_name = "energy"
            logger.info("VAD: using energy-based engine (test mode)")
            return

        # Try Silero first
        try:
            self.engine = SileroVAD(vad_config, audio_config)
            self._engine_name = "silero"
            logger.info("VAD: using Silero neural engine")
            return
        except (ImportError, Exception) as e:
            logger.warning(f"Silero VAD unavailable ({e}), using energy fallback")

        self.engine = EnergyVAD(vad_config, audio_config)
        self._engine_name = "energy"

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        return self.engine.is_speech(audio_frame)

    def segment_utterance(
        self,
        audio_frames: list[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Given a list of audio frames, trim leading/trailing silence
        and return only the speech portion. Returns None if no speech found.
        """
        speech_flags = [self.engine.is_speech(frame) for frame in audio_frames]

        # Find first and last speech frames (with padding)
        pad_frames = int(
            self.config.speech_pad_ms / (self.audio_config.chunk_duration_ms)
        )
        first_speech = None
        last_speech = None

        for i, is_sp in enumerate(speech_flags):
            if is_sp:
                if first_speech is None:
                    first_speech = max(0, i - pad_frames)
                last_speech = min(len(audio_frames) - 1, i + pad_frames)

        if first_speech is None:
            logger.debug("No speech detected in utterance")
            return None

        speech_frames = audio_frames[first_speech : last_speech + 1]
        result = np.concatenate(speech_frames)
        logger.debug(
            f"Segmented utterance: {len(result)/self.audio_config.sample_rate:.1f}s "
            f"(from {len(audio_frames)} frames)"
        )
        return result

    def reset(self):
        self.engine.reset()

    @property
    def engine_name(self) -> str:
        return self._engine_name
