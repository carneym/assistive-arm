"""
Wake word detection for the SO-101 assistive arm.

Two strategies:
  1. Porcupine (primary) — Picovoice's on-device wake word engine.
     Extremely efficient, <1% false activation rate.
     Requires a free API key from https://console.picovoice.ai/

  2. Keyword fallback — Runs Whisper on small rolling windows and checks
     for the wake phrase in the transcript. Higher latency but zero setup.

Both implement the same interface: process audio chunks, return True on detection.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..config.settings import WakeWordConfig, AudioConfig

logger = logging.getLogger(__name__)


class WakeWordEngine(ABC):
    """Base class for wake word engines."""

    @abstractmethod
    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """Process an audio frame. Returns True if wake word detected."""
        ...

    @abstractmethod
    def close(self):
        ...


class PorcupineWakeWord(WakeWordEngine):
    """
    Porcupine wake word detection (Picovoice).
    
    Runs entirely on-device, very low CPU usage.
    Supports custom wake words trained via Picovoice console.
    """

    def __init__(self, config: WakeWordConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self._porcupine = None

        try:
            import pvporcupine
            self._pvporcupine = pvporcupine
        except ImportError:
            raise ImportError(
                "pvporcupine not installed. Run: pip install pvporcupine\n"
                "Or use fallback wake word detection."
            )

        if not config.porcupine_access_key:
            raise ValueError(
                "Porcupine access key not set.\n"
                "Get a free key at https://console.picovoice.ai/\n"
                "Then: export PORCUPINE_ACCESS_KEY='your-key'"
            )

        # Initialise Porcupine
        if config.porcupine_keyword_path:
            # Custom trained keyword
            self._porcupine = pvporcupine.create(
                access_key=config.porcupine_access_key,
                keyword_paths=[config.porcupine_keyword_path],
                sensitivities=[config.sensitivity],
            )
        else:
            # Use built-in keyword (Porcupine has "hey google", "alexa", etc.)
            # For custom "hey arm", you'd train via Picovoice Console
            # For now, use "computer" as a placeholder built-in keyword
            builtin_keywords = pvporcupine.KEYWORDS
            logger.info(f"Available Porcupine keywords: {builtin_keywords}")

            # Use "computer" as stand-in; replace with trained "hey arm" .ppn file
            keyword = "computer"
            if keyword not in builtin_keywords:
                keyword = list(builtin_keywords)[0]
                logger.warning(f"Using '{keyword}' as placeholder wake word")

            self._porcupine = pvporcupine.create(
                access_key=config.porcupine_access_key,
                keywords=[keyword],
                sensitivities=[config.sensitivity],
            )

        self._frame_length = self._porcupine.frame_length
        logger.info(
            f"Porcupine initialised: frame_length={self._frame_length}, "
            f"sample_rate={self._porcupine.sample_rate}"
        )

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Process a single frame of audio.
        
        Note: Porcupine expects exactly `frame_length` samples (typically 512).
        The caller may need to buffer and rechunk audio.
        """
        if len(audio_frame) != self._frame_length:
            return False

        result = self._porcupine.process(audio_frame.tolist())
        if result >= 0:
            logger.info("Wake word detected (Porcupine)")
            return True
        return False

    @property
    def frame_length(self) -> int:
        return self._frame_length

    def close(self):
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None


class KeywordFallbackWakeWord(WakeWordEngine):
    """
    Fallback wake word detection using Whisper + fuzzy matching.

    Runs Whisper 'tiny' on a rolling audio buffer and checks for the wake
    phrase using fuzzy matching to handle transcription variations.

    Supports child voice mode with additional patterns for children's
    pronunciation and lower energy thresholds.
    """

    # Common transcription variations of "hey arm"
    WAKE_PATTERNS = [
        "hey arm", "hey, arm", "hey arm.", "hey arm!",
        "hay arm", "hay, arm",
        "hey on", "hey own",  # Common mishears
        "hey art", "hey arc",
        "hey um", "hey uhm",
        "a arm", "the arm",
        "hear arm", "here arm",
        "hey arms", "hey arm's",
        "hey arn", "hey arno",
        "heya arm", "hey a arm",
    ]

    # Additional patterns for children's pronunciation
    CHILD_WAKE_PATTERNS = [
        "hey ahm", "hey om", "hey awm",  # Children often soften the 'r'
        "hey aam", "hey am",
        "hi arm", "hi ahm", "hi om",     # Children may say "hi" instead of "hey"
        "hey yarm", "hey warm",          # 'y' or 'w' glide before vowels
        "heyam", "hey'm",                # Running words together
        "hey aum", "hey alm",
        "a yarm", "a arm",
        "hey arm please", "hey arm help",  # Polite additions
    ]

    def __init__(self, config: WakeWordConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self._child_mode = getattr(audio_config, "child_voice_mode", False)
        self._buffer: list[np.ndarray] = []
        self._buffer_duration_s = 2.0  # Rolling window
        self._buffer_max_samples = int(audio_config.sample_rate * self._buffer_duration_s)
        self._stt_model = None  # Lazy-loaded
        self._cooldown_until = 0  # Prevent rapid re-triggers

        # Speech burst detection - only run STT after speech ends
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._min_speech_frames = 10  # ~300ms of speech before checking
        self._silence_frames_to_end = 15  # ~450ms of silence = speech ended

        if self._child_mode:
            logger.info("Wake word: child voice mode enabled")

    def _get_stt(self):
        """Lazy-load a tiny Whisper model for wake word checking."""
        if self._stt_model is None:
            try:
                import whisper
                logger.info("Loading Whisper 'tiny.en' for wake word...")
                self._stt_model = whisper.load_model("tiny.en")
                logger.info("Whisper tiny.en loaded for wake word detection")
            except ImportError:
                logger.warning("Whisper not available — wake word fallback will use text-only mode")
                return None
        return self._stt_model

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching: lowercase, remove punctuation."""
        import re
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text)     # Collapse whitespace
        return text

    def _matches_wake_phrase(self, text: str) -> bool:
        """Check if text contains the wake phrase using fuzzy matching."""
        normalized = self._normalize_text(text)

        # Check exact wake phrase
        wake_normalized = self._normalize_text(self.config.wake_phrase)
        if wake_normalized in normalized:
            return True

        # Check common variations
        for pattern in self.WAKE_PATTERNS:
            if self._normalize_text(pattern) in normalized:
                return True

        # Check child-specific patterns
        if self._child_mode:
            for pattern in self.CHILD_WAKE_PATTERNS:
                if self._normalize_text(pattern) in normalized:
                    return True

        # Check for close phonetic match: "hey" + something starting with "ar"
        words = normalized.split()
        trigger_words = ["hey", "hay", "a", "the", "hear", "here", "heya"]
        if self._child_mode:
            trigger_words.extend(["hi", "hiya", "he"])

        for i, word in enumerate(words):
            if word in trigger_words:
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    # Standard matches
                    if next_word.startswith("ar") or next_word in ("on", "own", "um", "uhm"):
                        return True
                    # Child-specific: softer 'r' sounds
                    if self._child_mode:
                        if next_word.startswith(("ah", "aw", "om", "am", "al")):
                            return True
                        if next_word in ("yarm", "warm", "m", "ohm"):
                            return True

        return False

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Detect speech bursts and check for wake phrase only after speech ends.

        This is much more efficient than continuous STT - we only run Whisper
        once per speech burst instead of every few hundred milliseconds.
        """
        self._buffer.append(audio_frame)

        # Trim buffer to max size
        total = sum(len(f) for f in self._buffer)
        while total > self._buffer_max_samples and self._buffer:
            removed = self._buffer.pop(0)
            total -= len(removed)

        # Check cooldown
        now = time.time()
        if now < self._cooldown_until:
            return False

        # Detect speech energy in current frame
        frame_rms = np.sqrt(np.mean(audio_frame.astype(np.float32) ** 2))
        speech_threshold = 400 if self._child_mode else 600
        is_speech = frame_rms > speech_threshold

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            if not self._in_speech and self._speech_frames >= 3:
                self._in_speech = True
        else:
            if self._in_speech:
                self._silence_frames += 1

        # Check if speech burst just ended (speech detected, then silence)
        speech_ended = (
            self._in_speech
            and self._speech_frames >= self._min_speech_frames
            and self._silence_frames >= self._silence_frames_to_end
        )

        if not speech_ended:
            return False

        # Speech burst ended - now run STT once
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0

        # Get the audio to transcribe
        combined = np.concatenate(self._buffer)

        model = self._get_stt()
        if model is None:
            return False

        try:
            audio_float = combined.astype(np.float32) / 32768.0
            result = model.transcribe(
                audio_float,
                language="en",
                fp16=False,
                no_speech_threshold=0.5,
                condition_on_previous_text=False,
            )
            text = result.get("text", "")

            if self._matches_wake_phrase(text):
                logger.info(f"Wake word detected: '{text.strip()}'")
                self._buffer.clear()
                self._cooldown_until = now + 2.0
                return True

        except Exception as e:
            logger.error(f"Wake word STT error: {e}")

        return False

    def check_text(self, text: str) -> bool:
        """
        Direct text check — useful for testing or when STT runs externally.
        Uses the same fuzzy matching as audio-based detection.
        """
        if self._matches_wake_phrase(text):
            logger.info(f"Wake word detected in text: '{text.strip()}'")
            return True
        return False

    @property
    def frame_length(self) -> int:
        return self.audio_config.chunk_samples

    def close(self):
        self._stt_model = None


class SimulatedWakeWord(WakeWordEngine):
    """
    Simulated wake word for testing — triggers on audio energy above threshold.
    No actual speech recognition needed.
    """

    def __init__(self, config: WakeWordConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self._energy_threshold = 3000  # Simulated "speech" energy
        self._triggered = False
        self._cooldown_frames = 0

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1
            return False

        rms = np.sqrt(np.mean(audio_frame.astype(np.float32) ** 2))
        if rms > self._energy_threshold and not self._triggered:
            self._triggered = True
            self._cooldown_frames = 100  # Don't re-trigger for a while
            logger.info(f"Simulated wake word triggered (energy={rms:.0f})")
            return True

        if rms < 500:
            self._triggered = False

        return False

    @property
    def frame_length(self) -> int:
        return self.audio_config.chunk_samples

    def close(self):
        pass


class WakeWordDetector:
    """
    Factory + wrapper that selects the best available wake word engine.
    
    Priority:
      1. Porcupine (if key available)
      2. Keyword fallback (Whisper tiny)
      3. Simulated (test mode)
    """

    def __init__(
        self,
        wake_config: WakeWordConfig,
        audio_config: AudioConfig,
        test_mode: bool = False,
    ):
        self.engine: WakeWordEngine
        self._engine_name: str

        if test_mode:
            self.engine = SimulatedWakeWord(wake_config, audio_config)
            self._engine_name = "simulated"
            logger.info("Wake word: using simulated engine (test mode)")
            return

        # Try Porcupine first
        if wake_config.porcupine_access_key:
            try:
                self.engine = PorcupineWakeWord(wake_config, audio_config)
                self._engine_name = "porcupine"
                logger.info("Wake word: using Porcupine engine")
                return
            except Exception as e:
                logger.warning(f"Porcupine init failed ({e}), trying fallback")

        # Fallback to keyword matching
        if wake_config.use_fallback:
            self.engine = KeywordFallbackWakeWord(wake_config, audio_config)
            self._engine_name = "keyword_fallback"
            logger.info("Wake word: using keyword fallback engine")
            return

        # Last resort
        self.engine = SimulatedWakeWord(wake_config, audio_config)
        self._engine_name = "simulated"
        logger.warning("Wake word: no engine available, using simulated")

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        return self.engine.process_frame(audio_frame)

    def check_text(self, text: str) -> bool:
        """Check if text contains wake word (for text-based testing)."""
        if hasattr(self.engine, "check_text"):
            return self.engine.check_text(text)
        return self.engine.config.wake_phrase in text.lower()

    @property
    def engine_name(self) -> str:
        return self._engine_name

    @property
    def frame_length(self) -> int:
        return self.engine.frame_length

    def close(self):
        self.engine.close()
