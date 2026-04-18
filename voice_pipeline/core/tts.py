"""
Text-to-Speech module for the SO-101 assistive arm.

Provides audible feedback to the user: confirmations, status updates,
error messages, and prompts for clarification.

Backends:
  1. Piper TTS — fast, local, natural-sounding (recommended for Jetson)
  2. pyttsx3 — works everywhere, robotic but functional
  3. Simulated — just prints to console (for testing)
"""

import logging
from abc import ABC, abstractmethod

from ..config.settings import TTSConfig

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    @abstractmethod
    def speak(self, text: str):
        """Speak the given text aloud."""
        ...

    @abstractmethod
    def stop(self):
        """Stop any ongoing speech."""
        ...


class PiperTTSEngine(TTSEngine):
    """
    Piper TTS — local neural text-to-speech.
    
    Fast enough for real-time on Jetson. Natural-sounding voices.
    Download voice models from: https://github.com/rhasspy/piper
    
    Recommended voice: en_US-lessac-medium (good quality, fast)
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        try:
            # piper-tts has its own API
            from piper import PiperVoice

            if config.piper_model:
                self._voice = PiperVoice.load(config.piper_model)
                logger.info(f"Piper TTS loaded: {config.piper_model}")
            else:
                raise FileNotFoundError("No Piper model path specified")

        except (ImportError, FileNotFoundError) as e:
            raise ImportError(f"Piper TTS not available: {e}")

    def speak(self, text: str):
        import wave
        import subprocess
        import tempfile

        # Synthesise to WAV, then play
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            with wave.open(f.name, "w") as wav:
                self._voice.synthesize(text, wav)
            # Play via aplay (Linux) or similar
            subprocess.run(["aplay", "-q", f.name], check=False)

        logger.debug(f"TTS spoke: '{text}'")

    def stop(self):
        import subprocess
        subprocess.run(["pkill", "-f", "aplay"], check=False)


class Pyttsx3Engine(TTSEngine):
    """
    pyttsx3 — cross-platform TTS that works everywhere.
    
    Quality is robotic but it's zero-setup and reliable.
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", config.rate)
            self._engine.setProperty("volume", config.volume)
            logger.info("pyttsx3 TTS initialised")
        except Exception as e:
            raise ImportError(f"pyttsx3 not available: {e}")

    def speak(self, text: str):
        self._engine.say(text)
        self._engine.runAndWait()
        logger.debug(f"TTS spoke: '{text}'")

    def stop(self):
        self._engine.stop()


class SimulatedTTS(TTSEngine):
    """Console-only TTS for testing — prints instead of speaking."""

    def __init__(self, config: TTSConfig):
        self.config = config

    def speak(self, text: str):
        print(f"  🔊 ARM SAYS: \"{text}\"")
        logger.debug(f"Simulated TTS: '{text}'")

    def stop(self):
        pass


class TextToSpeech:
    """
    Factory + wrapper for TTS engines.
    
    Also provides convenience methods for common responses
    (ready, error, confirmation, etc.)
    """

    def __init__(self, config: TTSConfig, test_mode: bool = False):
        self.engine: TTSEngine
        self._engine_name: str

        if test_mode:
            self.engine = SimulatedTTS(config)
            self._engine_name = "simulated"
            return

        # Try configured engine
        if config.engine == "piper":
            try:
                self.engine = PiperTTSEngine(config)
                self._engine_name = "piper"
                return
            except ImportError as e:
                logger.warning(f"Piper TTS unavailable: {e}")

        # Try pyttsx3
        try:
            self.engine = Pyttsx3Engine(config)
            self._engine_name = "pyttsx3"
            return
        except ImportError as e:
            logger.warning(f"pyttsx3 unavailable: {e}")

        # Fallback to simulated
        self.engine = SimulatedTTS(config)
        self._engine_name = "simulated"
        logger.warning("TTS: no audio engine available, using console output")

    def speak(self, text: str):
        self.engine.speak(text)

    def stop(self):
        self.engine.stop()

    # === Convenience methods for common responses ===

    def say_ready(self):
        self.speak("Ready. Say 'Hey Arm' followed by your command.")

    def say_listening(self):
        self.speak("Listening.")

    def say_working(self, task_description: str = ""):
        if task_description:
            self.speak(f"{task_description}")
        else:
            self.speak("Working on it.")

    def say_done(self, task_description: str = ""):
        if task_description:
            self.speak(f"Done. {task_description}")
        else:
            self.speak("Done.")

    def say_error(self, message: str = ""):
        self.speak(f"Sorry, something went wrong. {message}".strip())

    def say_not_understood(self):
        self.speak("I didn't catch that. Could you say it again?")

    def say_confirm(self, prompt: str):
        self.speak(prompt)

    def say_cancelled(self):
        self.speak("Cancelled.")

    def say_stopped(self):
        self.speak("Stopped.")

    @property
    def engine_name(self) -> str:
        return self._engine_name
