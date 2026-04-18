"""
Configuration for the SO-101 Assistive Arm Voice Pipeline.

All tuneable parameters live here. Override via environment variables
where noted, or edit directly for your setup.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    """Audio capture settings."""
    sample_rate: int = 16000          # Hz — Whisper expects 16kHz
    channels: int = 1                  # Mono
    dtype: str = "int16"               # 16-bit PCM
    chunk_duration_ms: int = 30        # Frame size for VAD (10, 20, or 30ms)
    max_record_seconds: float = 10.0   # Max utterance length after wake word
    silence_timeout_seconds: float = 1.5  # Stop recording after this much silence
    # Child voice mode: adjusts thresholds for higher-pitched, quieter voices
    child_voice_mode: bool = False

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def max_record_samples(self) -> int:
        return int(self.sample_rate * self.max_record_seconds)


@dataclass
class WakeWordConfig:
    """Wake word detection settings."""
    wake_phrase: str = "hey arm"
    # Porcupine access key — get free at https://console.picovoice.ai/
    # Set via: export PORCUPINE_ACCESS_KEY="your-key"
    porcupine_access_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("PORCUPINE_ACCESS_KEY")
    )
    # Path to custom Porcupine .ppn keyword file (if trained)
    porcupine_keyword_path: Optional[str] = None
    # Sensitivity for Porcupine (0.0 to 1.0, higher = more sensitive but more false positives)
    sensitivity: float = 0.5
    # Fallback: simple keyword detection on continuous Whisper transcription
    use_fallback: bool = True


@dataclass
class STTConfig:
    """Speech-to-text (Whisper) settings."""
    # Model size: "tiny" (39M), "base" (74M), "small" (244M), "medium" (769M)
    # "small" recommended for Jetson Orin Nano; "tiny" for quick testing
    model_size: str = "small"
    # Language hint (improves accuracy)
    language: str = "en"
    # Use FP16 on GPU, FP32 on CPU
    fp16: bool = False  # Set True on Jetson with CUDA
    # Device: "cuda" on Jetson, "cpu" on dev machine
    device: str = "cpu"
    # Temperature for decoding (0 = greedy, most deterministic)
    temperature: float = 0.0
    # Suppress short nonsense outputs
    no_speech_threshold: float = 0.6
    # Initial prompt to guide Whisper toward expected vocabulary
    initial_prompt: str = (
        "Hey Arm, pick up my phone. Put it down. "
        "Get my cup. Turn the page. Press the button. "
        "Stop. Move that. Scratch my chin. Give that to someone."
    )


@dataclass
class VADConfig:
    """Voice Activity Detection settings."""
    # Silero VAD threshold (0.0 to 1.0)
    threshold: float = 0.5
    # Minimum speech duration to consider valid (seconds)
    min_speech_duration: float = 0.3
    # Padding around detected speech (seconds)
    speech_pad_ms: int = 300


@dataclass
class IntentConfig:
    """Intent parsing settings."""
    # Minimum confidence to execute without asking for confirmation
    confidence_threshold: float = 0.7
    # Tasks that ALWAYS require verbal confirmation before execution
    confirmation_required_tasks: list = field(default_factory=lambda: [
        "scratch",       # Near-face actions
        "bring_to_mouth", # Near-face actions
        "handover",       # Involves another person
    ])
    # Maximum number of times to ask for clarification before giving up
    max_clarification_attempts: int = 2
    # Use Claude API for complex/ambiguous commands (Phase 6)
    use_llm_fallback: bool = False
    claude_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )


@dataclass
class TTSConfig:
    """Text-to-speech settings."""
    # Use Piper TTS if available, else pyttsx3
    engine: str = "pyttsx3"  # "piper" or "pyttsx3"
    # Piper voice model path (download from https://github.com/rhasspy/piper)
    piper_model: Optional[str] = None
    # Speech rate for pyttsx3 (words per minute)
    rate: int = 175
    # Volume (0.0 to 1.0)
    volume: float = 0.9


@dataclass
class ArmConfig:
    """SO-ARM 101 robotic arm settings."""
    # Enable/disable arm control (set False for testing without hardware)
    enabled: bool = True
    # Serial port for Feetech servo bus
    # Set via: export ARM_SERIAL_PORT="/dev/ttyUSB0"
    # Common ports: /dev/ttyUSB0 (Linux), /dev/cu.usbserial-* (macOS)
    serial_port: Optional[str] = field(
        default_factory=lambda: os.environ.get("ARM_SERIAL_PORT")
    )
    # Baud rate for STS3215 servos (default 1000000)
    baudrate: int = 1000000
    # Movement speed (0-4095, lower = slower/safer)
    default_speed: int = 500


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)

    # Global
    debug: bool = field(
        default_factory=lambda: os.environ.get("DEBUG", "0") == "1"
    )
    test_mode: bool = False  # Set via --test flag

    def summary(self) -> str:
        """Print a human-readable config summary."""
        arm_status = "OFF"
        if self.arm.enabled:
            arm_status = self.arm.serial_port or "auto-detect"
        lines = [
            "=== SO-101 Voice Pipeline Configuration ===",
            f"  Wake word:      '{self.wake_word.wake_phrase}'",
            f"  Porcupine key:  {'set' if self.wake_word.porcupine_access_key else 'NOT SET (using fallback)'}",
            f"  Whisper model:  {self.stt.model_size} ({self.stt.device})",
            f"  Audio:          {self.audio.sample_rate}Hz, {self.audio.channels}ch",
            f"  VAD threshold:  {self.vad.threshold}",
            f"  TTS engine:     {self.tts.engine}",
            f"  Arm control:    {arm_status}",
            f"  LLM fallback:   {'ON' if self.intent.use_llm_fallback else 'OFF'}",
            f"  Test mode:      {'ON' if self.test_mode else 'OFF'}",
            f"  Child voice:    {'ON' if self.audio.child_voice_mode else 'OFF'}",
            f"  Debug:          {'ON' if self.debug else 'OFF'}",
            "============================================",
        ]
        return "\n".join(lines)
