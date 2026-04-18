"""
SO-101 Assistive Arm — Voice Pipeline Main Loop

This is the main orchestrator that ties together:
  Wake Word → VAD → STT → Intent Parser → (Action dispatch) → TTS Feedback

Run:
  python -m voice_pipeline.main          # Live mode (needs mic + dependencies)
  python -m voice_pipeline.main --test   # Test mode (no hardware needed)
"""

import argparse
import logging
import shutil
import sys
import time
from enum import Enum
from typing import Optional

import numpy as np

from .config.settings import PipelineConfig
from .core.audio_capture import AudioCapture, SimulatedAudioSource
from .core.wake_word import WakeWordDetector
from .core.vad import VoiceActivityDetector
from .core.stt import SpeechToText, TranscriptResult
from .core.intent_parser import IntentParser, Intent, Action
from .core.tts import TextToSpeech
from .core.arm_controller import (
    ArmController,
    scan_serial_ports,
    run_arm_diagnostic,
    print_diagnostic_report,
    print_positions,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


class PipelineState(str, Enum):
    """State machine for the voice pipeline."""
    IDLE = "idle"                     # Waiting for wake word
    LISTENING = "listening"           # Recording utterance after wake word
    PROCESSING = "processing"        # Running STT + intent parsing
    CONFIRMING = "confirming"        # Waiting for yes/no confirmation
    EXECUTING = "executing"          # Action in progress
    ERROR = "error"                  # Error state, will recover to IDLE


class LevelMeter:
    """Terminal-based audio level meter."""

    STATE_COLORS = {
        PipelineState.IDLE: "\033[90m",       # Gray
        PipelineState.LISTENING: "\033[92m",  # Green
        PipelineState.PROCESSING: "\033[93m", # Yellow
        PipelineState.CONFIRMING: "\033[94m", # Blue
        PipelineState.EXECUTING: "\033[95m",  # Magenta
        PipelineState.ERROR: "\033[91m",      # Red
    }
    RESET = "\033[0m"
    BAR_CHAR = "█"
    EMPTY_CHAR = "░"

    def __init__(self, width: int = 40, max_rms: float = 8000.0):
        self.width = width
        self.max_rms = max_rms
        self._peak = 0.0
        self._peak_decay = 0.95

    def render(self, audio_chunk: np.ndarray, state: PipelineState) -> str:
        """Render a level meter bar for the given audio chunk."""
        # Calculate RMS level
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

        # Update peak with decay
        self._peak = max(rms, self._peak * self._peak_decay)

        # Normalize to 0-1 range (with log scale for better visualization)
        level = min(1.0, np.log1p(rms) / np.log1p(self.max_rms))
        peak_pos = min(1.0, np.log1p(self._peak) / np.log1p(self.max_rms))

        # Build the bar
        filled = int(level * self.width)
        peak_idx = int(peak_pos * self.width)

        color = self.STATE_COLORS.get(state, self.RESET)
        state_label = f"[{state.value.upper():^11}]"

        bar = ""
        for i in range(self.width):
            if i < filled:
                bar += self.BAR_CHAR
            elif i == peak_idx:
                bar += "|"
            else:
                bar += self.EMPTY_CHAR

        # Get terminal width and truncate if needed
        term_width = shutil.get_terminal_size().columns
        output = f"{color}{state_label}{self.RESET} [{color}{bar}{self.RESET}]"

        return output

    def print(self, audio_chunk: np.ndarray, state: PipelineState):
        """Print the level meter, overwriting the current line."""
        meter = self.render(audio_chunk, state)
        sys.stdout.write(f"\r{meter}")
        sys.stdout.flush()

    def clear(self):
        """Clear the meter line."""
        term_width = shutil.get_terminal_size().columns
        sys.stdout.write("\r" + " " * term_width + "\r")
        sys.stdout.flush()


class VoicePipeline:
    """
    Main voice pipeline for the SO-101 assistive arm.

    State machine:
      IDLE → (wake word) → LISTENING → (silence/timeout) → PROCESSING
        → CONFIRMING → (yes) → EXECUTING → IDLE
        → CONFIRMING → (no) → IDLE
        → (high confidence, no confirm needed) → EXECUTING → IDLE
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = PipelineState.IDLE
        self._pending_intent: Optional[Intent] = None

        logger.info("Initialising voice pipeline components...")
        logger.info(config.summary())

        # Initialise components
        self.audio = AudioCapture(config.audio, test_mode=config.test_mode)
        self.source = self.audio.get_source()
        self.wake_word = WakeWordDetector(
            config.wake_word, config.audio, test_mode=config.test_mode
        )
        self.vad = VoiceActivityDetector(
            config.vad, config.audio, test_mode=config.test_mode
        )
        self.stt = SpeechToText(
            config.stt, config.audio, test_mode=config.test_mode
        )
        self.intent_parser = IntentParser(config.intent)
        self.tts = TextToSpeech(config.tts, test_mode=config.test_mode)

        # Arm controller (for physical arm movements)
        self.arm_controller = ArmController(
            port=config.arm.serial_port,
            enabled=config.arm.enabled and not config.test_mode,
            run_diagnostic=config.arm.enabled and not config.test_mode,
        )

        # Level meter for visual feedback
        self.level_meter = LevelMeter()

        if self.arm_controller.enabled:
            arm_status = "SIMULATION" if self.arm_controller.is_simulation else "CONNECTED"
        else:
            arm_status = "disabled"
        logger.info(
            f"Pipeline ready — wake_word={self.wake_word.engine_name}, "
            f"vad={self.vad.engine_name}, stt={self.stt.engine_name}, "
            f"tts={self.tts.engine_name}, arm={arm_status}"
        )

    def run(self):
        """Main pipeline loop."""
        logger.info("=" * 50)
        logger.info("SO-101 Voice Pipeline Started")
        logger.info(f"Say '{self.config.wake_word.wake_phrase}' to activate")
        logger.info("=" * 50)

        self.tts.say_ready()

        if self.config.test_mode:
            self._run_test_mode()
        else:
            self._run_live_mode()

    def _run_live_mode(self):
        """Run the pipeline with live audio input."""
        try:
            for audio_chunk in self.source.stream_chunks():
                # Show level meter
                self.level_meter.print(audio_chunk, self.state)
                self._process_state(audio_chunk)
        except KeyboardInterrupt:
            self.level_meter.clear()
            logger.info("Pipeline stopped by user (Ctrl+C)")
            self.tts.speak("Shutting down. Goodbye.")
        finally:
            self.level_meter.clear()
            self._cleanup()

    def _run_test_mode(self):
        """
        Run the pipeline in test mode with simulated audio.
        
        Simulates a sequence of wake-word → command cycles.
        """
        logger.info("Running in TEST MODE — simulating voice commands")
        print("\n" + "=" * 60)
        print("  SO-101 VOICE PIPELINE — TEST MODE")
        print("  Simulating voice command sequences")
        print("=" * 60 + "\n")

        # Simulated command sequence
        test_commands = [
            "hey arm pick up my phone",
            "put it down",
            "hey arm get my cup",
            "drink",
            "yes",
            "put it back",
            "hey arm scratch my chin",
            "yes",
            "stop",
            "hey arm what can you do",
        ]

        for i, command in enumerate(test_commands):
            print(f"\n{'─' * 50}")
            print(f"  SIM COMMAND {i+1}: \"{command}\"")
            print(f"{'─' * 50}")

            self._handle_simulated_command(command)
            time.sleep(0.3)  # Brief pause between commands

        print(f"\n{'=' * 60}")
        print("  TEST COMPLETE — All commands processed")
        print(f"{'=' * 60}\n")

    def _handle_simulated_command(self, command_text: str):
        """Process a single simulated command through the full pipeline."""
        # Step 1: Check for wake word
        has_wake = self.wake_word.check_text(command_text)
        if has_wake:
            logger.info("Wake word detected → LISTENING")
            self.state = PipelineState.LISTENING

        # Step 2: Create simulated audio and transcribe
        if self.state == PipelineState.LISTENING or self.state == PipelineState.CONFIRMING:
            # Simulate STT result
            transcript = TranscriptResult(
                text=command_text,
                confidence=-0.3,
                duration_s=2.0,
                processing_time_s=0.1,
                no_speech_prob=0.05,
            )

            if transcript.is_valid:
                self._handle_transcript(transcript)
            else:
                self.tts.say_not_understood()
                self.state = PipelineState.IDLE
        elif self.state == PipelineState.IDLE:
            # Commands without wake word in IDLE — check if it's a continuation
            # (e.g., "put it down" after a pick_up, or a stop command)
            stripped = command_text.lower().strip()

            # Always-active commands (stop, cancel)
            if any(w in stripped for w in ("stop", "cancel", "halt")):
                self.tts.say_stopped()
                self.state = PipelineState.IDLE
                return

            # If not idle-breaking, treat as if wake word was implicit
            self.state = PipelineState.LISTENING
            transcript = TranscriptResult(
                text=command_text,
                confidence=-0.3,
                duration_s=2.0,
                processing_time_s=0.1,
                no_speech_prob=0.05,
            )
            self._handle_transcript(transcript)

    def _process_state(self, audio_chunk: np.ndarray):
        """Process a single audio chunk based on current state."""
        if self.state == PipelineState.IDLE:
            # Listen for wake word
            if self.wake_word.process_frame(audio_chunk):
                logger.info("Wake word detected → LISTENING")
                self.state = PipelineState.LISTENING
                self.tts.say_listening()
                self.vad.reset()
                self._utterance_frames = []

        elif self.state == PipelineState.LISTENING:
            # Collect audio until silence
            self._utterance_frames.append(audio_chunk)
            total_samples = sum(len(f) for f in self._utterance_frames)

            # Check for end of speech or timeout
            is_speech = self.vad.is_speech(audio_chunk)
            duration = total_samples / self.config.audio.sample_rate

            if duration > self.config.audio.max_record_seconds:
                logger.info("Max recording time reached")
                self._process_utterance()
            elif not is_speech and duration > 1.0:
                # Count consecutive silence chunks
                if not hasattr(self, "_silence_count"):
                    self._silence_count = 0
                self._silence_count += 1
                silence_needed = int(
                    self.config.audio.silence_timeout_seconds
                    / (self.config.audio.chunk_duration_ms / 1000)
                )
                if self._silence_count >= silence_needed:
                    self._process_utterance()
            else:
                self._silence_count = 0

        elif self.state == PipelineState.CONFIRMING:
            # Listen for yes/no confirmation response
            if not hasattr(self, "_confirm_frames"):
                self._confirm_frames = []
                self._confirm_silence_count = 0

            self._confirm_frames.append(audio_chunk)
            total_samples = sum(len(f) for f in self._confirm_frames)
            duration = total_samples / self.config.audio.sample_rate

            # Timeout after 5 seconds of waiting
            if duration > 5.0:
                logger.info("Confirmation timeout")
                self.tts.speak("I didn't hear a response. Cancelling.")
                self._pending_intent = None
                self._confirm_frames = []
                self.state = PipelineState.IDLE
                return

            # Check for end of speech
            is_speech = self.vad.is_speech(audio_chunk)
            if not is_speech and duration > 0.3:
                self._confirm_silence_count += 1
                silence_needed = int(1.0 / (self.config.audio.chunk_duration_ms / 1000))
                if self._confirm_silence_count >= silence_needed:
                    self._process_confirmation()
            else:
                self._confirm_silence_count = 0

    def _process_confirmation(self):
        """Process the confirmation response (yes/no)."""
        frames = getattr(self, "_confirm_frames", [])
        self._confirm_frames = []
        self._confirm_silence_count = 0

        if not frames:
            self.tts.say_not_understood()
            self.state = PipelineState.IDLE
            return

        # Segment and transcribe
        audio = self.vad.segment_utterance(frames)
        if audio is None:
            logger.info("No speech in confirmation response")
            self.tts.speak("I didn't catch that. Please try again.")
            self._confirm_frames = []
            return  # Stay in CONFIRMING state

        transcript = self.stt.transcribe(audio)
        logger.info(f"Confirmation response: '{transcript.text}'")

        if not transcript.is_valid:
            self.tts.speak("I didn't understand. Say yes or no.")
            return  # Stay in CONFIRMING state

        # Check for yes/no
        confirmation = self.intent_parser.is_confirmation(transcript.text)
        if confirmation is True:
            logger.info("Confirmed → EXECUTING")
            self._execute_intent(self._pending_intent)
        elif confirmation is False:
            logger.info("Declined → IDLE")
            self.tts.say_cancelled()
            self._pending_intent = None
            self.state = PipelineState.IDLE
        else:
            # Not a clear yes/no - ask again
            self.tts.speak("Please say yes or no.")
            # Stay in CONFIRMING state

    def _process_utterance(self):
        """Process collected audio through STT and intent parsing."""
        self.state = PipelineState.PROCESSING
        self._silence_count = 0

        # Segment speech from collected frames
        audio = self.vad.segment_utterance(self._utterance_frames)
        if audio is None:
            logger.info("No speech detected in utterance")
            self.tts.say_not_understood()
            self.state = PipelineState.IDLE
            return

        # Transcribe
        transcript = self.stt.transcribe(audio)
        if not transcript.is_valid:
            self.tts.say_not_understood()
            self.state = PipelineState.IDLE
            return

        self._handle_transcript(transcript)

    def _handle_transcript(self, transcript: TranscriptResult):
        """Handle a valid transcript — parse intent and decide action."""
        logger.info(f"Transcript: '{transcript.text}'")

        # If we're in confirmation state, check for yes/no
        if self.state == PipelineState.CONFIRMING and self._pending_intent:
            confirmation = self.intent_parser.is_confirmation(transcript.text)
            if confirmation is True:
                logger.info("Confirmed → EXECUTING")
                self._execute_intent(self._pending_intent)
            elif confirmation is False:
                self.tts.say_cancelled()
                self._pending_intent = None
                self.state = PipelineState.IDLE
            else:
                # Not a clear yes/no — try parsing as new command
                self._pending_intent = None
                self._parse_and_dispatch(transcript)
            return

        self._parse_and_dispatch(transcript)

    def _parse_and_dispatch(self, transcript: TranscriptResult):
        """Parse intent and either execute or request confirmation."""
        intent = self.intent_parser.parse(transcript.text)

        # Handle immediate commands
        if intent.action == Action.STOP:
            self.tts.say_stopped()
            self.state = PipelineState.IDLE
            return

        if intent.action == Action.CANCEL:
            self.tts.say_cancelled()
            self.state = PipelineState.IDLE
            return

        if intent.action == Action.HELP:
            self._say_help()
            self.state = PipelineState.IDLE
            return

        if intent.action == Action.STATUS:
            self.tts.speak(f"I'm in {self.state.value} state. Ready for commands.")
            self.state = PipelineState.IDLE
            return

        if intent.action == Action.UNKNOWN:
            self.tts.say_not_understood()
            self.state = PipelineState.IDLE
            return

        # Check confidence
        if intent.confidence < self.config.intent.confidence_threshold:
            self.tts.speak(
                f"I think you want me to {intent.action.value.replace('_', ' ')}"
                f"{' the ' + intent.target if intent.target else ''}. Is that right?"
            )
            self._pending_intent = intent
            self._confirm_frames = []
            self._confirm_silence_count = 0
            self.vad.reset()
            self.state = PipelineState.CONFIRMING
            logger.info("Waiting for confirmation...")
            return

        # Check if confirmation required (safety)
        if intent.requires_confirmation:
            prompt = self.intent_parser.get_confirmation_prompt(intent)
            self.tts.say_confirm(prompt)
            self._pending_intent = intent
            self._confirm_frames = []
            self._confirm_silence_count = 0
            self.vad.reset()
            self.state = PipelineState.CONFIRMING
            logger.info("Waiting for confirmation...")
            return

        # High confidence, no confirmation needed → execute
        self._execute_intent(intent)

    def _execute_intent(self, intent: Intent):
        """
        Execute an intent by dispatching to the arm controller.
        """
        self.state = PipelineState.EXECUTING

        # Build description of what the arm will do
        action_descriptions = {
            Action.PICK_UP: f"Picking up {intent.target or 'the object'}.",
            Action.PUT_DOWN: "Putting it down.",
            Action.BRING_TO_MOUTH: f"Bringing the {intent.target or 'object'} to your mouth.",
            Action.MOVE: f"Moving {intent.target or 'that'} out of the way.",
            Action.PRESS: f"Pressing the {intent.target or 'button'}.",
            Action.TURN_PAGE: "Turning the page.",
            Action.SCRATCH: f"Scratching your {intent.target or 'chin'}.",
            Action.HANDOVER: f"Handing it to {intent.modifiers.get('person', 'them')}.",
            Action.OPEN: f"Opening the {intent.target or 'object'}.",
            Action.GO_HOME: "Returning to home position.",
            # Preprogrammed arm movements
            Action.WAVE: "Waving hello!",
            Action.RAISE_ARM: "Raising arm.",
            Action.LOWER_ARM: "Lowering arm.",
        }

        description = action_descriptions.get(
            intent.action, f"Executing {intent.action.value}."
        )

        logger.info(f"EXECUTING: {intent}")
        self.tts.say_working(description)

        # Execute via arm controller
        result = self.arm_controller.execute(intent)
        if result.success:
            logger.info(f"Arm execution: {result.message}")
        else:
            logger.warning(f"Arm execution failed: {result.error}")

        self.tts.say_done("Ready for next command.")
        self.state = PipelineState.IDLE
        self._pending_intent = None

    def _say_help(self):
        """List available commands."""
        self.tts.speak(
            "I can pick up objects, put them down, bring a cup to your mouth, "
            "press buttons, turn pages, scratch an itch, move things aside, "
            "wave hello, raise my arm, or hand objects to someone. "
            "Just say Hey Arm and tell me what you need."
        )

    def _cleanup(self):
        """Clean up resources."""
        self.wake_word.close()
        self.source.close()
        self.arm_controller.close()


def main():
    parser = argparse.ArgumentParser(
        description="SO-101 Assistive Arm — Voice Pipeline"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run in test mode (no mic/hardware needed)"
    )
    parser.add_argument(
        "--whisper-model", default="small",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (default: small)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--child-voice", action="store_true",
        help="Enable child voice mode (lower thresholds, child-specific pronunciations)"
    )
    parser.add_argument(
        "--no-arm", action="store_true",
        help="Disable arm control (voice-only mode for testing)"
    )
    parser.add_argument(
        "--arm-port", type=str, default=None,
        help="Serial port for arm (e.g., /dev/ttyUSB0 or /dev/cu.usbserial-*)"
    )
    parser.add_argument(
        "--scan-ports", action="store_true",
        help="Scan for available serial ports and test for arm, then exit"
    )
    parser.add_argument(
        "--test-arm", action="store_true",
        help="Run arm diagnostic test and exit"
    )
    parser.add_argument(
        "--read-positions", action="store_true",
        help="Read and display current servo positions, then exit"
    )
    args = parser.parse_args()

    # Handle --scan-ports: list all serial ports and exit
    if args.scan_ports:
        print("\n" + "=" * 50)
        print("  Available Serial Ports")
        print("=" * 50 + "\n")
        ports = scan_serial_ports()
        if not ports:
            print("  No serial ports found.\n")
            print("  Make sure:")
            print("    - USB cable is connected")
            print("    - USB-serial driver is installed")
            print("    - You have permission to access serial ports\n")
        else:
            for p in ports:
                print(f"  {p['port']}")
                print(f"    Description: {p['description']}")
                print(f"    Hardware ID: {p['hwid']}\n")
        print("=" * 50 + "\n")
        sys.exit(0)

    # Handle --test-arm: run diagnostic and exit
    if args.test_arm:
        result = run_arm_diagnostic(args.arm_port)
        print_diagnostic_report(result)
        sys.exit(0 if result.all_ok else 1)

    # Handle --read-positions: show current servo positions and exit
    if args.read_positions:
        print_positions(args.arm_port)
        sys.exit(0)

    # Configure
    config = PipelineConfig()
    config.test_mode = args.test
    config.stt.model_size = args.whisper_model
    if args.debug:
        config.debug = True
        logging.getLogger().setLevel(logging.DEBUG)
    if args.child_voice:
        config.audio.child_voice_mode = True
        logger.info("Child voice mode enabled")
    if args.no_arm:
        config.arm.enabled = False
        logger.info("Arm control disabled (voice-only mode)")
    if args.arm_port:
        config.arm.serial_port = args.arm_port
        logger.info(f"Arm serial port: {args.arm_port}")

    # Run
    pipeline = VoicePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
