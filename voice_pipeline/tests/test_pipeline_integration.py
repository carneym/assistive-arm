"""
Integration test for the full voice pipeline.

Runs the pipeline in test mode (no hardware, no ML models)
and verifies the state machine transitions and command handling.

Run: python voice_pipeline/tests/test_pipeline_integration.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from voice_pipeline.config.settings import PipelineConfig
from voice_pipeline.core.intent_parser import IntentParser, Action, Intent
from voice_pipeline.core.stt import TranscriptResult


def get_config() -> PipelineConfig:
    config = PipelineConfig()
    config.test_mode = True
    return config


def test_full_pipeline_state_machine():
    """Test that the pipeline runs through a simulated command sequence."""
    from voice_pipeline.main import VoicePipeline, PipelineState

    config = get_config()
    pipeline = VoicePipeline(config)

    # Verify initial state
    assert pipeline.state == PipelineState.IDLE
    print("  ✓ Initial state: IDLE")

    # Test: wake word + pick up command
    pipeline._handle_simulated_command("hey arm pick up my phone")
    assert pipeline.state == PipelineState.IDLE, \
        f"Expected IDLE after execution, got {pipeline.state}"
    print("  ✓ 'hey arm pick up my phone' → executed → back to IDLE")

    # Test: stop command (always active)
    pipeline._handle_simulated_command("stop")
    assert pipeline.state == PipelineState.IDLE
    print("  ✓ 'stop' → IDLE")

    # Test: command requiring confirmation (scratch)
    pipeline._handle_simulated_command("hey arm scratch my chin")
    assert pipeline.state == PipelineState.CONFIRMING, \
        f"Expected CONFIRMING for scratch, got {pipeline.state}"
    print("  ✓ 'scratch my chin' → CONFIRMING (safety)")

    # Test: confirm with "yes"
    pipeline._handle_simulated_command("yes")
    assert pipeline.state == PipelineState.IDLE
    print("  ✓ 'yes' → executed → IDLE")

    # Test: command requiring confirmation, then cancel
    pipeline._handle_simulated_command("hey arm scratch my nose")
    assert pipeline.state == PipelineState.CONFIRMING
    pipeline._handle_simulated_command("no")
    assert pipeline.state == PipelineState.IDLE
    print("  ✓ 'scratch my nose' → CONFIRMING → 'no' → IDLE (cancelled)")

    # Test: help command
    pipeline._handle_simulated_command("hey arm what can you do")
    assert pipeline.state == PipelineState.IDLE
    print("  ✓ 'help' → responded → IDLE")


def test_sequential_commands():
    """Test a realistic sequence of daily-use commands."""
    from voice_pipeline.main import VoicePipeline, PipelineState

    config = get_config()
    pipeline = VoicePipeline(config)

    sequence = [
        ("hey arm pick up my phone", PipelineState.IDLE),      # Direct execute
        ("put it down", PipelineState.IDLE),                    # Follow-up
        ("hey arm get my cup", PipelineState.IDLE),             # Another pick
        ("drink", PipelineState.CONFIRMING),                    # Bring-to-mouth needs confirm
        ("yes", PipelineState.IDLE),                            # Confirm
        ("put it back", PipelineState.IDLE),                    # Put down
        ("hey arm press the button", PipelineState.IDLE),       # Press
        ("hey arm turn the page", PipelineState.IDLE),          # Turn page
    ]

    for command, expected_state in sequence:
        pipeline._handle_simulated_command(command)
        assert pipeline.state == expected_state, \
            f"After '{command}': expected {expected_state}, got {pipeline.state}"
        print(f"  ✓ '{command}' → {pipeline.state.value}")


def test_edge_cases():
    """Test edge cases and unusual inputs."""
    parser = IntentParser(get_config().intent)

    # Empty / garbage input
    intent = parser.parse("")
    assert intent.action == Action.UNKNOWN
    print("  ✓ Empty string → UNKNOWN")

    intent = parser.parse("   ")
    assert intent.action == Action.UNKNOWN
    print("  ✓ Whitespace only → UNKNOWN")

    intent = parser.parse("asdfghjkl")
    assert intent.action == Action.UNKNOWN
    print("  ✓ Random letters → UNKNOWN")

    # Multiple actions in one sentence (should pick highest confidence)
    intent = parser.parse("stop and pick up my phone")
    assert intent.action == Action.STOP, \
        f"'stop and pick up' should prioritise STOP, got {intent.action}"
    print("  ✓ 'stop and pick up' → STOP (safety priority)")

    # Polite variations
    intent = parser.parse("hey arm could you please pick up my cup")
    assert intent.action == Action.PICK_UP
    assert intent.target == "cup"
    print("  ✓ Polite form parsed correctly")

    # Whisper artefacts (common transcription noise)
    intent = parser.parse("hey arm pick up my phone.")
    assert intent.action == Action.PICK_UP
    print("  ✓ Trailing period handled")

    intent = parser.parse("Hey Arm, Pick Up My Phone!")
    assert intent.action == Action.PICK_UP
    assert intent.target == "phone"
    print("  ✓ Mixed case + punctuation handled")

    # Demonstrative reference ("that one")
    intent = parser.parse("pick up that")
    assert intent.action == Action.PICK_UP
    assert intent.raw_target == "that"
    print(f"  ✓ 'pick up that' → needs visual disambiguation (raw_target='that')")


def test_object_coverage():
    """Verify all common household objects are recognised."""
    parser = IntentParser(get_config().intent)

    objects_to_test = {
        "phone": ["phone", "mobile", "iphone", "cellphone"],
        "cup": ["cup", "mug", "glass", "coffee", "tea", "water"],
        "remote": ["remote", "tv remote", "clicker", "controller"],
        "keys": ["keys", "key"],
        "pen": ["pen", "pencil"],
        "tissue": ["tissue", "tissues", "napkin"],
        "bottle": ["bottle"],
        "book": ["book"],
    }

    for canonical, aliases in objects_to_test.items():
        for alias in aliases:
            intent = parser.parse(f"pick up the {alias}")
            assert intent.target == canonical, \
                f"'pick up the {alias}' → expected '{canonical}', got '{intent.target}'"
        print(f"  ✓ {canonical}: {len(aliases)} aliases all resolve correctly")


def test_confirmation_prompts():
    """Test that confirmation prompts are generated correctly."""
    parser = IntentParser(get_config().intent)

    # Scratch
    intent = parser.parse("scratch my chin")
    prompt = parser.get_confirmation_prompt(intent)
    assert "chin" in prompt.lower()
    assert "yes" in prompt.lower() or "no" in prompt.lower()
    print(f"  ✓ Scratch prompt: '{prompt}'")

    # Bring to mouth
    intent = Intent(action=Action.BRING_TO_MOUTH, target="cup", requires_confirmation=True)
    prompt = parser.get_confirmation_prompt(intent)
    assert "cup" in prompt.lower()
    print(f"  ✓ Bring-to-mouth prompt: '{prompt}'")

    # Handover
    intent = Intent(
        action=Action.HANDOVER, target=None,
        modifiers={"person": "Mary"}, requires_confirmation=True,
    )
    prompt = parser.get_confirmation_prompt(intent)
    assert "mary" in prompt.lower()
    print(f"  ✓ Handover prompt: '{prompt}'")


if __name__ == "__main__":
    test_funcs = [
        ("Full pipeline state machine", test_full_pipeline_state_machine),
        ("Sequential daily-use commands", test_sequential_commands),
        ("Edge cases & unusual input", test_edge_cases),
        ("Object coverage", test_object_coverage),
        ("Confirmation prompts", test_confirmation_prompts),
    ]

    print("\n" + "=" * 60)
    print("  SO-101 VOICE PIPELINE — INTEGRATION TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, func in test_funcs:
        print(f"\n--- {name} ---")
        try:
            func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")

    sys.exit(1 if failed > 0 else 0)
