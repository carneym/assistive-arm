"""
Tests for the intent parser.

These tests verify command parsing logic without needing
any ML models, audio hardware, or network access.

Run: python -m pytest voice_pipeline/tests/test_intent_parser.py -v
  or: python voice_pipeline/tests/test_intent_parser.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from voice_pipeline.core.intent_parser import IntentParser, Intent, Action
from voice_pipeline.config.settings import IntentConfig


def get_parser() -> IntentParser:
    return IntentParser(IntentConfig())


def test_pick_up_commands():
    """Test various ways to say 'pick up'."""
    parser = get_parser()

    test_cases = [
        ("pick up my phone", Action.PICK_UP, "phone"),
        ("grab the cup", Action.PICK_UP, "cup"),
        ("get my remote", Action.PICK_UP, "remote"),
        ("take the keys", Action.PICK_UP, "keys"),
        ("fetch my water", Action.PICK_UP, "cup"),
        ("hand me my phone", Action.PICK_UP, "phone"),
        ("give me the pen", Action.PICK_UP, "pen"),
    ]

    for text, expected_action, expected_target in test_cases:
        intent = parser.parse(text)
        assert intent.action == expected_action, \
            f"'{text}' → expected {expected_action}, got {intent.action}"
        assert intent.target == expected_target, \
            f"'{text}' → expected target '{expected_target}', got '{intent.target}'"
        assert intent.confidence > 0.7, \
            f"'{text}' → confidence {intent.confidence} too low"
        print(f"  ✓ '{text}' → {intent.action.value} ({intent.target})")


def test_put_down_commands():
    parser = get_parser()

    test_cases = [
        ("put it down", Action.PUT_DOWN),
        ("release", Action.PUT_DOWN),
        ("drop it", Action.PUT_DOWN),
        ("let go", Action.PUT_DOWN),
        ("put it back", Action.PUT_DOWN),
        ("place it down", Action.PUT_DOWN),
    ]

    for text, expected_action in test_cases:
        intent = parser.parse(text)
        assert intent.action == expected_action, \
            f"'{text}' → expected {expected_action}, got {intent.action}"
        print(f"  ✓ '{text}' → {intent.action.value}")


def test_safety_commands():
    """Test that stop/cancel are always high confidence."""
    parser = get_parser()

    for text in ["stop", "freeze", "halt", "cancel", "never mind", "abort"]:
        intent = parser.parse(text)
        assert intent.action in (Action.STOP, Action.CANCEL), \
            f"'{text}' → expected STOP/CANCEL, got {intent.action}"
        assert intent.confidence >= 0.9, \
            f"'{text}' → safety command confidence {intent.confidence} < 0.9"
        print(f"  ✓ '{text}' → {intent.action.value} (conf={intent.confidence:.2f})")


def test_bring_to_mouth():
    parser = get_parser()

    test_cases = [
        "bring it to my mouth",
        "drink",
        "let me drink",
        "take a drink",
        "sip",
    ]

    for text in test_cases:
        intent = parser.parse(text)
        assert intent.action == Action.BRING_TO_MOUTH, \
            f"'{text}' → expected BRING_TO_MOUTH, got {intent.action}"
        print(f"  ✓ '{text}' → {intent.action.value}")


def test_scratch_requires_confirmation():
    parser = get_parser()

    intent = parser.parse("scratch my chin")
    assert intent.action == Action.SCRATCH
    assert intent.target == "chin"
    assert intent.requires_confirmation, "Scratch should require confirmation"
    print(f"  ✓ 'scratch my chin' → requires confirmation ✓")

    intent = parser.parse("scratch my nose")
    assert intent.target == "nose"
    print(f"  ✓ 'scratch my nose' → target='nose' ✓")


def test_wake_word_stripping():
    parser = get_parser()

    # "hey arm" prefix should be stripped
    intent = parser.parse("hey arm pick up my phone")
    assert intent.action == Action.PICK_UP
    assert intent.target == "phone"
    print(f"  ✓ 'hey arm pick up my phone' → wake word stripped ✓")

    # "can you" / "please" should be stripped
    intent = parser.parse("hey arm can you get my cup")
    assert intent.action == Action.PICK_UP
    assert intent.target == "cup"
    print(f"  ✓ 'hey arm can you get my cup' → politeness stripped ✓")


def test_object_aliases():
    parser = get_parser()

    alias_tests = [
        ("grab the mobile", "phone"),
        ("get my coffee", "cup"),
        ("pick up the clicker", "remote"),
        ("grab my iphone", "phone"),
        ("get the tv remote", "remote"),
    ]

    for text, expected_target in alias_tests:
        intent = parser.parse(text)
        assert intent.target == expected_target, \
            f"'{text}' → expected '{expected_target}', got '{intent.target}'"
        print(f"  ✓ '{text}' → target='{intent.target}'")


def test_modifier_extraction():
    parser = get_parser()

    intent = parser.parse("pick up the red cup on the left")
    assert intent.modifiers.get("colour") == "red"
    assert intent.modifiers.get("location") == "left"
    print(f"  ✓ Modifiers: colour={intent.modifiers.get('colour')}, location={intent.modifiers.get('location')}")

    intent = parser.parse("give that to John")
    assert intent.modifiers.get("person") == "john" or intent.modifiers.get("person") == "John"
    print(f"  ✓ Handover person: {intent.modifiers.get('person')}")


def test_confirmation_parsing():
    parser = get_parser()

    assert parser.is_confirmation("yes") is True
    assert parser.is_confirmation("yeah go ahead") is True
    assert parser.is_confirmation("no") is False
    assert parser.is_confirmation("nope cancel") is False
    assert parser.is_confirmation("pick up my phone") is None
    print("  ✓ Confirmation parsing: yes/no/ambiguous all correct")


def test_unknown_commands():
    parser = get_parser()

    intent = parser.parse("what is the meaning of life")
    assert intent.action == Action.UNKNOWN
    assert intent.confidence == 0.0
    print(f"  ✓ Unrecognised → UNKNOWN (conf=0.00)")


def test_press_commands():
    parser = get_parser()

    for text in ["press the button", "turn on the light", "switch off"]:
        intent = parser.parse(text)
        assert intent.action == Action.PRESS, \
            f"'{text}' → expected PRESS, got {intent.action}"
        print(f"  ✓ '{text}' → {intent.action.value}")


if __name__ == "__main__":
    test_funcs = [
        ("Pick-up commands", test_pick_up_commands),
        ("Put-down commands", test_put_down_commands),
        ("Safety commands (stop/cancel)", test_safety_commands),
        ("Bring-to-mouth commands", test_bring_to_mouth),
        ("Scratch (requires confirmation)", test_scratch_requires_confirmation),
        ("Wake word stripping", test_wake_word_stripping),
        ("Object aliases", test_object_aliases),
        ("Modifier extraction", test_modifier_extraction),
        ("Confirmation parsing", test_confirmation_parsing),
        ("Unknown commands", test_unknown_commands),
        ("Press commands", test_press_commands),
    ]

    print("\n" + "=" * 60)
    print("  SO-101 INTENT PARSER — TEST SUITE")
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
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")

    sys.exit(1 if failed > 0 else 0)
