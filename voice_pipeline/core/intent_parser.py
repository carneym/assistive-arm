"""
Intent parser for the SO-101 assistive arm.

Converts transcribed text into structured commands the arm can execute.

Architecture:
  Tier 1 (fast, local): Pattern matching with confidence scoring.
    Handles ~80% of daily commands with zero latency.
  
  Tier 2 (optional, Phase 6): LLM call (Claude API) for ambiguous
    or novel commands that don't match any pattern.

Each intent has:
  - action: what to do (pick_up, put_down, bring_to_mouth, etc.)
  - target: what object (phone, cup, etc.) — may be None for stop/cancel
  - modifiers: extra info (location hints, urgency, etc.)
  - confidence: how sure the parser is (0.0 to 1.0)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..config.settings import IntentConfig

logger = logging.getLogger(__name__)


class Action(str, Enum):
    """Supported arm actions."""
    PICK_UP = "pick_up"
    PUT_DOWN = "put_down"
    BRING_TO_MOUTH = "bring_to_mouth"
    MOVE = "move"
    PRESS = "press"
    TURN_PAGE = "turn_page"
    SCRATCH = "scratch"
    HANDOVER = "handover"
    OPEN = "open"
    STOP = "stop"
    CANCEL = "cancel"
    GO_HOME = "go_home"
    STATUS = "status"
    HELP = "help"
    # Preprogrammed arm movements
    WAVE = "wave"
    RAISE_ARM = "raise_arm"
    LOWER_ARM = "lower_arm"
    UNKNOWN = "unknown"


# Common object aliases → canonical names
OBJECT_ALIASES = {
    # Phone
    "phone": "phone", "mobile": "phone", "cell": "phone", "cellphone": "phone",
    "iphone": "phone", "smartphone": "phone", "my phone": "phone",
    # Cup/Drink
    "cup": "cup", "mug": "cup", "glass": "cup", "drink": "cup",
    "water": "cup", "coffee": "cup", "tea": "cup", "bottle": "bottle",
    # Remote
    "remote": "remote", "tv remote": "remote", "controller": "remote",
    "clicker": "remote",
    # Food
    "snack": "snack", "food": "food", "sandwich": "food",
    "bag": "snack_bag", "crisp bag": "snack_bag", "chip bag": "snack_bag",
    # Book
    "book": "book", "page": "book", "kindle": "kindle", "tablet": "tablet",
    # Other common objects
    "keys": "keys", "key": "keys",
    "pen": "pen", "pencil": "pen",
    "tissue": "tissue", "tissues": "tissue", "napkin": "tissue",
    "utensil": "utensil", "fork": "utensil", "spoon": "utensil",
    "knife": "utensil",
    "plate": "plate", "bowl": "bowl",
    # Body parts (for scratch)
    "chin": "chin", "nose": "nose", "cheek": "cheek",
    "forehead": "forehead", "ear": "ear",
    # Switches
    "button": "button", "switch": "switch", "light": "light_switch",
    "lamp": "light_switch",
}

# Intent patterns: (regex, action, confidence, optional object extractor)
INTENT_PATTERNS = [
    # === HIGH PRIORITY: Safety commands ===
    (r"\b(stop|freeze|halt|don'?t|wait)\b", Action.STOP, 0.95),
    (r"\b(cancel|never\s?mind|abort)\b", Action.CANCEL, 0.95),
    (r"\b(go\s+home|home\s+position|rest)\b", Action.GO_HOME, 0.90),

    # === PREPROGRAMMED ARM MOVEMENTS ===
    (r"\b(wave|wave\s+hello|say\s+hi|wave\s+hi|hello)\b", Action.WAVE, 0.90),
    (r"\b(raise\s+(your\s+)?arm|arm\s+up|lift\s+(your\s+)?arm|raise\s+up)\b", Action.RAISE_ARM, 0.90),
    (r"\b(lower\s+(your\s+)?arm|arm\s+down|put\s+(your\s+)?arm\s+down)\b", Action.LOWER_ARM, 0.90),

    # === BRING TO MOUTH (drink/eat) — checked BEFORE pick_up to win on "take a drink" ===
    (r"\b(bring\s+(it\s+)?to\s+m(y|e)\s+(mouth|face|lips))\b", Action.BRING_TO_MOUTH, 0.92),
    (r"\b(take\s+a\s+(drink|sip))\b", Action.BRING_TO_MOUTH, 0.90),
    (r"\b(drink|sip)\b", Action.BRING_TO_MOUTH, 0.88),
    (r"\b(let\s+me\s+(drink|sip|eat))\b", Action.BRING_TO_MOUTH, 0.88),

    # === PICK UP ===
    (r"\b(pick\s+up|grab|get|fetch)\b", Action.PICK_UP, 0.85),
    (r"\b(take)\b(?!\s+a\s+(drink|sip))", Action.PICK_UP, 0.80),  # "take" but NOT "take a drink"
    (r"\b(hand\s+me|give\s+me|pass\s+me)\b", Action.PICK_UP, 0.80),

    # === PUT DOWN ===
    (r"\b(put\s+(it\s+)?down|release|drop|let\s+go|place)\b", Action.PUT_DOWN, 0.90),
    (r"\b(put\s+(it\s+)?back)\b", Action.PUT_DOWN, 0.85),

    # === MOVE ===
    (r"\b(move|push|slide|shift)\b.*\b(out\s+of\s+the\s+way|aside|over)\b", Action.MOVE, 0.85),
    (r"\b(move|push|slide)\b", Action.MOVE, 0.70),

    # === PRESS ===
    (r"\b(press|push|tap|hit)\s+(the\s+)?(button|switch)\b", Action.PRESS, 0.90),
    (r"\b(turn\s+on|turn\s+off|switch\s+on|switch\s+off)\b", Action.PRESS, 0.85),

    # === TURN PAGE ===
    (r"\b(turn|flip)\s+(the\s+)?page\b", Action.TURN_PAGE, 0.90),
    (r"\b(next\s+page)\b", Action.TURN_PAGE, 0.90),

    # === SCRATCH ===
    (r"\b(scratch|itch|rub)\b", Action.SCRATCH, 0.85),

    # === HANDOVER ===
    (r"\b(give\s+(that|it|this)\s+to)\b", Action.HANDOVER, 0.85),
    (r"\b(hand\s+(it\s+)?over\s+to)\b", Action.HANDOVER, 0.85),

    # === OPEN ===
    (r"\b(open)\b", Action.OPEN, 0.75),

    # === STATUS/HELP ===
    (r"\b(status|what\s+are\s+you\s+doing|where\s+are\s+you)\b", Action.STATUS, 0.85),
    (r"\b(help|what\s+can\s+you\s+do)\b", Action.HELP, 0.85),
]


@dataclass
class Intent:
    """Parsed intent from a voice command."""
    action: Action
    target: Optional[str] = None       # Canonical object name
    raw_target: Optional[str] = None   # Original text for the target
    modifiers: dict = field(default_factory=dict)  # Extra context
    confidence: float = 0.0
    raw_text: str = ""                 # Full original transcript
    requires_confirmation: bool = False

    def __str__(self) -> str:
        parts = [f"Intent({self.action.value}"]
        if self.target:
            parts.append(f", target='{self.target}'")
        if self.modifiers:
            parts.append(f", mods={self.modifiers}")
        parts.append(f", conf={self.confidence:.2f})")
        return "".join(parts)


class IntentParser:
    """
    Rule-based intent parser for voice commands.
    
    Extracts (action, target, modifiers) from transcribed text.
    Falls back to UNKNOWN for unrecognised commands.
    """

    def __init__(self, config: IntentConfig):
        self.config = config

    def parse(self, text: str) -> Intent:
        """
        Parse a transcribed command into a structured Intent.
        
        Steps:
          1. Normalise text (lowercase, strip punctuation)
          2. Strip wake word if present
          3. Match against intent patterns (highest confidence wins)
          4. Extract target object
          5. Check if confirmation is required
        """
        raw_text = text
        text = self._normalise(text)
        text = self._strip_wake_word(text)

        if not text:
            return Intent(
                action=Action.UNKNOWN,
                confidence=0.0,
                raw_text=raw_text,
            )

        # Find best matching pattern
        best_action = Action.UNKNOWN
        best_confidence = 0.0
        best_match = None

        for pattern, action, base_confidence in INTENT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and base_confidence > best_confidence:
                best_action = action
                best_confidence = base_confidence
                best_match = match

        # Extract target object
        target, raw_target = self._extract_target(text, best_action)

        # Boost confidence if we found a known target
        if target and best_confidence > 0:
            best_confidence = min(best_confidence + 0.05, 1.0)

        # Extract modifiers
        modifiers = self._extract_modifiers(text)

        # Check if confirmation is required
        requires_confirmation = (
            best_action.value in self.config.confirmation_required_tasks
        )

        intent = Intent(
            action=best_action,
            target=target,
            raw_target=raw_target,
            modifiers=modifiers,
            confidence=best_confidence,
            raw_text=raw_text,
            requires_confirmation=requires_confirmation,
        )

        logger.info(f"Parsed: '{raw_text}' → {intent}")
        return intent

    def _normalise(self, text: str) -> str:
        """Normalise text for matching."""
        text = text.lower().strip()
        # Remove punctuation but keep apostrophes
        text = re.sub(r"[^\w\s']", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text

    def _strip_wake_word(self, text: str) -> str:
        """Remove the wake word from the beginning of the command."""
        wake_phrases = ["hey arm", "hey arm,", "arm,", "arm"]
        for phrase in wake_phrases:
            if text.startswith(phrase):
                text = text[len(phrase):].strip()
                # Also strip "can you" / "could you" / "please" niceties
                text = re.sub(r"^(can you|could you|please|would you)\s+", "", text)
                break
        return text

    def _extract_target(self, text: str, action: Action) -> tuple[Optional[str], Optional[str]]:
        """
        Extract the target object from the command text.
        Returns (canonical_name, raw_text) or (None, None).
        """
        # For body-part actions (scratch), look for body parts
        if action == Action.SCRATCH:
            for alias, canonical in OBJECT_ALIASES.items():
                if canonical in ("chin", "nose", "cheek", "forehead", "ear"):
                    if alias in text:
                        return canonical, alias

        # For other actions, look for object references
        # Try longest match first (e.g., "tv remote" before "remote")
        sorted_aliases = sorted(OBJECT_ALIASES.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if alias in text:
                canonical = OBJECT_ALIASES[alias]
                return canonical, alias

        # Check for demonstrative references ("that", "this", "it")
        if re.search(r"\b(that|this|it|the one)\b", text):
            return None, "that"  # Target needs visual disambiguation

        return None, None

    def _extract_modifiers(self, text: str) -> dict:
        """Extract additional context from the command."""
        modifiers = {}

        # Location hints
        location_match = re.search(
            r"\bon\s+(?:the\s+)?(left|right|desk|table|front|back)", text
        )
        if location_match:
            modifiers["location"] = location_match.group(1)

        # Colour hints (for disambiguation)
        colour_match = re.search(
            r"\b(red|blue|green|black|white|silver|grey|gray|yellow)\b", text
        )
        if colour_match:
            modifiers["colour"] = colour_match.group(1)

        # Size hints
        size_match = re.search(r"\b(big|small|large|little|tall|short)\b", text)
        if size_match:
            modifiers["size"] = size_match.group(1)

        # Person name (for handover)
        name_match = re.search(r"\bgive\s+(?:that|it)\s+to\s+(\w+)", text)
        if name_match:
            modifiers["person"] = name_match.group(1)

        return modifiers

    def get_confirmation_prompt(self, intent: Intent) -> str:
        """Generate a confirmation prompt for high-risk actions."""
        if intent.action == Action.SCRATCH:
            return f"I'll scratch your {intent.target or 'face area'}. Shall I go ahead? Say yes or no."
        elif intent.action == Action.BRING_TO_MOUTH:
            return f"I'll bring the {intent.target or 'object'} to your mouth. Ready? Say yes or no."
        elif intent.action == Action.HANDOVER:
            person = intent.modifiers.get("person", "them")
            return f"I'll hand this to {person}. Shall I proceed? Say yes or no."
        else:
            return f"I'll {intent.action.value.replace('_', ' ')} the {intent.target or 'object'}. OK?"

    def is_confirmation(self, text: str) -> Optional[bool]:
        """Check if text is a yes/no confirmation. Returns True/False/None."""
        text = text.lower().strip()
        if re.search(r"\b(yes|yeah|yep|go ahead|do it|sure|ok|okay|affirmative)\b", text):
            return True
        if re.search(r"\b(no|nope|nah|don'?t|cancel|stop|never\s?mind)\b", text):
            return False
        return None
