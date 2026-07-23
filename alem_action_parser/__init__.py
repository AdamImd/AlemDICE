"""Side-effect-free canonical ALEM action names and parsing helpers.

This module deliberately lives outside the :mod:`alem` package.  Importing the
package initializes the JAX environment and renderer, which is inappropriate
for offline artifact analysis.  Runtime wrappers re-export these helpers so
there remains one parser implementation and one canonical action vocabulary.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import get_close_matches

CANONICAL_ACTIONS = (
    "Noop",
    "Move West",
    "Move East",
    "Move North",
    "Move South",
    "Do",
    "Sleep",
    "Place Stone",
    "Place Table",
    "Place Furnace",
    "Place Plant",
    "Make Wood Pickaxe",
    "Make Stone Pickaxe",
    "Make Iron Pickaxe",
    "Make Wood Sword",
    "Make Stone Sword",
    "Make Iron Sword",
    "Rest",
    "Descend",
    "Ascend",
    "Make Diamond Pickaxe",
    "Make Diamond Sword",
    "Make Iron Armour",
    "Make Diamond Armour",
    "Shoot Arrow",
    "Make Arrow",
    "Cast Spell",
    "Place Torch",
    "Drink Potion Red",
    "Drink Potion Green",
    "Drink Potion Blue",
    "Drink Potion Pink",
    "Drink Potion Cyan",
    "Drink Potion Yellow",
    "Read Book",
    "Enchant Sword",
    "Enchant Armour",
    "Make Torch",
    "Level Up Dexterity",
    "Level Up Strength",
    "Level Up Intelligence",
    "Enchant Bow",
    "Request Food",
    "Request Drink",
    "Request Wood",
    "Request Stone",
    "Request Iron",
    "Request Coal",
    "Request Diamond",
    "Request Ruby",
    "Request Sapphire",
    "Build Shelter",
    "Build Forge",
    "Build Beacon",
    "Give",
)

_GIVE_EXTRACT_PATTERN = re.compile(
    r"^\s*give\s+(?:to\s+)?(?:agent|teammate)[_\s-]*(\d+)\s*$",
    re.IGNORECASE,
)


def _valid_actions_lower(valid_actions: Iterable[str]) -> dict[str, str]:
    return {action.lower(): action for action in valid_actions}


def validate_action(candidate: str | None, valid_actions: Iterable[str] = CANONICAL_ACTIONS):
    """Return the canonical action name when ``candidate`` is an exact match."""

    if not candidate or not candidate.strip():
        return None
    actions = tuple(valid_actions)
    candidate = candidate.strip()
    if candidate in actions:
        return candidate
    return _valid_actions_lower(actions).get(candidate.lower())


def fuzzy_match_action(
    candidate: str | None,
    valid_actions: Iterable[str] = CANONICAL_ACTIONS,
    cutoff: float = 0.6,
):
    """Return the closest canonical action, or ``None`` below ``cutoff``."""

    if not candidate or not candidate.strip():
        return None
    actions = tuple(valid_actions)
    valid_actions_lower = _valid_actions_lower(actions)
    matches = get_close_matches(
        candidate.strip().lower(),
        list(valid_actions_lower),
        n=1,
        cutoff=cutoff,
    )
    return valid_actions_lower[matches[0]] if matches else None


def _clean_extracted(text: str) -> str:
    return text.strip().strip("\"'`").rstrip(".,;:!?").strip()


def _normalize_give_target(text: str | None):
    if not text:
        return None
    match = _GIVE_EXTRACT_PATTERN.match(text)
    if not match:
        return None
    return f"Give to Agent {int(match.group(1))}"


def _try_extract_strict(text: str | None, valid_actions: Iterable[str]):
    if not text:
        return None
    give = _normalize_give_target(text)
    return give or validate_action(text, valid_actions)


def _try_extract_valid(text: str | None, valid_actions: Iterable[str]):
    if not text:
        return None
    actions = tuple(valid_actions)
    give = _normalize_give_target(text)
    if give:
        return give
    valid = validate_action(text, actions)
    if valid:
        return valid
    if ":" in text:
        before = text.split(":")[0].strip()
        give = _normalize_give_target(before)
        if give:
            return give
        valid = validate_action(before, actions)
        if valid:
            return valid
        fuzzy = fuzzy_match_action(before, actions)
        if fuzzy:
            return fuzzy
    return fuzzy_match_action(text, actions)


def extract_action_multistrategy(
    completion_text: str | None,
    valid_actions: Iterable[str] = CANONICAL_ACTIONS,
):
    """Extract a canonical action from model output using ordered fallbacks."""

    if not completion_text:
        return None
    actions = tuple(valid_actions)

    match = re.search(r"<action>(.*?)</action>", completion_text, re.DOTALL)
    if match:
        result = _try_extract_valid(_clean_extracted(match.group(1)), actions)
        if result:
            return result

    if not match and "<action>" in completion_text:
        match_open = re.search(r"<action>(.*?)(?=<[a-z]|$)", completion_text, re.DOTALL)
        if match_open:
            result = _try_extract_valid(_clean_extracted(match_open.group(1)), actions)
            if result:
                return result

    action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", completion_text, re.IGNORECASE)
    if action_match:
        result = _try_extract_valid(_clean_extracted(action_match.group(1)), actions)
        if result:
            return result

    exact = _try_extract_strict(completion_text.strip(), actions)
    if exact:
        return exact

    if ":" in completion_text and "\n" not in completion_text.strip():
        result = _try_extract_strict(completion_text.split(":")[0].strip(), actions)
        if result:
            return result

    if "<" in completion_text:
        before_tag = completion_text[: completion_text.index("<")].strip()
        if before_tag:
            result = _try_extract_strict(before_tag, actions)
            if result:
                return result

    first_line = completion_text.strip().split("\n")[0].strip()
    if first_line and first_line != completion_text.strip():
        result = _try_extract_strict(first_line, actions)
        if result:
            return result

    return None
