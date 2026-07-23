"""Compatibility re-export for the side-effect-free ALEM action parser."""

from alem_action_parser import (
    CANONICAL_ACTIONS,
    extract_action_multistrategy,
    fuzzy_match_action,
    validate_action,
)

__all__ = [
    "CANONICAL_ACTIONS",
    "extract_action_multistrategy",
    "fuzzy_match_action",
    "validate_action",
]
