"""Minimal contract test for model-action extraction."""

from alem.llm.action_parser import extract_action_multistrategy


def test_action_parser_keeps_supported_paths_and_rejects_noise():
    actions = (
        "Noop",
        "Move North",
        "Move South",
        "Move East",
        "Move West",
        "Do",
    )
    cases = {
        "Move North": "Move North",
        "<action>move south</action>": "Move South",
        "ACTION: Move East": "Move East",
        "<action>give agent 2</action>": "Give to Agent 2",
        "<action>Move Norht</action>": "Move North",
        "unrelated prose with no executable command": None,
    }

    for completion, expected in cases.items():
        assert extract_action_multistrategy(completion, actions) == expected
