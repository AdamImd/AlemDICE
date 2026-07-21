"""Focused tests for the bodyless team-leader protocol."""

import json

import pytest

from baselines.llm.eval_utils.client import LLMResponse
from baselines.llm.eval_utils.team_leader import (
    CommunicationTracker,
    TeamLeaderAgent,
    fallback_plan,
    format_assignment,
    leader_system_prompt,
    parse_team_plan,
)


def _plan_payload(ids=(0, 1, 2)):
    return {
        "objective": "Build tools and survive",
        "assignments": [
            {"agent_id": agent_id, "subgoal": f"work {agent_id}", "milestones": ["report"]}
            for agent_id in ids
        ],
    }


def _tagged(payload):
    return f"<team_plan>{json.dumps(payload)}</team_plan>"


def test_plan_parser_requires_one_assignment_per_physical_worker():
    parsed = parse_team_plan(_tagged(_plan_payload()))
    assert parsed is not None
    assert [assignment.agent_id for assignment in parsed.assignments] == [0, 1, 2]

    for ids in ((0, 1, 1), (0, 1), (0, 1, 3)):
        assert parse_team_plan(_tagged(_plan_payload(ids))) is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"objective": "", "assignments": _plan_payload()["assignments"]},
        {
            "objective": "x",
            "assignments": [
                {"agent_id": i, "subgoal": "x", "milestones": []} for i in range(3)
            ],
        },
    ],
)
def test_plan_parser_rejects_malformed_atomic_plans(payload):
    assert parse_team_plan(_tagged(payload)) is None
    assert parse_team_plan("not tagged") is None


def test_assignment_contains_only_selected_worker_details():
    plan = parse_team_plan(_tagged(_plan_payload()))
    assignment = format_assignment(plan, 1, version=2, issued_step=5, review_step=10)
    assert "work 1" in assignment
    assert "work 0" not in assignment
    assert "work 2" not in assignment
    assert "next review: 10" in assignment


def test_leader_prompt_reuses_public_rules_but_has_no_action_contract():
    prompt = leader_system_prompt(
        "You are Agent 0 (warrior).\n<game_rules>PUBLIC SENTINEL</game_rules>"
    )
    assert "bodyless Team Leader" in prompt
    assert "PUBLIC SENTINEL" in prompt
    assert "Agent 0 (warrior)" not in prompt
    assert "Do not output an action tag" in prompt


def test_communication_tracker_separates_payload_fanout_and_prompt_bytes():
    tracker = CommunicationTracker()
    tracker.eligible("worker_peer")
    tracker.route(
        sender=0,
        recipients=(1, 2),
        channel="worker_peer",
        content="Meet at x=4 now",
        sent_step=3,
        delivered_step=4,
    )
    tracker.prompt_injection("assignment")
    metrics = tracker.as_dict()
    assert metrics["worker_peer"]["payload_bytes"] == len(b"Meet at x=4 now")
    assert metrics["worker_peer"]["delivery_bytes"] == 2 * len(b"Meet at x=4 now")
    assert metrics["leader_assignment_prompt"]["prompt_injections"] == 1


class _FakeClient:
    def __init__(self):
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return LLMResponse(
            "fake",
            _tagged(_plan_payload()) + "<scratchpad>private</scratchpad>",
            "stop",
            10,
            5,
        )


def test_bodyless_leader_plans_without_action_or_environment_interface():
    fake = _FakeClient()
    leader = TeamLeaderAgent(lambda: fake)
    leader.set_instruction_prompt("rules")
    response, plan = leader.plan(
        step=0,
        observations=[
            {
                "obs_long_term": f"legal-{i}",
                "obs_short_term": f"status-{i}",
                "raw_obs": "HIDDEN-STATE-SENTINEL",
            }
            for i in range(3)
        ],
        reports={0: [], 1: [], 2: []},
        previous_plan=fallback_plan(),
        previous_version=0,
    )
    assert plan is not None
    assert response.completion.startswith("<team_plan>")
    assert not hasattr(leader, "act")
    assert leader.scratchpad == "private"
    sent_prompt = "\n".join(message.content for message in fake.messages)
    assert "legal-0" in sent_prompt
    assert "HIDDEN-STATE-SENTINEL" not in sent_prompt
