"""Minimal authority, lease, privacy, and executor-contract tests."""

import json

from baselines.llm.eval_utils.client import LLMResponse
from baselines.llm.eval_utils.team_commander import (
    EmbodiedCommanderPlanner,
    PlanProposal,
    ReviewRequest,
    SquadRuntime,
    SquadSpec,
    format_squad_directive,
    parse_squad_plan,
    parse_squad_status,
)


def _spec(commander_id=0):
    return SquadSpec(
        team_id="alpha",
        members=(0, 1, 2),
        commander_id=commander_id,
        review_interval=5,
        lease_steps=10,
        max_steps=30,
    )


def _replace_payload():
    return {
        "operation": "REPLACE",
        "objective": "Build one coherent team capability",
        "assignments": [
            {
                "agent_id": agent_id,
                "task_id": f"task-{agent_id}",
                "directive": f"Execute role {agent_id}",
                "target": f"target-{agent_id}",
                "completion": f"role {agent_id} complete",
                "dependencies": [],
            }
            for agent_id in range(3)
        ],
    }


def _tag(payload):
    return f"<squad_plan>{json.dumps(payload)}</squad_plan>"


def _proposal():
    return parse_squad_plan(
        _tag(_replace_payload()),
        spec=_spec(),
        step=0,
        canonical_actions={"Noop", "Do"},
    )


def test_plan_requires_one_unique_assignment_per_squad_member():
    assert _proposal() is not None
    missing = _replace_payload()
    missing["assignments"].pop()
    duplicate = _replace_payload()
    duplicate["assignments"][2]["task_id"] = "task-1"
    assert parse_squad_plan(_tag(missing), spec=_spec(), step=0) is None
    assert parse_squad_plan(_tag(duplicate), spec=_spec(), step=0) is None


def test_invalid_or_unauthorized_review_is_atomic_and_does_not_extend_lease():
    runtime = SquadRuntime(_spec())
    accepted = runtime.apply(_proposal(), step=0, review=ReviewRequest("scheduled", True))
    kept = runtime.apply(PlanProposal("KEEP"), step=5, review=ReviewRequest("scheduled", True))
    assert accepted.accepted and accepted.plan.version == 1
    assert kept.accepted and kept.plan.expiry_tick == 15

    rejected_keep = runtime.apply(
        PlanProposal("KEEP"), step=6, review=ReviewRequest("event:BLOCKED", False)
    )
    rejected_invalid = runtime.apply(None, step=6, review=ReviewRequest("invalid_retry", False))
    assert not rejected_keep.accepted
    assert not rejected_invalid.accepted
    assert rejected_invalid.plan.version == 1
    assert rejected_invalid.plan.expiry_tick == 15
    assert runtime.review_due(7).trigger == "invalid_retry"


def test_status_authority_rejects_stale_or_wrong_assignments_and_deduplicates_events():
    runtime = SquadRuntime(_spec())
    waiting = "SCP1|TEAM=alpha|VERSION=0|TASK=NONE|STATE=WAITING|EVIDENCE=no plan"
    assert runtime.ingest_status(2, waiting, step=0) is not None
    runtime.apply(_proposal(), step=0, review=ReviewRequest("scheduled", True))

    blocked = "SCP1|TEAM=alpha|VERSION=1|TASK=task-1|STATE=BLOCKED|EVIDENCE=rock wall"
    assert runtime.ingest_status(1, blocked, step=1) is not None
    assert runtime.ingest_status(1, blocked, step=1) is not None
    assert runtime.metrics["event_transitions"] == 1
    assert runtime.review_due(2).trigger == "event:BLOCKED"

    rejected = (
        "SCP1|TEAM=alpha|VERSION=0|TASK=task-1|STATE=ACTIVE|EVIDENCE=moving",
        "SCP1|TEAM=alpha|VERSION=1|TASK=task-2|STATE=ACTIVE|EVIDENCE=moving",
        "SCP1|TEAM=beta|VERSION=1|TASK=task-1|STATE=ACTIVE|EVIDENCE=moving",
    )
    assert all(runtime.ingest_status(1, status, step=2) is None for status in rejected)
    assert runtime.ingest_status(2, waiting, step=2) is None


class _PlannerClient:
    def __init__(self):
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return LLMResponse(
            model_id="scripted",
            completion=_tag(_replace_payload()) + "<scratchpad>private memory</scratchpad>",
            stop_reason="stop",
            input_tokens=10,
            output_tokens=5,
        )


def test_planner_receives_only_the_commanders_legal_view_and_validated_reports():
    client = _PlannerClient()
    planner = EmbodiedCommanderPlanner(lambda: client, spec=_spec())
    planner.set_instruction_prompt(
        "You are Agent 0. ROLE-PRIVATE\n<game_rules>PUBLIC RULES</game_rules>"
    )
    report = parse_squad_status(
        "SCP1|TEAM=alpha|VERSION=0|TASK=NONE|STATE=WAITING|EVIDENCE=waiting"
    )
    _, proposal = planner.plan(
        step=0,
        review=ReviewRequest("scheduled", True),
        commander_observation={
            "obs_long_term": "COMMANDER-LEGAL-LONG",
            "obs_short_term": "COMMANDER-LEGAL-SHORT",
            "raw_state": "HIDDEN-SENTINEL",
        },
        reports=[(1, report)],
        active_plan=None,
        canonical_actions={"Noop", "Do"},
    )

    sent = "\n".join(message.content for message in client.messages)
    assert proposal is not None
    assert "PUBLIC RULES" in sent and "COMMANDER-LEGAL-LONG" in sent
    assert 'Canonical physical actions allowed in an optional sync.action: ["Do", "Noop"]' in sent
    assert "ROLE-PRIVATE" not in sent and "HIDDEN-SENTINEL" not in sent
    assert "Never output <action>" in sent
    assert planner.scratchpad == "private memory"


def test_all_agents_receive_the_full_plan_with_distinct_authority_and_assignment():
    runtime = SquadRuntime(_spec(commander_id=1))
    plan = runtime.apply(_proposal(), step=0, review=ReviewRequest("scheduled", True)).plan
    directives = [
        format_squad_directive(plan, spec=runtime.spec, agent_id=agent_id)
        for agent_id in runtime.spec.members
    ]
    assert all("task-0" in directive and "task-2" in directive for directive in directives)
    assert "role=COMMANDER-EXECUTOR" in directives[1]
    assert "YOUR ASSIGNMENT: task=task-2" in directives[2]
    assert "Do not create competing plans" in directives[2]
    assert "STATE=ACTIVE" in directives[2]
    assert "STATE=ACK|ACTIVE" not in directives[2]
