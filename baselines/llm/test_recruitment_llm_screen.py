import copy
import gzip
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from baselines.llm.eval_utils.client import LLMResponse
from baselines.llm.eval_utils.openai_responses import OpenAIResponsesWrapper
from baselines.llm.eval_utils.team_formation import (
    RecordKind,
    RecruitmentMethod,
    RecruitmentRecord,
    TaskCard,
    TeamDirectory,
    parse_tfp1,
)
from baselines.llm.recruitment_arena import (
    AGENT_IDS,
    ScenarioFamily,
    generate_scenario,
)
from baselines.llm.recruitment_llm_screen import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_PROMPT_FRAMING_TOKENS,
    CampaignBudget,
    JointExactPublicApplicationSelector,
    PublicSelectorView,
    ScreenConfig,
    SharedCallBudget,
    build_agent_view,
    estimate_campaign,
    request_control_record,
    run_llm_recruitment_episode,
    selector_for_method,
)
from scripts import run_recruitment_llm_screen as recruitment_runner
from scripts.run_recruitment_llm_screen import (
    DEFAULT_LOGICAL_CALL_CAP,
    DEFAULT_PROVIDER_ATTEMPT_CAP,
    DEFAULT_TOKEN_EXPOSURE_CAP,
    DurableReservationLedger,
    _artifact_paths,
    _assert_artifacts_bound_to_ledger,
    _assert_reservation_coverage,
    _caps_conflicts,
    _compute_canary_gate,
    _launch_projection,
    _load_completed_or_pending,
    _load_passing_canary_gate,
    _normalize_output_root,
    _output_root_lock,
    _parse_args,
    _protocol_config,
    _require_hosted_credentials,
    _validate_debug_shard,
    _validate_managed_cell_inventory,
    _write_canary_gate,
    load_completed_marker,
    persist_completed_episode,
)


def _response(completion, *, response_id="resp_fake", latency=0.0):
    return LLMResponse(
        model_id="gpt-5.6-luna",
        completion=completion,
        stop_reason="stop",
        input_tokens=100,
        output_tokens=10,
        reasoning="",
        reasoning_tokens=4,
        response_id=response_id,
        status="completed",
        cached_tokens=50,
        cache_write_tokens=0,
        latency_seconds=latency,
    )


def _reservation_payload(
    index=0,
    *,
    seed=22000,
    family="single_complementary",
    method="open_volunteer",
    prompt_bytes=100,
):
    return {
        "reservation_key": {
            "seed": seed,
            "family": family,
            "method": method,
            "round_index": index // len(AGENT_IDS),
            "agent_id": index % len(AGENT_IDS),
            "semantic_attempt": 0,
        },
        "logical_calls": 1,
        "provider_attempts": 2,
        "tokens": (prompt_bytes + DEFAULT_PROMPT_FRAMING_TOKENS + 1024) * 2,
        "prompt_bytes": prompt_bytes,
        "prompt_framing_tokens": DEFAULT_PROMPT_FRAMING_TOKENS,
        "max_output_tokens": 1024,
    }


def _resolution_payload(reservation):
    return {
        "reservation_key": reservation["reservation_key"],
        "provider_attempts_actual": 1,
        "input_tokens": 100,
        "output_tokens": 10,
        "outcome": "abstain",
    }


class _SequenceClient:
    def __init__(self, completions):
        self.completions = list(completions)
        self.messages = []

    def generate(self, messages):
        self.messages.append(messages)
        return _response(
            self.completions.pop(0),
            response_id=f"resp_{len(self.messages)}",
        )


def _directory(scenario, method):
    directory = TeamDirectory(agent_ids=AGENT_IDS, method=method, seed=scenario.seed)
    for task in scenario.tasks:
        directory.register_task(task)
    directory.advance(0)
    directory.deliver_ordinary(0)
    return directory


class _CanaryFormationClient:
    def __init__(self, scenario, agent_id):
        self.scenario = scenario
        self.agent_id = agent_id

    def generate(self, messages):
        view = json.loads(messages[1].content.removeprefix("AGENT_VIEW_JSON="))
        cards = {card["task_id"]: card for card in view["task_cards"]}
        states = view["public_ledger"]["tasks"]
        advice = view["public_selector"]["advice"]
        for task_id, raw_members in sorted(advice.items()):
            members = tuple(raw_members)
            if self.agent_id == cards[task_id]["sponsor_id"] and set(members).issubset(
                states[task_id]["accepts"]
            ):
                return _response(
                    RecruitmentRecord(
                        RecordKind.LOCK,
                        task_id,
                        members=members,
                    ).render()
                )
        for task_id, raw_members in sorted(advice.items()):
            members = tuple(raw_members)
            if self.agent_id in members and self.agent_id not in states[task_id]["accepts"]:
                return _response(
                    RecruitmentRecord(
                        RecordKind.ACCEPT,
                        task_id,
                        members=members,
                    ).render()
                )
        task_id = sorted(cards)[0]
        if str(self.agent_id) not in states[task_id]["applications"]:
            profile = self.scenario.agents[self.agent_id]
            return _response(
                RecruitmentRecord(
                    RecordKind.APPLY,
                    task_id,
                    capabilities=profile.true_capabilities,
                    cost=profile.task_costs[task_id],
                ).render()
            )
        return _response("ABSTAIN")


def _persist_passing_canary(output):
    source_hashes = {"source.py": "abc"}
    markers = []
    for family in (
        ScenarioFamily.SINGLE_COMPLEMENTARY,
        ScenarioFamily.TWO_DISJOINT,
    ):
        scenario = generate_scenario(family, 22000)
        episode = run_llm_recruitment_episode(
            scenario,
            ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=8),
            client_factory=lambda agent_id, scenario=scenario: (
                _CanaryFormationClient(scenario, agent_id)
            ),
        )
        markers.append(
            persist_completed_episode(
                output,
                episode,
                config_sha256="config-hash",
                source_hashes=source_hashes,
            )
        )
    gate = _write_canary_gate(
        output,
        markers,
        config_sha256="config-hash",
        source_hashes=source_hashes,
    )
    return markers, gate, source_hashes


def _constructed_joint_conflict_view():
    task_cards = (
        {
            "task_id": "alpha",
            "demand": [70, 0, 0],
            "required_size": 2,
            "reward": 100,
            "deadline_round": 12,
            "sponsor_id": 0,
        },
        {
            "task_id": "beta",
            "demand": [0, 70, 0],
            "required_size": 2,
            "reward": 100,
            "deadline_round": 12,
            "sponsor_id": 1,
        },
    )
    capabilities = (
        (60, 40, 0),
        (60, 0, 0),
        (0, 30, 0),
        (40, 0, 0),
        (30, 0, 0),
        (0, 0, 0),
    )
    task_states = {}
    for card in task_cards:
        task_id = card["task_id"]
        task_states[task_id] = {
            "card": card,
            "phase": "forming",
            "applications": {
                str(agent_id): {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "claimed_capabilities": list(capability),
                    "cost": 0,
                    "arrival_round": 1,
                    # Deliberately extraneous: the selector must whitelist
                    # public claims and ignore any hidden-truth injection.
                    "true_capabilities": [999 - agent_id] * 3,
                    "true_cost": 999,
                }
                for agent_id, capability in enumerate(capabilities)
            },
            "declines": [],
            "lease": None,
        }
    return PublicSelectorView(
        method=RecruitmentMethod.OPEN_VOLUNTEER.value,
        round_index=2,
        agent_ids=AGENT_IDS,
        public_roles={str(agent_id): "public" for agent_id in AGENT_IDS},
        task_cards=task_cards,
        public_ledger={
            "round_index": 2,
            "agent_to_task": {},
            "tasks": task_states,
        },
    )


def test_open_joint_selector_adapts_constructed_e2d2_conflict():
    selector = JointExactPublicApplicationSelector()

    assert selector.select(_constructed_joint_conflict_view()) == {
        "alpha": (1, 3),
        "beta": (0, 2),
    }


def test_open_joint_selector_ignores_truth_perturbations_and_all_replicas_agree():
    selector = JointExactPublicApplicationSelector()
    original = _constructed_joint_conflict_view()
    altered_ledger = copy.deepcopy(original.public_ledger)
    for state in altered_ledger["tasks"].values():
        state["applications"] = dict(reversed(tuple(state["applications"].items())))
        for agent_id, application in state["applications"].items():
            application["true_capabilities"] = [-int(agent_id), int(agent_id), 100]
            application["true_cost"] = int(agent_id)
            application["oracle_label"] = "must-not-enter-selector"
    altered = replace(original, public_ledger=altered_ledger)

    peer_outputs = [selector.select(copy.deepcopy(original)) for _ in AGENT_IDS]
    assert all(output == peer_outputs[0] for output in peer_outputs)
    assert selector.select(altered) == peer_outputs[0]


def test_published_open_joint_plan_is_exclusive_and_replayable():
    directory = TeamDirectory(
        agent_ids=AGENT_IDS,
        method=RecruitmentMethod.OPEN_VOLUNTEER,
        seed=22000,
    )
    cards = (
        TaskCard("alpha", (70, 0, 0), 2, 100, 12, 0),
        TaskCard("beta", (0, 70, 0), 2, 100, 12, 1),
    )
    for card in cards:
        directory.register_task(card)
    capabilities = (
        (60, 40, 0),
        (60, 0, 0),
        (0, 30, 0),
        (40, 0, 0),
        (30, 0, 0),
        (0, 0, 0),
    )
    directory.advance(0)
    directory.deliver_ordinary(0)
    for agent_id, capability in enumerate(capabilities):
        directory.submit_control(
            sender=agent_id,
            sent_round=0,
            raw=RecruitmentRecord(
                RecordKind.APPLY,
                "alpha",
                capabilities=capability,
                cost=0,
            ).render(),
        )
    directory.advance(1)
    directory.deliver_ordinary(1)
    for agent_id, capability in enumerate(capabilities):
        directory.submit_control(
            sender=agent_id,
            sent_round=1,
            raw=RecruitmentRecord(
                RecordKind.APPLY,
                "beta",
                capabilities=capability,
                cost=0,
            ).render(),
        )
    directory.advance(2)
    directory.deliver_ordinary(2)
    snapshot = directory.snapshot()
    view = PublicSelectorView(
        method=directory.method.value,
        round_index=2,
        agent_ids=AGENT_IDS,
        public_roles={str(agent_id): "public" for agent_id in AGENT_IDS},
        task_cards=tuple(card.as_dict() for card in cards),
        public_ledger={
            "round_index": 2,
            "agent_to_task": snapshot.agent_to_task,
            "tasks": snapshot.tasks,
        },
    )
    plan = JointExactPublicApplicationSelector().select(view)
    directory.publish_open_roster_plan(
        round_index=2,
        selector="joint_exact_allocation",
        public_input_sha256="a" * 64,
        rosters={task_id: tuple(members) for task_id, members in plan.items()},
    )

    for agent_id, task_id, members in (
        (0, "beta", (0, 2)),
        (1, "alpha", (1, 3)),
    ):
        directory.submit_control(
            sender=agent_id,
            sent_round=2,
            raw=RecruitmentRecord(RecordKind.ACCEPT, task_id, members=members).render(),
        )
    directory.advance(3)
    directory.deliver_ordinary(3)
    for agent_id, task_id, members in (
        (2, "beta", (0, 2)),
        (3, "alpha", (1, 3)),
    ):
        directory.submit_control(
            sender=agent_id,
            sent_round=3,
            raw=RecruitmentRecord(RecordKind.ACCEPT, task_id, members=members).render(),
        )
    directory.advance(4)
    directory.deliver_ordinary(4)
    for agent_id, task_id, members in (
        (0, "alpha", (1, 3)),
        (1, "beta", (0, 2)),
    ):
        directory.submit_control(
            sender=agent_id,
            sent_round=4,
            raw=RecruitmentRecord(RecordKind.LOCK, task_id, members=members).render(),
        )
    transitions = directory.advance(5)
    directory.deliver_ordinary(5)

    assert [transition.code for transition in transitions] == ["team.locked", "team.locked"]
    assert directory.agent_to_task == {
        0: "beta",
        1: "alpha",
        2: "beta",
        3: "alpha",
    }
    replayed = TeamDirectory.replay(directory.export_replay())
    assert replayed.state_hash() == directory.state_hash()
    assert replayed.audit_chain_hash == directory.audit_chain_hash


def test_v4_open_only_cap_and_two_cell_canary_projection_are_frozen():
    protocol = _protocol_config(
        logical_call_cap=DEFAULT_LOGICAL_CALL_CAP,
        provider_attempt_cap=DEFAULT_PROVIDER_ATTEMPT_CAP,
        token_exposure_cap=DEFAULT_TOKEN_EXPOSURE_CAP,
    )

    assert protocol["selectors"] == {
        "open_volunteer": "joint_exact_allocation",
    }
    assert protocol["estimate"] == estimate_campaign(
        seed_count=3,
        family_count=4,
        method_count=1,
    )
    assert protocol["estimate"]["episodes"] == 12
    assert protocol["protocol_revision"] == (
        "e2b-v4-open-joint-confirmation-two-cell-canary"
    )
    assert protocol["methods"] == ["open_volunteer"]
    assert protocol["max_output_tokens"] == 4096
    assert len(protocol["canary_cells"]) == 2
    assert {cell["method"] for cell in protocol["canary_cells"]} == {
        "open_volunteer",
    }
    assert _launch_projection(2) == {
        "episodes": 2,
        "initial_logical_calls": 144,
        "semantic_repair_allowance": 36,
        "logical_calls": 180,
        "provider_attempts_reserved": 360,
        "provider_attempt_token_exposure": 7_603_200,
    }
    assert _launch_projection(12) == {
        "episodes": 12,
        "initial_logical_calls": 864,
        "semantic_repair_allowance": 216,
        "logical_calls": 1080,
        "provider_attempts_reserved": 2160,
        "provider_attempt_token_exposure": 45_619_200,
    }
    assert selector_for_method(RecruitmentMethod.OPEN_VOLUNTEER).name == ("joint_exact_allocation")
    assert selector_for_method(RecruitmentMethod.MUTUAL_NOMINATION).name == (
        "native_mutual_reciprocal"
    )


def test_v4_campaign_ledger_rejects_mutual_dispatch(tmp_path):
    ledger = DurableReservationLedger(
        tmp_path,
        launch_binding={"config_sha256": "a" * 64, "git_head": "b" * 40},
    )
    with pytest.raises(ValueError, match="reservation key has invalid values"):
        ledger.reserve(_reservation_payload(method="mutual_nomination"))
    assert ledger.reconcile()["records"] == 0


@pytest.mark.parametrize(
    "method",
    [RecruitmentMethod.OPEN_VOLUNTEER, RecruitmentMethod.MUTUAL_NOMINATION],
)
def test_prompt_projection_is_agent_local_and_selector_is_truth_free(method):
    scenario = generate_scenario(ScenarioFamily.TWO_DISJOINT, 22000)
    directory = _directory(scenario, method)
    view = build_agent_view(
        scenario,
        directory,
        agent_id=0,
        round_index=0,
    ).as_dict()
    profile = scenario.agents[0]

    assert view["own_private"] == {
        "true_capabilities": list(profile.true_capabilities),
        "task_costs": dict(sorted(profile.task_costs.items())),
    }
    assert "pending_control" not in json.dumps(view)
    assert json.dumps(view).count('"own_private"') == 1
    assert json.dumps(view).count('"true_capabilities"') == 1
    assert json.dumps(view).count('"task_costs"') == 1
    assert "oracle" not in json.dumps(view).lower()

    captured = {}

    class _CaptureSelector:
        name = "capture"

        def select(self, selector_view):
            captured.update(selector_view.as_dict())
            return {}

    run_llm_recruitment_episode(
        scenario,
        ScreenConfig(method=method, rounds=1),
        client_factory=lambda agent_id: _SequenceClient(["ABSTAIN"]),
        selector=_CaptureSelector(),
    )
    encoded_selector = json.dumps(captured)
    assert "true_capabilities" not in encoded_selector
    assert "task_costs" not in encoded_selector
    assert "oracle" not in encoded_selector.lower()


def test_eligible_agents_are_scheduled_concurrently_from_one_snapshot():
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    barrier = threading.Barrier(len(AGENT_IDS))

    class _BarrierClient:
        def generate(self, messages):
            barrier.wait(timeout=2)
            time.sleep(0.02)
            return _response("ABSTAIN")

    episode = run_llm_recruitment_episode(
        scenario,
        ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
        client_factory=lambda agent_id: _BarrierClient(),
    )
    round_zero = episode["rounds"][0]

    assert round_zero["eligible_agents"] == list(AGENT_IDS)
    assert round_zero["logical_model_calls"] == len(AGENT_IDS)
    assert round_zero["peak_concurrent_calls"] == len(AGENT_IDS)
    assert round_zero["selector"]["replica_count"] == len(AGENT_IDS)
    assert len(set(round_zero["selector"]["replica_advice_sha256"])) == 1
    projections = [
        record["prompt_projection"]["public_ledger"] for record in episode["_debug_call_records"]
    ]
    assert all(projection == projections[0] for projection in projections)


def test_two_no_progress_rounds_stop_future_calls_without_changing_requested_horizon():
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    episode = run_llm_recruitment_episode(
        scenario,
        ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER),
        client_factory=lambda agent_id: _SequenceClient(["ABSTAIN", "ABSTAIN"]),
    )

    assert episode["early_stop"] == {
        "requested_acting_rounds": 12,
        "executed_acting_rounds": 2,
        "stall_rounds": 2,
        "reason": "no_public_progress_2_rounds",
    }
    assert episode["drain"]["round_index"] == 2
    assert len(episode["call_ledger"]) == 12


@pytest.mark.parametrize(
    ("first", "second", "expected_record", "expected_code"),
    [
        (
            "not tfp1",
            "TFP1|TYPE=APPLY|TASK=single.t0|CAP=80,20,20|COST=7",
            RecordKind.APPLY,
            None,
        ),
        ("x" * 300, "y" * 300, None, "semantic_exhausted:parse.too_many_bytes"),
        ("TFP1|TYPE=CFP|TASK=single.t0", "ABSTAIN", None, "model.abstain"),
    ],
)
def test_one_semantic_repair_then_typed_record_or_safe_abstain(
    first,
    second,
    expected_record,
    expected_code,
):
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    directory = _directory(scenario, RecruitmentMethod.OPEN_VOLUNTEER)
    view = build_agent_view(
        scenario,
        directory,
        agent_id=0,
        round_index=0,
    )
    client = _SequenceClient([first, second])
    decision = request_control_record(
        client=client,
        view=view,
        config=ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
        episode_budget=SharedCallBudget(2),
    )

    assert len(decision.calls) == 2
    assert decision.semantic_repairs == 1
    assert (None if decision.record is None else decision.record.kind) is expected_record
    assert decision.abstain_code == expected_code
    assert "Repair it once" in client.messages[1][-1].content
    assert decision.calls[0]["_debug"]["raw_completion"] == first
    assert decision.calls[1]["_debug"]["raw_completion"] == second
    assert "_debug" not in decision.as_dict()["calls"][0]


def test_public_transition_preflight_gets_one_repair():
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    directory = _directory(scenario, RecruitmentMethod.OPEN_VOLUNTEER)
    view = build_agent_view(scenario, directory, agent_id=0, round_index=0)
    client = _SequenceClient(
        [
            "TFP1|TYPE=LOCK|TASK=single.t0|MEMBERS=0,4",
            "ABSTAIN",
        ]
    )
    decision = request_control_record(
        client=client,
        view=view,
        config=ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
        episode_budget=SharedCallBudget(2),
        transition_validator=lambda record: (False, "authority.wrong_locker"),
    )

    assert decision.record is None
    assert decision.semantic_repairs == 1
    assert decision.calls[0]["validation_code"] == ("transition_preflight:authority.wrong_locker")
    assert decision.calls[0]["parse"]["valid"]
    assert decision.abstain_code == "model.abstain"


def test_deterministic_semantics_atomic_resume_and_debug_hash_replay(tmp_path):
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    config = ScreenConfig(method=RecruitmentMethod.MUTUAL_NOMINATION, rounds=1)

    def run():
        return run_llm_recruitment_episode(
            scenario,
            config,
            client_factory=lambda agent_id: _SequenceClient(["ABSTAIN"]),
        )

    first = run()
    repeated = run()
    assert first["deterministic_episode_hash"] == repeated["deterministic_episode_hash"]
    assert first["terminal_state_hash"] == repeated["terminal_state_hash"]
    assert first["terminal_audit_chain_hash"] == repeated["terminal_audit_chain_hash"]
    assert first["replay_hash_match"] and repeated["replay_hash_match"]

    source_hashes = {"source.py": "abc"}
    marker = persist_completed_episode(
        tmp_path,
        first,
        config_sha256="config-hash",
        source_hashes=source_hashes,
    )
    loaded = load_completed_marker(
        tmp_path,
        seed=22000,
        family=ScenarioFamily.SINGLE_COMPLEMENTARY,
        method=RecruitmentMethod.MUTUAL_NOMINATION,
        config_sha256="config-hash",
        source_hashes=source_hashes,
    )
    assert loaded == marker
    debug_path = tmp_path / marker["debug_artifact"]
    assert _validate_debug_shard(
        debug_path,
        expected_count=marker["debug_record_count"],
        expected_content_sha256=marker["debug_content_sha256"],
        expected_gzip_sha256=marker["debug_gzip_sha256"],
    )
    with gzip.open(debug_path, "rt", encoding="utf-8") as handle:
        debug = json.loads(next(handle))
    assert debug["raw_completion"] == "ABSTAIN"
    assert debug["messages"][1]["content"].startswith("AGENT_VIEW_JSON=")
    assert debug["response"]["id"] == "resp_1"

    with debug_path.open("ab") as handle:
        handle.write(b"corruption")
    assert (
        load_completed_marker(
            tmp_path,
            seed=22000,
            family=ScenarioFamily.SINGLE_COMPLEMENTARY,
            method=RecruitmentMethod.MUTUAL_NOMINATION,
            config_sha256="config-hash",
            source_hashes=source_hashes,
        )
        is None
    )


def test_episode_call_budget_is_atomic_under_concurrency():
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    episode = run_llm_recruitment_episode(
        scenario,
        ScreenConfig(
            method=RecruitmentMethod.OPEN_VOLUNTEER,
            rounds=1,
            max_calls_per_episode=1,
        ),
        client_factory=lambda agent_id: _SequenceClient(["ABSTAIN"]),
    )

    assert episode["call_caps"]["episode_used"] == 1
    assert len(episode["call_ledger"]) == 1
    assert (
        sum(
            decision["abstain_code"] == "budget.episode_calls"
            for decision in episode["rounds"][0]["decisions"]
        )
        == 5
    )

    campaign_budget = CampaignBudget(
        logical_limit=1,
        provider_attempt_limit=2,
        token_limit=1_000_000,
    )
    campaign_limited = run_llm_recruitment_episode(
        scenario,
        ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
        client_factory=lambda agent_id: _SequenceClient(["ABSTAIN"]),
        campaign_budget=campaign_budget,
    )
    assert campaign_budget.snapshot()["logical_used"] == 1
    assert campaign_limited["call_caps"]["episode_used"] == 1
    assert len(campaign_limited["call_ledger"]) == 1
    assert (
        sum(
            decision["abstain_code"] == "budget.campaign_logical_calls"
            for decision in campaign_limited["rounds"][0]["decisions"]
        )
        == 5
    )


def test_default_hard_caps_admit_base_projection_but_not_theoretical_repair_ceiling():
    protocol = _protocol_config(
        logical_call_cap=DEFAULT_LOGICAL_CALL_CAP,
        provider_attempt_cap=DEFAULT_PROVIDER_ATTEMPT_CAP,
        token_exposure_cap=DEFAULT_TOKEN_EXPOSURE_CAP,
    )
    projection = _launch_projection(12)

    assert not _caps_conflicts(protocol=protocol, projection=projection)
    assert DEFAULT_LOGICAL_CALL_CAP < protocol["estimate"]["max_logical_calls"]
    assert DEFAULT_PROVIDER_ATTEMPT_CAP < protocol["estimate"]["max_provider_attempts"]
    assert DEFAULT_TOKEN_EXPOSURE_CAP < protocol["estimate"]["max_provider_total_token_exposure"]
    too_small = _protocol_config(
        logical_call_cap=projection["logical_calls"] - 1,
        provider_attempt_cap=projection["provider_attempts_reserved"] - 1,
        token_exposure_cap=projection["provider_attempt_token_exposure"] - 1,
    )
    assert len(_caps_conflicts(protocol=too_small, projection=projection)) == 3


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (
            CampaignBudget(
                logical_limit=1,
                provider_attempt_limit=1,
                token_limit=100_000,
            ),
            "provider_attempts",
        ),
        (
            CampaignBudget(
                logical_limit=1,
                provider_attempt_limit=2,
                token_limit=10,
            ),
            "tokens",
        ),
    ],
)
def test_campaign_budget_reserves_provider_and_token_worst_case_atomically(
    budget,
    expected,
):
    assert (
        budget.reserve(
            prompt_bytes=100,
            max_output_tokens=100,
            max_provider_attempts=2,
        )
        == expected
    )
    assert budget.snapshot()["logical_used"] == 0


def test_canary_gate_is_pure_recomputed_and_tampering_fails_closed(tmp_path):
    markers, gate, source_hashes = _persist_passing_canary(tmp_path)

    assert gate["status"] == "pass"
    assert all(gate["gates"].values())
    assert len(gate["cells"]) == 2
    assert {
        cell["cell"]["method"] for cell in gate["cells"]
    } == {"open_volunteer"}
    assert all(cell["status"] == "pass" for cell in gate["cells"])
    assert gate == _compute_canary_gate(
        tmp_path,
        markers,
        config_sha256="config-hash",
        source_hashes=source_hashes,
    )
    assert (
        _load_passing_canary_gate(
            tmp_path,
            markers,
            config_sha256="config-hash",
            source_hashes=source_hashes,
        )
        == gate
    )

    gate_path = tmp_path / "canary_gate.json"
    tampered = copy.deepcopy(gate)
    tampered["cells"][0]["diagnostics"]["logical_calls"] += 1
    gate_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert (
        _load_passing_canary_gate(
            tmp_path,
            markers,
            config_sha256="config-hash",
            source_hashes=source_hashes,
        )
        is None
    )


def test_canary_requires_both_open_family_cells_and_nonforming_cell_cannot_promote(
    tmp_path,
):
    markers, gate, source_hashes = _persist_passing_canary(tmp_path)
    assert gate["status"] == "pass"
    with pytest.raises(ValueError, match="exact frozen two-cell matrix"):
        _compute_canary_gate(
            tmp_path,
            markers[:-1],
            config_sha256="config-hash",
            source_hashes=source_hashes,
        )

    scenario = generate_scenario(ScenarioFamily.TWO_DISJOINT, 22000)
    episode = run_llm_recruitment_episode(
        scenario,
        ScreenConfig(
            method=RecruitmentMethod.OPEN_VOLUNTEER,
            rounds=2,
        ),
        client_factory=lambda agent_id: _SequenceClient(["ABSTAIN", "ABSTAIN"]),
    )
    nonforming = persist_completed_episode(
        tmp_path,
        episode,
        config_sha256="config-hash",
        source_hashes=source_hashes,
    )
    replaced = [
        nonforming
        if marker["family"] == "two_disjoint"
        and marker["method"] == "open_volunteer"
        else marker
        for marker in markers
    ]
    failed_gate = _write_canary_gate(
        tmp_path,
        replaced,
        config_sha256="config-hash",
        source_hashes=source_hashes,
    )
    assert failed_gate["status"] == "fail"
    failed_cell = next(
        cell
        for cell in failed_gate["cells"]
        if cell["cell"]["family"] == "two_disjoint"
        and cell["cell"]["method"] == "open_volunteer"
    )
    assert not failed_cell["gates"]["formed_true_feasible_team"]
    assert (
        _load_passing_canary_gate(
            tmp_path,
            replaced,
            config_sha256="config-hash",
            source_hashes=source_hashes,
        )
        is None
    )


def test_v4_open_only_offline_canary_full_resume_lifecycle(
    tmp_path,
    monkeypatch,
):
    class _OpenLifecycleClient:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        def generate(self, messages):
            view = json.loads(
                messages[1].content.removeprefix("AGENT_VIEW_JSON=")
            )
            cards = {card["task_id"]: card for card in view["task_cards"]}
            states = view["public_ledger"]["tasks"]
            advice = view["public_selector"]["advice"]
            for task_id, raw_members in sorted(advice.items()):
                members = tuple(raw_members)
                if (
                    self.agent_id == cards[task_id]["sponsor_id"]
                    and set(members).issubset(states[task_id]["accepts"])
                ):
                    return _response(
                        RecruitmentRecord(
                            RecordKind.LOCK,
                            task_id,
                            members=members,
                        ).render()
                    )
            for task_id, raw_members in sorted(advice.items()):
                members = tuple(raw_members)
                if (
                    self.agent_id in members
                    and self.agent_id not in states[task_id]["accepts"]
                ):
                    return _response(
                        RecruitmentRecord(
                            RecordKind.ACCEPT,
                            task_id,
                            members=members,
                        ).render()
                    )
            for task_id in sorted(cards):
                if (
                    states[task_id]["lease"] is None
                    and str(self.agent_id)
                    not in states[task_id]["applications"]
                ):
                    own = view["own_private"]
                    return _response(
                        RecruitmentRecord(
                            RecordKind.APPLY,
                            task_id,
                            capabilities=tuple(own["true_capabilities"]),
                            cost=own["task_costs"][task_id],
                        ).render()
                    )
            return _response("ABSTAIN")

    protocol = _protocol_config(
        logical_call_cap=DEFAULT_LOGICAL_CALL_CAP,
        provider_attempt_cap=DEFAULT_PROVIDER_ATTEMPT_CAP,
        token_exposure_cap=DEFAULT_TOKEN_EXPOSURE_CAP,
    )
    binding = {
        "config_sha256": "a" * 64,
        "git_head": "b" * 40,
        "source_hashes": {"synthetic.py": "c" * 64},
        "uv_lock_sha256": "d" * 64,
    }
    monkeypatch.setattr(recruitment_runner, "_require_hosted_credentials", lambda: None)
    monkeypatch.setattr(
        recruitment_runner,
        "_launch_binding",
        lambda candidate: binding if candidate == protocol else pytest.fail(),
    )
    monkeypatch.setattr(
        recruitment_runner,
        "_client_factory",
        lambda config: lambda agent_id: _OpenLifecycleClient(agent_id),
    )
    output = tmp_path / "v4"
    output.mkdir()
    canary_args = SimpleNamespace(
        stage="canary",
        resume=False,
        parallel_cells=True,
        workers=2,
    )
    assert (
        recruitment_runner._run_hosted_locked(
            args=canary_args,
            output=output,
            jobs=recruitment_runner.CANARY_CELLS,
            protocol=protocol,
            launch_binding=binding,
        )
        == 0
    )
    canary_manifest = json.loads(
        (output / "run_manifest_canary.json").read_text(encoding="utf-8")
    )
    assert canary_manifest["status"] == "complete"
    assert canary_manifest["canary_gate"]["status"] == "pass"
    assert canary_manifest["completed_episodes"] == 2
    assert canary_manifest["cell_workers"] == 2
    assert canary_manifest["campaign_budget"]["logical_used"] == 44

    all_jobs = tuple(
        (seed, family, method)
        for seed in recruitment_runner.FROZEN_SEEDS
        for family in recruitment_runner.FROZEN_FAMILIES
        for method in recruitment_runner.FROZEN_METHODS
    )
    full_args = SimpleNamespace(
        stage="full",
        resume=True,
        parallel_cells=True,
        workers=3,
    )
    assert (
        recruitment_runner._run_hosted_locked(
            args=full_args,
            output=output,
            jobs=all_jobs,
            protocol=protocol,
            launch_binding=binding,
        )
        == 0
    )
    full_manifest_path = output / "run_manifest_full.json"
    full_manifest = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    assert full_manifest["status"] == "complete"
    assert full_manifest["completed_episodes"] == 12
    assert full_manifest["cell_workers"] == 3
    assert full_manifest["campaign_budget"]["logical_used"] == 272
    assert full_manifest["campaign_budget"]["provider_attempts_reserved"] == 544
    assert full_manifest["campaign_budget"]["tokens_reserved"] == 4_555_934
    assert not full_manifest["reservation_ledger_final"]["unresolved"]
    assert not full_manifest["reservation_ledger_final"]["overages"]
    checkpoint = full_manifest["reservation_ledger_final"]["checkpoint"]

    monkeypatch.setattr(
        recruitment_runner,
        "_client_factory",
        lambda config: pytest.fail("completed resume must not construct a client"),
    )
    assert (
        recruitment_runner._run_hosted_locked(
            args=full_args,
            output=output,
            jobs=all_jobs,
            protocol=protocol,
            launch_binding=binding,
        )
        == 0
    )
    resumed = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    assert resumed["reservation_ledger_final"]["checkpoint"] == checkpoint


def test_completed_cell_rejects_copied_marker_and_tampered_counts(tmp_path):
    markers, _, source_hashes = _persist_passing_canary(tmp_path)
    marker = next(
        marker
        for marker in markers
        if marker["family"] == "single_complementary"
        and marker["method"] == "open_volunteer"
    )
    marker_path = (
        tmp_path / "markers" / "open_volunteer__single_complementary__seed_22000.complete.json"
    )
    tampered = json.loads(marker_path.read_text(encoding="utf-8"))
    tampered["logical_calls"] += 1
    marker_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert (
        load_completed_marker(
            tmp_path,
            seed=22000,
            family=ScenarioFamily.SINGLE_COMPLEMENTARY,
            method=RecruitmentMethod.OPEN_VOLUNTEER,
            config_sha256="config-hash",
            source_hashes=source_hashes,
        )
        is None
    )

    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    target_artifact, target_debug, target_marker = _artifact_paths(
        tmp_path,
        seed=22000,
        family=ScenarioFamily.SCARCE_CAPABILITY,
        method=RecruitmentMethod.OPEN_VOLUNTEER,
    )
    target_artifact.parent.mkdir(parents=True, exist_ok=True)
    target_debug.parent.mkdir(parents=True, exist_ok=True)
    target_marker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp_path / marker["artifact"], target_artifact)
    shutil.copyfile(tmp_path / marker["debug_artifact"], target_debug)
    shutil.copyfile(marker_path, target_marker)
    assert (
        load_completed_marker(
            tmp_path,
            seed=22000,
            family=ScenarioFamily.SCARCE_CAPABILITY,
            method=RecruitmentMethod.OPEN_VOLUNTEER,
            config_sha256="config-hash",
            source_hashes=source_hashes,
        )
        is None
    )


def test_output_lock_is_nonblocking_and_crash_reservation_stops_resume(tmp_path):
    output = tmp_path / "campaign"
    with _output_root_lock(output):
        with pytest.raises(RuntimeError, match="another hosted E2b invocation"):
            with _output_root_lock(output):
                pass

    binding = {"config_sha256": "a" * 64, "git_head": "b" * 40}
    ledger = DurableReservationLedger(output, launch_binding=binding)
    reservation = {
        "reservation_key": {
            "seed": 22000,
            "family": "single_complementary",
            "method": "open_volunteer",
            "round_index": 0,
            "agent_id": 0,
            "semantic_attempt": 0,
        },
        "logical_calls": 1,
        "provider_attempts": 2,
        "tokens": (100 + DEFAULT_PROMPT_FRAMING_TOKENS + 1024) * 2,
        "prompt_bytes": 100,
        "prompt_framing_tokens": DEFAULT_PROMPT_FRAMING_TOKENS,
        "max_output_tokens": 1024,
    }
    ledger.reserve(reservation)
    reopened = DurableReservationLedger(output, launch_binding=binding)
    reconciliation = reopened.reconcile()
    assert len(reconciliation["unresolved"]) == 1
    assert reconciliation["logical_used"] == 1
    with pytest.raises(RuntimeError, match="unresolved pre-dispatch reservations"):
        _assert_reservation_coverage(reconciliation, [])


def test_reservation_ledger_is_bound_and_hash_chained(tmp_path):
    output = tmp_path / "campaign"
    binding = {"config_sha256": "a" * 64, "git_head": "b" * 40}
    ledger = DurableReservationLedger(output, launch_binding=binding)
    reservation = {
        "reservation_key": {
            "seed": 22000,
            "family": "single_complementary",
            "method": "open_volunteer",
            "round_index": 0,
            "agent_id": 0,
            "semantic_attempt": 0,
        },
        "logical_calls": 1,
        "provider_attempts": 2,
        "tokens": (100 + DEFAULT_PROMPT_FRAMING_TOKENS + 1024) * 2,
        "prompt_bytes": 100,
        "prompt_framing_tokens": DEFAULT_PROMPT_FRAMING_TOKENS,
        "max_output_tokens": 1024,
    }
    ledger.reserve(reservation)

    with pytest.raises(ValueError, match="binding or sequence"):
        DurableReservationLedger(
            output,
            launch_binding={"config_sha256": "c" * 64, "git_head": "b" * 40},
        )

    path = output / "reservation_ledger.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["tokens"] += 1
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        DurableReservationLedger(output, launch_binding=binding)


def test_valid_prefix_ledger_truncation_cannot_repeat_18_call_canary(tmp_path):
    output = tmp_path / "campaign"
    binding = {"config_sha256": "a" * 64, "git_head": "b" * 40}
    ledger = DurableReservationLedger(output, launch_binding=binding)
    for index in range(18):
        reservation = _reservation_payload(index)
        ledger.reserve(reservation)
        ledger.resolve(_resolution_payload(reservation))
    checkpoint = ledger.reconcile()["checkpoint"]
    ledger.verify_checkpoint(checkpoint)

    lines = (output / "reservation_ledger.jsonl").read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    assert len(lines) == 36
    (output / "reservation_ledger.jsonl").write_text(
        "".join(lines[:-4]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="anchor high-water"):
        DurableReservationLedger(output, launch_binding=binding)


def test_existing_or_unanchored_cell_artifacts_are_never_pending(tmp_path):
    output = tmp_path / "campaign"
    artifact, _, _ = _artifact_paths(
        output,
        seed=22000,
        family=ScenarioFamily.SINGLE_COMPLEMENTARY,
        method=RecruitmentMethod.OPEN_VOLUNTEER,
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    jobs = (
        (
            22000,
            ScenarioFamily.SINGLE_COMPLEMENTARY,
            RecruitmentMethod.OPEN_VOLUNTEER,
        ),
    )
    present = _validate_managed_cell_inventory(output, jobs)
    empty = {
        "records": 0,
        "reservations": {},
        "resolutions": {},
        "unresolved": [],
        "overages": [],
    }
    with pytest.raises(RuntimeError, match="empty anchored ledger"):
        _assert_artifacts_bound_to_ledger(empty, present)
    with pytest.raises(RuntimeError, match="partial managed cell artifacts"):
        _load_completed_or_pending(
            output,
            seed=22000,
            family=ScenarioFamily.SINGLE_COMPLEMENTARY,
            method=RecruitmentMethod.OPEN_VOLUNTEER,
            config_sha256="config",
            source_hashes={},
            launch_binding={},
            protocol={},
            expected_resolved_model=None,
            reservation_reconciliation=empty,
        )


def test_empty_ledger_gate_and_unknown_root_entries_fail_before_dispatch(tmp_path):
    jobs = (
        (
            22000,
            ScenarioFamily.SINGLE_COMPLEMENTARY,
            RecruitmentMethod.OPEN_VOLUNTEER,
        ),
    )
    output = tmp_path / "gate_only"
    output.mkdir()
    (output / "canary_gate.json").write_text("{}\n", encoding="utf-8")
    managed = _validate_managed_cell_inventory(output, jobs)
    assert managed == 1
    with pytest.raises(RuntimeError, match="empty anchored ledger"):
        _assert_artifacts_bound_to_ledger(
            {
                "records": 0,
                "reservations": {},
                "resolutions": {},
                "unresolved": [],
                "overages": [],
            },
            managed,
        )

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / ".run_manifest_canary.json.tmp-crash").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unexpected managed output-root entry"):
        _validate_managed_cell_inventory(unknown, jobs)


def test_managed_roots_reject_symlink_hardlink_and_parent_aliases(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink component"):
        _normalize_output_root(alias_parent / "campaign")

    real_root = tmp_path / "root"
    real_root.mkdir()
    root_alias = tmp_path / "root_alias"
    root_alias.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink component"):
        with _output_root_lock(root_alias):
            pass

    binding = {"config_sha256": "a" * 64, "git_head": "b" * 40}
    first = tmp_path / "first"
    first.mkdir()
    ledger = DurableReservationLedger(first, launch_binding=binding)
    ledger.reserve(_reservation_payload())
    second = tmp_path / "second"
    second.mkdir()
    os.link(first / "reservation_ledger.jsonl", second / "reservation_ledger.jsonl")
    with pytest.raises(RuntimeError, match="multiple hard links"):
        DurableReservationLedger(second, launch_binding=binding)

    lock_a = tmp_path / "lock_a"
    lock_b = tmp_path / "lock_b"
    with _output_root_lock(lock_a):
        pass
    lock_b.mkdir()
    os.link(lock_a / ".e2b-hosted.lock", lock_b / ".e2b-hosted.lock")
    with pytest.raises(RuntimeError, match="multiple hard links"):
        with _output_root_lock(lock_b):
            pass

    shared_a = tmp_path / "shared_a"
    shared_b = tmp_path / "shared_b"
    jobs = (
        (
            22000,
            ScenarioFamily.SINGLE_COMPLEMENTARY,
            RecruitmentMethod.OPEN_VOLUNTEER,
        ),
    )
    artifact_a = _artifact_paths(
        shared_a,
        seed=22000,
        family=ScenarioFamily.SINGLE_COMPLEMENTARY,
        method=RecruitmentMethod.OPEN_VOLUNTEER,
    )[0]
    artifact_b = _artifact_paths(
        shared_b,
        seed=22000,
        family=ScenarioFamily.SINGLE_COMPLEMENTARY,
        method=RecruitmentMethod.OPEN_VOLUNTEER,
    )[0]
    artifact_a.parent.mkdir(parents=True)
    artifact_b.parent.mkdir(parents=True)
    artifact_a.write_text("{}\n", encoding="utf-8")
    os.link(artifact_a, artifact_b)
    for root in (shared_a, shared_b):
        with pytest.raises(RuntimeError, match="multiple hard links"):
            _validate_managed_cell_inventory(root, jobs)

    linked_root = tmp_path / "linked_artifacts"
    linked_root.mkdir()
    (linked_root / "episodes").symlink_to(
        artifact_a.parent,
        target_is_directory=True,
    )
    with pytest.raises(RuntimeError, match="output directory is linked"):
        _validate_managed_cell_inventory(linked_root, jobs)


def test_every_new_directory_ancestor_is_fsynced(tmp_path, monkeypatch):
    observed = []
    original = recruitment_runner._fsync_directory

    def tracked(path):
        observed.append(path)
        original(path)

    monkeypatch.setattr(recruitment_runner, "_fsync_directory", tracked)
    target = tmp_path / "one" / "two" / "three"
    recruitment_runner._ensure_directory(target)

    for directory in (tmp_path / "one", tmp_path / "one" / "two", target):
        assert directory in observed
        assert directory.parent in observed


def test_budget_poison_is_sticky_but_inflight_resolution_can_finish():
    resolutions = []
    budget = CampaignBudget(
        logical_limit=4,
        provider_attempt_limit=8,
        token_limit=100_000,
        resolution_callback=resolutions.append,
    )
    first = _reservation_payload(0)["reservation_key"]
    second = _reservation_payload(1)["reservation_key"]
    assert (
        budget.reserve(
            prompt_bytes=100,
            max_output_tokens=1024,
            max_provider_attempts=2,
            reservation_key=first,
        )
        is None
    )
    assert (
        budget.reserve(
            prompt_bytes=100,
            max_output_tokens=1024,
            max_provider_attempts=2,
            reservation_key=second,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="actual provider usage exceeds"):
        budget.resolve(
            reservation_key=first,
            provider_attempts_actual=1,
            input_tokens=100,
            output_tokens=1025,
            outcome="abstain",
        )
    assert budget.snapshot()["poisoned"]
    assert (
        budget.reserve(
            prompt_bytes=100,
            max_output_tokens=1024,
            max_provider_attempts=2,
            reservation_key=_reservation_payload(2)["reservation_key"],
        )
        == "poisoned"
    )
    budget.resolve(
        reservation_key=second,
        provider_attempts_actual=1,
        input_tokens=100,
        output_tokens=10,
        outcome="abstain",
    )
    assert budget.snapshot()["unresolved_in_process"] == 0
    assert [value["outcome"] for value in resolutions] == [
        "usage_exceeded",
        "abstain",
    ]


def test_resolution_callback_failure_also_poison_stops_dispatch():
    def fail_resolution(payload):
        del payload
        raise OSError("simulated durable resolution failure")

    budget = CampaignBudget(
        logical_limit=2,
        provider_attempt_limit=4,
        token_limit=100_000,
        resolution_callback=fail_resolution,
    )
    key = _reservation_payload()["reservation_key"]
    assert (
        budget.reserve(
            prompt_bytes=100,
            max_output_tokens=1024,
            max_provider_attempts=2,
            reservation_key=key,
        )
        is None
    )
    with pytest.raises(OSError, match="durable resolution"):
        budget.resolve(
            reservation_key=key,
            provider_attempts_actual=1,
            input_tokens=100,
            output_tokens=10,
            outcome="abstain",
        )
    assert budget.snapshot()["poisoned_reason"] == "resolution_callback_failure"
    assert (
        budget.reserve(
            prompt_bytes=100,
            max_output_tokens=1024,
            max_provider_attempts=2,
            reservation_key=_reservation_payload(1)["reservation_key"],
        )
        == "poisoned"
    )


def test_malformed_raw_provider_envelope_cannot_persist_or_gate(tmp_path):
    class _Responses:
        def create(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                id="resp_malformed",
                model="gpt-5.6-luna",
                output_text="ABSTAIN",
                status="completed",
                incomplete_details=None,
                usage=None,
            )

    class _SDK:
        responses = _Responses()

    client_config = SimpleNamespace(
        client_name="openai_responses",
        model_id="gpt-5.6-luna",
        base_url=None,
        timeout=30,
        generate_kwargs={
            "max_output_tokens": 1024,
            "preserve_completion_whitespace": True,
            "strict_response_envelope": True,
        },
        max_retries=0,
        delay=0,
        alternate_roles=False,
        enable_thinking=False,
    )
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    episode = run_llm_recruitment_episode(
        scenario,
        ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
        client_factory=lambda agent_id: OpenAIResponsesWrapper(
            client_config,
            sdk_client=_SDK(),
        ),
    )
    assert episode["provider_model"]["resolved"] is None
    assert {
        call["validation_code"] for call in episode["call_ledger"]
    } == {"transport.exception"}
    with pytest.raises(RuntimeError, match="comprehensive validation"):
        persist_completed_episode(
            tmp_path,
            episode,
            config_sha256="config",
            source_hashes={"source": "hash"},
        )
    assert not (tmp_path / "canary_gate.json").exists()


def test_reasoning_only_completion_at_v4_cap_cannot_persist_or_promote(tmp_path):
    class _TruncatedClient:
        def generate(self, messages):
            del messages
            return _response("")._replace(
                status="incomplete",
                incomplete_reason="max_output_tokens",
                output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                reasoning_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            )

    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    episode = run_llm_recruitment_episode(
        scenario,
        ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
        client_factory=lambda agent_id: _TruncatedClient(),
    )

    assert {
        call["validation_code"] for call in episode["call_ledger"]
    } == {"provider.status_not_completed"}
    assert all(
        call["status"] == "incomplete"
        and call["incomplete_reason"] == "max_output_tokens"
        and call["output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
        and call["reasoning_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS
        for call in episode["call_ledger"]
    )
    with pytest.raises(RuntimeError, match="comprehensive validation"):
        persist_completed_episode(
            tmp_path,
            episode,
            config_sha256="config",
            source_hashes={"source": "hash"},
        )
    assert not (tmp_path / "canary_gate.json").exists()


def test_poisoned_manifest_cannot_resume_before_provider_dispatch(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "campaign"
    output.mkdir()
    manifest_path = output / "run_manifest_canary.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    binding = {
        "config_sha256": "a" * 64,
        "git_head": "b" * 40,
        "source_hashes": {},
    }
    checkpoint = DurableReservationLedger(
        output,
        launch_binding=binding,
    ).reconcile()["checkpoint"]
    failed_manifest = {
        "status": "failed",
        "reservation_ledger_at_launch": {"checkpoint": checkpoint},
        "reservation_ledger_final": {"checkpoint": checkpoint},
        "campaign_budget": {
            "poisoned": True,
            "poisoned_reason": "provider_usage_exceeded",
        },
    }
    monkeypatch.setattr(recruitment_runner, "_require_hosted_credentials", lambda: None)
    monkeypatch.setattr(recruitment_runner, "_launch_binding", lambda protocol: binding)
    monkeypatch.setattr(
        recruitment_runner,
        "_load_bound_manifest",
        lambda *args, **kwargs: failed_manifest,
    )
    monkeypatch.setattr(
        recruitment_runner,
        "_client_factory",
        lambda config: pytest.fail("provider client must not be constructed"),
    )
    args = SimpleNamespace(stage="canary", resume=True)
    with pytest.raises(RuntimeError, match="poisoned or failed run"):
        recruitment_runner._run_hosted_locked(
            args=args,
            output=output,
            jobs=(),
            protocol={},
            launch_binding=binding,
        )


def test_hosted_credentials_fail_closed_and_parallel_cells_are_explicit(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _require_hosted_credentials()
    monkeypatch.setenv("OPENAI_API_KEY", " ")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _require_hosted_credentials()

    monkeypatch.setattr(sys, "argv", ["run_recruitment_llm_screen.py"])
    default = _parse_args()
    assert not default.parallel_cells
    assert default.workers == 1
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_recruitment_llm_screen.py",
            "--parallel-cells",
            "--workers",
            "3",
        ],
    )
    explicit = _parse_args()
    assert explicit.parallel_cells
    assert explicit.workers == 3


def test_cell_validation_allows_other_active_cell_but_campaign_gate_stops(
    tmp_path,
):
    binding = {"config_sha256": "a" * 64, "git_head": "b" * 40}
    ledger = DurableReservationLedger(tmp_path, launch_binding=binding)
    campaign = CampaignBudget(
        logical_limit=20,
        provider_attempt_limit=40,
        token_limit=1_000_000,
        reservation_callback=ledger.reserve,
        resolution_callback=ledger.resolve,
    )
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    episode = run_llm_recruitment_episode(
        scenario,
        ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
        client_factory=lambda agent_id: _SequenceClient(["ABSTAIN"]),
        campaign_budget=campaign,
    )
    other = {
        "reservation_key": {
            "seed": 22001,
            "family": "single_complementary",
            "method": "open_volunteer",
            "round_index": 0,
            "agent_id": 0,
            "semantic_attempt": 0,
        },
        "logical_calls": 1,
        "provider_attempts": 2,
        "tokens": (100 + DEFAULT_PROMPT_FRAMING_TOKENS + 1024) * 2,
        "prompt_bytes": 100,
        "prompt_framing_tokens": DEFAULT_PROMPT_FRAMING_TOKENS,
        "max_output_tokens": 1024,
    }
    ledger.reserve(other)
    reconciliation = ledger.reconcile()
    marker = persist_completed_episode(
        tmp_path,
        episode,
        config_sha256="config-hash",
        source_hashes={"source.py": "abc"},
        launch_binding=binding,
        reservation_reconciliation=reconciliation,
        require_reservations=True,
    )
    assert marker["logical_calls"] == 6
    with pytest.raises(RuntimeError, match="unresolved pre-dispatch reservations"):
        _assert_reservation_coverage(reconciliation, [marker])


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (
            _response("ABSTAIN")._replace(model_id="gpt-5.6-luna-wrong"),
            "provider.model_not_accepted",
        ),
        (
            _response("ABSTAIN")._replace(model_id="gpt-5.6-luna-2026-99-99"),
            "provider.model_not_accepted",
        ),
        (
            _response("ABSTAIN")._replace(
                status="incomplete",
                incomplete_reason="max_output_tokens",
            ),
            "provider.status_not_completed",
        ),
        (
            _response("ABSTAIN")._replace(response_id=""),
            "provider.missing_response_id",
        ),
    ],
)
def test_wrong_model_incomplete_or_missing_id_fails_without_semantic_repair(
    response,
    expected_code,
):
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    view = build_agent_view(
        scenario,
        _directory(scenario, RecruitmentMethod.OPEN_VOLUNTEER),
        agent_id=0,
        round_index=0,
    )

    class _Client:
        def generate(self, messages):
            del messages
            return response

    decision = request_control_record(
        client=_Client(),
        view=view,
        config=ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
        episode_budget=SharedCallBudget(2),
    )

    assert decision.record is None
    assert decision.abstain_code == expected_code
    assert decision.semantic_repairs == 0
    assert decision.calls[0]["validation_code"] == expected_code


@pytest.mark.parametrize(
    "raw",
    (
        "ABSTAIN\n",
        " ABSTAIN",
        "TFP1|TYPE=APPLY|TASK=single.t0|CAP=80,20,20|COST=7\n",
    ),
)
def test_raw_whitespace_is_archived_exactly_and_rejected(raw):
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    view = build_agent_view(
        scenario,
        _directory(scenario, RecruitmentMethod.OPEN_VOLUNTEER),
        agent_id=0,
        round_index=0,
    )
    client = _SequenceClient([raw, "ABSTAIN"])
    decision = request_control_record(
        client=client,
        view=view,
        config=ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
        episode_budget=SharedCallBudget(2),
    )

    assert decision.calls[0]["_debug"]["raw_completion"] == raw
    assert decision.calls[0]["raw_completion_sha256"]
    assert not decision.calls[0]["valid"]
    assert decision.semantic_repairs == 1
    assert decision.abstain_code == "model.abstain"


def test_tfp1_byte_boundary_and_framing_usage_overage_fail_closed():
    assert parse_tfp1("x" * 256).payload_bytes == 256
    assert parse_tfp1("x" * 256).code != "parse.too_many_bytes"
    assert parse_tfp1("x" * 257).code == "parse.too_many_bytes"

    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    view = build_agent_view(
        scenario,
        _directory(scenario, RecruitmentMethod.OPEN_VOLUNTEER),
        agent_id=0,
        round_index=0,
    )
    events = []

    class _Client:
        def generate(self, messages):
            del messages
            assert events and events[0][0] == "reserve"
            return _response("ABSTAIN")._replace(
                output_tokens=DEFAULT_MAX_OUTPUT_TOKENS + 1
            )

    budget = CampaignBudget(
        logical_limit=2,
        provider_attempt_limit=4,
        token_limit=100_000,
        reservation_callback=lambda payload: events.append(("reserve", payload)),
        resolution_callback=lambda payload: events.append(("resolve", payload)),
    )
    with pytest.raises(RuntimeError, match="actual provider usage exceeds"):
        request_control_record(
            client=_Client(),
            view=view,
            config=ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
            episode_budget=SharedCallBudget(2),
            campaign_budget=budget,
            reservation_context={
                "seed": 22000,
                "family": "single_complementary",
                "method": "open_volunteer",
            },
        )
    assert events[0][1]["prompt_framing_tokens"] == DEFAULT_PROMPT_FRAMING_TOKENS
    assert events[1][1]["outcome"] == "usage_exceeded"


def test_actual_input_usage_above_prompt_plus_framing_fails_closed():
    scenario = generate_scenario(ScenarioFamily.SINGLE_COMPLEMENTARY, 22000)
    view = build_agent_view(
        scenario,
        _directory(scenario, RecruitmentMethod.OPEN_VOLUNTEER),
        agent_id=0,
        round_index=0,
    )

    class _Client:
        def generate(self, messages):
            prompt_bytes = sum(len(message.content.encode("utf-8")) for message in messages)
            return _response("ABSTAIN")._replace(
                input_tokens=prompt_bytes + DEFAULT_PROMPT_FRAMING_TOKENS + 1
            )

    budget = CampaignBudget(
        logical_limit=2,
        provider_attempt_limit=4,
        token_limit=100_000,
    )
    with pytest.raises(RuntimeError, match="actual provider usage exceeds"):
        request_control_record(
            client=_Client(),
            view=view,
            config=ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=1),
            episode_budget=SharedCallBudget(2),
            campaign_budget=budget,
            reservation_context={
                "seed": 22000,
                "family": "single_complementary",
                "method": "open_volunteer",
            },
        )


def test_completed_cell_recomputes_analysis_after_artifact_rehash(tmp_path):
    markers, _, source_hashes = _persist_passing_canary(tmp_path)
    marker = next(
        marker
        for marker in markers
        if marker["family"] == "single_complementary"
        and marker["method"] == "open_volunteer"
    )
    artifact_path = tmp_path / marker["artifact"]
    marker_path = (
        tmp_path / "markers" / "open_volunteer__single_complementary__seed_22000.complete.json"
    )
    episode = json.loads(artifact_path.read_text(encoding="utf-8"))
    episode["analysis_only"]["true_feasible_locked_tasks"] += 1
    artifact_path.write_text(json.dumps(episode), encoding="utf-8")
    tampered_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    tampered_marker["artifact_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(tampered_marker), encoding="utf-8")

    assert (
        load_completed_marker(
            tmp_path,
            seed=22000,
            family=ScenarioFamily.SINGLE_COMPLEMENTARY,
            method=RecruitmentMethod.OPEN_VOLUNTEER,
            config_sha256="config-hash",
            source_hashes=source_hashes,
        )
        is None
    )


def test_llm_formed_team_preserves_exclusivity_private_routing_and_replay():
    scenario = generate_scenario(ScenarioFamily.TWO_DISJOINT, 22000)
    task_id = "disjoint.t0"

    class _FormationClient:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        def generate(self, messages):
            payload = json.loads(messages[1].content.removeprefix("AGENT_VIEW_JSON="))
            round_index = payload["round_index"]
            if round_index == 0 and self.agent_id in {0, 4}:
                profile = scenario.agents[self.agent_id]
                return _response(
                    RecruitmentRecord(
                        RecordKind.APPLY,
                        task_id,
                        capabilities=profile.true_capabilities,
                        cost=profile.task_costs[task_id],
                    ).render()
                )
            if round_index == 1 and self.agent_id in {0, 4}:
                return _response(
                    RecruitmentRecord(
                        RecordKind.ACCEPT,
                        task_id,
                        members=(0, 4),
                    ).render()
                )
            if round_index == 2 and self.agent_id == 4:
                return _response(
                    RecruitmentRecord(
                        RecordKind.LOCK,
                        task_id,
                        members=(0, 4),
                    ).render()
                )
            return _response("ABSTAIN")

    episode = run_llm_recruitment_episode(
        scenario,
        ScreenConfig(method=RecruitmentMethod.OPEN_VOLUNTEER, rounds=4),
        client_factory=lambda agent_id: _FormationClient(agent_id),
    )

    assert episode["analysis_only"]["locked_rosters"][task_id] == [0, 4]
    assert episode["replay_hash_match"]
    directory = TeamDirectory.replay(episode["directory_replay"])
    assert directory.agent_to_task == {0: task_id, 4: task_id}

    profile = scenario.agents[0]
    application = RecruitmentRecord(
        RecordKind.APPLY,
        "disjoint.t1",
        capabilities=profile.true_capabilities,
        cost=profile.task_costs["disjoint.t1"],
    )
    directory.submit_control(sender=0, sent_round=4, raw=application.render())
    transition = directory.advance(5)
    directory.deliver_ordinary(5)
    assert transition[-1].code == "state.agent_already_teamed"
    assert not transition[-1].accepted

    routed = directory.submit_ordinary(sender=0, sent_round=5, content="team-only")
    rejected = directory.submit_ordinary(sender=1, sent_round=5, content="outsider")
    assert routed.accepted and routed.recipients == (4,)
    assert not rejected.accepted
    directory.advance(6)
    delivered = directory.deliver_ordinary(6)
    assert delivered[0].accepted and delivered[0].recipients == (4,)
    replayed = TeamDirectory.replay(directory.export_replay())
    assert replayed.state_hash() == directory.state_hash()
    assert replayed.audit_chain_hash == directory.audit_chain_hash
