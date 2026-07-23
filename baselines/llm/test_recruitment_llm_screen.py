import gzip
import json
import threading
import time

import pytest

from baselines.llm.eval_utils.client import LLMResponse
from baselines.llm.eval_utils.team_formation import (
    RecordKind,
    RecruitmentMethod,
    RecruitmentRecord,
    TeamDirectory,
)
from baselines.llm.recruitment_arena import (
    AGENT_IDS,
    ScenarioFamily,
    generate_scenario,
)
from baselines.llm.recruitment_llm_screen import (
    CampaignBudget,
    ScreenConfig,
    SharedCallBudget,
    build_agent_view,
    request_control_record,
    run_llm_recruitment_episode,
)
from scripts.run_recruitment_llm_screen import (
    DEFAULT_LOGICAL_CALL_CAP,
    DEFAULT_PROVIDER_ATTEMPT_CAP,
    DEFAULT_TOKEN_EXPOSURE_CAP,
    _caps_conflicts,
    _launch_projection,
    _load_passing_canary_gate,
    _protocol_config,
    _validate_debug_shard,
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
    projection = _launch_projection(24)

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


def test_llm_formed_team_preserves_exclusivity_private_routing_replay_and_canary_gate(
    tmp_path,
):
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

    source_hashes = {"source.py": "abc"}
    marker = persist_completed_episode(
        tmp_path,
        episode,
        config_sha256="config-hash",
        source_hashes=source_hashes,
    )
    gate = _write_canary_gate(
        tmp_path,
        marker,
        config_sha256="config-hash",
        source_hashes=source_hashes,
    )
    assert gate["status"] == "pass"
    assert (
        _load_passing_canary_gate(
            tmp_path,
            marker,
            config_sha256="config-hash",
            source_hashes=source_hashes,
        )
        == gate
    )
