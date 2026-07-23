from itertools import combinations

import pytest

from baselines.llm.eval_utils.recruitment_selection import (
    SelectionMethod,
    true_information_oracle,
)
from baselines.llm.eval_utils.team_formation import RecruitmentMethod
from baselines.llm.recruitment_arena import (
    AGENT_IDS,
    ContractSelectionPolicy,
    ScenarioFamily,
    TaskChoicePolicy,
    generate_scenario,
    run_scripted_episode,
)


@pytest.mark.parametrize(
    ("family", "oracle_reward", "oracle_tasks", "individually_feasible"),
    [
        (ScenarioFamily.SINGLE_COMPLEMENTARY, 100, 1, 1),
        (ScenarioFamily.TWO_DISJOINT, 200, 2, 2),
        (ScenarioFamily.SCARCE_CAPABILITY, 100, 1, 2),
        (ScenarioFamily.OVERSUBSCRIBED, 100, 1, 1),
    ],
)
def test_scenario_family_oracle_invariants(
    family,
    oracle_reward,
    oracle_tasks,
    individually_feasible,
):
    scenario = generate_scenario(family, 20000)
    oracle = true_information_oracle(scenario.tasks, scenario.agents)
    profiles = {agent.agent_id: agent for agent in scenario.agents}

    assert len(scenario.agents) == 6
    assert sorted(agent.public_role for agent in scenario.agents) == [
        "forager",
        "forager",
        "miner",
        "miner",
        "warrior",
        "warrior",
    ]
    assert int(oracle.total_completed_reward) == oracle_reward
    assert len(oracle.completed_tasks) == oracle_tasks
    assert (
        sum(
            any(
                all(
                    sum(profiles[member].true_capabilities[dimension] for member in members)
                    >= demand
                    for dimension, demand in enumerate(task.demand)
                    if demand > 0
                )
                for members in combinations(AGENT_IDS, task.required_size)
            )
            for task in scenario.tasks
        )
        == individually_feasible
    )


@pytest.mark.parametrize("family", list(ScenarioFamily))
@pytest.mark.parametrize("method", list(RecruitmentMethod))
@pytest.mark.parametrize("task_choice", list(TaskChoicePolicy))
def test_scripted_arena_integrity_and_replay(family, method, task_choice):
    scenario = generate_scenario(family, 20000)
    oracle = true_information_oracle(scenario.tasks, scenario.agents)
    episode = run_scripted_episode(
        scenario,
        method,
        task_choice=task_choice,
        oracle=oracle,
    )
    metrics = episode.metrics

    assert metrics.valid_control_submissions == metrics.control_submissions
    assert metrics.rejected_control_transitions == 0
    assert metrics.unauthorized_ordinary_deliveries == 0
    assert metrics.roster_agreement_rate == 1
    assert metrics.replay_hash_match
    assert metrics.model_calls == metrics.provider_requests == 0
    assert metrics.transport_errors == 0
    assert 0 <= metrics.normalized_reward <= 1
    assert 0 <= metrics.formation_rate <= 1
    assert 0 <= metrics.oracle_allocation_coverage <= 1


def test_public_sweep_recovers_disjoint_allocation_deterministically():
    scenario = generate_scenario(ScenarioFamily.TWO_DISJOINT, 20000)
    local = run_scripted_episode(
        scenario,
        RecruitmentMethod.OPEN_VOLUNTEER,
        task_choice=TaskChoicePolicy.LOCAL_COMMIT,
    )
    sweep = run_scripted_episode(
        scenario,
        RecruitmentMethod.OPEN_VOLUNTEER,
        task_choice=TaskChoicePolicy.PUBLIC_SWEEP,
    )
    repeated = run_scripted_episode(
        scenario,
        RecruitmentMethod.OPEN_VOLUNTEER,
        task_choice=TaskChoicePolicy.PUBLIC_SWEEP,
    )

    assert local.metrics.achieved_reward == 100
    assert sweep.metrics.achieved_reward == 200
    assert sweep.metrics.control_delivered_bytes > local.metrics.control_delivered_bytes
    assert sweep.metrics.terminal_state_hash == repeated.metrics.terminal_state_hash
    assert sweep.metrics.terminal_audit_chain_hash == repeated.metrics.terminal_audit_chain_hash


@pytest.mark.parametrize(
    "selector",
    [
        *SelectionMethod,
        ContractSelectionPolicy.JOINT_EXACT_ALLOCATION,
    ],
)
def test_contract_selector_ablation_is_deterministic_and_auditable(selector):
    scenario = generate_scenario(ScenarioFamily.TWO_DISJOINT, 20000)
    first = run_scripted_episode(
        scenario,
        RecruitmentMethod.CONTRACT_NET,
        task_choice=TaskChoicePolicy.PUBLIC_SWEEP,
        selector=selector,
    )
    repeated = run_scripted_episode(
        scenario,
        RecruitmentMethod.CONTRACT_NET,
        task_choice=TaskChoicePolicy.PUBLIC_SWEEP,
        selector=selector,
    )
    selector_event = next(event for event in first.events if event["phase"] == "selector_audit")

    assert first.method.endswith(f"__{selector.value}")
    assert first.as_dict() == repeated.as_dict()
    assert selector_event["information_source"] == "claimed"
    assert selector_event["decisions"]
    assert all(
        decision["information_source"] == "claimed"
        and "true_capabilities" not in decision
        and "true_costs" not in decision
        for decision in selector_event["decisions"]
    )
    assert first.metrics.valid_control_submissions == first.metrics.control_submissions
    assert first.metrics.rejected_control_transitions == 0
    assert first.metrics.unauthorized_ordinary_deliveries == 0
    assert first.metrics.overstaff_agent_slots == 0
    assert first.metrics.replay_hash_match
    if selector == ContractSelectionPolicy.JOINT_EXACT_ALLOCATION:
        size_by_task = {task.task_id: task.required_size for task in scenario.tasks}
        for decision in selector_event["decisions"]:
            assignments = decision["selected_assignments"]
            members = [member for assignment in assignments for member in assignment["members"]]
            assert len(members) == len(set(members))
            assert all(
                len(assignment["members"]) == size_by_task[assignment["task_id"]]
                for assignment in assignments
            )
            assert decision["objective_order"][0] == "maximize_public_reward"


def test_default_contract_selector_preserves_e2a_behavior_and_label():
    scenario = generate_scenario(ScenarioFamily.TWO_DISJOINT, 20000)
    legacy = run_scripted_episode(
        scenario,
        RecruitmentMethod.CONTRACT_NET,
        task_choice=TaskChoicePolicy.PUBLIC_SWEEP,
    )
    explicit = run_scripted_episode(
        scenario,
        RecruitmentMethod.CONTRACT_NET,
        task_choice=TaskChoicePolicy.PUBLIC_SWEEP,
        selector=SelectionMethod.FIRST_VALID,
    )

    assert legacy.method == "contract_net__public_sweep"
    assert not any(event["phase"] == "selector_audit" for event in legacy.events)
    assert legacy.metrics.terminal_state_hash == explicit.metrics.terminal_state_hash
    assert legacy.metrics.terminal_audit_chain_hash == explicit.metrics.terminal_audit_chain_hash
