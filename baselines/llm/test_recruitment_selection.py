"""Single mathematical contract test for E2 roster selection and its oracle."""

import random
import sys
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import pytest

EVAL_UTILS = Path(__file__).resolve().parent / "eval_utils"
if str(EVAL_UTILS) not in sys.path:
    sys.path.insert(0, str(EVAL_UTILS))

from recruitment_selection import (  # noqa: E402
    InformationSource,
    claimed_feasibility,
    exact_utility,
    first_valid,
    random_valid,
    roster_utility,
    true_information_oracle,
)


def _f(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _reference_feasible(task, roster, agents, information):
    if len(roster) != task["team_size"]:
        return False
    key = f"{information}_capabilities"
    return all(
        sum(_f(agents[agent_id][key][dimension]) for agent_id in roster) >= _f(demand)
        for dimension, demand in enumerate(task["demand"])
        if demand > 0
    )


def _reference_utility(task, roster, agents, information):
    capability_key = f"{information}_capabilities"
    cost_key = f"{information}_costs"
    ratios = [
        sum(_f(agents[agent_id][capability_key][dimension]) for agent_id in roster)
        / _f(demand)
        for dimension, demand in enumerate(task["demand"])
        if demand > 0
    ]
    raw_cost = sum(
        _f(agents[agent_id][cost_key][task["task_id"]]) for agent_id in roster
    )
    return min(ratios) - Fraction(1, 4) * raw_cost / (100 * len(roster))


def _reference_candidate_rosters(task, candidates):
    earliest = {}
    for candidate in candidates:
        agent_id = candidate["agent_id"]
        arrival = candidate["arrival_round"]
        earliest[agent_id] = min(arrival, earliest.get(agent_id, arrival))
    ordered = sorted(earliest)
    return [
        (tuple(roster), max(earliest[agent_id] for agent_id in roster))
        for roster in combinations(ordered, task["team_size"])
    ]


def _reference_selectors(task, candidates, agents, seed):
    feasible = [
        (roster, arrival)
        for roster, arrival in _reference_candidate_rosters(task, candidates)
        if _reference_feasible(task, roster, agents, "claimed")
    ]
    unique = sorted({roster for roster, _ in feasible})
    first = min(feasible, key=lambda item: (item[1], item[0]))[0]
    random_choice = random.Random(seed).choice(unique)
    exact = min(
        unique,
        key=lambda roster: (
            -_reference_utility(task, roster, agents, "claimed"),
            roster,
        ),
    )
    return first, random_choice, exact, len(unique)


def _assignment_key(labels):
    return tuple((1, "") if label is None else (0, label) for label in labels)


def _reference_oracle(tasks, agents):
    agent_ids = tuple(sorted(agents))
    task_by_id = {task["task_id"]: task for task in tasks}
    task_ids = tuple(sorted(task_by_id))
    evaluated = 0
    feasible_count = 0
    best = None
    for labels in product((*task_ids, None), repeat=6):
        evaluated += 1
        rosters = {
            task_id: tuple(
                agent_id
                for agent_id, label in zip(agent_ids, labels, strict=True)
                if label == task_id
            )
            for task_id in task_ids
        }
        if any(
            roster
            and not _reference_feasible(
                task_by_id[task_id],
                roster,
                agents,
                "true",
            )
            for task_id, roster in rosters.items()
        ):
            continue
        feasible_count += 1
        completed = {task_id: roster for task_id, roster in rosters.items() if roster}
        total_reward = sum(
            (_f(task_by_id[task_id]["reward"]) for task_id in completed),
            Fraction(0),
        )
        total_utility = sum(
            (
                _reference_utility(
                    task_by_id[task_id],
                    roster,
                    agents,
                    "true",
                )
                for task_id, roster in completed.items()
            ),
            Fraction(0),
        )
        total_cost = sum(
            (
                _f(agents[agent_id]["true_costs"][task_id])
                for task_id, roster in completed.items()
                for agent_id in roster
            ),
            Fraction(0),
        )
        score = (total_reward, total_utility, -total_cost)
        candidate = (score, _assignment_key(labels), labels, completed)
        if (
            best is None
            or candidate[0] > best[0]
            or (candidate[0] == best[0] and candidate[1] < best[1])
        ):
            best = candidate
    assert best is not None
    score, _, labels, completed = best
    return {
        "agent_assignments": tuple(zip(agent_ids, labels, strict=True)),
        "completed": completed,
        "reward": score[0],
        "utility": score[1],
        "cost": -score[2],
        "evaluated": evaluated,
        "feasible": feasible_count,
    }


def _agents(*, scarce_true_miner=False):
    capabilities = (
        (40, 0, 0),
        (85, 0, 0),
        (0, 90, 0),
        (0, 75, 0),
        (0, 0, 80),
        (0, 0, 95),
    )
    result = {}
    for agent_id, capability in enumerate(capabilities):
        true_capability = capability
        claimed_capability = capability
        if scarce_true_miner and agent_id == 4:
            true_capability = (0, 0, 0)
            claimed_capability = (0, 0, 95)
        if scarce_true_miner and agent_id == 5:
            claimed_capability = (0, 0, 40)
        result[agent_id] = {
            # Mapping keys supply IDs, exercising the structural fallback.
            "claimed_capabilities": claimed_capability,
            "true_capabilities": true_capability,
            "claimed_costs": {
                "alpha": 10 + 3 * agent_id,
                "beta": 20 + 2 * agent_id,
            },
            "true_costs": {
                "alpha": 12 + 2 * agent_id,
                "beta": 18 + 3 * agent_id,
            },
        }
    return result


CASES = (
    {
        "name": "disjoint_complementary",
        "agents": _agents(),
        "tasks": (
            {
                "task_id": "alpha",
                "demand": (70, 70, 0),
                "team_size": 2,
                "reward": 100,
            },
            {
                "task_id": "beta",
                "demand": (0, 70, 70),
                "team_size": 2,
                "reward": 100,
            },
        ),
        "seed": 20003,
    },
    {
        "name": "scarce_true_capability",
        "agents": _agents(scarce_true_miner=True),
        "tasks": (
            {
                "task_id": "alpha",
                "demand": (70, 0, 70),
                "team_size": 2,
                "reward": 120,
            },
            {
                "task_id": "beta",
                "demand": (70, 0, 70),
                "team_size": 2,
                "reward": 100,
            },
        ),
        "seed": 20777,
    },
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_selection_rules_and_true_oracle_match_bruteforce(case):
    agents = case["agents"]
    tasks = case["tasks"]
    task = tasks[0]
    candidates = [
        {"agent_id": agent_id, "arrival_round": round_number}
        for agent_id, round_number in ((0, 2), (1, 1), (2, 1), (3, 3), (4, 1), (5, 2))
    ]
    # A duplicate later bid must not bias random selection or formation order.
    candidates.append({"agent_id": 0, "arrival_round": 9})

    for roster in combinations(range(6), task["team_size"]):
        assert claimed_feasibility(task, roster, agents) is _reference_feasible(
            task,
            roster,
            agents,
            "claimed",
        )
        if claimed_feasibility(task, roster, agents):
            assert roster_utility(task, roster, agents) == _reference_utility(
                task,
                roster,
                agents,
                "claimed",
            )

    # Regression: nonzero coverage is not enough; it must meet demand (40 < 70).
    insufficient = SimpleNamespace(
        task_id="alpha",
        demand=(70, 0, 0),
        required_size=1,
        reward=1,
    )
    assert not claimed_feasibility(insufficient, (0,), agents)

    expected_first, expected_random, expected_exact, feasible_count = (
        _reference_selectors(task, candidates, agents, case["seed"])
    )
    first = first_valid(task, candidates, agents)
    sampled = random_valid(task, candidates, agents, seed=case["seed"])
    optimized = exact_utility(task, candidates, agents)
    assert first.roster == expected_first
    assert sampled.roster == expected_random
    assert optimized.roster == expected_exact
    assert {
        first.feasible_roster_count,
        sampled.feasible_roster_count,
        optimized.feasible_roster_count,
    } == {feasible_count}
    assert optimized.utility == _reference_utility(
        task,
        expected_exact,
        agents,
        "claimed",
    )

    oracle = true_information_oracle(tasks, agents)
    reference = _reference_oracle(tasks, agents)
    assert oracle.agent_assignments == reference["agent_assignments"]
    assert {
        assignment.task_id: assignment.roster for assignment in oracle.completed_tasks
    } == reference["completed"]
    assert oracle.total_completed_reward == reference["reward"]
    assert oracle.total_roster_utility == reference["utility"]
    assert oracle.total_raw_cost == reference["cost"]
    assert oracle.assignments_evaluated == reference["evaluated"] == 3**6
    assert oracle.feasible_assignments == reference["feasible"]

    # The public utility supports true information without requiring arena types.
    for assignment in oracle.completed_tasks:
        assert assignment.roster_utility == roster_utility(
            next(task for task in tasks if task["task_id"] == assignment.task_id),
            assignment.roster,
            agents,
            information=InformationSource.TRUE,
        )
