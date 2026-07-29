"""Deterministic roster selection and a true-information E2 oracle.

The public functions deliberately use structural inputs. Tasks, agents, and
candidate rosters may be dataclasses, ordinary objects, or mappings with the
documented field aliases. This keeps the mathematical slice independent of the
TFP1 runtime and RecruitmentArena implementations.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import combinations, product
from typing import Any

AgentId = Hashable
TaskId = Hashable

__all__ = [
    "AgentCandidate",
    "InformationSource",
    "OracleResult",
    "OracleTaskAssignment",
    "PublicJointAllocationResult",
    "PublicJointTaskAssignment",
    "RosterCandidate",
    "SelectionMethod",
    "SelectionResult",
    "claimed_feasibility",
    "exact_utility",
    "first_valid",
    "random_valid",
    "roster_utility",
    "public_joint_allocation",
    "true_information_oracle",
]

_MISSING = object()
_NO_DEFAULT = object()


class SelectionMethod(StrEnum):
    """Preregistered Peer Contract Net roster-selection rules."""

    FIRST_VALID = "first_valid"
    RANDOM_VALID = "random_valid"
    EXACT_UTILITY = "exact_utility"


class InformationSource(StrEnum):
    """Capability/cost information available to a calculation."""

    CLAIMED = "claimed"
    TRUE = "true"


@dataclass(frozen=True)
class RosterCandidate:
    """One public candidate roster and its first ledger arrival round."""

    members: tuple[AgentId, ...]
    arrival_round: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.arrival_round, bool) or not isinstance(self.arrival_round, int):
            raise TypeError("arrival_round must be an integer")
        if self.arrival_round < 0:
            raise ValueError("arrival_round must be non-negative")
        canonical = _canonical_roster(self.members)
        object.__setattr__(self, "members", canonical)


@dataclass(frozen=True)
class AgentCandidate:
    """One candidate agent/bid and its first ledger arrival round."""

    agent_id: AgentId
    arrival_round: int = 0

    def __post_init__(self) -> None:
        _stable_id_key(self.agent_id)
        if isinstance(self.arrival_round, bool) or not isinstance(self.arrival_round, int):
            raise TypeError("arrival_round must be an integer")
        if self.arrival_round < 0:
            raise ValueError("arrival_round must be non-negative")


@dataclass(frozen=True)
class SelectionResult:
    """Auditable output from one decentralized selection rule."""

    method: SelectionMethod
    roster: tuple[AgentId, ...] | None
    utility: Fraction | None
    feasible_roster_count: int


@dataclass(frozen=True)
class OracleTaskAssignment:
    """One task completed in the true-information oracle assignment."""

    task_id: TaskId
    roster: tuple[AgentId, ...]
    reward: Fraction
    roster_utility: Fraction
    raw_cost: Fraction


@dataclass(frozen=True)
class OracleResult:
    """Exact six-agent oracle, including every lexicographic objective term."""

    agent_assignments: tuple[tuple[AgentId, TaskId | None], ...]
    completed_tasks: tuple[OracleTaskAssignment, ...]
    total_completed_reward: Fraction
    total_roster_utility: Fraction
    total_raw_cost: Fraction
    assignments_evaluated: int
    feasible_assignments: int


@dataclass(frozen=True)
class PublicJointTaskAssignment:
    """One task selected from the shared claimed-information ledger."""

    task_id: TaskId
    roster: tuple[AgentId, ...]
    reward: Fraction
    roster_utility: Fraction
    raw_cost: Fraction


@dataclass(frozen=True)
class PublicJointAllocationResult:
    """Deterministic bounded allocation computed only from public task/bid state.

    This is the deployable comparison method. Unlike ``OracleResult``, it never
    reads private true capabilities or costs.
    """

    agent_assignments: tuple[tuple[AgentId, TaskId | None], ...]
    completed_tasks: tuple[PublicJointTaskAssignment, ...]
    total_public_reward: Fraction
    total_claimed_utility: Fraction
    total_claimed_cost: Fraction
    assignments_evaluated: int
    feasible_assignments: int


def _field(value: Any, names: Sequence[str], default: Any = _NO_DEFAULT) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not _NO_DEFAULT:
        return default
    raise TypeError(f"{type(value).__name__} must expose one of: {', '.join(names)}")


def _fraction(value: Any, *, label: str, non_negative: bool = True) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, not bool")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, float):
        if value != value or abs(value) == float("inf"):
            raise ValueError(f"{label} must be finite")
        result = Fraction(str(value))
    elif isinstance(value, (int, str)):
        try:
            result = Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{label} must be numeric") from exc
    else:
        raise TypeError(f"{label} must be int, float, Fraction, or numeric string")
    if non_negative and result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _stable_id_key(identifier: Hashable) -> tuple[str, str]:
    if isinstance(identifier, bool):
        raise TypeError("bool is not a valid stable ID")
    try:
        hash(identifier)
    except TypeError as exc:
        raise TypeError(f"ID must be hashable: {identifier!r}") from exc
    return type(identifier).__qualname__, str(identifier)


def _canonical_roster(members: Iterable[AgentId]) -> tuple[AgentId, ...]:
    try:
        values = tuple(members)
    except TypeError as exc:
        raise TypeError("roster members must be iterable") from exc
    if len(set(values)) != len(values):
        raise ValueError(f"roster contains duplicate agent IDs: {values!r}")
    return tuple(sorted(values, key=_stable_id_key))


def _candidate(candidate: Any) -> RosterCandidate:
    if isinstance(candidate, RosterCandidate):
        return candidate
    members = _field(
        candidate,
        ("members", "roster", "agent_ids"),
        default=_MISSING,
    )
    if members is _MISSING:
        members = candidate
    arrival_round = _field(
        candidate,
        ("arrival_round", "round", "received_round"),
        default=0,
    )
    return RosterCandidate(tuple(members), arrival_round)


def _has_field(value: Any, names: Sequence[str]) -> bool:
    return any(
        (isinstance(value, Mapping) and name in value) or hasattr(value, name) for name in names
    )


def _agent_candidate(candidate: Any) -> AgentCandidate:
    if isinstance(candidate, AgentCandidate):
        return candidate
    if isinstance(candidate, (int, str)) and not isinstance(candidate, bool):
        return AgentCandidate(candidate)
    identifier = _field(candidate, ("agent_id", "id"))
    arrival_round = _field(
        candidate,
        ("arrival_round", "round", "received_round"),
        default=0,
    )
    return AgentCandidate(identifier, arrival_round)


def _task_id(task: Any) -> TaskId:
    identifier = _field(task, ("task_id", "id"))
    _stable_id_key(identifier)
    return identifier


def _team_size(task: Any) -> int:
    value = _field(
        task,
        ("team_size", "required_size", "required_team_size", "roster_size"),
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("task team size must be an integer")
    if value < 1:
        raise ValueError("task team size must be positive")
    return value


def _demand_items(task: Any) -> tuple[tuple[Hashable, Fraction], ...]:
    demand = _field(task, ("demand", "capability_demand", "requirements"))
    if isinstance(demand, Mapping):
        raw_items = sorted(demand.items(), key=lambda item: _stable_id_key(item[0]))
    else:
        try:
            raw_items = list(enumerate(demand))
        except TypeError as exc:
            raise TypeError("task demand must be a sequence or mapping") from exc
    items = tuple(
        (dimension, _fraction(value, label=f"demand[{dimension!r}]"))
        for dimension, value in raw_items
    )
    if not items:
        raise ValueError("task demand must contain at least one dimension")
    if not any(value > 0 for _, value in items):
        raise ValueError("task demand must include at least one positive dimension")
    return items


def _agent_id(agent: Any, fallback: Any = _MISSING) -> AgentId:
    identifier = _field(agent, ("agent_id", "id"), default=fallback)
    if identifier is _MISSING:
        raise TypeError("agent must expose agent_id or id")
    _stable_id_key(identifier)
    return identifier


def _agent_map(agents: Mapping[AgentId, Any] | Iterable[Any]) -> dict[AgentId, Any]:
    result: dict[AgentId, Any] = {}
    if isinstance(agents, Mapping):
        iterator = (
            (_agent_id(agent, fallback=identifier), agent) for identifier, agent in agents.items()
        )
    else:
        iterator = ((_agent_id(agent), agent) for agent in agents)
    for identifier, agent in iterator:
        if identifier in result:
            raise ValueError(f"duplicate agent ID: {identifier!r}")
        result[identifier] = agent
    if not result:
        raise ValueError("agents must not be empty")
    return result


def _capabilities(agent: Any, source: InformationSource) -> Any:
    if source is InformationSource.CLAIMED:
        return _field(
            agent,
            (
                "claimed_capabilities",
                "claimed_capability",
                "capability_claim",
                "capabilities",
            ),
        )
    return _field(
        agent,
        ("true_capabilities", "true_capability", "capabilities"),
    )


def _capability(
    agent: Any,
    dimension: Hashable,
    source: InformationSource,
) -> Fraction:
    capabilities = _capabilities(agent, source)
    if isinstance(capabilities, Mapping):
        value = capabilities.get(dimension, 0)
    else:
        if not isinstance(dimension, int):
            raise TypeError("sequence capabilities require integer-indexed task demand")
        try:
            value = capabilities[dimension]
        except (IndexError, TypeError) as exc:
            raise ValueError(f"capability vector has no dimension {dimension!r}") from exc
    return _fraction(value, label=f"{source.value} capability[{dimension!r}]")


def _cost(agent: Any, task: Any, source: InformationSource) -> Fraction:
    task_identifier = _task_id(task)
    if source is InformationSource.CLAIMED:
        costs = _field(
            agent,
            ("claimed_costs", "claimed_task_costs", "costs", "task_costs"),
        )
    else:
        costs = _field(
            agent,
            ("true_costs", "true_task_costs", "costs", "task_costs"),
        )
    if isinstance(costs, Mapping):
        if task_identifier not in costs:
            raise ValueError(f"agent cost mapping has no entry for task {task_identifier!r}")
        value = costs[task_identifier]
    else:
        value = costs
    return _fraction(value, label=f"{source.value} cost for {task_identifier!r}")


def _feasible(
    task: Any,
    roster: Iterable[AgentId],
    agents: Mapping[AgentId, Any] | Iterable[Any],
    source: InformationSource,
) -> bool:
    members = _canonical_roster(roster)
    if len(members) != _team_size(task):
        return False
    agent_by_id = _agent_map(agents)
    if any(identifier not in agent_by_id for identifier in members):
        return False
    for dimension, demand in _demand_items(task):
        if demand <= 0:
            continue
        coverage = sum(
            (_capability(agent_by_id[identifier], dimension, source) for identifier in members),
            Fraction(0),
        )
        if coverage < demand:
            return False
    return True


def claimed_feasibility(
    task: Any,
    roster: Iterable[AgentId],
    agents: Mapping[AgentId, Any] | Iterable[Any],
) -> bool:
    """Whether a roster has the required size and claimed demanded coverage."""

    return _feasible(task, roster, agents, InformationSource.CLAIMED)


def roster_utility(
    task: Any,
    roster: Iterable[AgentId],
    agents: Mapping[AgentId, Any] | Iterable[Any],
    *,
    information: InformationSource = InformationSource.CLAIMED,
) -> Fraction:
    """Compute preregistered U(S,t) exactly, without floating-point ties."""

    members = _canonical_roster(roster)
    if not members:
        raise ValueError("utility requires a non-empty roster")
    agent_by_id = _agent_map(agents)
    missing = [identifier for identifier in members if identifier not in agent_by_id]
    if missing:
        raise ValueError(f"unknown roster agent IDs: {missing!r}")
    demanded_ratios = []
    for dimension, demand in _demand_items(task):
        if demand <= 0:
            continue
        coverage = sum(
            (
                _capability(agent_by_id[identifier], dimension, information)
                for identifier in members
            ),
            Fraction(0),
        )
        demanded_ratios.append(coverage / demand)
    coverage_term = min(demanded_ratios)
    total_cost = sum(
        (_cost(agent_by_id[identifier], task, information) for identifier in members),
        Fraction(0),
    )
    cost_penalty = Fraction(1, 4) * total_cost / (100 * len(members))
    return coverage_term - cost_penalty


def _candidate_rosters(
    task: Any,
    candidates: Iterable[Any],
) -> tuple[RosterCandidate, ...]:
    """Expand individual candidates into all subsets, or accept explicit rosters."""

    raw = tuple(candidates)
    if not raw:
        return ()
    roster_fields = ("members", "roster", "agent_ids")
    explicit_rosters = [
        isinstance(candidate, RosterCandidate)
        or _has_field(candidate, roster_fields)
        or (
            isinstance(candidate, Sequence)
            and not isinstance(candidate, (str, bytes, bytearray))
            and not _has_field(candidate, ("agent_id", "id"))
        )
        for candidate in raw
    ]
    if any(explicit_rosters) and not all(explicit_rosters):
        raise TypeError("candidate input cannot mix individual agents and explicit rosters")
    if all(explicit_rosters):
        return tuple(_candidate(candidate) for candidate in raw)

    earliest: dict[AgentId, AgentCandidate] = {}
    for raw_candidate in raw:
        candidate = _agent_candidate(raw_candidate)
        previous = earliest.get(candidate.agent_id)
        if previous is None or candidate.arrival_round < previous.arrival_round:
            earliest[candidate.agent_id] = candidate
    ordered = tuple(sorted(earliest.values(), key=lambda item: _stable_id_key(item.agent_id)))
    return tuple(
        RosterCandidate(
            members=tuple(candidate.agent_id for candidate in subset),
            arrival_round=max(candidate.arrival_round for candidate in subset),
        )
        for subset in combinations(ordered, _team_size(task))
    )


def _feasible_candidates(
    task: Any,
    candidates: Iterable[Any],
    agents: Mapping[AgentId, Any] | Iterable[Any],
) -> tuple[RosterCandidate, ...]:
    earliest: dict[tuple[AgentId, ...], RosterCandidate] = {}
    for candidate in _candidate_rosters(task, candidates):
        if not claimed_feasibility(task, candidate.members, agents):
            continue
        previous = earliest.get(candidate.members)
        if previous is None or candidate.arrival_round < previous.arrival_round:
            earliest[candidate.members] = candidate
    return tuple(
        sorted(
            earliest.values(),
            key=lambda candidate: (
                candidate.arrival_round,
                tuple(_stable_id_key(identifier) for identifier in candidate.members),
            ),
        )
    )


def first_valid(
    task: Any,
    candidates: Iterable[Any],
    agents: Mapping[AgentId, Any] | Iterable[Any],
) -> SelectionResult:
    """Select earliest claimed-feasible roster, then lexicographic member IDs."""

    feasible = _feasible_candidates(task, candidates, agents)
    selected = feasible[0] if feasible else None
    return SelectionResult(
        method=SelectionMethod.FIRST_VALID,
        roster=selected.members if selected is not None else None,
        utility=(roster_utility(task, selected.members, agents) if selected is not None else None),
        feasible_roster_count=len(feasible),
    )


def random_valid(
    task: Any,
    candidates: Iterable[Any],
    agents: Mapping[AgentId, Any] | Iterable[Any],
    *,
    seed: int,
) -> SelectionResult:
    """Seeded sampling over the canonical unique claimed-feasible rosters."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    feasible_by_roster = {
        candidate.members: candidate for candidate in _feasible_candidates(task, candidates, agents)
    }
    rosters = tuple(
        sorted(
            feasible_by_roster,
            key=lambda roster: tuple(_stable_id_key(identifier) for identifier in roster),
        )
    )
    selected = random.Random(seed).choice(rosters) if rosters else None
    return SelectionResult(
        method=SelectionMethod.RANDOM_VALID,
        roster=selected,
        utility=(roster_utility(task, selected, agents) if selected is not None else None),
        feasible_roster_count=len(rosters),
    )


def exact_utility(
    task: Any,
    candidates: Iterable[Any],
    agents: Mapping[AgentId, Any] | Iterable[Any],
) -> SelectionResult:
    """Maximize claimed U(S,t), resolving exact ties lexicographically."""

    rosters = {candidate.members for candidate in _feasible_candidates(task, candidates, agents)}
    scored = [(roster_utility(task, roster, agents), roster) for roster in rosters]
    selected_utility: Fraction | None = None
    selected_roster: tuple[AgentId, ...] | None = None
    for utility, roster in scored:
        if (
            selected_utility is None
            or utility > selected_utility
            or (
                utility == selected_utility
                and tuple(_stable_id_key(identifier) for identifier in roster)
                < tuple(_stable_id_key(identifier) for identifier in selected_roster or ())
            )
        ):
            selected_utility = utility
            selected_roster = roster
    return SelectionResult(
        method=SelectionMethod.EXACT_UTILITY,
        roster=selected_roster,
        utility=selected_utility,
        feasible_roster_count=len(rosters),
    )


def _task_reward(task: Any) -> Fraction:
    return _fraction(_field(task, ("reward", "task_reward")), label="task reward")


def _assignment_key(
    labels: tuple[TaskId | None, ...],
) -> tuple[tuple[int, tuple[str, str]], ...]:
    # Task labels sort before idle, so symmetric ties allocate lower agent IDs.
    return tuple((1, ("", "")) if label is None else (0, _stable_id_key(label)) for label in labels)


def public_joint_allocation(
    tasks: Iterable[Any],
    bids_by_task: Mapping[TaskId, Mapping[AgentId, Any] | Iterable[Any]],
    *,
    eligible_agent_ids: Iterable[AgentId] | None = None,
    max_agents: int = 6,
    max_tasks: int = 2,
) -> PublicJointAllocationResult:
    """Exactly allocate bounded public tasks from delivered claimed bids.

    Every replica with the same task cards, bid records, and eligible IDs
    obtains the same result. The lexicographic objective is:

    1. maximize summed public task reward;
    2. maximize summed claimed-information roster utility;
    3. minimize summed claimed raw cost; and
    4. minimize the canonical agent-assignment tuple.

    Empty task rosters are allowed. Every non-empty roster must have the exact
    requested size, contain only agents with a delivered bid for that task,
    and satisfy claimed capability demand. Since each agent receives one label
    in the enumeration, cross-task membership is exclusive by construction.
    """

    if isinstance(max_agents, bool) or not isinstance(max_agents, int):
        raise TypeError("max_agents must be an integer")
    if max_agents < 1:
        raise ValueError("max_agents must be positive")
    if isinstance(max_tasks, bool) or not isinstance(max_tasks, int):
        raise TypeError("max_tasks must be an integer")
    if max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    task_list = tuple(tasks)
    if not task_list:
        raise ValueError("joint allocation tasks must not be empty")
    if len(task_list) > max_tasks:
        raise ValueError(f"joint allocation is bounded to {max_tasks} tasks; got {len(task_list)}")
    task_by_id: dict[TaskId, Any] = {}
    for task in task_list:
        identifier = _task_id(task)
        if identifier in task_by_id:
            raise ValueError(f"duplicate task ID: {identifier!r}")
        task_by_id[identifier] = task
    unknown_tasks = set(bids_by_task).difference(task_by_id)
    if unknown_tasks:
        raise ValueError(
            f"bids supplied for unknown tasks: {sorted(unknown_tasks, key=_stable_id_key)!r}"
        )
    task_ids = tuple(sorted(task_by_id, key=_stable_id_key))

    agents_for_task: dict[TaskId, dict[AgentId, Any]] = {}
    discovered_ids: set[AgentId] = set()
    for task_id in task_ids:
        raw_agents = bids_by_task.get(task_id, ())
        values = (
            tuple(raw_agents.values()) if isinstance(raw_agents, Mapping) else tuple(raw_agents)
        )
        agent_map = (
            _agent_map(raw_agents if isinstance(raw_agents, Mapping) else values) if values else {}
        )
        agents_for_task[task_id] = agent_map
        discovered_ids.update(agent_map)
    if eligible_agent_ids is None:
        agent_ids = tuple(sorted(discovered_ids, key=_stable_id_key))
    else:
        eligible = tuple(eligible_agent_ids)
        if len(set(eligible)) != len(eligible):
            raise ValueError("eligible_agent_ids contains duplicates")
        for identifier in eligible:
            _stable_id_key(identifier)
        agent_ids = tuple(sorted(eligible, key=_stable_id_key))
        unexpected = discovered_ids.difference(agent_ids)
        if unexpected:
            raise ValueError(
                f"bids include ineligible agent IDs: {sorted(unexpected, key=_stable_id_key)!r}"
            )
    if len(agent_ids) > max_agents:
        raise ValueError(
            f"joint allocation is bounded to {max_agents} agents; got {len(agent_ids)}"
        )

    assignments_evaluated = 0
    feasible_assignments = 0
    best_labels: tuple[TaskId | None, ...] | None = None
    best_completed: tuple[PublicJointTaskAssignment, ...] = ()
    best_score: tuple[Fraction, Fraction, Fraction] | None = None
    for labels in product((*task_ids, None), repeat=len(agent_ids)):
        assignments_evaluated += 1
        rosters = {
            task_id: tuple(
                agent_id
                for agent_id, label in zip(agent_ids, labels, strict=True)
                if label == task_id
            )
            for task_id in task_ids
        }
        valid = True
        for task_id, roster in rosters.items():
            if not roster:
                continue
            if any(member not in agents_for_task[task_id] for member in roster) or not _feasible(
                task_by_id[task_id],
                roster,
                agents_for_task[task_id],
                InformationSource.CLAIMED,
            ):
                valid = False
                break
        if not valid:
            continue
        feasible_assignments += 1
        completed = []
        for task_id in task_ids:
            roster = rosters[task_id]
            if not roster:
                continue
            task = task_by_id[task_id]
            raw_cost = sum(
                (
                    _cost(
                        agents_for_task[task_id][agent_id],
                        task,
                        InformationSource.CLAIMED,
                    )
                    for agent_id in roster
                ),
                Fraction(0),
            )
            completed.append(
                PublicJointTaskAssignment(
                    task_id=task_id,
                    roster=_canonical_roster(roster),
                    reward=_task_reward(task),
                    roster_utility=roster_utility(
                        task,
                        roster,
                        agents_for_task[task_id],
                        information=InformationSource.CLAIMED,
                    ),
                    raw_cost=raw_cost,
                )
            )
        completed_tuple = tuple(completed)
        total_reward = sum(
            (assignment.reward for assignment in completed_tuple),
            Fraction(0),
        )
        total_utility = sum(
            (assignment.roster_utility for assignment in completed_tuple),
            Fraction(0),
        )
        total_cost = sum(
            (assignment.raw_cost for assignment in completed_tuple),
            Fraction(0),
        )
        score = (total_reward, total_utility, -total_cost)
        if (
            best_score is None
            or score > best_score
            or (
                score == best_score and _assignment_key(labels) < _assignment_key(best_labels or ())
            )
        ):
            best_score = score
            best_labels = labels
            best_completed = completed_tuple

    assert best_score is not None and best_labels is not None
    return PublicJointAllocationResult(
        agent_assignments=tuple(zip(agent_ids, best_labels, strict=True)),
        completed_tasks=best_completed,
        total_public_reward=best_score[0],
        total_claimed_utility=best_score[1],
        total_claimed_cost=-best_score[2],
        assignments_evaluated=assignments_evaluated,
        feasible_assignments=feasible_assignments,
    )


def true_information_oracle(
    tasks: Iterable[Any],
    agents: Mapping[AgentId, Any] | Iterable[Any],
) -> OracleResult:
    """Enumerate ``(tasks + idle)^6`` under private true capabilities/costs.

    Objective order is fixed and lexicographic:

    1. maximize total completed task reward;
    2. maximize summed exact true-information roster utility;
    3. minimize total raw true cost; and
    4. minimize the canonical agent-assignment tuple.
    """

    agent_by_id = _agent_map(agents)
    agent_ids = tuple(sorted(agent_by_id, key=_stable_id_key))
    if len(agent_ids) != 6:
        raise ValueError(
            f"the E2 true-information oracle requires exactly six agents; got {len(agent_ids)}"
        )
    task_list = tuple(tasks)
    if not task_list:
        raise ValueError("oracle tasks must not be empty")
    task_by_id: dict[TaskId, Any] = {}
    for task in task_list:
        identifier = _task_id(task)
        if identifier in task_by_id:
            raise ValueError(f"duplicate task ID: {identifier!r}")
        task_by_id[identifier] = task
    task_ids = tuple(sorted(task_by_id, key=_stable_id_key))

    assignments_evaluated = 0
    feasible_assignments = 0
    best_labels: tuple[TaskId | None, ...] | None = None
    best_completed: tuple[OracleTaskAssignment, ...] = ()
    best_score: tuple[Fraction, Fraction, Fraction] | None = None

    for labels in product((*task_ids, None), repeat=len(agent_ids)):
        assignments_evaluated += 1
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
            and not _feasible(
                task_by_id[task_id],
                roster,
                agent_by_id,
                InformationSource.TRUE,
            )
            for task_id, roster in rosters.items()
        ):
            continue
        feasible_assignments += 1
        completed = []
        for task_id in task_ids:
            roster = rosters[task_id]
            if not roster:
                continue
            task = task_by_id[task_id]
            raw_cost = sum(
                (_cost(agent_by_id[agent_id], task, InformationSource.TRUE) for agent_id in roster),
                Fraction(0),
            )
            completed.append(
                OracleTaskAssignment(
                    task_id=task_id,
                    roster=_canonical_roster(roster),
                    reward=_task_reward(task),
                    roster_utility=roster_utility(
                        task,
                        roster,
                        agent_by_id,
                        information=InformationSource.TRUE,
                    ),
                    raw_cost=raw_cost,
                )
            )
        completed_tuple = tuple(completed)
        total_reward = sum(
            (assignment.reward for assignment in completed_tuple),
            Fraction(0),
        )
        total_utility = sum(
            (assignment.roster_utility for assignment in completed_tuple),
            Fraction(0),
        )
        total_cost = sum(
            (assignment.raw_cost for assignment in completed_tuple),
            Fraction(0),
        )
        score = (total_reward, total_utility, -total_cost)
        if (
            best_score is None
            or score > best_score
            or (
                score == best_score and _assignment_key(labels) < _assignment_key(best_labels or ())
            )
        ):
            best_score = score
            best_labels = labels
            best_completed = completed_tuple

    assert best_score is not None and best_labels is not None
    return OracleResult(
        agent_assignments=tuple(zip(agent_ids, best_labels, strict=True)),
        completed_tasks=best_completed,
        total_completed_reward=best_score[0],
        total_roster_utility=best_score[1],
        total_raw_cost=-best_score[2],
        assignments_evaluated=assignments_evaluated,
        feasible_assignments=feasible_assignments,
    )
