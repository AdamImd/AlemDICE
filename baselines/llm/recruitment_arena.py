"""Deterministic six-agent arena for E2 coalition-formation experiments.

The arena exposes only public task cards, public role labels, an agent's own
capabilities/costs, delivered TFP1 records, and current team membership to live
policies. Private capability truth is consulted only when a locked roster is
scored and by the analysis-only oracle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import combinations, product
from typing import Any

from baselines.llm.eval_utils.recruitment_selection import (
    InformationSource,
    RosterCandidate,
    SelectionMethod,
    SelectionResult,
    exact_utility,
    first_valid,
    random_valid,
    roster_utility,
    true_information_oracle,
)
from baselines.llm.eval_utils.team_formation import (
    RecordKind,
    RecruitmentMethod,
    RecruitmentRecord,
    TaskCard,
    TaskPhase,
    TeamDirectory,
)

SCHEMA_VERSION = "alem-dice-recruitment-arena-v1"
AGENT_IDS = tuple(range(6))
DEFAULT_ROUNDS = 12
# Twelve acting rounds are indexed 0..11. Round 12 is a delivery-only drain
# and common allocation close; expiry would occur at the start of round 13.
DEFAULT_DEADLINE = 13


class ScenarioFamily(StrEnum):
    SINGLE_COMPLEMENTARY = "single_complementary"
    TWO_DISJOINT = "two_disjoint"
    SCARCE_CAPABILITY = "scarce_capability"
    OVERSUBSCRIBED = "oversubscribed"


class ControlArm(StrEnum):
    PREFORMED_ALL_SIX = "preformed_all_six"
    FIXED_3_PLUS_3 = "fixed_3_plus_3"
    NO_TEAM = "no_team"
    RANDOM_ALLOCATION = "random_allocation"


class TaskChoicePolicy(StrEnum):
    LOCAL_COMMIT = "local_commit"
    PUBLIC_SWEEP = "public_sweep"


@dataclass(frozen=True)
class AgentProfile:
    agent_id: int
    public_role: str
    true_capabilities: tuple[int, int, int]
    task_costs: dict[str, int]

    @property
    def claimed_capabilities(self) -> tuple[int, int, int]:
        """E2a scripted agents report truthfully."""

        return self.true_capabilities

    @property
    def claimed_costs(self) -> dict[str, int]:
        """E2a scripted agents report truthfully."""

        return self.task_costs

    @property
    def true_costs(self) -> dict[str, int]:
        return self.task_costs

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "public_role": self.public_role,
            "true_capabilities": list(self.true_capabilities),
            "task_costs": dict(sorted(self.task_costs.items())),
        }


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: ScenarioFamily
    seed: int
    agents: tuple[AgentProfile, ...]
    tasks: tuple[TaskCard, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "family": self.family.value,
            "seed": self.seed,
            "agents": [agent.as_dict() for agent in self.agents],
            "tasks": [task.as_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class EpisodeMetrics:
    method: str
    scenario_id: str
    family: ScenarioFamily
    seed: int
    rounds: int
    task_count: int
    oracle_reward: int
    achieved_reward: int
    normalized_reward: Fraction | None
    reward_regret: int | None
    oracle_utility: Fraction
    achieved_utility: Fraction | None
    utility_regret: Fraction | None
    oracle_raw_cost: Fraction
    achieved_raw_cost: Fraction | None
    oracle_assignment: tuple[dict[str, Any], ...]
    completed_tasks: int
    true_feasible_locks: int
    true_infeasible_locks: int
    formation_opportunities: int
    formation_rate: Fraction
    oracle_allocation_opportunities: int
    oracle_allocation_coverage: Fraction | None
    lock_rounds: tuple[int, ...]
    control_submissions: int
    valid_control_submissions: int
    accepted_control_transitions: int
    rejected_control_transitions: int
    control_payload_bytes: int
    control_delivered_bytes: int
    ordinary_deliveries: int
    ordinary_payload_bytes: int
    ordinary_delivered_bytes: int
    unauthorized_ordinary_deliveries: int
    roster_agreement_rate: Fraction | None
    overlapping_roster_rounds: int
    multi_offer_agent_rounds: int
    roster_revision_count: int
    overstaff_agent_slots: int
    exact_roster_comparator: bool
    replay_hash_match: bool | None
    terminal_state_hash: str
    terminal_audit_chain_hash: str | None
    model_calls: int = 0
    provider_requests: int = 0
    transport_errors: int = 0

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["family"] = self.family.value
        for name in (
            "normalized_reward",
            "oracle_utility",
            "achieved_utility",
            "utility_regret",
            "oracle_raw_cost",
            "achieved_raw_cost",
            "formation_rate",
            "oracle_allocation_coverage",
            "roster_agreement_rate",
        ):
            fraction = getattr(self, name)
            value[name] = (
                None
                if fraction is None
                else {
                    "numerator": fraction.numerator,
                    "denominator": fraction.denominator,
                    "float": float(fraction),
                }
            )
        value["lock_rounds"] = list(self.lock_rounds)
        return value


@dataclass(frozen=True)
class ArenaEpisode:
    schema_version: str
    scenario: Scenario
    method: str
    metrics: EpisodeMetrics
    events: tuple[dict[str, Any], ...]
    directory_replay: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario": self.scenario.as_dict(),
            "method": self.method,
            "metrics": self.metrics.as_dict(),
            "events": list(self.events),
            "directory_replay": self.directory_replay,
        }


_ARCHETYPES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("warrior", (80, 20, 20)),
    ("warrior", (80, 20, 20)),
    ("forager", (20, 80, 20)),
    ("forager", (20, 80, 20)),
    ("miner", (20, 20, 90)),
    # Deliberately below the scarce-task threshold even after pairing with a
    # non-specialist (35 + 20 < 80). The earlier 65 draft would not create a
    # scarce capability under additive team coverage.
    ("miner", (20, 20, 35)),
)

_PUBLIC_ROLE_CAPABILITIES: dict[str, tuple[int, int, int]] = {
    "warrior": (80, 20, 20),
    "forager": (20, 80, 20),
    # Mutual nomination sees only the coarse public role, not which miner owns
    # the private 90 versus 35 mining capability.
    "miner": (20, 20, 62),
}


def _stable_int(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _task(
    task_id: str,
    *,
    demand: tuple[int, int, int],
    required_size: int,
    reward: int = 100,
) -> TaskCard:
    return TaskCard(
        task_id=task_id,
        demand=demand,
        required_size=required_size,
        reward=reward,
        deadline_round=DEFAULT_DEADLINE,
        sponsor_id=_stable_int("sponsor", task_id) % len(AGENT_IDS),
    )


def generate_scenario(family: ScenarioFamily | str, seed: int) -> Scenario:
    """Build one paired six-agent scenario without rejection sampling."""

    family = ScenarioFamily(family)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    indexed_archetypes = list(enumerate(_ARCHETYPES))
    indexed_archetypes.sort(
        key=lambda item: (_stable_int("role-permutation", seed, item[0]), item[0])
    )
    role_by_agent = {agent_id: indexed_archetypes[agent_id][1] for agent_id in AGENT_IDS}

    if family is ScenarioFamily.SINGLE_COMPLEMENTARY:
        tasks = (_task("single.t0", demand=(70, 70, 0), required_size=2),)
    elif family is ScenarioFamily.TWO_DISJOINT:
        tasks = (
            _task("disjoint.t0", demand=(70, 70, 0), required_size=2),
            _task("disjoint.t1", demand=(0, 70, 70), required_size=2),
        )
    elif family is ScenarioFamily.SCARCE_CAPABILITY:
        tasks = (
            _task("scarce.t0", demand=(70, 0, 80), required_size=2),
            _task("scarce.t1", demand=(0, 70, 80), required_size=2),
        )
    else:
        tasks = (_task("over.t0", demand=(70, 70, 0), required_size=3),)

    agents = []
    for agent_id in AGENT_IDS:
        role, capabilities = role_by_agent[agent_id]
        costs = {
            task.task_id: _stable_int("cost", seed, agent_id, task.task_id) % 101 for task in tasks
        }
        agents.append(
            AgentProfile(
                agent_id=agent_id,
                public_role=role,
                true_capabilities=capabilities,
                task_costs=costs,
            )
        )
    return Scenario(
        scenario_id=f"{family.value}:{seed}",
        family=family,
        seed=seed,
        agents=tuple(agents),
        tasks=tasks,
    )


def _true_feasible(
    task: TaskCard,
    members: Iterable[int],
    profiles: dict[int, AgentProfile],
) -> bool:
    roster = tuple(members)
    if len(roster) != task.required_size or len(set(roster)) != len(roster):
        return False
    for dimension, demand in enumerate(task.demand):
        if demand <= 0:
            continue
        if sum(profiles[member].true_capabilities[dimension] for member in roster) < demand:
            return False
    return True


def _individually_feasible_tasks(scenario: Scenario) -> tuple[str, ...]:
    profiles = {agent.agent_id: agent for agent in scenario.agents}
    return tuple(
        task.task_id
        for task in scenario.tasks
        if any(
            _true_feasible(task, members, profiles)
            for members in combinations(AGENT_IDS, task.required_size)
        )
    )


def _oracle_assignment(oracle: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "task_id": str(assignment.task_id),
            "members": list(assignment.roster),
            "reward": str(assignment.reward),
            "roster_utility": str(assignment.roster_utility),
            "raw_cost": str(assignment.raw_cost),
        }
        for assignment in oracle.completed_tasks
    )


def _public_role_roster(
    task: TaskCard,
    scenario: Scenario,
    candidates: Iterable[int],
) -> tuple[int, ...] | None:
    candidate_ids = tuple(sorted(set(candidates)))
    for members in combinations(candidate_ids, task.required_size):
        if all(
            sum(
                _PUBLIC_ROLE_CAPABILITIES[
                    next(agent.public_role for agent in scenario.agents if agent.agent_id == member)
                ][dimension]
                for member in members
            )
            >= demand
            for dimension, demand in enumerate(task.demand)
            if demand > 0
        ):
            return members
    return None


def _candidate_rosters_from_bids(
    state: Any,
    bids: Iterable[Any] | None = None,
) -> tuple[RosterCandidate, ...]:
    ordered = sorted(
        state.bids.values() if bids is None else bids,
        key=lambda bid: (bid.arrival_round, bid.agent_id),
    )
    return tuple(
        RosterCandidate(
            members=tuple(sorted(bid.agent_id for bid in group)),
            arrival_round=max(bid.arrival_round for bid in group),
        )
        for group in combinations(ordered, state.card.required_size)
    )


def _selection_agents(
    state: Any,
    bids: Iterable[Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "agent_id": bid.agent_id,
            "claimed_capabilities": bid.claimed_capabilities,
            "claimed_costs": {state.card.task_id: bid.cost},
        }
        for bid in (state.bids.values() if bids is None else bids)
    )


def _contract_selection(
    state: Any,
    bids: Iterable[Any],
    *,
    episode_seed: int,
    selector: SelectionMethod,
    audit: list[dict[str, Any]] | None,
    round_index: int,
) -> SelectionResult | None:
    """Select from delivered bids using claimed information only.

    The random arm derives its seed from public episode/task identity and the
    currently available bidder IDs. Repeating an identical public state
    therefore repeats the draw across processes and Python invocations.
    """

    bid_values = tuple(bids)
    if len(bid_values) < state.card.required_size:
        return None
    candidates = _candidate_rosters_from_bids(state, bid_values)
    agents = _selection_agents(state, bid_values)
    random_seed = _stable_int(
        "e2d-contract-selector-v1",
        episode_seed,
        state.card.task_id,
        ",".join(str(bid.agent_id) for bid in sorted(bid_values, key=lambda item: item.agent_id)),
    )
    if selector is SelectionMethod.FIRST_VALID:
        selection = first_valid(state.card, candidates, agents)
    elif selector is SelectionMethod.RANDOM_VALID:
        selection = random_valid(
            state.card,
            candidates,
            agents,
            seed=random_seed,
        )
    else:
        selection = exact_utility(state.card, candidates, agents)
    if audit is not None:
        audit.append(
            {
                "round": round_index,
                "task_id": state.card.task_id,
                "method": selector.value,
                "information_source": InformationSource.CLAIMED.value,
                "candidate_agent_ids": sorted(bid.agent_id for bid in bid_values),
                "random_seed": (random_seed if selector is SelectionMethod.RANDOM_VALID else None),
                "selected_roster": (None if selection.roster is None else list(selection.roster)),
                "selected_claimed_utility": (
                    None if selection.utility is None else str(selection.utility)
                ),
                "feasible_roster_count": selection.feasible_roster_count,
            }
        )
    return selection


@dataclass(frozen=True)
class _ControlProposal:
    priority: int
    sender: int
    task_id: str
    record: RecruitmentRecord


def _task_preferences(
    scenario: Scenario,
    *,
    public_roles_only: bool,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Choose one task per agent using only locally available information."""

    preferences: dict[int, str] = {}
    audit: dict[str, Any] = {}
    for profile in scenario.agents:
        capabilities = (
            _PUBLIC_ROLE_CAPABILITIES[profile.public_role]
            if public_roles_only
            else profile.true_capabilities
        )
        scored = []
        for task in scenario.tasks:
            coverage = sum(
                (
                    Fraction(capabilities[dimension], demand)
                    for dimension, demand in enumerate(task.demand)
                    if demand > 0
                ),
                Fraction(0),
            )
            cost_penalty = (
                Fraction(0)
                if public_roles_only
                else Fraction(profile.task_costs[task.task_id], 400)
            )
            score = coverage - cost_penalty
            tie_break = _stable_int(
                "task-preference",
                scenario.seed,
                profile.agent_id,
                task.task_id,
            )
            scored.append((score, tie_break, task.task_id, coverage, cost_penalty))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        preferences[profile.agent_id] = scored[0][2]
        audit[str(profile.agent_id)] = {
            "information": (
                "public_role_proxy" if public_roles_only else "own_capability_and_cost"
            ),
            "selected_task": scored[0][2],
            "scores": {
                task_id: {
                    "score": str(score),
                    "coverage": str(coverage),
                    "cost_penalty": str(cost_penalty),
                    "tie_break": tie_break,
                }
                for score, tie_break, task_id, coverage, cost_penalty in scored
            },
        }
    return preferences, audit


def _forming_states(directory: TeamDirectory) -> tuple[Any, ...]:
    return tuple(
        directory.tasks[task_id]
        for task_id in sorted(directory.tasks)
        if directory.tasks[task_id].phase
        in {TaskPhase.ANNOUNCED, TaskPhase.FORMING, TaskPhase.AWARDED}
    )


def _observable_candidate_rosters(
    directory: TeamDirectory,
) -> dict[str, tuple[int, ...]]:
    rosters: dict[str, tuple[int, ...]] = {}
    for state in _forming_states(directory):
        roster: tuple[int, ...] | None
        if directory.method is RecruitmentMethod.OPEN_VOLUNTEER:
            roster = directory._open_roster(state)
        elif directory.method is RecruitmentMethod.CONTRACT_NET:
            roster = state.award
        else:
            counts: dict[tuple[int, ...], int] = {}
            for nominated in state.nominations.values():
                counts[nominated] = counts.get(nominated, 0) + 1
            roster = (
                min(counts, key=lambda members: (-counts[members], members)) if counts else None
            )
        if roster is not None:
            rosters[state.card.task_id] = roster
    return rosters


def _add_proposal(
    proposals: dict[int, list[_ControlProposal]],
    *,
    priority: int,
    sender: int,
    state: Any,
    record: RecruitmentRecord,
) -> None:
    proposals.setdefault(sender, []).append(
        _ControlProposal(
            priority=priority,
            sender=sender,
            task_id=state.card.task_id,
            record=record,
        )
    )


def _submit_proposals(
    directory: TeamDirectory,
    round_index: int,
    proposals: dict[int, list[_ControlProposal]],
) -> None:
    for sender in sorted(proposals):
        selected = min(
            proposals[sender],
            key=lambda item: (
                item.priority,
                item.task_id,
                item.record.kind.value,
            ),
        )
        directory.submit_control(
            sender=sender,
            sent_round=round_index,
            raw=selected.record.render(),
        )


def _emit_open_volunteer(
    directory: TeamDirectory,
    scenario: Scenario,
    round_index: int,
    preferences: dict[int, str],
) -> None:
    profiles = {agent.agent_id: agent for agent in scenario.agents}
    proposals: dict[int, list[_ControlProposal]] = {}
    for state in _forming_states(directory):
        task = state.card
        if round_index + 1 >= task.deadline_round:
            continue
        roster = directory._open_roster(state)
        if (
            roster is not None
            and set(roster).issubset(state.accepts)
            and not any(member in directory.agent_to_task for member in roster)
        ):
            _add_proposal(
                proposals,
                priority=0,
                sender=task.sponsor_id,
                state=state,
                record=RecruitmentRecord(
                    RecordKind.LOCK,
                    task.task_id,
                    members=roster,
                ),
            )
        if roster is not None:
            for member in roster:
                if member not in state.accepts and member not in directory.agent_to_task:
                    _add_proposal(
                        proposals,
                        priority=1,
                        sender=member,
                        state=state,
                        record=RecruitmentRecord(
                            RecordKind.ACCEPT,
                            task.task_id,
                            members=roster,
                        ),
                    )
        for agent_id in AGENT_IDS:
            if (
                preferences[agent_id] != task.task_id
                or agent_id in state.applications
                or agent_id in directory.agent_to_task
            ):
                continue
            profile = profiles[agent_id]
            _add_proposal(
                proposals,
                priority=3,
                sender=agent_id,
                state=state,
                record=RecruitmentRecord(
                    RecordKind.APPLY,
                    task.task_id,
                    capabilities=profile.true_capabilities,
                    cost=profile.task_costs[task.task_id],
                ),
            )
    _submit_proposals(directory, round_index, proposals)


def _emit_mutual_nomination(
    directory: TeamDirectory,
    scenario: Scenario,
    round_index: int,
    preferences: dict[int, str],
) -> None:
    proposals: dict[int, list[_ControlProposal]] = {}
    for state in _forming_states(directory):
        if round_index + 1 >= state.card.deadline_round:
            continue
        candidates = tuple(
            agent_id
            for agent_id, task_id in preferences.items()
            if task_id == state.card.task_id and agent_id not in directory.agent_to_task
        )
        roster = _public_role_roster(state.card, scenario, candidates)
        if roster is None:
            continue
        if all(state.nominations.get(member) == roster for member in roster):
            _add_proposal(
                proposals,
                priority=0,
                sender=min(roster),
                state=state,
                record=RecruitmentRecord(
                    RecordKind.LOCK,
                    state.card.task_id,
                    members=roster,
                ),
            )
            continue
        for member in roster:
            if state.nominations.get(member) != roster:
                _add_proposal(
                    proposals,
                    priority=1,
                    sender=member,
                    state=state,
                    record=RecruitmentRecord(
                        RecordKind.NOMINATE,
                        state.card.task_id,
                        members=roster,
                    ),
                )
    _submit_proposals(directory, round_index, proposals)


def _emit_contract_net(
    directory: TeamDirectory,
    scenario: Scenario,
    round_index: int,
    preferences: dict[int, str],
    selector: SelectionMethod,
    selector_audit: list[dict[str, Any]] | None,
) -> None:
    profiles = {agent.agent_id: agent for agent in scenario.agents}
    proposals: dict[int, list[_ControlProposal]] = {}
    for state in _forming_states(directory):
        task = state.card
        if round_index + 1 >= task.deadline_round:
            continue
        if not state.cfp_open:
            _add_proposal(
                proposals,
                priority=3,
                sender=task.sponsor_id,
                state=state,
                record=RecruitmentRecord(RecordKind.CFP, task.task_id),
            )
        if state.award is None and round_index >= 3 and len(state.bids) >= task.required_size:
            selection = _contract_selection(
                state,
                state.bids.values(),
                episode_seed=directory.seed,
                selector=selector,
                audit=selector_audit,
                round_index=round_index,
            )
            if selection is not None and selection.roster is not None:
                _add_proposal(
                    proposals,
                    priority=2,
                    sender=task.sponsor_id,
                    state=state,
                    record=RecruitmentRecord(
                        RecordKind.AWARD,
                        task.task_id,
                        members=tuple(int(member) for member in selection.roster),
                    ),
                )
        if state.award is not None:
            if set(state.award).issubset(state.accepts) and not any(
                member in directory.agent_to_task for member in state.award
            ):
                _add_proposal(
                    proposals,
                    priority=0,
                    sender=task.sponsor_id,
                    state=state,
                    record=RecruitmentRecord(
                        RecordKind.LOCK,
                        task.task_id,
                        members=state.award,
                    ),
                )
            for member in state.award:
                if member not in state.accepts and member not in directory.agent_to_task:
                    _add_proposal(
                        proposals,
                        priority=1,
                        sender=member,
                        state=state,
                        record=RecruitmentRecord(
                            RecordKind.ACCEPT,
                            task.task_id,
                            members=state.award,
                        ),
                    )
        for agent_id in AGENT_IDS:
            if (
                not state.cfp_open
                or preferences[agent_id] != task.task_id
                or agent_id in state.bids
                or agent_id in directory.agent_to_task
            ):
                continue
            profile = profiles[agent_id]
            _add_proposal(
                proposals,
                priority=4,
                sender=agent_id,
                state=state,
                record=RecruitmentRecord(
                    RecordKind.BID,
                    task.task_id,
                    capabilities=profile.true_capabilities,
                    cost=profile.task_costs[task.task_id],
                ),
            )
    _submit_proposals(directory, round_index, proposals)


def _emit_open_sweep(
    directory: TeamDirectory,
    scenario: Scenario,
    round_index: int,
) -> None:
    """Public application sweep followed by deterministic response waves."""

    states = _forming_states(directory)
    if not states:
        return
    profiles = {agent.agent_id: agent for agent in scenario.agents}
    proposals: dict[int, list[_ControlProposal]] = {}
    task_count = len(scenario.tasks)
    if round_index < task_count:
        state = next(
            (
                item
                for item in states
                if item.card.task_id == sorted(task.task_id for task in scenario.tasks)[round_index]
            ),
            None,
        )
        if state is not None:
            for agent_id in AGENT_IDS:
                if agent_id in directory.agent_to_task or agent_id in state.applications:
                    continue
                profile = profiles[agent_id]
                _add_proposal(
                    proposals,
                    priority=3,
                    sender=agent_id,
                    state=state,
                    record=RecruitmentRecord(
                        RecordKind.APPLY,
                        state.card.task_id,
                        capabilities=profile.true_capabilities,
                        cost=profile.task_costs[state.card.task_id],
                    ),
                )
        _submit_proposals(directory, round_index, proposals)
        return

    rosters = {state.card.task_id: directory._open_roster(state) for state in states}
    reserved: set[int] = set()
    locking_tasks: set[str] = set()
    for state in states:
        roster = rosters[state.card.task_id]
        if (
            roster is None
            or not set(roster).issubset(state.accepts)
            or reserved.intersection(roster)
            or any(member in directory.agent_to_task for member in roster)
        ):
            continue
        reserved.update(roster)
        locking_tasks.add(state.card.task_id)
        _add_proposal(
            proposals,
            priority=0,
            sender=state.card.sponsor_id,
            state=state,
            record=RecruitmentRecord(
                RecordKind.LOCK,
                state.card.task_id,
                members=roster,
            ),
        )
    for agent_id in AGENT_IDS:
        if agent_id in directory.agent_to_task:
            continue
        choices = [
            (state, rosters[state.card.task_id])
            for state in states
            if state.card.task_id not in locking_tasks
            and rosters[state.card.task_id] is not None
            and not reserved.intersection(rosters[state.card.task_id])
            and agent_id in rosters[state.card.task_id]
            and agent_id not in state.accepts
        ]
        if not choices:
            continue
        state, roster = min(choices, key=lambda item: item[0].card.task_id)
        _add_proposal(
            proposals,
            priority=1,
            sender=agent_id,
            state=state,
            record=RecruitmentRecord(
                RecordKind.ACCEPT,
                state.card.task_id,
                members=roster,
            ),
        )
    _submit_proposals(directory, round_index, proposals)


def _emit_mutual_sweep(
    directory: TeamDirectory,
    scenario: Scenario,
    round_index: int,
) -> None:
    """Derive all public-role rosters before per-agent nomination arbitration."""

    states = _forming_states(directory)
    proposals: dict[int, list[_ControlProposal]] = {}
    available = tuple(agent_id for agent_id in AGENT_IDS if agent_id not in directory.agent_to_task)
    rosters = {
        state.card.task_id: _public_role_roster(state.card, scenario, available) for state in states
    }
    reserved: set[int] = set()
    locking_tasks: set[str] = set()
    for state in states:
        roster = rosters[state.card.task_id]
        if (
            roster is None
            or not all(state.nominations.get(member) == roster for member in roster)
            or reserved.intersection(roster)
        ):
            continue
        reserved.update(roster)
        locking_tasks.add(state.card.task_id)
        _add_proposal(
            proposals,
            priority=0,
            sender=min(roster),
            state=state,
            record=RecruitmentRecord(
                RecordKind.LOCK,
                state.card.task_id,
                members=roster,
            ),
        )
    for agent_id in available:
        choices = [
            (state, rosters[state.card.task_id])
            for state in states
            if state.card.task_id not in locking_tasks
            and rosters[state.card.task_id] is not None
            and not reserved.intersection(rosters[state.card.task_id])
            and agent_id in rosters[state.card.task_id]
            and state.nominations.get(agent_id) != rosters[state.card.task_id]
        ]
        if not choices:
            continue
        state, roster = min(choices, key=lambda item: item[0].card.task_id)
        _add_proposal(
            proposals,
            priority=1,
            sender=agent_id,
            state=state,
            record=RecruitmentRecord(
                RecordKind.NOMINATE,
                state.card.task_id,
                members=roster,
            ),
        )
    _submit_proposals(directory, round_index, proposals)


def _available_contract_selection(
    state: Any,
    directory: TeamDirectory,
    *,
    selector: SelectionMethod,
    selector_audit: list[dict[str, Any]] | None,
    round_index: int,
) -> SelectionResult | None:
    bids = tuple(bid for bid in state.bids.values() if bid.agent_id not in directory.agent_to_task)
    return _contract_selection(
        state,
        bids,
        episode_seed=directory.seed,
        selector=selector,
        audit=selector_audit,
        round_index=round_index,
    )


def _emit_contract_sweep(
    directory: TeamDirectory,
    scenario: Scenario,
    round_index: int,
    selector: SelectionMethod,
    selector_audit: list[dict[str, Any]] | None,
) -> None:
    """Public CFP/bid barrier followed by award/accept/lock recovery waves."""

    states = _forming_states(directory)
    if not states:
        return
    profiles = {agent.agent_id: agent for agent in scenario.agents}
    proposals: dict[int, list[_ControlProposal]] = {}
    if any(not state.cfp_open for state in states):
        for state in states:
            if not state.cfp_open:
                _add_proposal(
                    proposals,
                    priority=3,
                    sender=state.card.sponsor_id,
                    state=state,
                    record=RecruitmentRecord(
                        RecordKind.CFP,
                        state.card.task_id,
                    ),
                )
        _submit_proposals(directory, round_index, proposals)
        return

    missing_bid = False
    for agent_id in AGENT_IDS:
        if agent_id in directory.agent_to_task:
            continue
        choices = [state for state in states if state.award is None and agent_id not in state.bids]
        if not choices:
            continue
        missing_bid = True
        state = min(choices, key=lambda item: item.card.task_id)
        profile = profiles[agent_id]
        _add_proposal(
            proposals,
            priority=4,
            sender=agent_id,
            state=state,
            record=RecruitmentRecord(
                RecordKind.BID,
                state.card.task_id,
                capabilities=profile.true_capabilities,
                cost=profile.task_costs[state.card.task_id],
            ),
        )
    if missing_bid:
        _submit_proposals(directory, round_index, proposals)
        return

    selections = {
        state.card.task_id: _available_contract_selection(
            state,
            directory,
            selector=selector,
            selector_audit=selector_audit,
            round_index=round_index,
        )
        for state in states
    }
    reserved: set[int] = set()
    locking_tasks: set[str] = set()
    for state in states:
        award = state.award
        if (
            award is None
            or any(member in directory.agent_to_task for member in award)
            or not set(award).issubset(state.accepts)
            or reserved.intersection(award)
        ):
            continue
        reserved.update(award)
        locking_tasks.add(state.card.task_id)
        _add_proposal(
            proposals,
            priority=0,
            sender=state.card.sponsor_id,
            state=state,
            record=RecruitmentRecord(
                RecordKind.LOCK,
                state.card.task_id,
                members=award,
            ),
        )
    for state in states:
        award = state.award
        selection = selections[state.card.task_id]
        needs_reaward = award is None or any(member in directory.agent_to_task for member in award)
        if (
            needs_reaward
            and selection is not None
            and selection.roster is not None
            and not reserved.intersection(selection.roster)
        ):
            _add_proposal(
                proposals,
                priority=2,
                sender=state.card.sponsor_id,
                state=state,
                record=RecruitmentRecord(
                    RecordKind.AWARD,
                    state.card.task_id,
                    members=tuple(int(member) for member in selection.roster),
                ),
            )
        if (
            award is None
            or needs_reaward
            or state.card.task_id in locking_tasks
            or reserved.intersection(award)
        ):
            continue
        for member in award:
            if member not in state.accepts:
                _add_proposal(
                    proposals,
                    priority=1,
                    sender=member,
                    state=state,
                    record=RecruitmentRecord(
                        RecordKind.ACCEPT,
                        state.card.task_id,
                        members=award,
                    ),
                )
    _submit_proposals(directory, round_index, proposals)


def _submit_scripted_controls(
    directory: TeamDirectory,
    scenario: Scenario,
    round_index: int,
    preferences: dict[int, str],
    task_choice: TaskChoicePolicy,
    selector: SelectionMethod,
    selector_audit: list[dict[str, Any]] | None,
) -> None:
    if task_choice is TaskChoicePolicy.PUBLIC_SWEEP:
        if directory.method is RecruitmentMethod.OPEN_VOLUNTEER:
            _emit_open_sweep(directory, scenario, round_index)
        elif directory.method is RecruitmentMethod.MUTUAL_NOMINATION:
            _emit_mutual_sweep(directory, scenario, round_index)
        else:
            _emit_contract_sweep(
                directory,
                scenario,
                round_index,
                selector,
                selector_audit,
            )
        return
    if directory.method is RecruitmentMethod.OPEN_VOLUNTEER:
        _emit_open_volunteer(directory, scenario, round_index, preferences)
    elif directory.method is RecruitmentMethod.MUTUAL_NOMINATION:
        _emit_mutual_nomination(directory, scenario, round_index, preferences)
    else:
        _emit_contract_net(
            directory,
            scenario,
            round_index,
            preferences,
            selector,
            selector_audit,
        )


def run_scripted_episode(
    scenario: Scenario,
    method: RecruitmentMethod | str,
    *,
    rounds: int = DEFAULT_ROUNDS,
    oracle: Any | None = None,
    task_choice: TaskChoicePolicy | str = TaskChoicePolicy.LOCAL_COMMIT,
    selector: SelectionMethod | str | None = None,
) -> ArenaEpisode:
    """Run one provider-free scripted E2 episode.

    ``selector=None`` preserves the E2a Contract Net behavior and arm label.
    An explicit selector is available only to Contract Net and adds the
    selector to the arm label for paired E2d comparisons.
    """

    method = RecruitmentMethod(method)
    task_choice = TaskChoicePolicy(task_choice)
    explicit_selector = selector is not None
    selection_method = (
        SelectionMethod.FIRST_VALID if selector is None else SelectionMethod(selector)
    )
    if explicit_selector and method is not RecruitmentMethod.CONTRACT_NET:
        raise ValueError("explicit roster selectors are supported only by Contract Net")
    if rounds != DEFAULT_ROUNDS:
        raise ValueError(f"E2a requires exactly {DEFAULT_ROUNDS} recruitment rounds")
    method_label = f"{method.value}__{task_choice.value}"
    if explicit_selector:
        method_label = f"{method_label}__{selection_method.value}"
    directory = TeamDirectory(agent_ids=AGENT_IDS, method=method, seed=scenario.seed)
    for task in scenario.tasks:
        directory.register_task(task)

    profiles = {agent.agent_id: agent for agent in scenario.agents}
    if task_choice is TaskChoicePolicy.LOCAL_COMMIT:
        preferences, preference_audit = _task_preferences(
            scenario,
            public_roles_only=method is RecruitmentMethod.MUTUAL_NOMINATION,
        )
    else:
        preferences = {}
        preference_audit = {
            "policy": TaskChoicePolicy.PUBLIC_SWEEP.value,
            "description": (
                "All agents sweep public tasks; per-agent conflicts resolve by "
                "stable task ID using delivered public state."
            ),
        }
    completed: set[str] = set()
    evaluated_leases: dict[int, bool] = {}
    feasible_rosters: dict[str, tuple[int, ...]] = {}
    lock_rounds: list[int] = []
    true_feasible_locks = 0
    true_infeasible_locks = 0
    previous_rosters: dict[str, tuple[int, ...]] = {}
    overlapping_roster_rounds = 0
    multi_offer_agent_rounds = 0
    roster_revision_count = 0
    selector_audit: list[dict[str, Any]] | None = [] if explicit_selector else None
    events: list[dict[str, Any]] = [
        {
            "round": -1,
            "phase": "local_task_commitment",
            "preferences": preference_audit,
        }
    ]

    for round_index in range(rounds):
        transitions = directory.advance(round_index)
        ordinary = directory.deliver_ordinary(round_index)
        events.append(
            {
                "round": round_index,
                "phase": "delivery",
                "transitions": [item.as_dict() for item in transitions],
                "ordinary": [item.as_dict() for item in ordinary],
                "state_hash": directory.state_hash(),
            }
        )
        for transition in transitions:
            if not transition.accepted or transition.code != "team.locked":
                continue
            state = directory.tasks[transition.task_id]
            lease = state.lease
            if lease is None or lease.version in evaluated_leases:
                continue
            lock_rounds.append(round_index)
            feasible = _true_feasible(state.card, lease.members, profiles)
            evaluated_leases[lease.version] = feasible
            if feasible:
                true_feasible_locks += 1
                feasible_rosters[lease.task_id] = lease.members
            else:
                true_infeasible_locks += 1
            events.append(
                {
                    "round": round_index,
                    "phase": "terminal_scoring",
                    "task_id": lease.task_id,
                    "members": list(lease.members),
                    "true_feasible": feasible,
                }
            )

        candidate_rosters = _observable_candidate_rosters(directory)
        for task_id, roster in candidate_rosters.items():
            previous = previous_rosters.get(task_id)
            if previous is not None and previous != roster:
                roster_revision_count += 1
            previous_rosters[task_id] = roster
        memberships: dict[int, int] = {}
        for roster in candidate_rosters.values():
            for member in roster:
                memberships[member] = memberships.get(member, 0) + 1
        offered_multiple = sum(count > 1 for count in memberships.values())
        if offered_multiple:
            overlapping_roster_rounds += 1
            multi_offer_agent_rounds += offered_multiple
        events.append(
            {
                "round": round_index,
                "phase": "candidate_rosters",
                "rosters": {
                    task_id: list(roster) for task_id, roster in sorted(candidate_rosters.items())
                },
                "multi_offer_agents": sorted(
                    member for member, count in memberships.items() if count > 1
                ),
            }
        )

        _submit_scripted_controls(
            directory,
            scenario,
            round_index,
            preferences,
            task_choice,
            selection_method,
            selector_audit,
        )
        events.append(
            {
                "round": round_index,
                "phase": "emission",
                "state_hash": directory.state_hash(),
            }
        )

    # Delivery-only round 12 applies records sent in the final acting round.
    # Formation-only teams retain their members until this common allocation
    # close, matching the static exclusive `(tasks + idle)^6` oracle.
    transitions = directory.advance(rounds)
    ordinary = directory.deliver_ordinary(rounds)
    events.append(
        {
            "round": rounds,
            "phase": "drain",
            "transitions": [item.as_dict() for item in transitions],
            "ordinary": [item.as_dict() for item in ordinary],
            "state_hash": directory.state_hash(),
        }
    )
    for transition in transitions:
        if not transition.accepted or transition.code != "team.locked":
            continue
        state = directory.tasks[transition.task_id]
        lease = state.lease
        if lease is None or lease.version in evaluated_leases:
            continue
        lock_rounds.append(rounds)
        feasible = _true_feasible(state.card, lease.members, profiles)
        evaluated_leases[lease.version] = feasible
        if feasible:
            true_feasible_locks += 1
            feasible_rosters[lease.task_id] = lease.members
        else:
            true_infeasible_locks += 1
        events.append(
            {
                "round": rounds,
                "phase": "terminal_scoring",
                "task_id": lease.task_id,
                "members": list(lease.members),
                "true_feasible": feasible,
            }
        )
    for task_id in sorted(directory.tasks):
        state = directory.tasks[task_id]
        lease = state.lease
        if (
            lease is None
            or state.phase is not TaskPhase.LOCKED
            or not evaluated_leases.get(lease.version, False)
        ):
            continue
        evidence = hashlib.sha256(
            f"{scenario.scenario_id}|{lease.task_id}|{lease.version}".encode()
        ).hexdigest()
        completion = directory.confirm_complete(
            task_id=lease.task_id,
            round_index=rounds,
            evidence_digest=evidence,
        )
        if not completion.accepted:
            raise RuntimeError(f"true completion rejected: {completion.code}")
        completed.add(lease.task_id)
        events.append(
            {
                "round": rounds,
                "phase": "completion_confirmation",
                "task_id": lease.task_id,
                "transition": completion.as_dict(),
            }
        )

    replay_payload = directory.export_replay()
    replayed = TeamDirectory.replay(replay_payload)
    replay_hash_match = replayed.state_hash() == directory.state_hash()
    oracle = true_information_oracle(scenario.tasks, scenario.agents) if oracle is None else oracle
    oracle_reward = int(oracle.total_completed_reward)
    achieved_reward = sum(task.reward for task in scenario.tasks if task.task_id in completed)
    normalized = Fraction(achieved_reward, oracle_reward) if oracle_reward else Fraction(1)
    formation_opportunities = len(_individually_feasible_tasks(scenario))
    formation_rate = (
        Fraction(true_feasible_locks, formation_opportunities)
        if formation_opportunities
        else Fraction(1)
    )
    oracle_opportunities = len(oracle.completed_tasks)
    oracle_coverage = (
        Fraction(len(completed), oracle_opportunities) if oracle_opportunities else Fraction(1)
    )
    task_by_id = {task.task_id: task for task in scenario.tasks}
    achieved_utility = sum(
        (
            roster_utility(
                task_by_id[task_id],
                members,
                profiles,
                information=InformationSource.TRUE,
            )
            for task_id, members in feasible_rosters.items()
            if task_id in completed
        ),
        Fraction(0),
    )
    achieved_cost = sum(
        (
            Fraction(profiles[member].task_costs[task_id])
            for task_id, members in feasible_rosters.items()
            if task_id in completed
            for member in members
        ),
        Fraction(0),
    )

    submissions = [
        item for item in directory.audit_records() if item["category"] == "control_submission"
    ]
    transition_records = [
        item["payload"]
        for item in directory.audit_records()
        if item["category"] == "transition" and item["payload"]["sender"] >= 0
    ]
    ordinary_records = [
        item["payload"]
        for item in directory.audit_records()
        if item["category"] == "ordinary_delivery"
    ]
    control_payload_bytes = sum(item["payload"]["parse"]["payload_bytes"] for item in submissions)
    payload_bytes_by_sequence = {
        int(item["payload"]["sequence"]): int(item["payload"]["parse"]["payload_bytes"])
        for item in submissions
    }
    control_delivered_bytes = sum(
        payload_bytes_by_sequence.get(int(item["sequence"]), 0) * len(item["delivered_recipients"])
        for item in transition_records
    )
    lock_sequences = {
        int(item["payload"]["sequence"])
        for item in submissions
        if (
            item["payload"]["parse"]["record"] is not None
            and item["payload"]["parse"]["record"]["kind"] == RecordKind.LOCK.value
        )
    }
    lock_results = [item for item in transition_records if int(item["sequence"]) in lock_sequences]
    if selector_audit is not None:
        events.append(
            {
                "round": rounds,
                "phase": "selector_audit",
                "method": selection_method.value,
                "information_source": InformationSource.CLAIMED.value,
                "random_seed_scheme": (
                    "sha256(e2d-contract-selector-v1,episode_seed,task_id,"
                    "sorted_available_bidder_ids)"
                ),
                "decisions": selector_audit,
            }
        )
    metrics = EpisodeMetrics(
        method=method_label,
        scenario_id=scenario.scenario_id,
        family=scenario.family,
        seed=scenario.seed,
        rounds=rounds,
        task_count=len(scenario.tasks),
        oracle_reward=oracle_reward,
        achieved_reward=achieved_reward,
        normalized_reward=normalized,
        reward_regret=oracle_reward - achieved_reward,
        oracle_utility=oracle.total_roster_utility,
        achieved_utility=achieved_utility,
        utility_regret=(
            oracle.total_roster_utility - achieved_utility
            if achieved_reward == oracle_reward
            else None
        ),
        oracle_raw_cost=oracle.total_raw_cost,
        achieved_raw_cost=achieved_cost,
        oracle_assignment=_oracle_assignment(oracle),
        completed_tasks=len(completed),
        true_feasible_locks=true_feasible_locks,
        true_infeasible_locks=true_infeasible_locks,
        formation_opportunities=formation_opportunities,
        formation_rate=formation_rate,
        oracle_allocation_opportunities=oracle_opportunities,
        oracle_allocation_coverage=oracle_coverage,
        lock_rounds=tuple(lock_rounds),
        control_submissions=len(submissions),
        valid_control_submissions=sum(
            bool(item["payload"]["parse"]["valid"]) for item in submissions
        ),
        accepted_control_transitions=sum(bool(item["accepted"]) for item in transition_records),
        rejected_control_transitions=sum(not bool(item["accepted"]) for item in transition_records),
        control_payload_bytes=control_payload_bytes,
        control_delivered_bytes=control_delivered_bytes,
        ordinary_deliveries=sum(
            len(item["recipients"]) for item in ordinary_records if item["accepted"]
        ),
        ordinary_payload_bytes=sum(
            len(item["content"].encode("utf-8")) for item in ordinary_records if item["accepted"]
        ),
        ordinary_delivered_bytes=sum(
            len(item["content"].encode("utf-8")) * len(item["recipients"])
            for item in ordinary_records
            if item["accepted"]
        ),
        unauthorized_ordinary_deliveries=sum(
            bool(item["accepted"])
            and any(
                directory.agent_to_task.get(recipient)
                != directory.agent_to_task.get(item["sender"])
                for recipient in item["recipients"]
            )
            for item in ordinary_records
        ),
        roster_agreement_rate=(
            Fraction(
                sum(bool(item["accepted"]) for item in lock_results),
                len(lock_results),
            )
            if lock_results
            else Fraction(1)
        ),
        overlapping_roster_rounds=overlapping_roster_rounds,
        multi_offer_agent_rounds=multi_offer_agent_rounds,
        roster_revision_count=roster_revision_count,
        overstaff_agent_slots=0,
        exact_roster_comparator=True,
        replay_hash_match=replay_hash_match,
        terminal_state_hash=directory.state_hash(),
        terminal_audit_chain_hash=directory.audit_chain_hash,
    )
    return ArenaEpisode(
        schema_version=SCHEMA_VERSION,
        scenario=scenario,
        method=method_label,
        metrics=metrics,
        events=tuple(events),
        directory_replay=replay_payload,
    )


def run_control_episode(
    scenario: Scenario,
    arm: ControlArm | str,
    *,
    rounds: int = DEFAULT_ROUNDS,
    oracle: Any | None = None,
) -> ArenaEpisode:
    """Evaluate one deterministic topology/allocation reference.

    The all-six and fixed-3+3 arms are deliberately marked as non-exact-roster
    comparators because their communication coalitions can exceed a task's
    requested roster size. Random allocation uses an exact-size roster and is
    therefore directly comparable to the oracle's allocation regret.
    """

    arm = ControlArm(arm)
    if rounds != DEFAULT_ROUNDS:
        raise ValueError(f"E2a requires exactly {DEFAULT_ROUNDS} recruitment rounds")
    profiles = {agent.agent_id: agent for agent in scenario.agents}
    oracle = true_information_oracle(scenario.tasks, scenario.agents) if oracle is None else oracle
    oracle_reward = int(oracle.total_completed_reward)
    rosters: list[tuple[TaskCard, tuple[int, ...], bool]] = []
    exact_roster_comparator = arm in {
        ControlArm.NO_TEAM,
        ControlArm.RANDOM_ALLOCATION,
    }

    random_rosters: dict[str, tuple[int, ...]] = {}
    if arm is ControlArm.RANDOM_ALLOCATION:
        choices_by_task = [
            tuple(combinations(AGENT_IDS, task.required_size)) for task in scenario.tasks
        ]
        disjoint_allocations = tuple(
            allocation
            for allocation in product(*choices_by_task)
            if len({member for roster in allocation for member in roster})
            == sum(len(roster) for roster in allocation)
        )
        selected = disjoint_allocations[
            _stable_int("random-allocation", scenario.seed, scenario.family.value)
            % len(disjoint_allocations)
        ]
        random_rosters = {
            task.task_id: roster for task, roster in zip(scenario.tasks, selected, strict=True)
        }

    for task_index, task in enumerate(scenario.tasks):
        if arm is ControlArm.NO_TEAM:
            continue
        if arm is ControlArm.PREFORMED_ALL_SIX:
            members = AGENT_IDS
        elif arm is ControlArm.FIXED_3_PLUS_3:
            groups = ((0, 1, 2), (3, 4, 5))
            # Public fixed topology: task-to-group mapping is independent of
            # private capability truth and each group receives at most one
            # simultaneous task in the frozen two-task scenarios.
            members = groups[task_index]
        else:
            members = random_rosters[task.task_id]
        feasible = _coverage_feasible(task, members, profiles)
        rosters.append((task, members, feasible))

    completed = [(task, members) for task, members, feasible in rosters if feasible]
    achieved_reward = sum(task.reward for task, _ in completed)
    normalized = (
        Fraction(achieved_reward, oracle_reward)
        if exact_roster_comparator and oracle_reward
        else Fraction(1)
        if exact_roster_comparator
        else None
    )
    formation_opportunities = len(_individually_feasible_tasks(scenario))
    formation_rate = (
        Fraction(len(completed), formation_opportunities)
        if formation_opportunities
        else Fraction(1)
    )
    oracle_opportunities = len(oracle.completed_tasks)
    oracle_coverage = (
        Fraction(len(completed), oracle_opportunities)
        if exact_roster_comparator and oracle_opportunities
        else Fraction(1)
        if exact_roster_comparator
        else None
    )
    achieved_utility = (
        sum(
            (
                roster_utility(
                    task,
                    members,
                    profiles,
                    information=InformationSource.TRUE,
                )
                for task, members in completed
            ),
            Fraction(0),
        )
        if exact_roster_comparator
        else None
    )
    achieved_cost = (
        sum(
            (
                Fraction(profiles[member].task_costs[task.task_id])
                for task, members in completed
                for member in members
            ),
            Fraction(0),
        )
        if exact_roster_comparator
        else None
    )
    messages = [
        (
            len(f"EXEC|TASK={task.task_id}".encode()),
            len(members),
        )
        for task, members in completed
    ]
    ordinary_payload_bytes = sum(payload * size for payload, size in messages)
    ordinary_deliveries = sum(size * (size - 1) for _, size in messages)
    ordinary_delivered_bytes = sum(payload * size * (size - 1) for payload, size in messages)
    events = tuple(
        {
            "round": task_index,
            "phase": "control_reference",
            "arm": arm.value,
            "task_id": task.task_id,
            "members": list(members),
            "requested_size": task.required_size,
            "true_feasible": feasible,
        }
        for task_index, (task, members, feasible) in enumerate(rosters)
    )
    terminal_projection = {
        "scenario_id": scenario.scenario_id,
        "arm": arm.value,
        "events": events,
        "achieved_reward": achieved_reward,
    }
    terminal_hash = hashlib.sha256(canonical_json(terminal_projection).encode("ascii")).hexdigest()
    metrics = EpisodeMetrics(
        method=arm.value,
        scenario_id=scenario.scenario_id,
        family=scenario.family,
        seed=scenario.seed,
        rounds=rounds,
        task_count=len(scenario.tasks),
        oracle_reward=oracle_reward,
        achieved_reward=achieved_reward,
        normalized_reward=normalized,
        reward_regret=(oracle_reward - achieved_reward if exact_roster_comparator else None),
        oracle_utility=oracle.total_roster_utility,
        achieved_utility=achieved_utility,
        utility_regret=(
            oracle.total_roster_utility - achieved_utility
            if exact_roster_comparator
            and achieved_utility is not None
            and achieved_reward == oracle_reward
            else None
        ),
        oracle_raw_cost=oracle.total_raw_cost,
        achieved_raw_cost=achieved_cost,
        oracle_assignment=_oracle_assignment(oracle),
        completed_tasks=len(completed),
        true_feasible_locks=len(completed),
        true_infeasible_locks=sum(not feasible for _, _, feasible in rosters),
        formation_opportunities=formation_opportunities,
        formation_rate=formation_rate,
        oracle_allocation_opportunities=oracle_opportunities,
        oracle_allocation_coverage=oracle_coverage,
        lock_rounds=tuple(range(len(rosters))),
        control_submissions=0,
        valid_control_submissions=0,
        accepted_control_transitions=0,
        rejected_control_transitions=0,
        control_payload_bytes=0,
        control_delivered_bytes=0,
        ordinary_deliveries=ordinary_deliveries,
        ordinary_payload_bytes=ordinary_payload_bytes,
        ordinary_delivered_bytes=ordinary_delivered_bytes,
        unauthorized_ordinary_deliveries=0,
        roster_agreement_rate=None,
        overlapping_roster_rounds=0,
        multi_offer_agent_rounds=0,
        roster_revision_count=0,
        overstaff_agent_slots=sum(
            max(0, len(members) - task.required_size) for task, members in completed
        ),
        exact_roster_comparator=exact_roster_comparator,
        replay_hash_match=None,
        terminal_state_hash=terminal_hash,
        terminal_audit_chain_hash=None,
    )
    return ArenaEpisode(
        schema_version=SCHEMA_VERSION,
        scenario=scenario,
        method=arm.value,
        metrics=metrics,
        events=events,
        directory_replay={
            "schema_version": "control-reference-v1",
            "terminal_state_hash": terminal_hash,
            "arm": arm.value,
        },
    )


def _coverage_feasible(
    task: TaskCard,
    members: Iterable[int],
    profiles: dict[int, AgentProfile],
) -> bool:
    roster = tuple(members)
    return all(
        sum(profiles[member].true_capabilities[dimension] for member in roster) >= demand
        for dimension, demand in enumerate(task.demand)
        if demand > 0
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


__all__ = [
    "AGENT_IDS",
    "DEFAULT_ROUNDS",
    "SCHEMA_VERSION",
    "AgentProfile",
    "ArenaEpisode",
    "ControlArm",
    "EpisodeMetrics",
    "Scenario",
    "ScenarioFamily",
    "TaskChoicePolicy",
    "canonical_json",
    "generate_scenario",
    "run_control_episode",
    "run_scripted_episode",
]
