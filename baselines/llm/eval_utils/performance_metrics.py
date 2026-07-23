"""Derived, behavior-preserving episode metrics for population studies.

The environment already owns the authoritative achievement and event counters.
This module only gives those counters a stable analysis schema; it does not
observe actions, alter rewards, or introduce another notion of success.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

PERFORMANCE_METRICS_SCHEMA = "alem-dice-performance-v1"

_PAPER_SCORE_KEYS = {
    "base": "Team/normal_reward_pct_of_max",
    "coord": "Team/coord_reward_pct_of_max",
    "total": "Team/reward_pct_of_max",
}
_ACHIEVEMENT_COVERAGE_KEYS = {
    "base": "Team/normal_achievement_pct",
    "coord": "Team/coordination_achievement_pct",
    "total": "Team/achievement_pct",
}
_TEAM_ACHIEVEMENT_KEYS = {
    "base": "Team/normal_achievements",
    "coord": "Team/coordination_achievements",
    "total": "Team/total_achievements",
}
_AGENT_ACHIEVEMENT_SUFFIXES = {
    "base": "normal_achievements",
    "coord": "coordination_achievements",
    "total": "total_achievements",
}
_EVENT_KEYS = {
    # This is the environment's non-overlapping high-level coordination total.
    # Its domain components still have heterogeneous units (per timestep for
    # sync-style attempts and per setup for handovers).
    "coordination_attempts": "Coordination/total_attempts",
    "coordination_resolved_attempts": "Coordination/total_resolved_attempts",
    "coordination_successes": "Coordination/total_successes",
    # Cooperation counters are cumulative successful/attempted interactions.
    "give_attempts": "Cooperation/give_attempt_count",
    "successful_transfers": "Cooperation/trade_count",
    "requests": "Cooperation/request_count",
    "revives": "Cooperation/revives",
}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _count(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _percent(user_info: Mapping[str, Any], key: str) -> float | None:
    fraction = _finite_number(user_info.get(key))
    return None if fraction is None else 100.0 * fraction


def _summed_agent_count(
    user_info: Mapping[str, Any],
    num_agents: int,
    suffix: str,
) -> int | None:
    values = [
        _count(user_info.get(f"Agent{agent_idx}/{suffix}")) for agent_idx in range(num_agents)
    ]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def build_performance_metrics(
    episode_log: Mapping[str, Any],
    num_agents: int,
    *,
    environment_steps_completed: int | None = None,
    agent_turns_submitted: int | None = None,
    alive_agent_turns: int | None = None,
    actionable_agent_turns: int | None = None,
) -> dict[str, Any]:
    """Build the stable E1 metric block from canonical episode counters.

    Achievement state is cumulative and binary per agent/type in Alem. Thus a
    final team-union count is also the number of distinct team first-unlocks,
    while the sum over agents is the number of agent/type first-unlocks. It is
    not a count of repeated world interactions.
    """

    if isinstance(num_agents, bool) or not isinstance(num_agents, int) or num_agents < 1:
        raise ValueError("num_agents must be a positive integer")

    user_info_value = episode_log.get("user_info", {})
    user_info: Mapping[str, Any] = user_info_value if isinstance(user_info_value, Mapping) else {}

    paper_score_percent = {
        name: _percent(user_info, key) for name, key in _PAPER_SCORE_KEYS.items()
    }
    achievement_coverage_percent = {
        name: _percent(user_info, key) for name, key in _ACHIEVEMENT_COVERAGE_KEYS.items()
    }
    team_unique = {name: _count(user_info.get(key)) for name, key in _TEAM_ACHIEVEMENT_KEYS.items()}
    summed_across_agents = {
        name: _summed_agent_count(user_info, num_agents, suffix)
        for name, suffix in _AGENT_ACHIEVEMENT_SUFFIXES.items()
    }
    event_counters = {name: _count(user_info.get(key)) for name, key in _EVENT_KEYS.items()}

    completed_steps = _count(environment_steps_completed)
    if completed_steps is None:
        completed_steps = _count(episode_log.get("num_steps"))
    submitted_turns = _count(agent_turns_submitted)
    alive_turns = _count(alive_agent_turns)
    if alive_turns is None:
        alive_turns = _count(user_info.get("Cooperation/alive_agent_steps"))
    actionable_turns = _count(actionable_agent_turns)
    if actionable_turns is None:
        actionable_turns = _count(user_info.get("Cooperation/actionable_agent_steps"))

    completed_agent_turn_capacity = (
        num_agents * completed_steps if completed_steps is not None else None
    )
    classified_values = [
        _count(episode_log.get(key))
        for key in (
            "action_parse_success",
            "action_parse_fail",
            "action_parse_skipped_inactive",
        )
    ]
    classified_action_turns = (
        sum(value or 0 for value in classified_values)
        if any(value is not None for value in classified_values)
        else None
    )
    survival_fraction = (
        alive_turns / completed_agent_turn_capacity
        if alive_turns is not None and completed_agent_turn_capacity
        else None
    )
    actionable_fraction = (
        actionable_turns / completed_agent_turn_capacity
        if actionable_turns is not None and completed_agent_turn_capacity
        else None
    )

    return {
        "schema_version": PERFORMANCE_METRICS_SCHEMA,
        "paper_score_percent": paper_score_percent,
        "achievement_coverage_percent": achievement_coverage_percent,
        "achievement_first_unlock_count": {
            "team_unique": team_unique,
            "summed_across_agents": summed_across_agents,
        },
        "event_counters": event_counters,
        "exposure": {
            "environment_steps_completed": completed_steps,
            "completed_agent_turn_capacity": completed_agent_turn_capacity,
            "agent_turns_submitted": submitted_turns,
            "classified_action_turns": classified_action_turns,
            "alive_agent_turns": alive_turns,
            "actionable_agent_turns": actionable_turns,
            "survival_fraction": survival_fraction,
            "actionable_fraction": actionable_fraction,
        },
        "semantics": {
            "achievement_counts": (
                "Cumulative binary first-unlocks. team_unique counts each achievement type "
                "once across the team; summed_across_agents counts each agent/type unlock."
            ),
            "event_counters": (
                "Cumulative raw environment counters. Coordination totals combine "
                "domain counters with heterogeneous units; fields must not be summed."
            ),
            "score_percent": (
                "Reward-weighted Base/Coord/Total leaderboard scores on a 0-100 scale."
            ),
        },
    }
