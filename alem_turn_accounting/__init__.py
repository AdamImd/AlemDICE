"""Side-effect-free validation for versioned ALEM turn accounting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TURN_ACCOUNTING_SCHEMA_VERSION = "alem-dice-turn-accounting-v2"
TURN_ACCOUNTING_FEATURES = (
    "pre_step_parse_classification",
    "canonical_submitted_vs_effective_noop",
    "exhaustive_effective_noop_partition",
    "per_physical_worker_counters",
)
TURN_ACCOUNTING_SEMANTICS = {
    "parse_state": "pre-step worker inactivity",
    "canonical_submitted_noop": "canonical Noop passed to env.step after action validation",
    "effective_environment_noop": (
        "canonical submitted Noop, or any action masked to Noop because the "
        "worker was inactive in the pre-step observation"
    ),
    "effective_noop_partition": (
        "intentional actionable + parse fallback + active action-validation fallback "
        "+ inactive effective + active residual"
    ),
}

# Aggregate field -> physical-worker field suffix.
TURN_ACCOUNTING_AGENT_SUFFIXES = {
    "action_parse_success": "parse_success",
    "action_parse_fail": "parse_fail",
    "action_parse_skipped_inactive": "parse_skipped_inactive",
    "intentional_actionable_noop_count": "intentional_actionable_noop_count",
    "parse_fallback_noop_count": "parse_fallback_noop_count",
    "active_action_validation_fallback_noop_count": (
        "active_action_validation_fallback_noop_count"
    ),
    "active_residual_effective_noop_count": "active_residual_effective_noop_count",
    "inactive_submitted_turn_count": "inactive_submitted_turn_count",
    "inactive_effective_noop_count": "inactive_effective_noop_count",
    "canonical_submitted_noop_count": "canonical_submitted_noop_count",
    "effective_environment_noop_count": "effective_environment_noop_count",
    "executed_noop_count": "executed_noop_count",
}


def _count(record: Mapping[str, Any], key: str, context: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}: invalid or missing {key}")
    return value


def _validate_scope(
    values: Mapping[str, int],
    *,
    expected_turns: int,
    context: str,
) -> None:
    success = values["action_parse_success"]
    failure = values["action_parse_fail"]
    skipped = values["action_parse_skipped_inactive"]
    inactive = values["inactive_submitted_turn_count"]
    inactive_effective = values["inactive_effective_noop_count"]
    intentional = values["intentional_actionable_noop_count"]
    parse_fallback = values["parse_fallback_noop_count"]
    validation_fallback = values["active_action_validation_fallback_noop_count"]
    active_residual = values["active_residual_effective_noop_count"]
    canonical = values["canonical_submitted_noop_count"]
    effective = values["effective_environment_noop_count"]
    executed_alias = values["executed_noop_count"]

    if success + failure + skipped != expected_turns:
        raise ValueError(
            f"{context}: parse success+fail+skipped does not equal {expected_turns} turns"
        )
    if skipped != inactive or inactive != inactive_effective:
        raise ValueError(
            f"{context}: skipped-inactive, inactive-submitted, and "
            "inactive-effective counters disagree"
        )

    active_success_noops = intentional + validation_fallback + active_residual
    active_noops = active_success_noops + parse_fallback
    active_turns = success + failure
    if active_success_noops > success:
        raise ValueError(f"{context}: successful active-Noop causes exceed parse successes")
    if parse_fallback != failure:
        raise ValueError(f"{context}: parse-fallback Noops do not equal active parse failures")
    if active_noops > active_turns:
        raise ValueError(f"{context}: active-Noop partition exceeds active turns")
    if effective != inactive + active_noops:
        raise ValueError(f"{context}: effective-Noop identity is inconsistent")
    if not active_noops <= canonical <= active_noops + inactive:
        raise ValueError(f"{context}: canonical submitted Noops violate active/inactive bounds")
    if canonical > expected_turns or effective > expected_turns:
        raise ValueError(f"{context}: Noop counters exceed submitted turns")
    if executed_alias != canonical:
        raise ValueError(
            f"{context}: deprecated executed-Noop alias disagrees with canonical Noops"
        )


def validate_turn_accounting(
    record: Mapping[str, Any],
    *,
    allow_unfinalized: bool = False,
    context: str = "turn accounting",
) -> bool:
    """Validate a complete v2 record, or return ``False`` for a true legacy record.

    A record that declares v2 is never downgraded to legacy.  Unfinalized v2
    records are accepted only when ``allow_unfinalized`` is true and the
    explicit finalization marker is ``False``.
    """

    schema_declared = "turn_accounting_schema_version" in record
    schema = record.get("turn_accounting_schema_version")
    if schema is None and not schema_declared:
        orphaned_v2_metadata = {
            "turn_accounting_features",
            "turn_accounting_complete",
            "turn_accounting_semantics",
            "turn_accounting_provenance",
        }.intersection(record)
        if orphaned_v2_metadata:
            raise ValueError(f"{context}: versioned accounting metadata lacks a schema declaration")
        return False
    if schema != TURN_ACCOUNTING_SCHEMA_VERSION:
        raise ValueError(f"{context}: unsupported declared turn-accounting schema {schema!r}")

    complete = record.get("turn_accounting_complete")
    if complete is not True:
        if allow_unfinalized and complete is False:
            return False
        raise ValueError(f"{context}: declared v2 accounting is not finalized")
    if list(record.get("turn_accounting_features") or ()) != list(TURN_ACCOUNTING_FEATURES):
        raise ValueError(f"{context}: invalid or missing v2 feature declaration")
    if record.get("turn_accounting_semantics") != TURN_ACCOUNTING_SEMANTICS:
        raise ValueError(f"{context}: invalid or missing v2 semantics declaration")
    if record.get("turn_accounting_provenance") != "evaluator_exact_pre_step":
        raise ValueError(f"{context}: invalid or missing v2 provenance")

    worker_count = _count(record, "physical_worker_count", context)
    if worker_count < 1:
        raise ValueError(f"{context}: physical_worker_count must be positive")
    num_steps = _count(record, "num_steps", context)

    aggregates = {field: _count(record, field, context) for field in TURN_ACCOUNTING_AGENT_SUFFIXES}
    worker_scopes: list[dict[str, int]] = []
    for worker_id in range(worker_count):
        worker_context = f"{context}, worker {worker_id}"
        scope = {
            aggregate: _count(record, f"agent_{worker_id}_{suffix}", worker_context)
            for aggregate, suffix in TURN_ACCOUNTING_AGENT_SUFFIXES.items()
        }
        _validate_scope(scope, expected_turns=num_steps, context=worker_context)
        worker_scopes.append(scope)

    for aggregate, aggregate_value in aggregates.items():
        worker_sum = sum(scope[aggregate] for scope in worker_scopes)
        if worker_sum != aggregate_value:
            raise ValueError(
                f"{context}: aggregate {aggregate}={aggregate_value} "
                f"disagrees with physical-worker sum={worker_sum}"
            )
    _validate_scope(
        aggregates,
        expected_turns=num_steps * worker_count,
        context=f"{context}, aggregate",
    )
    return True


__all__ = [
    "TURN_ACCOUNTING_AGENT_SUFFIXES",
    "TURN_ACCOUNTING_FEATURES",
    "TURN_ACCOUNTING_SCHEMA_VERSION",
    "TURN_ACCOUNTING_SEMANTICS",
    "validate_turn_accounting",
]
