#!/usr/bin/env python3
"""Replay, classify, and export commander-call evidence for human review."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.llm.eval_utils.team_commander import (  # noqa: E402
    SquadSpec,
    validate_squad_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if isinstance(record, dict):
            records.append(record)
    return records


def _legacy_records(path: Path) -> list[dict[str, Any]]:
    records = []
    ordinal = 0
    spec = SquadSpec(
        team_id="squad-0",
        members=(0, 1, 2),
        commander_id=0,
        review_interval=5,
        lease_steps=10,
        max_steps=100,
    )
    for step_record in _read_jsonl(path):
        call = step_record.get("commander_planning")
        if not isinstance(call, dict):
            continue
        step = int(step_record.get("step", 0))
        validation = validate_squad_plan(
            call.get("raw_output"),
            spec=spec,
            step=step,
        )
        accepted = bool(call.get("plan_valid"))
        failure_code = None
        if not accepted:
            failure_code = (
                validation.code
                if not validation.valid
                else f"policy.{call.get('plan_result') or 'rejected'}"
            )
        records.append(
            {
                "schema_version": "alem-dice-commander-call-legacy-v1",
                "call_id": f"legacy:{path.name}:{ordinal}",
                "attempt_id": None,
                "episode_index": 0,
                "seed": None,
                "planner_call_index": ordinal,
                "step": step,
                "review": {
                    "trigger": call.get("review_trigger"),
                    "scheduled": call.get("review_trigger") == "scheduled",
                },
                "request": {"prompt_messages": call.get("prompt_messages")},
                "response": {
                    "model_id": call.get("provider_model"),
                    "response_id": call.get("provider_response_id"),
                    "raw_output": call.get("raw_output"),
                    "input_tokens": call.get("input_tokens", 0),
                    "output_tokens": call.get("output_tokens", 0),
                    "reasoning_tokens": call.get("reasoning_tokens", 0),
                    "cached_tokens": call.get("cached_tokens", 0),
                    "latency_seconds": call.get("latency_seconds", 0.0),
                },
                "validation": asdict(validation),
                "application": {
                    "accepted": accepted,
                    "reason": call.get("plan_result"),
                    "effective_plan_after": call.get("parsed_plan"),
                    "fallback": "unknown_legacy",
                },
                "failure_code": failure_code,
                "exception": None,
                "source_file": str(path),
                "legacy_missing_fields": True,
            }
        )
        ordinal += 1
    return records


def _native_records(path: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    for record in records:
        if record.get("schema_version") != "alem-dice-commander-call-v1":
            raise ValueError(f"{path}: unsupported commander-call schema")
        context = ((record.get("request") or {}).get("context") or {})
        spec_payload = context.get("squad_spec")
        response = record.get("response") or {}
        if spec_payload and response:
            spec = SquadSpec(
                **{
                    **spec_payload,
                    "members": tuple(spec_payload["members"]),
                }
            )
            replay = validate_squad_plan(
                response.get("raw_output"),
                spec=spec,
                step=int(record["step"]),
                canonical_actions=context.get("canonical_actions"),
            )
            stored = record.get("validation") or {}
            if replay.valid != stored.get("valid") or replay.code != stored.get("code"):
                raise ValueError(
                    f"{path}: replay mismatch for {record.get('call_id')}: "
                    f"stored={stored.get('code')} replay={replay.code}"
                )
        record["source_file"] = str(path)
    return records


def _collect(input_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.name.endswith("_commander_calls.jsonl"):
            return input_path.parent, _native_records(input_path)
        if input_path.name.endswith("_debug.jsonl"):
            return input_path.parent, _legacy_records(input_path)
        raise ValueError("Input file must be a commander-call journal or debug JSONL")
    journals = sorted(input_path.rglob("*_commander_calls.jsonl"))
    if not journals:
        raise FileNotFoundError(f"No commander-call journals found under {input_path}")
    records = []
    for path in journals:
        records.extend(_native_records(path))
    return input_path, records


def _with_recovery(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record.get("attempt_id"),
            record.get("episode_index"),
            record.get("source_file", ""),
        )
        grouped.setdefault(key, []).append(record)
    failures = []
    for calls in grouped.values():
        calls.sort(key=lambda value: int(value.get("planner_call_index", 0)))
        for index, record in enumerate(calls):
            if not record.get("failure_code"):
                continue
            next_success = next(
                (
                    candidate
                    for candidate in calls[index + 1 :]
                    if (candidate.get("application") or {}).get("accepted")
                ),
                None,
            )
            enriched = dict(record)
            enriched["recovery"] = {
                "next_success_call_id": (
                    next_success.get("call_id") if next_success is not None else None
                ),
                "ticks_to_next_success": (
                    int(next_success["step"]) - int(record["step"])
                    if next_success is not None
                    else None
                ),
            }
            failures.append(enriched)
    return failures


def analyze(input_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    default_output, records = _collect(input_path)
    output_dir = (output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    call_ids = [record.get("call_id") for record in records]
    if None in call_ids or len(call_ids) != len(set(call_ids)):
        raise ValueError("Commander call IDs must be present and unique")
    failures = _with_recovery(records)
    codes = Counter(str(record["failure_code"]) for record in failures)
    fallbacks = Counter(
        str((record.get("application") or {}).get("fallback"))
        for record in failures
    )
    summary = {
        "schema_version": "alem-dice-commander-failure-summary-v1",
        "input": str(input_path.resolve()),
        "call_count": len(records),
        "accepted_call_count": sum(
            bool((record.get("application") or {}).get("accepted"))
            for record in records
        ),
        "failed_call_count": len(failures),
        "failure_counts": dict(sorted(codes.items())),
        "fallback_counts": dict(sorted(fallbacks.items())),
        "replay_verified_native_calls": sum(
            record.get("schema_version") == "alem-dice-commander-call-v1"
            for record in records
        ),
    }
    failed_path = output_dir / "commander_failures.jsonl"
    failed_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in failures),
        encoding="utf-8",
    )
    annotations = [
        {
            "call_id": record["call_id"],
            "groundedness": None,
            "physical_feasibility": None,
            "role_compliance": None,
            "dependency_coherence": None,
            "objective_usefulness": None,
            "format_only_repairable": None,
            "severity": None,
            "evidence": "",
            "reviewer": "",
            "notes": "",
        }
        for record in failures
    ]
    (output_dir / "commander_failure_annotations.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in annotations),
        encoding="utf-8",
    )
    (output_dir / "commander_failure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Commander Failure Audit",
        "",
        f"- Calls: {summary['call_count']}",
        f"- Accepted: {summary['accepted_call_count']}",
        f"- Failed: {summary['failed_call_count']}",
        "",
        "## Failure taxonomy",
        "",
    ]
    lines.extend(
        f"- `{code}`: {count}" for code, count in summary["failure_counts"].items()
    )
    lines.extend(["", "## Failed calls", ""])
    for record in failures:
        recovery = record["recovery"]["ticks_to_next_success"]
        lines.append(
            f"- Tick {record['step']} — `{record['failure_code']}`; "
            f"fallback `{(record.get('application') or {}).get('fallback')}`; "
            f"recovery {recovery if recovery is not None else 'not observed'} ticks; "
            f"call `{record['call_id']}`."
        )
    (output_dir / "commander_failure_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    args = _parser().parse_args()
    result = analyze(args.input, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
