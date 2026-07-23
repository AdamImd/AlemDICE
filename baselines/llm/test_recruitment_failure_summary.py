import gzip
import hashlib
import json

import pytest

from scripts.summarize_recruitment_llm_failure import (
    _write_text_outside_root,
    render_markdown,
    summarize_failed_campaign,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_cell(root, *, seed, family, method, rows, budget_exhausted):
    stem = f"{method}__{family}__seed_{seed}"
    artifact_path = root / "episodes" / f"{stem}.json"
    debug_path = root / "debug" / f"{stem}.calls.jsonl.gz"
    marker_path = root / "markers" / f"{stem}.complete.json"
    for path in (artifact_path, debug_path, marker_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    episode = {
        "rounds": [
            {
                "decisions": [
                    {
                        "abstain_code": (
                            "budget.campaign_poisoned"
                            if budget_exhausted
                            else "model.abstain"
                        )
                    }
                ]
            }
        ],
        "analysis_only": {
            "oracle_allocation_coverage": 1.0,
            "achieved_reward": 100,
        },
        "replay_hash_match": True,
    }
    artifact_path.write_text(json.dumps(episode), encoding="utf-8")
    content = "".join(_canonical(row) + "\n" for row in rows).encode("ascii")
    compressed = gzip.compress(content, mtime=0)
    debug_path.write_bytes(compressed)
    marker = {
        "seed": seed,
        "family": family,
        "method": method,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "debug_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
        "debug_content_sha256": hashlib.sha256(content).hexdigest(),
        "debug_record_count": len(rows),
        "logical_calls": len(rows),
        "budget_exhausted_decisions": budget_exhausted,
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    return marker


def test_failed_campaign_summary_excludes_failed_cancelled_and_poisoned_cells(
    tmp_path,
):
    completed_row = {
        "normalized_parse": {"validation_code": "abstain"},
        "response": {
            "status": "completed",
            "incomplete_reason": None,
            "usage": {"output_tokens": 10, "reasoning_tokens": 5},
        },
        "raw_completion": "ABSTAIN",
    }
    truncated_row = {
        "round_index": 0,
        "agent_id": 1,
        "attempt": 0,
        "normalized_parse": {"validation_code": "provider.status_not_completed"},
        "response": {
            "status": "incomplete",
            "incomplete_reason": "max_output_tokens",
            "usage": {"output_tokens": 1024, "reasoning_tokens": 1024},
        },
        "raw_completion": "",
    }
    eligible = _write_cell(
        tmp_path,
        seed=1,
        family="single",
        method="open",
        rows=[completed_row],
        budget_exhausted=0,
    )
    poisoned = _write_cell(
        tmp_path,
        seed=1,
        family="single",
        method="mutual",
        rows=[],
        budget_exhausted=1,
    )
    _write_cell(
        tmp_path,
        seed=1,
        family="two",
        method="open",
        rows=[truncated_row],
        budget_exhausted=0,
    )
    ledger = b"{\"event\":\"synthetic\"}\n"
    (tmp_path / "reservation_ledger.jsonl").write_bytes(ledger)
    anchor_dir = tmp_path / "reservation_anchors"
    anchor_dir.mkdir()
    (anchor_dir / "00000000000000000000.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    full = {
        "status": "failed",
        "protocol": {
            "seeds": [1],
            "scenario_families": ["single", "two"],
            "methods": ["open", "mutual"],
            "max_output_tokens": 1024,
        },
        "markers": [eligible, poisoned],
        "failed_episodes": [
            {"seed": 1, "family": "two", "method": "open"},
        ],
        "cancelled_cells": [
            {"seed": 1, "family": "two", "method": "mutual"},
        ],
        "campaign_budget": {
            "logical_used": 2,
            "provider_attempts_reserved": 4,
            "tokens_reserved": 100,
            "poisoned": True,
            "poisoned_reason": "cell_failure:RuntimeError",
        },
        "reservation_ledger_final": {
            "ledger_sha256": hashlib.sha256(ledger).hexdigest(),
            "records": 1,
            "anchor_count": 1,
            "unresolved": [],
            "overages": [],
        },
        "source_commit": "a" * 40,
        "config_sha256": "b" * 64,
    }
    (tmp_path / "run_manifest_full.json").write_text(
        json.dumps(full),
        encoding="utf-8",
    )
    (tmp_path / "run_manifest_canary.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "canary_gate.json").write_text("{}\n", encoding="utf-8")

    summary = summarize_failed_campaign(tmp_path)

    assert summary["campaign_status"] == "failed_non_promotable"
    assert summary["classification_counts"] == {
        "cancelled_no_evidence": 1,
        "eligible_partial_diagnostic": 1,
        "excluded_zero_call_poisoned": 1,
        "failed_diagnostic_only": 1,
    }
    assert summary["diagnostic_totals"]["debug_calls"] == 2
    assert summary["diagnostic_totals"]["max_output_truncations"] == 1
    assert (
        summary["diagnostic_totals"]["exact_cap_reasoning_only_truncations"]
        == 1
    )
    assert len(summary["eligible_partial_outcomes"]) == 1
    markdown = render_markdown(summary)
    assert "failed and permanently non-promotable" in markdown
    assert "not a between-method efficacy result" in markdown
    with pytest.raises(ValueError, match="outside the preserved campaign root"):
        _write_text_outside_root(
            tmp_path,
            tmp_path / "must_not_write.md",
            markdown,
        )
