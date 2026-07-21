"""Tests that per-difficulty provenance is preserved across resume."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from baselines.llm.eval_alem import _write_run_provenance


def _config():
    return OmegaConf.create(
        {
            "experiment": {"profile": "test"},
            "alem": {"coordination_difficulty": "easy"},
            "clients": [{"model_id": "gpt-test"}],
            "wandb": {"run_id": "first"},
        }
    )


def test_resume_appends_history_and_keeps_initial_manifest(tmp_path):
    # Use the actual repository for Git/lock provenance while isolating outputs.
    repo = str(Path(__file__).resolve().parents[2])
    started = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
    _write_run_provenance(_config(), str(tmp_path), repo, started)
    initial = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))

    resumed_config = _config()
    resumed_config.wandb.run_id = "second"
    _write_run_provenance(
        resumed_config,
        str(tmp_path),
        repo,
        started + timedelta(minutes=1),
    )
    resumed = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))

    assert resumed["started_at"] == initial["started_at"]
    assert resumed["resolved_config_file"] == "resolved_config.yaml"
    assert (tmp_path / "resolved_config.yaml").is_file()
    assert len(resumed["resume_history"]) == 1
    resume_entry = resumed["resume_history"][0]
    assert resume_entry["resolved_config_file"].startswith("resolved_config.resume-")
    assert (tmp_path / resume_entry["resolved_config_file"]).is_file()
    assert not list(tmp_path.glob(".*.tmp"))


def test_corrupt_resume_manifest_is_rejected_before_config_snapshot(tmp_path):
    repo = str(Path(__file__).resolve().parents[2])
    (tmp_path / "run_manifest.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be a JSON object"):
        _write_run_provenance(
            _config(),
            str(tmp_path),
            repo,
            datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
        )

    assert not list(tmp_path.glob("resolved_config.resume-*.yaml"))
