"""Tests for the sequential, resumable OpenAI matrix launcher."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts import run_openai_matrix


@pytest.fixture(autouse=True)
def _clean_source_for_launcher_tests(monkeypatch):
    monkeypatch.setattr(run_openai_matrix, "_git_status_lines", lambda: ())


def test_dry_run_prints_cap_without_key_or_files(tmp_path, monkeypatch, capsys):
    output_root = tmp_path / "dry"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_openai_matrix.run_matrix(
        ["--profile", "openai_reduced", "--output-root", str(output_root), "--dry-run"]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Nominal decision-call cap: 5,400" in captured.out
    assert "Maximum concurrent decision calls: 9" in captured.out
    assert not output_root.exists()


def test_real_profile_requires_api_key_before_creating_output(tmp_path, monkeypatch, capsys):
    output_root = tmp_path / "missing-key"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_openai_matrix.run_matrix(
        ["--profile", "openai_reduced", "--output-root", str(output_root)]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "OPENAI_API_KEY is required" in captured.err
    assert not output_root.exists()


def test_matrix_runs_difficulties_sequentially_and_writes_manifest(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "matrix"
    calls = []

    def fake_run(command):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(run_openai_matrix, "_run_eval", fake_run)

    result = run_openai_matrix.run_matrix(
        ["--profile", "openai_reduced", "--output-root", str(output_root)]
    )

    assert result == 0
    assert len(calls) == 3
    assert [
        next(arg for arg in command if arg.startswith("alem.coordination_difficulty="))
        for command in calls
    ] == [
        "alem.coordination_difficulty=easy",
        "alem.coordination_difficulty=medium",
        "alem.coordination_difficulty=hard",
    ]
    assert all(command[0] == run_openai_matrix.sys.executable for command in calls)
    manifest = json.loads(
        (output_root / run_openai_matrix.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["profile"] == "openai_reduced"
    assert manifest["seeds"] == [9999, 10000, 10001]
    assert len(manifest["resolved_config_sha256"]) == 64
    assert manifest["source_commit"]
    assert len(manifest["uv_lock_sha256"]) == 64
    assert (output_root / run_openai_matrix.RESOLVED_CONFIG_NAME).is_file()


def test_resume_skips_a_fully_completed_difficulty(tmp_path, monkeypatch, capsys):
    output_root = tmp_path / "resume"
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(
        run_openai_matrix,
        "_run_eval",
        lambda command: subprocess.CompletedProcess(command, 0),
    )
    assert (
        run_openai_matrix.run_matrix(
            ["--profile", "openai_reduced", "--output-root", str(output_root)]
        )
        == 0
    )

    episode_dir = output_root / "easy" / "alem" / "default"
    episode_dir.mkdir(parents=True, exist_ok=True)
    for episode_index in range(3):
        (episode_dir / f"default_run_{episode_index:02d}.json").write_text(
            json.dumps(
                {
                    "schema_version": "alem-dice-episode-v1",
                    "artifact_status": "complete",
                    "termination_reason": "environment_terminated",
                }
            ),
            encoding="utf-8",
        )

    calls = []

    def fake_resume_run(command):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_openai_matrix, "_run_eval", fake_resume_run)
    result = run_openai_matrix.run_matrix(
        ["--profile", "openai_reduced", "--resume", str(output_root)]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert len(calls) == 2
    assert "Skipping easy: all episode outputs already exist." in captured.out
    assert "Remaining decision-call cap: 3,600" in captured.out


def test_failed_or_corrupt_episode_markers_are_not_counted_complete(tmp_path):
    output_root = tmp_path / "incomplete"
    episode_dir = output_root / "easy" / "alem" / "default"
    episode_dir.mkdir(parents=True)
    (episode_dir / "default_run_00.json").write_text(
        '{"error": "provider unavailable"}\n', encoding="utf-8"
    )
    (episode_dir / "default_run_01.json").write_text("not json\n", encoding="utf-8")
    (episode_dir / "default_run_02.json").write_text("{}\n", encoding="utf-8")

    matrix, _ = run_openai_matrix.build_matrix_run(
        profile="openai_reduced", output_root=output_root
    )

    easy = matrix.difficulties[0]
    assert easy.completed_episode_units == 0
    assert easy.remaining_episode_units == 3
    assert easy.remaining_decision_call_cap == 1800


def test_only_explicit_terminal_marker_counts_as_complete(tmp_path):
    output_root = tmp_path / "complete"
    episode_dir = output_root / "easy" / "alem" / "default"
    episode_dir.mkdir(parents=True)
    (episode_dir / "default_run_00.json").write_text(
        json.dumps(
            {
                "schema_version": "alem-dice-episode-v1",
                "artifact_status": "complete",
                "termination_reason": "evaluator_step_cap",
            }
        ),
        encoding="utf-8",
    )

    matrix, _ = run_openai_matrix.build_matrix_run(
        profile="openai_reduced", output_root=output_root
    )

    assert matrix.difficulties[0].completed_episode_units == 1


def test_resume_rejects_profile_mismatch(tmp_path, monkeypatch, capsys):
    output_root = tmp_path / "mismatch"
    output_root.mkdir()
    (output_root / run_openai_matrix.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "fake_smoke",
                "ablation": None,
                "difficulties": ["easy"],
                "seeds": [9999],
                "num_agents": 3,
                "max_steps_per_episode": 3,
                "num_workers": 1,
                "model_ids": ["fake-smoke", "fake-smoke", "fake-smoke"],
                "generate_debriefs": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")

    result = run_openai_matrix.run_matrix(
        ["--profile", "openai_reduced", "--resume", str(output_root), "--dry-run"]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "Resume manifest does not match" in captured.err


def test_paid_run_rejects_dirty_source_before_creating_output(
    tmp_path, monkeypatch, capsys
):
    output_root = tmp_path / "dirty"
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(
        run_openai_matrix,
        "_git_status_lines",
        lambda: (" M baselines/llm/eval_alem.py",),
    )

    result = run_openai_matrix.run_matrix(
        ["--profile", "openai_reduced", "--output-root", str(output_root)]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "require a clean Git worktree" in captured.err
    assert not output_root.exists()
