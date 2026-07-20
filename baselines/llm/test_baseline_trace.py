"""Pinned current-main environment traces for AlemDICE baseline fidelity."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from baselines.llm.alem_env import make_env
from baselines.llm.experiment_config import compose_experiment

EXPECTED_TRACE_SHA256 = {
    "easy": "e6b7aac26fb134ad0ed813b0fbdf8b09d0e4cb7d14928a7fbe6127a94cf3a0fc",
    "medium": "6b09372c5b2dc583f19b6400d6dca61238298e3a09e53809a07ad39c6e7b6b5e",
    "hard": "8aeb089409b96a7f93d24ff06c29d29cfff67b029a5c6b357036d8ce148b3511",
}


def _snapshot(env, step, observations, rewards=None, dones=None):
    raw = env.env.get_obs(env.state)
    return {
        "step": step,
        "texts": [item["text"]["long_term_context"] for item in observations],
        "short": [item["text"]["short_term_context"] for item in observations],
        # Round only to make this regression fixture tolerant of inconsequential
        # accelerator-level float noise. commands.sh runs the gate on CPU.
        "raw": [
            np.asarray(raw[agent]).round(7).tolist() for agent in env.env.agents
        ],
        "rewards": rewards,
        "dones": dones,
    }


@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_seed_9999_three_noop_trace_matches_pinned_upstream(difficulty):
    config = compose_experiment(
        "fake_smoke", overrides=[f"alem.coordination_difficulty={difficulty}"]
    )
    env = make_env("alem", "default", config)
    observations, _ = env.reset(seed=9999)
    trace = [_snapshot(env, 0, observations)]

    for step in range(1, 4):
        observations, rewards, terminated, truncated, _ = env.step(["Noop"] * 3)
        trace.append(
            _snapshot(
                env,
                step,
                observations,
                rewards=[float(value) for value in rewards],
                dones=[
                    bool(is_terminated or is_truncated)
                    for is_terminated, is_truncated in zip(terminated, truncated)
                ],
            )
        )

    payload = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_TRACE_SHA256[difficulty]
