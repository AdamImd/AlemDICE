"""Three-step integration checks for bodyless leader evaluator topology."""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.llm.eval_utils.agents import AgentFactory
from baselines.llm.eval_utils.client import LLMResponse
from baselines.llm.eval_utils.evaluator import Evaluator
from baselines.llm.eval_utils.team_leader import TeamLeaderAgent
from baselines.llm.experiment_config import compose_experiment


class _ScriptedLeaderClient:
    def generate(self, messages):
        payload = {
            "objective": "Survive",
            "assignments": [
                {
                    "agent_id": agent_id,
                    "subgoal": f"Hold role {agent_id}",
                    "milestones": ["Report blockers"],
                }
                for agent_id in range(3)
            ],
        }
        return LLMResponse(
            model_id="scripted-team-leader",
            completion=f"<team_plan>{json.dumps(payload)}</team_plan>",
            stop_reason="stop",
            input_tokens=10,
            output_tokens=5,
        )


@pytest.mark.parametrize("topology", ["leader_peer", "leader_no_peer"])
def test_bodyless_leader_never_changes_physical_trajectory_shapes(tmp_path, topology):
    config = compose_experiment("fake_smoke")
    config.team.topology = topology
    config.team.leader_replan_interval = 1
    config.clients.append(
        OmegaConf.create(OmegaConf.to_container(config.clients[0], resolve=True))
    )
    factory = AgentFactory(config)

    def create_scripted_leader():
        return TeamLeaderAgent(lambda: _ScriptedLeaderClient())

    factory.create_leader = create_scripted_leader
    evaluator = Evaluator("alem", config, output_dir=str(tmp_path))
    result = evaluator.run_episode("default", factory, episode_idx=0)

    assert result["artifact_status"] == "complete"
    assert result["num_agents"] == 3
    assert result["physical_worker_count"] == 3
    assert result["logical_participant_count"] == 4
    assert result["leader"]["bodyless"] is True
    assert result["leader"]["plan_calls"] == 3
    trajectory = np.load(
        tmp_path / "alem" / "default" / "default_run_00_trajectory.npz",
        allow_pickle=True,
    )
    assert trajectory["actions"].shape == (3, 3)
    assert trajectory["rewards"].shape == (3, 3)
    assert trajectory["text_obs"].shape == (3, 3)
    assert trajectory["text_actions"].shape == (3, 3)

    communication = result["communication_metrics"]
    assert communication["observation_to_leader"]["delivered_messages"] == 9
    assert communication["leader_to_worker"]["delivered_messages"] == 9
    if topology == "leader_no_peer":
        assert communication.get("worker_peer", {}).get("delivered_messages", 0) == 0
