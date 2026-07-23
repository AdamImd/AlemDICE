"""One end-to-end commander comparison plus one cheap client-isolation check."""

import json
import threading
import types

import numpy as np

import baselines.llm.eval_utils.agents as agent_module
from baselines.llm.eval_utils.agents import AgentFactory
from baselines.llm.eval_utils.client import LLMResponse
from baselines.llm.eval_utils.evaluator import Evaluator
from baselines.llm.eval_utils.team_commander import EmbodiedCommanderPlanner, SquadSpec
from baselines.llm.experiment_config import compose_experiment


class _ScriptedPlannerClient:
    def generate(self, messages):
        payload = {
            "operation": "REPLACE",
            "objective": "Survive as one squad",
            "assignments": [
                {
                    "agent_id": agent_id,
                    "task_id": f"hold-{agent_id}",
                    "directive": f"Hold role {agent_id}",
                    "target": "current safe area",
                    "completion": "survived this lease",
                    "dependencies": [],
                }
                for agent_id in range(3)
            ],
        }
        return LLMResponse(
            model_id="scripted-commander",
            completion=f"<squad_plan>{json.dumps(payload)}</squad_plan>",
            stop_reason="stop",
            input_tokens=10,
            output_tokens=5,
        )


def test_commander_preserves_source_trajectory_shape_and_plans_before_parallel_actions(
    tmp_path,
):
    trajectories = {}
    treatment_result = None
    observed_directives = []
    for topology in ("baseline", "embodied_commander_broadcast"):
        config = compose_experiment("fake_smoke")
        config.team.topology = topology
        factory = AgentFactory(config)
        if topology != "baseline":
            factory.create_commander_planner = lambda spec: EmbodiedCommanderPlanner(
                lambda: _ScriptedPlannerClient(), spec=spec
            )
            original_create_agent = factory.create_agent
            barrier = threading.Barrier(3, timeout=3)

            def create_agent(agent_idx):
                agent = original_create_agent(agent_idx)
                original_act = agent.act

                def set_directive(self, directive):
                    self.test_directive = directive

                def concurrent_act(self, obs, prev_action=None):
                    assert "Full accepted plan:" in self.test_directive
                    observed_directives.append((self.agent_id, self.test_directive))
                    barrier.wait()
                    return original_act(obs, prev_action=prev_action)

                agent.set_squad_directive = types.MethodType(set_directive, agent)
                agent.act = types.MethodType(concurrent_act, agent)
                return agent

            factory.create_agent = create_agent

        root = tmp_path / topology
        result = Evaluator("alem", config, output_dir=str(root)).run_episode(
            "default", factory, episode_idx=0
        )
        with np.load(
            root / "alem" / "default" / "default_run_00_trajectory.npz",
            allow_pickle=True,
        ) as trajectory:
            trajectories[topology] = {
                field: trajectory[field].copy()
                for field in ("obs", "actions", "rewards", "dones", "text_actions")
            }
        if topology != "baseline":
            treatment_result = result
            journal_path = (
                root
                / "alem"
                / "default"
                / "default_run_00_commander_calls.jsonl"
            )
            journal = [
                json.loads(line)
                for line in journal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert len(journal) == 1
            assert journal[0]["application"]["accepted"] is True
            assert journal[0]["validation"]["code"] == "valid.replace"

    assert treatment_result["artifact_status"] == "complete"
    assert treatment_result["physical_worker_count"] == 3
    assert treatment_result["logical_participant_count"] == 3
    assert treatment_result["commander_plan_model_call_count"] == 1
    assert treatment_result["squad_commander"]["valid_plans"] == 1
    assert 0.0 <= treatment_result["squad_commander"]["valid_status_coverage"] <= 1.0
    assert {agent_id for agent_id, _ in observed_directives} == {0, 1, 2}
    assert len(observed_directives) == 9
    assert trajectories["embodied_commander_broadcast"]["actions"].shape == (3, 3)
    for field in trajectories["baseline"]:
        np.testing.assert_array_equal(
            trajectories["baseline"][field],
            trajectories["embodied_commander_broadcast"][field],
        )


def test_planner_clones_commander_client_with_an_isolated_cache_key(monkeypatch):
    config = compose_experiment("embodied_commander_30")
    captured = {}

    def fake_create(client_config):
        captured["config"] = client_config
        return lambda: object()

    monkeypatch.setattr(agent_module, "create_llm_client", fake_create)
    spec = SquadSpec(team_id="alpha", members=(0, 1, 2), commander_id=0, max_steps=30)
    planner = AgentFactory(config).create_commander_planner(spec)
    planner_config = captured["config"]
    source_config = config.clients[0]

    assert planner.spec == spec
    assert planner_config.client_name == source_config.client_name
    assert planner_config.model_id == source_config.model_id
    assert planner_config.generate_kwargs.reasoning_effort == "none"
    assert str(planner_config.generate_kwargs.prompt_cache_key).endswith(":planner-alpha-agent0")


def test_dedicated_planner_client_routes_luna_without_a_fourth_worker(monkeypatch):
    config = compose_experiment("embodied_commander_nano_luna_100")
    captured = {}

    def fake_create(client_config):
        captured["config"] = client_config
        return lambda: object()

    monkeypatch.setattr(agent_module, "create_llm_client", fake_create)
    spec = SquadSpec(team_id="alpha", members=(0, 1, 2), commander_id=0, max_steps=100)
    AgentFactory(config).create_commander_planner(spec)
    planner_config = captured["config"]

    assert len(config.clients) == 3
    assert {client.model_id for client in config.clients} == {"gpt-5.4-nano"}
    assert planner_config.model_id == "gpt-5.6-luna"
    assert planner_config.generate_kwargs.reasoning_effort == "high"
    assert str(planner_config.generate_kwargs.prompt_cache_key).endswith(
        ":planner-alpha-agent0"
    )
