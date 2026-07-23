import pytest

from scripts.run_e0_agent_compatibility import (
    PopulationResult,
    ScriptedProbeAgent,
    _expected_roles,
    build_e0_config,
)


@pytest.mark.parametrize("num_agents", [1, 2, 3, 4, 6])
def test_e0_config_has_one_client_per_agent(num_agents):
    config = build_e0_config(
        num_agents=num_agents,
        seeds=(13000, 13001),
        steps=10,
    )

    assert config.alem.num_agents == num_agents
    assert len(config.clients) == num_agents
    assert config.experiment.seeds == [13000, 13001]
    assert config.eval.num_episodes.alem == 2
    assert config.eval.max_steps_per_episode == 10
    assert config.eval.num_workers == 1
    assert config.team.topology == "baseline"
    assert config.WANDB_MODE == "disabled"
    assert len(
        {
            client.generate_kwargs.prompt_cache_key
            for client in config.clients
        }
    ) == num_agents


def test_e0_seed_sequence_must_be_contiguous():
    with pytest.raises(ValueError, match="contiguous"):
        build_e0_config(num_agents=3, seeds=(13000, 13002), steps=10)


def test_scripted_probe_emits_no_usage_and_one_controlled_message():
    agent = ScriptedProbeAgent(4)
    agent.reset()
    agent.receive_communication({0: "hello"})
    response = agent.act({}, prev_action=None)

    assert response.completion == "Noop"
    assert response.input_tokens == response.output_tokens == 0
    assert agent.current_communication == "E0|AGENT=4|TICK=0|STATE=ACTIVE"
    assert agent.communication_history == [{0: "hello"}]
    assert "<action>Noop</action>" in agent._last_raw_completion


def test_expected_roles_cycle_warrior_forager_miner():
    assert _expected_roles(6) == [2, 1, 3, 2, 1, 3]


def test_population_result_reports_step_weighted_turn_times():
    result = PopulationResult(
        num_agents=2,
        passed=False,
        structural_passed=True,
        full_horizon=False,
        validation_notes=("seed 1 ended early",),
        episodes=(
            {
                "steps": 100,
                "episode_wall_seconds": 20.0,
                "mean_tick_wall_seconds": 0.2,
                "max_tick_wall_seconds": 1.0,
                "worker_round_wall_seconds": 0.1,
            },
            {
                "steps": 200,
                "episode_wall_seconds": 30.0,
                "mean_tick_wall_seconds": 0.1,
                "max_tick_wall_seconds": 2.0,
                "worker_round_wall_seconds": 0.2,
            },
        ),
        observation_shape=(10,),
        action_space_size=55,
        give_mapping_count=2,
        role_values=(2, 1),
        expected_role_values=(2, 1),
    )

    summary = result.as_dict()

    assert summary["timing"]["total_episode_wall_seconds"] == 50.0
    assert summary["timing"]["mean_episode_wall_seconds"] == 25.0
    assert summary["timing"]["mean_tick_wall_seconds"] == pytest.approx(2 / 15)
    assert summary["timing"]["max_tick_wall_seconds"] == 2.0
    assert summary["timing"]["total_worker_round_wall_seconds"] == pytest.approx(0.3)
