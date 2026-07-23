import pytest

from scripts.run_e0_agent_compatibility import (
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
