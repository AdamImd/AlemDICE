"""Minimal contracts for the retained artifact experiment profiles."""

from baselines.llm.experiment_config import compose_experiment, validate_experiment_config


def test_fake_smoke_is_free_and_deterministic():
    config = compose_experiment("fake_smoke")
    spec = validate_experiment_config(config)

    assert config.agent.type == "dummy"
    assert spec.requires_api_key is False
    assert spec.difficulties == ("easy",)
    assert spec.seeds == (9999,)
    assert spec.max_steps_per_episode == 3
    assert spec.nominal_decision_call_cap == 0
    assert config.WANDB_MODE == "disabled"


def test_local_vllm_profile_is_32_agents_and_1000_steps():
    config = compose_experiment("gemma4_e4b_vllm_32x1000")
    spec = validate_experiment_config(config)

    assert spec.seeds == (14100,)
    assert spec.max_steps_per_episode == 1000
    assert len(config.clients) == spec.num_agents == 32
    assert all(client.client_name == "vllm" for client in config.clients)
    assert config.team.topology == "baseline"
    assert config.eval.generate_debriefs is False


def test_source_scaling_profile_is_unchanged_high_reasoning_source():
    config = compose_experiment("source_scaling_200")
    spec = validate_experiment_config(config)

    assert spec.difficulties == ("easy",)
    assert spec.seeds == (13100, 13101, 13102)
    assert spec.max_steps_per_episode == 200
    assert spec.num_workers == 1
    assert spec.num_agents == 1
    assert len(config.clients) == 6
    assert all(client.client_name == "openai_responses" for client in config.clients)
    assert all(client.model_id == "gpt-5.4-nano" for client in config.clients)
    assert all(client.generate_kwargs.reasoning_effort == "high" for client in config.clients)
    assert all(client.generate_kwargs.prompt_cache_traffic_shards == 1 for client in config.clients)
    assert config.agent.type == "robust_all"
    assert config.agent.prompt_mode == "specific_collaborative"
    assert config.agent.use_cot is True
    assert config.agent.use_scratchpad is True
    assert config.agent.use_communication is True
    assert config.coordination.strategy == "free"
    assert config.team.topology == "baseline"
    assert config.eval.generate_debriefs is False
