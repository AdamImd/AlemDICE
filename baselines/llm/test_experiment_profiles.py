"""Minimal contracts for free smoke and paid embodied-commander profiles."""

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


def test_embodied_commander_profiles_match_source_luna_contract():
    source = compose_experiment("upstream_main_full")
    stages = (
        ("embodied_commander_30", (12000,), 30, 1),
        ("embodied_commander_100", (12000,), 100, 1),
        ("embodied_commander_200", (12000, 12001, 12002), 200, 3),
    )

    for profile, seeds, steps, workers in stages:
        config = compose_experiment(profile)
        spec = validate_experiment_config(config)
        assert spec.difficulties == ("easy",)
        assert spec.seeds == seeds
        assert spec.max_steps_per_episode == steps
        assert spec.num_workers == workers
        assert len(config.clients) == 3
        assert config.team.topology == "baseline"
        assert tuple(config.team.members) == (0, 1, 2)
        assert config.team.commander_agent_id == 0
        assert config.team.commander_review_interval == 5
        assert config.team.commander_lease_steps == 10
        assert config.eval.generate_debriefs is False
        for actual, canonical in zip(config.clients, source.clients, strict=True):
            assert actual.client_name == canonical.client_name == "openai_responses"
            assert actual.model_id == canonical.model_id == "gpt-5.6-luna"
            assert actual.generate_kwargs == canonical.generate_kwargs
            assert actual.timeout == canonical.timeout
            assert actual.max_retries == canonical.max_retries
            assert actual.delay == canonical.delay


def test_nano_luna_profile_keeps_planner_outside_the_three_workers():
    config = compose_experiment("embodied_commander_nano_luna_100")
    spec = validate_experiment_config(config)

    assert spec.seeds == (12100,)
    assert spec.max_steps_per_episode == 100
    assert len(config.clients) == spec.num_agents == 3
    assert spec.model_ids == ("gpt-5.4-nano",) * 3
    assert all(
        client.generate_kwargs.reasoning_effort == "none"
        for client in config.clients
    )
    assert spec.commander_planner_model_id == "gpt-5.6-luna"
    assert spec.commander_planner_reasoning_effort == "high"
    assert "prompt_cache_options" not in config.clients[0].generate_kwargs
