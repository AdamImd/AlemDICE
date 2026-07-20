"""Tests for named LLM experiment and ablation configuration."""

import pytest

from baselines.llm.experiment_config import (
    ExperimentConfigError,
    compose_experiment,
    validate_experiment_config,
)


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


def test_openai_reduced_matches_paid_matrix_contract():
    config = compose_experiment("openai_reduced")
    spec = validate_experiment_config(config)

    assert spec.difficulties == ("easy", "medium", "hard")
    assert spec.seeds == (9999, 10000, 10001)
    assert spec.num_agents == 3
    assert spec.num_workers == 3
    assert spec.max_steps_per_episode == 200
    assert spec.nominal_decision_call_cap == 5400
    assert spec.nominal_debrief_call_cap == 0
    assert config.eval.generate_debriefs is False
    assert config.WANDB_MODE == "disabled"
    cache_keys = []
    roles = ("warrior", "forager", "miner")
    for client, role in zip(config.clients, roles, strict=True):
        assert client.client_name == "openai_responses"
        assert client.model_id == "gpt-5.6-luna"
        assert client.generate_kwargs.reasoning_effort == "none"
        assert client.generate_kwargs.prompt_cache_key
        assert client.generate_kwargs.prompt_cache_traffic_shards == 3
        assert client.generate_kwargs.prompt_cache_options == {
            "mode": "explicit",
            "ttl": "30m",
        }
        cache_key = str(client.generate_kwargs.prompt_cache_key)
        assert cache_key.endswith(f":role-{role}")
        assert ":traffic-" not in cache_key
        cache_keys.append(cache_key)
    assert len(set(cache_keys)) == 3


def test_upstream_full_has_twenty_seeds_and_canonical_cap():
    config = compose_experiment("upstream_main_full")
    reduced_config = compose_experiment("openai_reduced")
    spec = validate_experiment_config(config)

    assert spec.seeds == tuple(range(9999, 10019))
    assert spec.max_steps_per_episode == 10000
    assert spec.nominal_decision_call_cap == 1_800_000
    assert spec.nominal_debrief_call_cap == 180
    assert config.eval.generate_debriefs is True
    assert all(client.generate_kwargs.prompt_cache_traffic_shards == 3 for client in config.clients)
    assert [str(client.generate_kwargs.prompt_cache_key) for client in config.clients] == [
        str(client.generate_kwargs.prompt_cache_key) for client in reduced_config.clients
    ]


@pytest.mark.parametrize(
    ("ablation", "flag", "expected"),
    [
        ("hard_no_communication", "use_communication", False),
        ("hard_no_scratchpad", "use_scratchpad", False),
        ("hard_no_visible_cot", "use_cot", False),
    ],
)
def test_hard_ablation_manifests(ablation, flag, expected):
    config = compose_experiment("openai_reduced", ablation=ablation)
    spec = validate_experiment_config(config)

    assert spec.difficulties == ("hard",)
    assert config.alem.coordination_difficulty == "hard"
    assert config.agent[flag] is expected
    if ablation == "hard_no_visible_cot":
        assert config.agent.remember_cot is False


def test_validation_rejects_seed_episode_mismatch():
    config = compose_experiment("openai_reduced")
    config.eval.num_episodes.alem = 2

    with pytest.raises(ExperimentConfigError, match="must equal len"):
        validate_experiment_config(config)


def test_validation_rejects_non_responses_client():
    config = compose_experiment("openai_reduced")
    config.clients[1].client_name = "openai"

    with pytest.raises(ExperimentConfigError, match="openai_responses"):
        validate_experiment_config(config)


def test_validation_rejects_invalid_cache_traffic_shards():
    config = compose_experiment("openai_reduced")
    config.clients[0].generate_kwargs.prompt_cache_traffic_shards = 0

    with pytest.raises(ExperimentConfigError, match="prompt_cache_traffic_shards"):
        validate_experiment_config(config)
