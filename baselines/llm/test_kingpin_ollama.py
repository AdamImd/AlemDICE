"""Small contracts for the Kingpin Ollama deployment profile and launcher."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from scripts.run_kingpin_ollama import _uv_executable, evaluator_command, parse_args


def test_profile_routes_eight_agents_to_each_owned_port():
    config_dir = str((Path(__file__).parent / "config").resolve())
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        config = compose(
            config_name="config",
            overrides=["experiment=gemma4_31b_ollama_32x25"],
        )

    assert config.alem.num_agents == 32
    assert config.eval.max_steps_per_episode == 25
    assert config.eval.debug is False
    assert config.eval.save_images is False
    assert config.team.topology == "baseline"
    assert len(config.clients) == 32
    assert [client.base_url for client in config.clients] == [
        f"http://127.0.0.1:{port}" for port in (11534, 11535, 11536, 11537) for _ in range(8)
    ]
    assert all(client.client_name == "ollama" for client in config.clients)


def test_smoke_command_caps_every_agents_output():
    args = parse_args(["smoke", "--dry-run"])
    command = evaluator_command(args, smoke=True)

    assert "eval.max_steps_per_episode=1" in command
    assert "alem.max_timesteps=1" in command
    assert sum("generate_kwargs.max_tokens=128" in item for item in command) == 32


def test_default_ports_do_not_overlap_system_services():
    args = parse_args(["start"])
    assert tuple(args.ports) == (11534, 11535, 11536, 11537)


def test_uv_discovery_returns_an_absolute_executable():
    uv = Path(_uv_executable())
    assert uv.is_absolute()
    assert uv.is_file()
