"""Minimal public environment-factory contract."""

from alem.alem_coop.envs.alem_pixels_env import AlemCoopPixelsEnv
from alem.alem_coop.envs.alem_symbolic_env import AlemCoopSymbolicEnv
from alem.alem_coop.envs.alem_symbolic_env_debug import AlemCoopSymbolicEnvDebug
from alem.alem_env import make_alem_env_from_name


def test_environment_factory_dispatches_supported_names():
    expected = {
        "Alem-Coop-Symbolic": AlemCoopSymbolicEnv,
        "Alem-Coop-Symbolic-Debug": AlemCoopSymbolicEnvDebug,
        "Alem-Coop-Pixels": AlemCoopPixelsEnv,
    }

    for name, environment_type in expected.items():
        assert isinstance(make_alem_env_from_name(name), environment_type)
