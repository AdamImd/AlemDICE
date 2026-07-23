"""One end-to-end simulator smoke test; exhaustive dynamics tests are intentionally omitted."""

import jax
import jax.numpy as jnp

from alem.alem_coop.constants import Action
from alem.alem_coop.envs.alem_symbolic_env_debug import AlemCoopSymbolicEnvDebug


def test_debug_environment_resets_and_steps_three_agents():
    environment = AlemCoopSymbolicEnvDebug(num_agents=3)
    reset_key, step_key = jax.random.split(jax.random.PRNGKey(0))
    observations, state = environment.reset(reset_key)
    actions = {agent: jnp.int32(Action.NOOP.value) for agent in environment.agents}

    next_observations, next_state, rewards, dones, info = environment.step(step_key, state, actions)

    assert set(observations) == set(environment.agents)
    assert set(next_observations) == set(environment.agents)
    assert set(rewards) == set(environment.agents)
    assert set(environment.agents).issubset(dones)
    assert "__all__" in dones
    assert isinstance(info, dict)
    assert int(next_state.timestep) == int(state.timestep) + 1
