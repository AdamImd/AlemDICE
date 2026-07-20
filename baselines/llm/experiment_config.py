"""Composition and validation helpers for named Alem LLM experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parent / "config"
PROFILE_NAMES = ("fake_smoke", "openai_reduced", "upstream_main_full", "team_leader_200")
ABLATION_NAMES = (
    "hard_no_communication",
    "hard_no_scratchpad",
    "hard_no_visible_cot",
)
DIFFICULTIES = ("easy", "medium", "hard")


class ExperimentConfigError(ValueError):
    """Raised when an experiment profile cannot be executed safely."""


@dataclass(frozen=True)
class ExperimentSpec:
    """Validated matrix dimensions used by the launcher."""

    profile: str
    difficulties: tuple[str, ...]
    seeds: tuple[int, ...]
    num_agents: int
    max_steps_per_episode: int
    num_workers: int
    model_ids: tuple[str, ...]
    requires_api_key: bool
    generate_debriefs: bool

    @property
    def episodes_per_difficulty(self) -> int:
        return len(self.seeds)

    @property
    def nominal_decision_call_cap(self) -> int:
        if not self.requires_api_key:
            return 0
        return (
            len(self.difficulties)
            * self.episodes_per_difficulty
            * self.max_steps_per_episode
            * self.num_agents
        )

    @property
    def nominal_debrief_call_cap(self) -> int:
        if not self.generate_debriefs:
            return 0
        return len(self.difficulties) * self.episodes_per_difficulty * self.num_agents


def compose_experiment(
    profile: str,
    *,
    ablation: str | None = None,
    overrides: Iterable[str] = (),
) -> DictConfig:
    """Compose a named profile using Alem's Hydra configuration tree."""

    if profile not in PROFILE_NAMES:
        raise ExperimentConfigError(
            f"Unknown experiment profile {profile!r}; choose one of {', '.join(PROFILE_NAMES)}"
        )
    if ablation is not None and ablation not in ABLATION_NAMES:
        raise ExperimentConfigError(
            f"Unknown ablation {ablation!r}; choose one of {', '.join(ABLATION_NAMES)}"
        )

    hydra_overrides = [f"experiment={profile}"]
    if ablation is not None:
        hydra_overrides.append(f"ablation={ablation}")
    hydra_overrides.extend(overrides)

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR), version_base="1.1", job_name="alem_experiment"
    ):
        config = compose(config_name="config", overrides=hydra_overrides)
    validate_experiment_config(config, expected_profile=profile)
    return config


def _require_positive_int(config: DictConfig, key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExperimentConfigError(f"{key} must be a positive integer; got {value!r}")
    return value


def validate_experiment_config(
    config: DictConfig, *, expected_profile: str | None = None
) -> ExperimentSpec:
    """Validate invariants relied on by the evaluator and matrix launcher."""

    experiment = config.get("experiment")
    if experiment is None:
        raise ExperimentConfigError("Missing experiment metadata; select experiment=<profile>")

    profile = str(experiment.get("profile", "")).strip()
    if not profile:
        raise ExperimentConfigError("experiment.profile must be a non-empty string")
    if expected_profile is not None and profile != expected_profile:
        raise ExperimentConfigError(
            f"Composed profile {profile!r} does not match requested profile {expected_profile!r}"
        )

    difficulties = tuple(str(value) for value in experiment.get("difficulties", ()))
    if not difficulties:
        raise ExperimentConfigError("experiment.difficulties must not be empty")
    invalid_difficulties = [value for value in difficulties if value not in DIFFICULTIES]
    if invalid_difficulties:
        raise ExperimentConfigError(
            "experiment.difficulties contains unsupported values: "
            + ", ".join(invalid_difficulties)
        )
    if len(set(difficulties)) != len(difficulties):
        raise ExperimentConfigError("experiment.difficulties must not contain duplicates")

    raw_seeds = OmegaConf.to_container(experiment.get("seeds", []), resolve=True)
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ExperimentConfigError("experiment.seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise ExperimentConfigError("experiment.seeds must contain only integers")
    seeds = tuple(raw_seeds)
    expected_seeds = tuple(range(seeds[0], seeds[0] + len(seeds)))
    if seeds != expected_seeds:
        raise ExperimentConfigError(
            "experiment.seeds must be contiguous because Alem derives episode seed as "
            "EVAL_SEED + episode_idx"
        )
    if int(config.EVAL_SEED) != seeds[0]:
        raise ExperimentConfigError(
            f"EVAL_SEED ({config.EVAL_SEED}) must equal the first experiment seed ({seeds[0]})"
        )

    episodes = _require_positive_int(
        config, "eval.num_episodes.alem", config.eval.num_episodes.alem
    )
    if episodes != len(seeds):
        raise ExperimentConfigError(
            f"eval.num_episodes.alem ({episodes}) must equal len(experiment.seeds) ({len(seeds)})"
        )

    num_agents = _require_positive_int(config, "alem.num_agents", config.alem.num_agents)
    max_steps = _require_positive_int(
        config, "eval.max_steps_per_episode", config.eval.max_steps_per_episode
    )
    num_workers = _require_positive_int(config, "eval.num_workers", config.eval.num_workers)
    if num_workers > episodes:
        raise ExperimentConfigError(
            f"eval.num_workers ({num_workers}) cannot exceed episodes per difficulty ({episodes})"
        )

    clients = config.get("clients")
    if clients is None or len(clients) < num_agents:
        count = 0 if clients is None else len(clients)
        raise ExperimentConfigError(
            f"clients must provide at least one entry per agent; got {count} for {num_agents} agents"
        )

    model_ids: list[str] = []
    for index in range(num_agents):
        client = clients[index]
        client_name = str(client.get("client_name", "")).strip()
        model_id = str(client.get("model_id", "")).strip()
        if client_name != "openai_responses":
            raise ExperimentConfigError(
                f"clients[{index}].client_name must be 'openai_responses'; got {client_name!r}"
            )
        if not model_id:
            raise ExperimentConfigError(f"clients[{index}].model_id must not be empty")
        kwargs = client.get("generate_kwargs", {})
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort not in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ExperimentConfigError(
                f"clients[{index}].generate_kwargs.reasoning_effort is invalid: "
                f"{reasoning_effort!r}"
            )
        cache_key = str(kwargs.get("prompt_cache_key", "")).strip()
        if not cache_key:
            raise ExperimentConfigError(
                f"clients[{index}].generate_kwargs.prompt_cache_key must not be empty"
            )
        _require_positive_int(
            config,
            f"clients[{index}].generate_kwargs.prompt_cache_traffic_shards",
            kwargs.get("prompt_cache_traffic_shards", 1),
        )
        model_ids.append(model_id)

    if len(set(model_ids)) != 1:
        raise ExperimentConfigError(
            f"Baseline profiles must use the same model for every agent; got {model_ids!r}"
        )
    if str(config.get("WANDB_MODE", "")).lower() != "disabled":
        raise ExperimentConfigError("Named profiles must default WANDB_MODE to 'disabled'")

    return ExperimentSpec(
        profile=profile,
        difficulties=difficulties,
        seeds=seeds,
        num_agents=num_agents,
        max_steps_per_episode=max_steps,
        num_workers=num_workers,
        model_ids=tuple(model_ids),
        requires_api_key=bool(experiment.get("requires_api_key", False)),
        generate_debriefs=bool(config.eval.get("generate_debriefs", False)),
    )
