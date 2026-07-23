"""Auditable LLM policy layer for the frozen E2b recruitment screen.

The policy layer is intentionally smaller than the formation runtime.  It
projects one privacy-bounded prompt per eligible agent, obtains at most one
TFP1 control record (plus one semantic repair), and submits valid records to
``TeamDirectory``.  Every agent called in a round sees the same delayed public
snapshot; calls may execute concurrently, but submissions are applied in
agent-ID order.

Private capability truth and costs enter only the owning agent's prompt.
Analysis-only feasibility and oracle values are computed after formation and
are never passed to a client or selector.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Protocol

from baselines.llm.eval_utils.client import LLMResponse
from baselines.llm.eval_utils.prompt_builder import Message
from baselines.llm.eval_utils.recruitment_selection import true_information_oracle
from baselines.llm.eval_utils.team_formation import (
    MAX_CONTROL_BYTES,
    ParseResult,
    RecordKind,
    RecruitmentMethod,
    RecruitmentRecord,
    TaskPhase,
    TeamDirectory,
    parse_tfp1,
)
from baselines.llm.recruitment_arena import (
    AGENT_IDS,
    DEFAULT_ROUNDS,
    Scenario,
)

SCHEMA_VERSION = "alem-dice-e2b-llm-screen-v1"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_MAX_PROMPT_BYTES = 16_000
DEFAULT_SEMANTIC_REPAIRS = 1
DEFAULT_TRANSPORT_RETRIES = 1
DEFAULT_STALL_ROUNDS = 2
SUPPORTED_METHODS = (
    RecruitmentMethod.OPEN_VOLUNTEER,
    RecruitmentMethod.MUTUAL_NOMINATION,
)

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:api[_-]?key|authorization)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def redact_failure_excerpt(value: str, *, limit: int = 160) -> str:
    """Return a bounded diagnostic excerpt without common credential forms."""

    excerpt = value.replace("\r", "\\r").replace("\n", "\\n")
    for pattern in _SECRET_PATTERNS:
        excerpt = pattern.sub("[REDACTED]", excerpt)
    encoded = excerpt.encode("utf-8", errors="replace")[:limit]
    while True:
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError:
            encoded = encoded[:-1]


@dataclass(frozen=True)
class ScreenConfig:
    """One E2b episode's frozen generation and safety bounds."""

    method: RecruitmentMethod
    rounds: int = DEFAULT_ROUNDS
    model_id: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES
    max_semantic_repairs: int = DEFAULT_SEMANTIC_REPAIRS
    max_transport_retries: int = DEFAULT_TRANSPORT_RETRIES
    max_calls_per_episode: int | None = None
    max_round_workers: int = len(AGENT_IDS)

    def __post_init__(self) -> None:
        method = RecruitmentMethod(self.method)
        object.__setattr__(self, "method", method)
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"E2b does not support method {method.value!r}")
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if self.reasoning_effort != DEFAULT_REASONING_EFFORT:
            raise ValueError("canonical E2b reasoning effort is high")
        if self.max_output_tokens < 1 or self.max_prompt_bytes < 1:
            raise ValueError("token and prompt bounds must be positive")
        if self.max_semantic_repairs not in {0, 1}:
            raise ValueError("E2b permits at most one semantic repair")
        if self.max_transport_retries < 0:
            raise ValueError("max_transport_retries must be non-negative")
        if self.max_round_workers < 1:
            raise ValueError("max_round_workers must be positive")
        default_cap = len(AGENT_IDS) * self.rounds * (1 + self.max_semantic_repairs)
        if self.max_calls_per_episode is None:
            object.__setattr__(self, "max_calls_per_episode", default_cap)
        elif self.max_calls_per_episode < 1:
            raise ValueError("max_calls_per_episode must be positive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["method"] = self.method.value
        return value


class RecruitmentClient(Protocol):
    """Minimal injectable client surface shared by fake and Responses clients."""

    def generate(self, messages: Sequence[Message]) -> LLMResponse: ...


@dataclass(frozen=True)
class PublicSelectorView:
    """Truth-free state supplied to an optional roster selector."""

    method: str
    round_index: int
    agent_ids: tuple[int, ...]
    public_roles: dict[str, str]
    task_cards: tuple[dict[str, Any], ...]
    public_ledger: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "round_index": self.round_index,
            "agent_ids": list(self.agent_ids),
            "public_roles": self.public_roles,
            "task_cards": list(self.task_cards),
            "public_ledger": self.public_ledger,
        }


class PublicRosterSelector(Protocol):
    """Extension point for a promoted public-ledger joint selector."""

    name: str

    def select(self, view: PublicSelectorView) -> Mapping[str, Sequence[int]]: ...


@dataclass(frozen=True)
class NoRosterSelector:
    name: str = "none"

    def select(self, view: PublicSelectorView) -> Mapping[str, Sequence[int]]:
        del view
        return {}


@dataclass(frozen=True)
class AgentPromptView:
    """The exact structured payload visible to one policy client."""

    protocol: str
    method: str
    policy: str
    round_index: int
    agent_id: int
    public_roles: dict[str, str]
    task_cards: tuple[dict[str, Any], ...]
    public_ledger: dict[str, Any]
    public_selector: dict[str, Any]
    own_private: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["task_cards"] = list(self.task_cards)
        return value


@dataclass(frozen=True)
class Decision:
    agent_id: int
    round_index: int
    record: RecruitmentRecord | None
    abstain_code: str | None
    calls: tuple[dict[str, Any], ...]
    semantic_repairs: int
    deterministic_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "round_index": self.round_index,
            "record": None if self.record is None else self.record.as_dict(),
            "abstain_code": self.abstain_code,
            "calls": [
                {key: value for key, value in call.items() if key != "_debug"}
                for call in self.calls
            ],
            "semantic_repairs": self.semantic_repairs,
            "deterministic_hash": self.deterministic_hash,
        }


class SharedCallBudget:
    """Thread-safe reservation ledger for episode or campaign call ceilings."""

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("call budget must be positive")
        self.limit = int(limit)
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def reserve(self) -> bool:
        with self._lock:
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Release a reservation that never reached a client call."""

        with self._lock:
            if self._used < 1:
                raise RuntimeError("cannot refund an unused call budget")
            self._used -= 1


class CampaignBudget:
    """Atomic logical/provider/token reservations shared across episodes."""

    def __init__(
        self,
        *,
        logical_limit: int,
        provider_attempt_limit: int,
        token_limit: int,
        logical_used: int = 0,
        provider_attempts_reserved: int = 0,
        tokens_reserved: int = 0,
    ):
        limits = (logical_limit, provider_attempt_limit, token_limit)
        used = (logical_used, provider_attempts_reserved, tokens_reserved)
        if any(value < 1 for value in limits):
            raise ValueError("campaign limits must be positive")
        if any(value < 0 for value in used):
            raise ValueError("campaign usage must be non-negative")
        if any(current > limit for current, limit in zip(used, limits, strict=True)):
            raise ValueError("campaign usage exceeds a configured limit")
        self.logical_limit = int(logical_limit)
        self.provider_attempt_limit = int(provider_attempt_limit)
        self.token_limit = int(token_limit)
        self._logical_used = int(logical_used)
        self._provider_attempts_reserved = int(provider_attempts_reserved)
        self._tokens_reserved = int(tokens_reserved)
        self._lock = threading.Lock()

    def reserve(
        self,
        *,
        prompt_bytes: int,
        max_output_tokens: int,
        max_provider_attempts: int,
    ) -> str | None:
        """Reserve one call's worst exposure, or return the conflicting resource."""

        token_reservation = (prompt_bytes + max_output_tokens) * max_provider_attempts
        with self._lock:
            if self._logical_used + 1 > self.logical_limit:
                return "logical_calls"
            if (
                self._provider_attempts_reserved + max_provider_attempts
                > self.provider_attempt_limit
            ):
                return "provider_attempts"
            if self._tokens_reserved + token_reservation > self.token_limit:
                return "tokens"
            self._logical_used += 1
            self._provider_attempts_reserved += max_provider_attempts
            self._tokens_reserved += token_reservation
        return None

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "logical_limit": self.logical_limit,
                "logical_used": self._logical_used,
                "provider_attempt_limit": self.provider_attempt_limit,
                "provider_attempts_reserved": self._provider_attempts_reserved,
                "token_limit": self.token_limit,
                "tokens_reserved": self._tokens_reserved,
            }


def _public_task_cards(scenario: Scenario) -> tuple[dict[str, Any], ...]:
    return tuple(task.as_dict() for task in sorted(scenario.tasks, key=lambda item: item.task_id))


def _public_ledger(directory: TeamDirectory) -> dict[str, Any]:
    snapshot = directory.snapshot()
    # Pending controls are deliberately omitted: records become public only
    # after the one-round delayed delivery applies them to task state.
    return {
        "round_index": snapshot.round_index,
        "agent_to_task": snapshot.agent_to_task,
        "tasks": snapshot.tasks,
    }


def selector_view(
    scenario: Scenario,
    directory: TeamDirectory,
    *,
    round_index: int,
) -> PublicSelectorView:
    return PublicSelectorView(
        method=directory.method.value,
        round_index=round_index,
        agent_ids=AGENT_IDS,
        public_roles={
            str(agent.agent_id): agent.public_role
            for agent in sorted(scenario.agents, key=lambda item: item.agent_id)
        },
        task_cards=_public_task_cards(scenario),
        public_ledger=_public_ledger(directory),
    )


def _sanitize_selector_output(
    raw: Mapping[str, Sequence[int]],
    view: PublicSelectorView,
) -> dict[str, list[int]]:
    task_sizes = {card["task_id"]: int(card["required_size"]) for card in view.task_cards}
    output: dict[str, list[int]] = {}
    for task_id, members in sorted(raw.items()):
        roster = tuple(int(member) for member in members)
        if (
            task_id not in task_sizes
            or len(roster) != task_sizes[task_id]
            or tuple(sorted(set(roster))) != roster
            or any(member not in view.agent_ids for member in roster)
        ):
            raise ValueError(f"selector returned an invalid public roster for {task_id!r}")
        output[task_id] = list(roster)
    return output


def build_agent_view(
    scenario: Scenario,
    directory: TeamDirectory,
    *,
    agent_id: int,
    round_index: int,
    selector_name: str = "none",
    selector_advice: Mapping[str, Sequence[int]] | None = None,
) -> AgentPromptView:
    """Project public state plus exactly one agent's private profile."""

    profiles = {agent.agent_id: agent for agent in scenario.agents}
    if agent_id not in profiles:
        raise ValueError(f"unknown agent {agent_id}")
    profile = profiles[agent_id]
    return AgentPromptView(
        protocol="TFP1",
        method=directory.method.value,
        policy="public_sweep",
        round_index=round_index,
        agent_id=agent_id,
        public_roles={
            str(agent.agent_id): agent.public_role
            for agent in sorted(scenario.agents, key=lambda item: item.agent_id)
        },
        task_cards=_public_task_cards(scenario),
        public_ledger=_public_ledger(directory),
        public_selector={
            "name": selector_name,
            "advice": {
                task_id: list(members)
                for task_id, members in sorted((selector_advice or {}).items())
            },
        },
        own_private={
            "true_capabilities": list(profile.true_capabilities),
            "task_costs": dict(sorted(profile.task_costs.items())),
        },
    )


_SYSTEM_PROMPT = """You are one of six decentralized recruitment agents.
Use only the supplied view. Other agents' roles are public, but their true
capabilities and costs are private and absent. Records are delivered publicly
one round later. A locked agent cannot join another task.

Return exactly one line: either ABSTAIN or one canonical TFP1 record no longer
than 256 UTF-8 bytes. Do not add prose, Markdown, quotes, or code fences.
Canonical forms you may need:
TFP1|TYPE=APPLY|TASK=id|CAP=a,b,c|COST=n
TFP1|TYPE=ACCEPT|TASK=id|MEMBERS=0,1
TFP1|TYPE=NOMINATE|TASK=id|MEMBERS=0,1
TFP1|TYPE=LOCK|TASK=id|MEMBERS=0,1
Integer capability/cost values are 0..100; member IDs are sorted and unique.
Open Volunteer permits APPLY, ACCEPT, and LOCK. Mutual Nomination permits
NOMINATE and LOCK. For Open Volunteer, agents first APPLY; each selected member
then ACCEPTs the exact same claimed-feasible roster; after all accepts are
delivered, the task sponsor LOCKs it. For Mutual Nomination, every listed
member NOMINATEs the exact same roster; after all nominations are delivered,
the lowest-ID member LOCKs it. You may emit at most one record per round.
Sweep all public tasks over successive rounds, coordinate from the delayed
ledger, and abstain when no valid progress record is ready."""


def build_messages(view: AgentPromptView, *, repair_code: str | None = None) -> list[Message]:
    messages = [
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(role="user", content=f"AGENT_VIEW_JSON={canonical_json(view.as_dict())}"),
    ]
    if repair_code is not None:
        messages.append(
            Message(
                role="user",
                content=(
                    f"Your previous output failed validation ({repair_code}). "
                    "Repair it once. Return only ABSTAIN or one canonical permitted TFP1 record."
                ),
            )
        )
    return messages


def _prompt_bytes(messages: Sequence[Message]) -> int:
    return sum(len(message.content.encode("utf-8")) for message in messages)


def _validate_completion(
    raw: str,
    *,
    view: AgentPromptView,
) -> tuple[RecruitmentRecord | None, ParseResult | None, str]:
    if raw == "ABSTAIN":
        return None, None, "abstain"
    parsed = parse_tfp1(raw, max_bytes=MAX_CONTROL_BYTES)
    if not parsed.valid or parsed.record is None:
        return None, parsed, parsed.code
    record = parsed.record
    allowed = (
        {RecordKind.APPLY, RecordKind.ACCEPT, RecordKind.LOCK}
        if view.method == RecruitmentMethod.OPEN_VOLUNTEER.value
        else {RecordKind.NOMINATE, RecordKind.LOCK}
    )
    if record.kind not in allowed:
        return None, parsed, "semantic.kind_not_allowed"
    task_ids = {card["task_id"] for card in view.task_cards}
    if record.task_id not in task_ids:
        return None, parsed, "semantic.unknown_task"
    if record.kind in {RecordKind.ACCEPT, RecordKind.NOMINATE} and (
        view.agent_id not in record.members
    ):
        return None, parsed, "semantic.sender_missing"
    return record, parsed, "valid"


def _response_audit(
    response: LLMResponse,
    *,
    attempt: int,
    messages: Sequence[Message],
    view: AgentPromptView,
    prompt_bytes: int,
    validation_code: str,
    valid: bool,
    protocol_parse: ParseResult | None,
    normalized_record: RecruitmentRecord | None,
    started: float,
    ended: float,
) -> dict[str, Any]:
    completion = response.completion if isinstance(response.completion, str) else ""
    failure_excerpt = None if valid else redact_failure_excerpt(completion)
    return {
        "attempt": attempt,
        "semantic_repair": attempt > 0,
        "model_id": response.model_id,
        "response_id_hash": (
            None if response.response_id is None else sha256_text(str(response.response_id))
        ),
        "raw_completion_sha256": sha256_text(completion),
        "completion_bytes": len(completion.encode("utf-8", errors="replace")),
        "failure_excerpt": failure_excerpt,
        "validation_code": validation_code,
        "valid": valid,
        "parse": (
            None
            if protocol_parse is None
            else {
                "valid": protocol_parse.valid,
                "code": protocol_parse.code,
                "payload_bytes": protocol_parse.payload_bytes,
            }
        ),
        "prompt_bytes": prompt_bytes,
        "input_tokens": int(response.input_tokens),
        "output_tokens": int(response.output_tokens),
        "reasoning_tokens": int(response.reasoning_tokens),
        "cached_tokens": int(response.cached_tokens),
        "cache_write_tokens": int(response.cache_write_tokens),
        "stop_reason": response.stop_reason,
        "status": response.status,
        "incomplete_reason": response.incomplete_reason,
        "latency_seconds": float(response.latency_seconds or ended - started),
        "transport_attempt_count": int(response.transport_attempt_count),
        "transport_error_count": int(response.transport_error_count),
        "transport_error_types": list(response.transport_error_types),
        "started_monotonic": started,
        "ended_monotonic": ended,
        "_debug": {
            "prompt_projection": view.as_dict(),
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "raw_completion": completion,
            "normalized_parse": {
                "validation_code": validation_code,
                "protocol_parse": (
                    None
                    if protocol_parse is None
                    else {
                        "valid": protocol_parse.valid,
                        "code": protocol_parse.code,
                        "payload_bytes": protocol_parse.payload_bytes,
                    }
                ),
                "record": (
                    None
                    if normalized_record is None
                    else {
                        "canonical": normalized_record.render(),
                        "typed": normalized_record.as_dict(),
                    }
                ),
            },
            "response": {
                "id": response.response_id,
                "model_id": response.model_id,
                "status": response.status,
                "stop_reason": response.stop_reason,
                "incomplete_reason": response.incomplete_reason,
                "usage": {
                    "input_tokens": int(response.input_tokens),
                    "output_tokens": int(response.output_tokens),
                    "reasoning_tokens": int(response.reasoning_tokens),
                    "cached_tokens": int(response.cached_tokens),
                    "cache_write_tokens": int(response.cache_write_tokens),
                },
                "transport_attempt_count": int(response.transport_attempt_count),
                "transport_error_count": int(response.transport_error_count),
                "transport_error_types": list(response.transport_error_types),
            },
            "hashes": {
                "prompt_projection_sha256": sha256_text(canonical_json(view.as_dict())),
                "messages_sha256": sha256_text(
                    canonical_json(
                        [{"role": message.role, "content": message.content} for message in messages]
                    )
                ),
                "raw_completion_sha256": sha256_text(completion),
            },
        },
    }


def request_control_record(
    *,
    client: RecruitmentClient,
    view: AgentPromptView,
    config: ScreenConfig,
    episode_budget: SharedCallBudget,
    campaign_budget: CampaignBudget | None = None,
    transition_validator: (Callable[[RecruitmentRecord], tuple[bool, str]] | None) = None,
    clock: Callable[[], float] = time.monotonic,
) -> Decision:
    """Generate one record with at most one semantic repair and safe fallback."""

    calls: list[dict[str, Any]] = []
    repair_code: str | None = None
    final_record: RecruitmentRecord | None = None
    abstain_code = "semantic_exhausted"
    repairs = 0

    for attempt in range(1 + config.max_semantic_repairs):
        messages = build_messages(view, repair_code=repair_code)
        prompt_bytes = _prompt_bytes(messages)
        if prompt_bytes > config.max_prompt_bytes:
            abstain_code = "budget.prompt_bytes"
            break
        if not episode_budget.reserve():
            abstain_code = "budget.episode_calls"
            break
        if campaign_budget is not None:
            conflict = campaign_budget.reserve(
                prompt_bytes=prompt_bytes,
                max_output_tokens=config.max_output_tokens,
                max_provider_attempts=1 + config.max_transport_retries,
            )
            if conflict is not None:
                episode_budget.refund()
                abstain_code = f"budget.campaign_{conflict}"
                break
        started = clock()
        try:
            response = client.generate(messages)
        except Exception as exc:  # transport accounting may be unavailable on an exception
            ended = clock()
            calls.append(
                {
                    "attempt": attempt,
                    "semantic_repair": attempt > 0,
                    "model_id": config.model_id,
                    "response_id_hash": None,
                    "raw_completion_sha256": None,
                    "completion_bytes": 0,
                    "failure_excerpt": redact_failure_excerpt(f"{type(exc).__name__}: {exc}"),
                    "validation_code": "transport.exception",
                    "valid": False,
                    "parse": None,
                    "prompt_bytes": prompt_bytes,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                    "stop_reason": None,
                    "status": "exception",
                    "incomplete_reason": None,
                    "latency_seconds": ended - started,
                    "transport_attempt_count": int(
                        getattr(client, "last_transport_attempt_count", 1) or 1
                    ),
                    "transport_error_count": int(
                        getattr(client, "last_transport_error_count", 1) or 1
                    ),
                    "transport_error_types": list(
                        getattr(client, "last_transport_error_types", (type(exc).__name__,))
                    ),
                    "started_monotonic": started,
                    "ended_monotonic": ended,
                    "_debug": {
                        "prompt_projection": view.as_dict(),
                        "messages": [
                            {"role": message.role, "content": message.content}
                            for message in messages
                        ],
                        "raw_completion": None,
                        "normalized_parse": {
                            "validation_code": "transport.exception",
                            "protocol_parse": None,
                            "record": None,
                        },
                        "response": {
                            "id": None,
                            "model_id": config.model_id,
                            "status": "exception",
                            "stop_reason": None,
                            "incomplete_reason": None,
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "reasoning_tokens": 0,
                                "cached_tokens": 0,
                                "cache_write_tokens": 0,
                            },
                            "transport_attempt_count": int(
                                getattr(client, "last_transport_attempt_count", 1) or 1
                            ),
                            "transport_error_count": int(
                                getattr(client, "last_transport_error_count", 1) or 1
                            ),
                            "transport_error_types": list(
                                getattr(
                                    client,
                                    "last_transport_error_types",
                                    (type(exc).__name__,),
                                )
                            ),
                        },
                        "hashes": {
                            "prompt_projection_sha256": sha256_text(canonical_json(view.as_dict())),
                            "messages_sha256": sha256_text(
                                canonical_json(
                                    [
                                        {"role": message.role, "content": message.content}
                                        for message in messages
                                    ]
                                )
                            ),
                            "raw_completion_sha256": None,
                        },
                    },
                }
            )
            abstain_code = "transport.exception"
            break
        ended = clock()
        record, parsed, code = _validate_completion(response.completion, view=view)
        if code == "valid" and record is not None and transition_validator is not None:
            accepted, transition_code = transition_validator(record)
            if not accepted:
                code = f"transition_preflight:{transition_code}"
        valid = code in {"valid", "abstain"}
        calls.append(
            _response_audit(
                response,
                attempt=attempt,
                messages=messages,
                view=view,
                prompt_bytes=prompt_bytes,
                validation_code=code,
                valid=valid,
                protocol_parse=parsed,
                normalized_record=record,
                started=started,
                ended=ended,
            )
        )
        if code == "valid":
            final_record = record
            abstain_code = None
            break
        if code == "abstain":
            abstain_code = "model.abstain"
            break
        if attempt < config.max_semantic_repairs:
            repairs += 1
            repair_code = code
            continue
        abstain_code = f"semantic_exhausted:{code}"

    semantic_payload = {
        "agent_id": view.agent_id,
        "round_index": view.round_index,
        "record": None if final_record is None else final_record.render(),
        "abstain_code": abstain_code,
        "completion_hashes": [call["raw_completion_sha256"] for call in calls],
        "validation_codes": [call["validation_code"] for call in calls],
    }
    return Decision(
        agent_id=view.agent_id,
        round_index=view.round_index,
        record=final_record,
        abstain_code=abstain_code,
        calls=tuple(calls),
        semantic_repairs=repairs,
        deterministic_hash=sha256_text(canonical_json(semantic_payload)),
    )


def _eligible_agents(directory: TeamDirectory) -> tuple[int, ...]:
    if not any(
        state.phase in {TaskPhase.ANNOUNCED, TaskPhase.FORMING, TaskPhase.AWARDED}
        for state in directory.tasks.values()
    ):
        return ()
    # Public control remains available to uncommitted agents. Open-volunteer
    # sponsors stay eligible so they can issue LOCK after member acceptance.
    eligible = {agent_id for agent_id in AGENT_IDS if agent_id not in directory.agent_to_task}
    if directory.method is RecruitmentMethod.OPEN_VOLUNTEER:
        eligible.update(
            state.card.sponsor_id
            for state in directory.tasks.values()
            if state.phase in {TaskPhase.ANNOUNCED, TaskPhase.FORMING, TaskPhase.AWARDED}
        )
    return tuple(sorted(eligible))


def _peak_concurrency(calls: Sequence[dict[str, Any]]) -> int:
    boundaries: list[tuple[float, int]] = []
    for call in calls:
        boundaries.append((float(call["started_monotonic"]), 1))
        boundaries.append((float(call["ended_monotonic"]), -1))
    # Starts sort before ends at equal timestamps.
    boundaries.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    peak = 0
    for _, delta in boundaries:
        active += delta
        peak = max(peak, active)
    return peak


def _true_feasible(
    scenario: Scenario,
    task_id: str,
    members: Sequence[int],
) -> bool:
    task = next(task for task in scenario.tasks if task.task_id == task_id)
    profiles = {agent.agent_id: agent for agent in scenario.agents}
    if len(members) != task.required_size or len(set(members)) != len(members):
        return False
    return all(
        sum(profiles[member].true_capabilities[dimension] for member in members) >= demand
        for dimension, demand in enumerate(task.demand)
        if demand > 0
    )


def run_llm_recruitment_episode(
    scenario: Scenario,
    config: ScreenConfig,
    *,
    client_factory: Callable[[int], RecruitmentClient],
    selector: PublicRosterSelector | None = None,
    campaign_budget: CampaignBudget | None = None,
    stall_rounds: int = DEFAULT_STALL_ROUNDS,
) -> dict[str, Any]:
    """Run one concurrent, delayed-ledger E2b formation episode."""

    if config.rounds > min(task.deadline_round for task in scenario.tasks):
        raise ValueError("acting rounds must not exceed the task deadline")
    if stall_rounds < 1:
        raise ValueError("stall_rounds must be positive")
    selector = selector or NoRosterSelector()
    directory = TeamDirectory(agent_ids=AGENT_IDS, method=config.method, seed=scenario.seed)
    for task in scenario.tasks:
        directory.register_task(task)
    clients = {agent_id: client_factory(agent_id) for agent_id in AGENT_IDS}
    episode_budget = SharedCallBudget(int(config.max_calls_per_episode))
    rounds: list[dict[str, Any]] = []
    transition_ledger: list[dict[str, Any]] = []
    lock_rounds: dict[str, int] = {}
    debug_call_records: list[dict[str, Any]] = []
    consecutive_stalls = 0
    executed_rounds = 0
    stop_reason = "acting_horizon"

    for round_index in range(config.rounds):
        transitions = directory.advance(round_index)
        ordinary = directory.deliver_ordinary(round_index)
        transition_ledger.extend(item.as_dict() for item in transitions)
        for transition in transitions:
            if transition.accepted and transition.code == "team.locked":
                lock_rounds[str(transition.task_id)] = round_index

        public_selector_view = selector_view(
            scenario,
            directory,
            round_index=round_index,
        )
        advice = _sanitize_selector_output(
            selector.select(public_selector_view),
            public_selector_view,
        )
        eligible = _eligible_agents(directory)
        views = {
            agent_id: build_agent_view(
                scenario,
                directory,
                agent_id=agent_id,
                round_index=round_index,
                selector_name=selector.name,
                selector_advice=advice,
            )
            for agent_id in eligible
        }
        public_replay = directory.export_replay()

        def transition_validator(
            agent_id: int,
        ) -> Callable[[RecruitmentRecord], tuple[bool, str]]:
            def validate(record: RecruitmentRecord) -> tuple[bool, str]:
                candidate = TeamDirectory.replay(public_replay)
                envelope = candidate.submit_control(
                    sender=agent_id,
                    sent_round=round_index,
                    raw=record.render(),
                )
                transitions = candidate.advance(round_index + 1)
                result = next(item for item in transitions if item.sequence == envelope.sequence)
                return result.accepted, result.code

            return validate

        decisions: dict[int, Decision] = {}
        started = time.monotonic()
        with ThreadPoolExecutor(
            max_workers=min(config.max_round_workers, max(1, len(eligible)))
        ) as pool:
            futures = {
                pool.submit(
                    request_control_record,
                    client=clients[agent_id],
                    view=views[agent_id],
                    config=config,
                    episode_budget=episode_budget,
                    campaign_budget=campaign_budget,
                    transition_validator=transition_validator(agent_id),
                ): agent_id
                for agent_id in eligible
            }
            for future in as_completed(futures):
                agent_id = futures[future]
                decisions[agent_id] = future.result()
        decision_wall = time.monotonic() - started

        submitted: list[dict[str, Any]] = []
        for agent_id in sorted(decisions):
            record = decisions[agent_id].record
            if record is None:
                continue
            envelope = directory.submit_control(
                sender=agent_id,
                sent_round=round_index,
                raw=record.render(),
            )
            submitted.append(envelope.as_dict())
        round_calls = [call for agent_id in sorted(decisions) for call in decisions[agent_id].calls]
        for agent_id in sorted(decisions):
            for call in decisions[agent_id].calls:
                debug_call_records.append(
                    {
                        "schema_version": "alem-dice-e2b-call-debug-v1",
                        "scenario_id": scenario.scenario_id,
                        "family": scenario.family.value,
                        "seed": scenario.seed,
                        "method": config.method.value,
                        "selector": selector.name,
                        "round_index": round_index,
                        "agent_id": agent_id,
                        "attempt": int(call["attempt"]),
                        **call["_debug"],
                    }
                )
        rounds.append(
            {
                "round_index": round_index,
                "delivery_transitions": [item.as_dict() for item in transitions],
                "ordinary_deliveries": [item.as_dict() for item in ordinary],
                "eligible_agents": list(eligible),
                "selector": {"name": selector.name, "advice": advice},
                "decisions": [decisions[agent_id].as_dict() for agent_id in sorted(decisions)],
                "submissions": submitted,
                "logical_model_calls": len(round_calls),
                "provider_requests": sum(
                    int(call["transport_attempt_count"]) for call in round_calls
                ),
                "peak_concurrent_calls": _peak_concurrency(round_calls),
                "decision_wall_seconds": decision_wall,
                "state_hash_after_submission": directory.state_hash(),
            }
        )
        executed_rounds = round_index + 1
        accepted_public_transition = any(
            transition.accepted and transition.sender >= 0 for transition in transitions
        )
        if accepted_public_transition or submitted:
            consecutive_stalls = 0
        else:
            consecutive_stalls += 1
        forming_remains = any(
            state.phase in {TaskPhase.ANNOUNCED, TaskPhase.FORMING, TaskPhase.AWARDED}
            for state in directory.tasks.values()
        )
        if not forming_remains:
            stop_reason = "all_tasks_closed_or_locked"
            break
        if consecutive_stalls >= stall_rounds:
            stop_reason = f"no_public_progress_{stall_rounds}_rounds"
            break

    drain_round = executed_rounds
    transitions = directory.advance(drain_round)
    ordinary = directory.deliver_ordinary(drain_round)
    transition_ledger.extend(item.as_dict() for item in transitions)
    for transition in transitions:
        if transition.accepted and transition.code == "team.locked":
            lock_rounds[str(transition.task_id)] = drain_round

    replay = directory.export_replay()
    replayed = TeamDirectory.replay(replay)
    replay_match = (
        replayed.state_hash() == directory.state_hash()
        and replayed.audit_chain_hash == directory.audit_chain_hash
    )
    locked = {
        task_id: list(state.lease.members)
        for task_id, state in sorted(directory.tasks.items())
        if state.phase is TaskPhase.LOCKED and state.lease is not None
    }
    true_feasible = {
        task_id: _true_feasible(scenario, task_id, members) for task_id, members in locked.items()
    }
    oracle = true_information_oracle(scenario.tasks, scenario.agents)
    oracle_reward = int(oracle.total_completed_reward)
    oracle_completed_tasks = len(oracle.completed_tasks)
    true_feasible_locked_tasks = sum(true_feasible.values())
    achieved_reward = sum(
        task.reward for task in scenario.tasks if true_feasible.get(task.task_id, False)
    )
    all_calls = [
        call
        for round_record in rounds
        for decision in round_record["decisions"]
        for call in decision["calls"]
    ]
    deterministic_projection = {
        "scenario_id": scenario.scenario_id,
        "method": config.method.value,
        "selector": selector.name,
        "decisions": [
            decision["deterministic_hash"]
            for round_record in rounds
            for decision in round_record["decisions"]
        ],
        "early_stop": {
            "executed_acting_rounds": executed_rounds,
            "stall_rounds": stall_rounds,
            "reason": stop_reason,
            "drain_round": drain_round,
        },
        "terminal_state_hash": directory.state_hash(),
        "audit_chain_hash": directory.audit_chain_hash,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "family": scenario.family.value,
            "seed": scenario.seed,
        },
        "config": config.as_dict(),
        "selector": selector.name,
        "rounds": rounds,
        "drain": {
            "round_index": drain_round,
            "transitions": [item.as_dict() for item in transitions],
            "ordinary_deliveries": [item.as_dict() for item in ordinary],
        },
        "early_stop": {
            "requested_acting_rounds": config.rounds,
            "executed_acting_rounds": executed_rounds,
            "stall_rounds": stall_rounds,
            "reason": stop_reason,
        },
        "transition_ledger": transition_ledger,
        "call_ledger": all_calls,
        "retry_ledger": {
            "semantic_repairs": sum(
                decision["semantic_repairs"]
                for round_record in rounds
                for decision in round_record["decisions"]
            ),
            "transport_attempts": sum(int(call["transport_attempt_count"]) for call in all_calls),
            "transport_errors": sum(int(call["transport_error_count"]) for call in all_calls),
        },
        "token_ledger": {
            key: sum(int(call[key]) for call in all_calls)
            for key in (
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cached_tokens",
                "cache_write_tokens",
            )
        },
        "latency_ledger": {
            "sum_call_latency_seconds": sum(float(call["latency_seconds"]) for call in all_calls),
            "round_decision_wall_seconds": [
                float(round_record["decision_wall_seconds"]) for round_record in rounds
            ],
            "round_peak_concurrent_calls": [
                int(round_record["peak_concurrent_calls"]) for round_record in rounds
            ],
        },
        "call_caps": {
            "episode_limit": int(config.max_calls_per_episode),
            "episode_used": episode_budget.used,
            "campaign": (None if campaign_budget is None else campaign_budget.snapshot()),
        },
        "analysis_only": {
            "locked_rosters": locked,
            "lock_rounds": lock_rounds,
            "true_feasible_locks": true_feasible,
            "true_feasible_locked_tasks": true_feasible_locked_tasks,
            "oracle_completed_tasks": oracle_completed_tasks,
            "oracle_allocation_coverage": (
                true_feasible_locked_tasks / oracle_completed_tasks
                if oracle_completed_tasks
                else 1.0
            ),
            "oracle_reward": oracle_reward,
            "achieved_reward": achieved_reward,
            "normalized_reward": (
                float(Fraction(achieved_reward, oracle_reward)) if oracle_reward else 1.0
            ),
        },
        "directory_replay": replay,
        "replay_hash_match": replay_match,
        "terminal_state_hash": directory.state_hash(),
        "terminal_audit_chain_hash": directory.audit_chain_hash,
        "deterministic_episode_hash": sha256_text(canonical_json(deterministic_projection)),
        "_debug_call_records": debug_call_records,
    }


def estimate_campaign(
    *,
    seed_count: int,
    family_count: int,
    method_count: int,
    rounds: int = DEFAULT_ROUNDS,
    agent_count: int = len(AGENT_IDS),
    semantic_repairs: int = DEFAULT_SEMANTIC_REPAIRS,
    transport_retries: int = DEFAULT_TRANSPORT_RETRIES,
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, int]:
    """Conservative protocol ceiling; one ASCII byte is charged as one token."""

    episodes = seed_count * family_count * method_count
    initial_calls = episodes * rounds * agent_count
    logical_calls = initial_calls * (1 + semantic_repairs)
    provider_attempts = logical_calls * (1 + transport_retries)
    return {
        "episodes": episodes,
        "initial_logical_calls": initial_calls,
        "semantic_repair_calls": logical_calls - initial_calls,
        "max_logical_calls": logical_calls,
        "max_provider_attempts": provider_attempts,
        "max_recorded_input_tokens": logical_calls * max_prompt_bytes,
        "max_recorded_output_tokens": logical_calls * max_output_tokens,
        "max_recorded_total_tokens": logical_calls * (max_prompt_bytes + max_output_tokens),
        "max_provider_input_token_exposure": provider_attempts * max_prompt_bytes,
        "max_provider_output_token_exposure": provider_attempts * max_output_tokens,
        "max_provider_total_token_exposure": provider_attempts
        * (max_prompt_bytes + max_output_tokens),
    }


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_MAX_PROMPT_BYTES",
    "DEFAULT_MODEL",
    "DEFAULT_REASONING_EFFORT",
    "DEFAULT_SEMANTIC_REPAIRS",
    "DEFAULT_STALL_ROUNDS",
    "DEFAULT_TRANSPORT_RETRIES",
    "CampaignBudget",
    "NoRosterSelector",
    "PublicRosterSelector",
    "RecruitmentClient",
    "SCHEMA_VERSION",
    "SUPPORTED_METHODS",
    "ScreenConfig",
    "SharedCallBudget",
    "build_agent_view",
    "build_messages",
    "canonical_json",
    "estimate_campaign",
    "redact_failure_excerpt",
    "request_control_record",
    "run_llm_recruitment_episode",
    "selector_view",
    "sha256_text",
]
