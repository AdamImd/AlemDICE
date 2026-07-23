"""Deterministic TFP1 task-team formation and communication routing.

The runtime is deliberately domain independent. It validates public records,
maintains task-bound membership, and computes message recipients, but it never
selects an environment action or reads private capability truth. Sender identity
and round are trusted envelope metadata rather than model-controlled fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import combinations
from typing import Any

PROTOCOL_VERSION = "TFP1"
MAX_CONTROL_BYTES = 256
SYSTEM_AGENT_ID = -1
CAPABILITY_DIMENSIONS = 3

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,32}$")
_REASON_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,32}$")


class RecordKind(StrEnum):
    ANNOUNCE = "ANNOUNCE"
    CFP = "CFP"
    APPLY = "APPLY"
    NOMINATE = "NOMINATE"
    BID = "BID"
    AWARD = "AWARD"
    ACCEPT = "ACCEPT"
    DECLINE = "DECLINE"
    LOCK = "LOCK"
    CANCEL = "CANCEL"
    COMPLETE = "COMPLETE"
    EXPIRE = "EXPIRE"


class RecruitmentMethod(StrEnum):
    OPEN_VOLUNTEER = "open_volunteer"
    MUTUAL_NOMINATION = "mutual_nomination"
    CONTRACT_NET = "contract_net"


class TaskPhase(StrEnum):
    ANNOUNCED = "announced"
    FORMING = "forming"
    AWARDED = "awarded"
    LOCKED = "locked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


TERMINAL_PHASES = frozenset({TaskPhase.COMPLETED, TaskPhase.CANCELLED, TaskPhase.EXPIRED})


@dataclass(frozen=True)
class TaskCard:
    task_id: str
    demand: tuple[int, int, int]
    required_size: int
    reward: int
    deadline_round: int
    sponsor_id: int

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("task_id must contain 1-32 identifier characters")
        _validate_vector(self.demand, "demand")
        if not any(self.demand):
            raise ValueError("demand must contain at least one positive component")
        if self.required_size not in {2, 3}:
            raise ValueError("required_size must be 2 or 3")
        if isinstance(self.reward, bool) or not isinstance(self.reward, int) or self.reward <= 0:
            raise ValueError("reward must be a positive integer")
        if (
            isinstance(self.deadline_round, bool)
            or not isinstance(self.deadline_round, int)
            or self.deadline_round <= 0
        ):
            raise ValueError("deadline_round must be a positive integer")
        _validate_agent_id(self.sponsor_id, "sponsor_id")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["demand"] = list(self.demand)
        return value


@dataclass(frozen=True)
class CandidateBid:
    task_id: str
    agent_id: int
    claimed_capabilities: tuple[int, int, int]
    cost: int
    arrival_round: int

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        _validate_agent_id(self.agent_id, "agent_id")
        _validate_vector(self.claimed_capabilities, "claimed_capabilities")
        _validate_score(self.cost, "cost")
        if (
            isinstance(self.arrival_round, bool)
            or not isinstance(self.arrival_round, int)
            or self.arrival_round < 0
        ):
            raise ValueError("arrival_round must be a non-negative integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "claimed_capabilities": list(self.claimed_capabilities),
            "cost": self.cost,
            "arrival_round": self.arrival_round,
        }


@dataclass(frozen=True)
class RecruitmentRecord:
    kind: RecordKind
    task_id: str
    members: tuple[int, ...] = ()
    capabilities: tuple[int, int, int] | None = None
    cost: int | None = None
    demand: tuple[int, int, int] | None = None
    required_size: int | None = None
    reward: int | None = None
    deadline_round: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("invalid task_id")
        expected = _RECORD_FIELDS[self.kind]
        present = {
            "MEMBERS": bool(self.members),
            "CAP": self.capabilities is not None,
            "COST": self.cost is not None,
            "DEMAND": self.demand is not None,
            "SIZE": self.required_size is not None,
            "REWARD": self.reward is not None,
            "DEADLINE": self.deadline_round is not None,
            "REASON": self.reason is not None,
        }
        expected_optional = set(expected) - {"TYPE", "TASK"}
        actual_optional = {key for key, value in present.items() if value}
        if actual_optional != expected_optional:
            raise ValueError(
                f"{self.kind.value} requires exactly {sorted(expected_optional)}; "
                f"got {sorted(actual_optional)}"
            )
        if self.members:
            _validate_members(self.members)
        if self.capabilities is not None:
            _validate_vector(self.capabilities, "capabilities")
        if self.cost is not None:
            _validate_score(self.cost, "cost")
        if self.demand is not None:
            _validate_vector(self.demand, "demand")
            if not any(self.demand):
                raise ValueError("demand must contain at least one positive component")
        if self.required_size is not None and self.required_size not in {2, 3}:
            raise ValueError("required_size must be 2 or 3")
        if self.reward is not None and (
            isinstance(self.reward, bool) or not isinstance(self.reward, int) or self.reward <= 0
        ):
            raise ValueError("reward must be a positive integer")
        if self.deadline_round is not None and (
            isinstance(self.deadline_round, bool)
            or not isinstance(self.deadline_round, int)
            or self.deadline_round <= 0
        ):
            raise ValueError("deadline_round must be a positive integer")
        if self.reason is not None and not _REASON_RE.fullmatch(self.reason):
            raise ValueError("reason must contain 1-32 identifier characters")

    def render(self) -> str:
        values = {
            "TYPE": self.kind.value,
            "TASK": self.task_id,
            "MEMBERS": ",".join(str(value) for value in self.members),
            "CAP": _render_vector(self.capabilities),
            "COST": None if self.cost is None else str(self.cost),
            "DEMAND": _render_vector(self.demand),
            "SIZE": None if self.required_size is None else str(self.required_size),
            "REWARD": None if self.reward is None else str(self.reward),
            "DEADLINE": None if self.deadline_round is None else str(self.deadline_round),
            "REASON": self.reason,
        }
        return "|".join(
            [PROTOCOL_VERSION] + [f"{key}={values[key]}" for key in _RECORD_FIELDS[self.kind]]
        )

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": self.kind.value,
            "task_id": self.task_id,
        }
        if self.members:
            value["members"] = list(self.members)
        if self.capabilities is not None:
            value["capabilities"] = list(self.capabilities)
        for name in ("cost", "required_size", "reward", "deadline_round", "reason"):
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        if self.demand is not None:
            value["demand"] = list(self.demand)
        return value


_RECORD_FIELDS: dict[RecordKind, tuple[str, ...]] = {
    RecordKind.ANNOUNCE: ("TYPE", "TASK", "DEMAND", "SIZE", "REWARD", "DEADLINE"),
    RecordKind.CFP: ("TYPE", "TASK"),
    RecordKind.APPLY: ("TYPE", "TASK", "CAP", "COST"),
    RecordKind.NOMINATE: ("TYPE", "TASK", "MEMBERS"),
    RecordKind.BID: ("TYPE", "TASK", "CAP", "COST"),
    RecordKind.AWARD: ("TYPE", "TASK", "MEMBERS"),
    RecordKind.ACCEPT: ("TYPE", "TASK", "MEMBERS"),
    RecordKind.DECLINE: ("TYPE", "TASK", "MEMBERS"),
    RecordKind.LOCK: ("TYPE", "TASK", "MEMBERS"),
    RecordKind.CANCEL: ("TYPE", "TASK", "REASON"),
    RecordKind.COMPLETE: ("TYPE", "TASK"),
    RecordKind.EXPIRE: ("TYPE", "TASK"),
}


@dataclass(frozen=True)
class ParseResult:
    valid: bool
    code: str
    record: RecruitmentRecord | None
    payload_bytes: int


def parse_tfp1(raw: str | None, *, max_bytes: int = MAX_CONTROL_BYTES) -> ParseResult:
    """Parse one strict, canonical TFP1 record."""

    if not isinstance(raw, str) or not raw:
        return ParseResult(False, "parse.empty", None, 0)
    try:
        payload_bytes = len(raw.encode("utf-8"))
    except UnicodeEncodeError:
        return ParseResult(False, "parse.invalid_utf8", None, 0)
    if payload_bytes > max_bytes:
        return ParseResult(False, "parse.too_many_bytes", None, payload_bytes)
    if "\n" in raw or "\r" in raw:
        return ParseResult(False, "parse.multiline", None, payload_bytes)
    if raw != raw.strip():
        return ParseResult(False, "parse.outer_whitespace", None, payload_bytes)
    parts = raw.split("|")
    if not parts or parts[0] != PROTOCOL_VERSION:
        return ParseResult(False, "parse.version", None, payload_bytes)
    fields: dict[str, str] = {}
    field_order: list[str] = []
    for item in parts[1:]:
        if "=" not in item:
            return ParseResult(False, "parse.field_syntax", None, payload_bytes)
        key, value = item.split("=", 1)
        if not key or not value or key != key.upper():
            return ParseResult(False, "parse.field_syntax", None, payload_bytes)
        if key in fields:
            return ParseResult(False, "schema.duplicate_field", None, payload_bytes)
        fields[key] = value
        field_order.append(key)
    kind_text = fields.get("TYPE")
    try:
        kind = RecordKind(kind_text)
    except (TypeError, ValueError):
        return ParseResult(False, "schema.invalid_type", None, payload_bytes)
    expected = _RECORD_FIELDS[kind]
    if tuple(field_order) != expected:
        expected_set = set(expected)
        actual_set = set(field_order)
        if expected_set - actual_set:
            code = "schema.missing_field"
        elif actual_set - expected_set:
            code = "schema.unexpected_field"
        else:
            code = "schema.noncanonical_order"
        return ParseResult(False, code, None, payload_bytes)
    task_id = fields["TASK"]
    if not _TASK_ID_RE.fullmatch(task_id):
        return ParseResult(False, "schema.invalid_task_id", None, payload_bytes)
    try:
        kwargs: dict[str, Any] = {}
        if "MEMBERS" in fields:
            kwargs["members"] = _parse_members(fields["MEMBERS"])
        if "CAP" in fields:
            kwargs["capabilities"] = _parse_vector(fields["CAP"])
        if "COST" in fields:
            kwargs["cost"] = _parse_score(fields["COST"])
        if "DEMAND" in fields:
            demand = _parse_vector(fields["DEMAND"])
            if not any(demand):
                raise ValueError("zero demand")
            kwargs["demand"] = demand
        if "SIZE" in fields:
            size = _parse_int(fields["SIZE"])
            if size not in {2, 3}:
                raise ValueError("invalid size")
            kwargs["required_size"] = size
        if "REWARD" in fields:
            reward = _parse_int(fields["REWARD"])
            if reward <= 0:
                raise ValueError("invalid reward")
            kwargs["reward"] = reward
        if "DEADLINE" in fields:
            deadline = _parse_int(fields["DEADLINE"])
            if deadline <= 0:
                raise ValueError("invalid deadline")
            kwargs["deadline_round"] = deadline
        if "REASON" in fields:
            if not _REASON_RE.fullmatch(fields["REASON"]):
                raise ValueError("invalid reason")
            kwargs["reason"] = fields["REASON"]
        record = RecruitmentRecord(kind=kind, task_id=task_id, **kwargs)
    except (TypeError, ValueError):
        return ParseResult(False, "schema.invalid_value", None, payload_bytes)
    if record.render() != raw:
        return ParseResult(False, "schema.noncanonical_value", None, payload_bytes)
    return ParseResult(True, "valid", record, payload_bytes)


@dataclass(frozen=True)
class TeamLease:
    task_id: str
    members: tuple[int, ...]
    start_round: int
    deadline_round: int
    version: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskState:
    card: TaskCard
    phase: TaskPhase = TaskPhase.ANNOUNCED
    cfp_open: bool = False
    applications: dict[int, CandidateBid] = field(default_factory=dict)
    bids: dict[int, CandidateBid] = field(default_factory=dict)
    nominations: dict[int, tuple[int, ...]] = field(default_factory=dict)
    award: tuple[int, ...] | None = None
    accepts: set[int] = field(default_factory=set)
    declines: set[int] = field(default_factory=set)
    completion_reports: set[int] = field(default_factory=set)
    lease: TeamLease | None = None
    close_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "card": self.card.as_dict(),
            "phase": self.phase.value,
            "cfp_open": self.cfp_open,
            "applications": {
                str(key): value.as_dict() for key, value in sorted(self.applications.items())
            },
            "bids": {str(key): value.as_dict() for key, value in sorted(self.bids.items())},
            "nominations": {
                str(key): list(value) for key, value in sorted(self.nominations.items())
            },
            "award": None if self.award is None else list(self.award),
            "accepts": sorted(self.accepts),
            "declines": sorted(self.declines),
            "completion_reports": sorted(self.completion_reports),
            "lease": None if self.lease is None else self.lease.as_dict(),
            "close_reason": self.close_reason,
        }


@dataclass(frozen=True)
class ControlEnvelope:
    sequence: int
    sender: int
    sent_round: int
    deliver_round: int
    raw: str
    parse: ParseResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "sender": self.sender,
            "sent_round": self.sent_round,
            "deliver_round": self.deliver_round,
            "raw": self.raw,
            "parse": {
                "valid": self.parse.valid,
                "code": self.parse.code,
                "payload_bytes": self.parse.payload_bytes,
                "record": (None if self.parse.record is None else self.parse.record.as_dict()),
            },
        }


@dataclass(frozen=True)
class OrdinaryEnvelope:
    sequence: int
    sender: int
    recipients: tuple[int, ...]
    task_id: str
    lease_version: int
    content: str
    sent_round: int
    deliver_round: int

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["recipients"] = list(self.recipients)
        return value


@dataclass(frozen=True)
class TransitionResult:
    sequence: int
    accepted: bool
    code: str
    sender: int
    round_index: int
    task_id: str | None
    delivered_recipients: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["delivered_recipients"] = list(self.delivered_recipients)
        return value


@dataclass(frozen=True)
class RouteDecision:
    accepted: bool
    code: str
    sender: int
    recipients: tuple[int, ...]
    content: str
    sent_round: int
    deliver_round: int
    task_id: str | None = None
    lease_version: int | None = None
    sequence: int | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["recipients"] = list(self.recipients)
        return value


@dataclass(frozen=True)
class DirectorySnapshot:
    protocol_version: str
    agent_ids: tuple[int, ...]
    seed: int
    max_control_bytes: int
    round_index: int
    method: str
    agent_to_task: dict[str, str]
    tasks: dict[str, dict[str, Any]]
    pending_control: tuple[dict[str, Any], ...]
    pending_ordinary: tuple[dict[str, Any], ...]
    next_sequence: int
    lease_version: int
    current_round_emissions: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "agent_ids": list(self.agent_ids),
            "seed": self.seed,
            "max_control_bytes": self.max_control_bytes,
            "round_index": self.round_index,
            "method": self.method,
            "agent_to_task": dict(self.agent_to_task),
            "tasks": self.tasks,
            "pending_control": list(self.pending_control),
            "pending_ordinary": list(self.pending_ordinary),
            "next_sequence": self.next_sequence,
            "lease_version": self.lease_version,
            "current_round_emissions": [list(value) for value in self.current_round_emissions],
        }


class TeamDirectory:
    """Deterministic public formation directory and team-private router."""

    def __init__(
        self,
        *,
        agent_ids: tuple[int, ...],
        method: RecruitmentMethod | str,
        seed: int,
        max_control_bytes: int = MAX_CONTROL_BYTES,
    ):
        if not agent_ids or tuple(sorted(set(agent_ids))) != agent_ids:
            raise ValueError("agent_ids must be non-empty, sorted, and unique")
        for agent_id in agent_ids:
            _validate_agent_id(agent_id, "agent_id")
        if max_control_bytes < 1:
            raise ValueError("max_control_bytes must be positive")
        self.agent_ids = agent_ids
        self.method = RecruitmentMethod(method)
        self.seed = int(seed)
        self.max_control_bytes = int(max_control_bytes)
        self.current_round = -1
        self.tasks: dict[str, TaskState] = {}
        self.agent_to_task: dict[int, str] = {}
        self._pending_control: list[ControlEnvelope] = []
        self._pending_ordinary: list[OrdinaryEnvelope] = []
        self._emissions: set[tuple[int, int]] = set()
        self._sequence = 0
        self._lease_version = 0
        self._audit: list[dict[str, Any]] = []
        self._operations: list[dict[str, Any]] = []
        self._chain_hash = hashlib.sha256(
            _canonical_json(
                {
                    "protocol": PROTOCOL_VERSION,
                    "agent_ids": list(agent_ids),
                    "method": self.method.value,
                    "seed": self.seed,
                    "max_control_bytes": self.max_control_bytes,
                }
            ).encode("ascii")
        ).hexdigest()

    def register_task(self, card: TaskCard) -> TransitionResult:
        """Register a public card before round zero."""

        if self.current_round != -1:
            raise ValueError("tasks must be registered before round zero")
        if card.sponsor_id not in self.agent_ids:
            raise ValueError("task sponsor is not a configured agent")
        if card.required_size > len(self.agent_ids):
            raise ValueError("task required_size exceeds agent population")
        if card.task_id in self.tasks:
            raise ValueError(f"duplicate task_id {card.task_id!r}")
        self.tasks[card.task_id] = TaskState(card=card)
        sequence = self._next_sequence()
        result = TransitionResult(
            sequence=sequence,
            accepted=True,
            code="task.registered",
            sender=SYSTEM_AGENT_ID,
            round_index=-1,
            task_id=card.task_id,
            delivered_recipients=self.agent_ids,
        )
        self._operations.append({"op": "register_task", "card": card.as_dict()})
        self._record_audit("transition", result.as_dict())
        return result

    def submit_control(self, *, sender: int, sent_round: int, raw: str) -> ControlEnvelope:
        self._validate_submission(sender, sent_round)
        self._claim_emission(sender, sent_round)
        envelope = ControlEnvelope(
            sequence=self._next_sequence(),
            sender=sender,
            sent_round=sent_round,
            deliver_round=sent_round + 1,
            raw=raw,
            parse=parse_tfp1(raw, max_bytes=self.max_control_bytes),
        )
        self._pending_control.append(envelope)
        self._operations.append(
            {
                "op": "submit_control",
                "sender": sender,
                "sent_round": sent_round,
                "raw": raw,
            }
        )
        self._record_audit("control_submission", envelope.as_dict())
        return envelope

    def advance(self, round_index: int) -> tuple[TransitionResult, ...]:
        if round_index != self.current_round + 1:
            raise ValueError(
                f"advance must be consecutive: current={self.current_round}, got={round_index}"
            )
        if any(envelope.deliver_round <= self.current_round for envelope in self._pending_ordinary):
            raise ValueError("deliver pending ordinary messages before advancing")
        self.current_round = round_index
        results: list[TransitionResult] = []
        for task_id in sorted(self.tasks):
            state = self.tasks[task_id]
            if state.phase not in TERMINAL_PHASES and state.card.deadline_round <= round_index:
                results.append(self._expire(state, round_index))
        due = sorted(
            (
                envelope
                for envelope in self._pending_control
                if envelope.deliver_round == round_index
            ),
            key=lambda item: (item.deliver_round, item.sent_round, item.sender, item.sequence),
        )
        self._pending_control = [
            envelope for envelope in self._pending_control if envelope.deliver_round != round_index
        ]
        for envelope in due:
            results.append(self._apply_control(envelope, round_index))
        self._operations.append({"op": "advance", "round_index": round_index})
        return tuple(results)

    def submit_ordinary(self, *, sender: int, sent_round: int, content: str) -> RouteDecision:
        self._validate_submission(sender, sent_round)
        self._claim_emission(sender, sent_round)
        task_id = self.agent_to_task.get(sender)
        state = self.tasks.get(task_id) if task_id is not None else None
        if (
            not isinstance(content, str)
            or not content
            or state is None
            or state.phase != TaskPhase.LOCKED
            or state.lease is None
        ):
            result = RouteDecision(
                accepted=False,
                code=(
                    "route.empty_content"
                    if not isinstance(content, str) or not content
                    else "route.sender_unteamed"
                ),
                sender=sender,
                recipients=(),
                content=content if isinstance(content, str) else "",
                sent_round=sent_round,
                deliver_round=sent_round + 1,
            )
            self._operations.append(
                {
                    "op": "submit_ordinary",
                    "sender": sender,
                    "sent_round": sent_round,
                    "content": content if isinstance(content, str) else "",
                }
            )
            self._record_audit("ordinary_submission", result.as_dict())
            return result
        lease = state.lease
        recipients = tuple(member for member in lease.members if member != sender)
        envelope = OrdinaryEnvelope(
            sequence=self._next_sequence(),
            sender=sender,
            recipients=recipients,
            task_id=lease.task_id,
            lease_version=lease.version,
            content=content,
            sent_round=sent_round,
            deliver_round=sent_round + 1,
        )
        self._pending_ordinary.append(envelope)
        result = RouteDecision(
            accepted=True,
            code="route.queued",
            sender=sender,
            recipients=recipients,
            content=content,
            sent_round=sent_round,
            deliver_round=sent_round + 1,
            task_id=lease.task_id,
            lease_version=lease.version,
            sequence=envelope.sequence,
        )
        self._operations.append(
            {
                "op": "submit_ordinary",
                "sender": sender,
                "sent_round": sent_round,
                "content": content,
            }
        )
        self._record_audit("ordinary_submission", result.as_dict())
        return result

    def deliver_ordinary(self, round_index: int) -> tuple[RouteDecision, ...]:
        if round_index != self.current_round:
            raise ValueError("deliver_ordinary must follow advance for the same round")
        due = sorted(
            (
                envelope
                for envelope in self._pending_ordinary
                if envelope.deliver_round == round_index
            ),
            key=lambda item: (item.deliver_round, item.sender, item.sequence),
        )
        self._pending_ordinary = [
            envelope for envelope in self._pending_ordinary if envelope.deliver_round != round_index
        ]
        results: list[RouteDecision] = []
        for envelope in due:
            state = self.tasks.get(envelope.task_id)
            lease = None if state is None else state.lease
            active = (
                state is not None
                and state.phase == TaskPhase.LOCKED
                and lease is not None
                and lease.version == envelope.lease_version
                and self.agent_to_task.get(envelope.sender) == envelope.task_id
                and all(
                    self.agent_to_task.get(recipient) == envelope.task_id
                    for recipient in envelope.recipients
                )
            )
            result = RouteDecision(
                accepted=active,
                code="route.delivered" if active else "route.lease_inactive_at_delivery",
                sender=envelope.sender,
                recipients=envelope.recipients if active else (),
                content=envelope.content,
                sent_round=envelope.sent_round,
                deliver_round=envelope.deliver_round,
                task_id=envelope.task_id,
                lease_version=envelope.lease_version,
                sequence=envelope.sequence,
            )
            results.append(result)
            self._record_audit("ordinary_delivery", result.as_dict())
        self._operations.append({"op": "deliver_ordinary", "round_index": round_index})
        return tuple(results)

    def confirm_complete(
        self, *, task_id: str, round_index: int, evidence_digest: str
    ) -> TransitionResult:
        if round_index != self.current_round:
            raise ValueError("completion must be confirmed in the current round")
        state = self.tasks.get(task_id)
        sequence = self._next_sequence()
        if state is None:
            result = TransitionResult(
                sequence,
                False,
                "state.unknown_task",
                SYSTEM_AGENT_ID,
                round_index,
                task_id,
            )
        elif state.phase != TaskPhase.LOCKED:
            result = TransitionResult(
                sequence,
                False,
                "state.task_not_locked",
                SYSTEM_AGENT_ID,
                round_index,
                task_id,
            )
        elif not isinstance(evidence_digest, str) or not re.fullmatch(
            r"[A-Fa-f0-9]{8,64}", evidence_digest
        ):
            result = TransitionResult(
                sequence,
                False,
                "completion.invalid_evidence_digest",
                SYSTEM_AGENT_ID,
                round_index,
                task_id,
            )
        else:
            self._release(state)
            state.phase = TaskPhase.COMPLETED
            state.close_reason = f"evidence:{evidence_digest.lower()}"
            result = TransitionResult(
                sequence,
                True,
                "task.completed",
                SYSTEM_AGENT_ID,
                round_index,
                task_id,
                self.agent_ids,
            )
        self._operations.append(
            {
                "op": "confirm_complete",
                "task_id": task_id,
                "round_index": round_index,
                "evidence_digest": evidence_digest,
            }
        )
        self._record_audit("transition", result.as_dict())
        return result

    def active_team(self, agent_id: int) -> TeamLease | None:
        task_id = self.agent_to_task.get(agent_id)
        state = self.tasks.get(task_id) if task_id is not None else None
        return None if state is None else state.lease

    def snapshot(self) -> DirectorySnapshot:
        return DirectorySnapshot(
            protocol_version=PROTOCOL_VERSION,
            agent_ids=self.agent_ids,
            seed=self.seed,
            max_control_bytes=self.max_control_bytes,
            round_index=self.current_round,
            method=self.method.value,
            agent_to_task={str(key): value for key, value in sorted(self.agent_to_task.items())},
            tasks={key: value.as_dict() for key, value in sorted(self.tasks.items())},
            pending_control=tuple(
                envelope.as_dict()
                for envelope in sorted(
                    self._pending_control,
                    key=lambda item: (item.deliver_round, item.sender, item.sequence),
                )
            ),
            pending_ordinary=tuple(
                envelope.as_dict()
                for envelope in sorted(
                    self._pending_ordinary,
                    key=lambda item: (item.deliver_round, item.sender, item.sequence),
                )
            ),
            next_sequence=self._sequence,
            lease_version=self._lease_version,
            current_round_emissions=tuple(
                sorted(value for value in self._emissions if value[1] == self.current_round)
            ),
        )

    def state_hash(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.snapshot().as_dict()).encode("ascii")
        ).hexdigest()

    @property
    def audit_chain_hash(self) -> str:
        return self._chain_hash

    def audit_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._audit)

    def export_replay(self) -> dict[str, Any]:
        return {
            "schema_version": "tfp1-directory-replay-v1",
            "agent_ids": list(self.agent_ids),
            "method": self.method.value,
            "seed": self.seed,
            "max_control_bytes": self.max_control_bytes,
            "operations": list(self._operations),
            "final_state_hash": self.state_hash(),
            "audit_chain_hash": self.audit_chain_hash,
        }

    @classmethod
    def replay(cls, payload: dict[str, Any]) -> TeamDirectory:
        if payload.get("schema_version") != "tfp1-directory-replay-v1":
            raise ValueError("unsupported replay schema")
        directory = cls(
            agent_ids=tuple(int(value) for value in payload["agent_ids"]),
            method=payload["method"],
            seed=int(payload["seed"]),
            max_control_bytes=int(payload["max_control_bytes"]),
        )
        for operation in payload["operations"]:
            name = operation["op"]
            if name == "register_task":
                raw = operation["card"]
                directory.register_task(
                    TaskCard(
                        task_id=raw["task_id"],
                        demand=tuple(raw["demand"]),
                        required_size=raw["required_size"],
                        reward=raw["reward"],
                        deadline_round=raw["deadline_round"],
                        sponsor_id=raw["sponsor_id"],
                    )
                )
            elif name == "submit_control":
                directory.submit_control(
                    sender=operation["sender"],
                    sent_round=operation["sent_round"],
                    raw=operation["raw"],
                )
            elif name == "advance":
                directory.advance(operation["round_index"])
            elif name == "submit_ordinary":
                directory.submit_ordinary(
                    sender=operation["sender"],
                    sent_round=operation["sent_round"],
                    content=operation["content"],
                )
            elif name == "deliver_ordinary":
                directory.deliver_ordinary(operation["round_index"])
            elif name == "confirm_complete":
                directory.confirm_complete(
                    task_id=operation["task_id"],
                    round_index=operation["round_index"],
                    evidence_digest=operation["evidence_digest"],
                )
            else:
                raise ValueError(f"unknown replay operation {name!r}")
        if directory.state_hash() != payload["final_state_hash"]:
            raise ValueError("replay final state hash mismatch")
        if directory.audit_chain_hash != payload["audit_chain_hash"]:
            raise ValueError("replay audit chain hash mismatch")
        return directory

    def _apply_control(self, envelope: ControlEnvelope, round_index: int) -> TransitionResult:
        record = envelope.parse.record
        task_id = None if record is None else record.task_id
        if not envelope.parse.valid or record is None:
            return self._transition(
                envelope,
                round_index,
                False,
                envelope.parse.code,
                task_id,
            )
        if record.kind == RecordKind.ANNOUNCE:
            return self._apply_announcement(envelope, record, round_index)
        state = self.tasks.get(record.task_id)
        if state is None:
            return self._transition(envelope, round_index, False, "state.unknown_task", task_id)
        if state.phase in TERMINAL_PHASES:
            return self._transition(envelope, round_index, False, "state.task_closed", task_id)
        if state.card.deadline_round <= round_index:
            return self._transition(envelope, round_index, False, "state.task_expired", task_id)
        if state.phase == TaskPhase.LOCKED and record.kind not in {
            RecordKind.CANCEL,
            RecordKind.COMPLETE,
        }:
            return self._transition(envelope, round_index, False, "state.task_locked", task_id)
        handler = {
            RecordKind.CFP: self._handle_cfp,
            RecordKind.APPLY: self._handle_apply,
            RecordKind.NOMINATE: self._handle_nominate,
            RecordKind.BID: self._handle_bid,
            RecordKind.AWARD: self._handle_award,
            RecordKind.ACCEPT: self._handle_accept,
            RecordKind.DECLINE: self._handle_decline,
            RecordKind.LOCK: self._handle_lock,
            RecordKind.CANCEL: self._handle_cancel,
            RecordKind.COMPLETE: self._handle_complete_report,
            RecordKind.EXPIRE: self._handle_expire_record,
        }.get(record.kind)
        if handler is None:
            return self._transition(
                envelope, round_index, False, "state.unsupported_record", task_id
            )
        accepted, code = handler(state, envelope.sender, record, round_index)
        recipients = (
            tuple(agent for agent in self.agent_ids if agent != envelope.sender) if accepted else ()
        )
        return self._transition(
            envelope,
            round_index,
            accepted,
            code,
            task_id,
            recipients,
        )

    def _apply_announcement(
        self,
        envelope: ControlEnvelope,
        record: RecruitmentRecord,
        round_index: int,
    ) -> TransitionResult:
        if record.task_id in self.tasks:
            return self._transition(
                envelope,
                round_index,
                False,
                "state.duplicate_task",
                record.task_id,
            )
        if record.required_size > len(self.agent_ids):
            return self._transition(
                envelope,
                round_index,
                False,
                "task.population_too_small",
                record.task_id,
            )
        if record.deadline_round <= round_index:
            return self._transition(
                envelope,
                round_index,
                False,
                "task.deadline_not_future",
                record.task_id,
            )
        card = TaskCard(
            task_id=record.task_id,
            demand=record.demand,
            required_size=record.required_size,
            reward=record.reward,
            deadline_round=record.deadline_round,
            sponsor_id=envelope.sender,
        )
        self.tasks[card.task_id] = TaskState(card=card)
        recipients = tuple(agent for agent in self.agent_ids if agent != envelope.sender)
        return self._transition(
            envelope,
            round_index,
            True,
            "task.announced",
            card.task_id,
            recipients,
        )

    def _handle_cfp(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        del record, round_index
        if self.method != RecruitmentMethod.CONTRACT_NET:
            return False, "method.record_not_allowed"
        if sender != state.card.sponsor_id:
            return False, "authority.wrong_sponsor"
        state.cfp_open = True
        state.phase = TaskPhase.FORMING
        return True, "contract.cfp_opened"

    def _handle_apply(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        if self.method != RecruitmentMethod.OPEN_VOLUNTEER:
            return False, "method.record_not_allowed"
        if sender in self.agent_to_task:
            return False, "state.agent_already_teamed"
        state.applications[sender] = CandidateBid(
            task_id=record.task_id,
            agent_id=sender,
            claimed_capabilities=record.capabilities,
            cost=record.cost,
            arrival_round=round_index,
        )
        state.phase = TaskPhase.FORMING
        state.accepts.clear()
        state.declines.discard(sender)
        return True, "volunteer.application_recorded"

    def _handle_nominate(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        del round_index
        if self.method != RecruitmentMethod.MUTUAL_NOMINATION:
            return False, "method.record_not_allowed"
        if sender not in record.members:
            return False, "nomination.sender_missing"
        valid, code = self._validate_roster(state, record.members)
        if not valid:
            return False, code
        state.nominations[sender] = record.members
        state.phase = TaskPhase.FORMING
        return True, "nomination.recorded"

    def _handle_bid(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        if self.method != RecruitmentMethod.CONTRACT_NET:
            return False, "method.record_not_allowed"
        if not state.cfp_open:
            return False, "contract.cfp_not_open"
        if state.award is not None:
            return False, "contract.award_active"
        if sender in self.agent_to_task:
            return False, "state.agent_already_teamed"
        state.bids[sender] = CandidateBid(
            task_id=record.task_id,
            agent_id=sender,
            claimed_capabilities=record.capabilities,
            cost=record.cost,
            arrival_round=round_index,
        )
        state.phase = TaskPhase.FORMING
        return True, "contract.bid_recorded"

    def _handle_award(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        del round_index
        if self.method != RecruitmentMethod.CONTRACT_NET:
            return False, "method.record_not_allowed"
        if sender != state.card.sponsor_id:
            return False, "authority.wrong_sponsor"
        valid, code = self._validate_roster(state, record.members)
        if not valid:
            return False, code
        if not all(member in state.bids for member in record.members):
            return False, "contract.member_without_bid"
        if not _bids_cover(state.card, [state.bids[member] for member in record.members]):
            return False, "roster.claimed_infeasible"
        state.award = record.members
        state.accepts.clear()
        state.declines.clear()
        state.phase = TaskPhase.AWARDED
        return True, "contract.award_recorded"

    def _handle_accept(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        del round_index
        if self.method == RecruitmentMethod.OPEN_VOLUNTEER:
            roster = self._open_roster(state)
        elif self.method == RecruitmentMethod.CONTRACT_NET:
            roster = state.award
        else:
            return False, "method.record_not_allowed"
        if roster is None:
            return False, "state.roster_not_ready"
        if record.members != roster:
            return False, "state.roster_disagreement"
        if sender not in roster:
            return False, "authority.sender_not_roster_member"
        if sender in self.agent_to_task:
            return False, "state.agent_already_teamed"
        state.accepts.add(sender)
        state.declines.discard(sender)
        return True, "roster.accepted"

    def _handle_decline(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        del round_index
        roster = (
            self._open_roster(state)
            if self.method == RecruitmentMethod.OPEN_VOLUNTEER
            else state.award
            if self.method == RecruitmentMethod.CONTRACT_NET
            else None
        )
        if roster is None:
            return False, "state.roster_not_ready"
        if record.members != roster:
            return False, "state.roster_disagreement"
        if sender not in roster:
            return False, "authority.sender_not_roster_member"
        state.declines.add(sender)
        state.accepts.clear()
        return True, "roster.declined"

    def _handle_lock(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        if self.method == RecruitmentMethod.OPEN_VOLUNTEER:
            roster = self._open_roster(state)
            locker = state.card.sponsor_id
            ready = roster is not None and set(roster).issubset(state.accepts)
        elif self.method == RecruitmentMethod.MUTUAL_NOMINATION:
            roster = record.members
            locker = min(roster)
            ready = all(state.nominations.get(member) == roster for member in roster)
        else:
            roster = state.award
            locker = state.card.sponsor_id
            ready = roster is not None and set(roster).issubset(state.accepts)
        if sender != locker:
            return False, "authority.wrong_locker"
        if roster is None or record.members != roster:
            return False, "state.roster_disagreement"
        if not ready or state.declines.intersection(roster):
            return False, "state.roster_not_confirmed"
        valid, code = self._validate_roster(state, roster)
        if not valid:
            return False, code
        self._lease_version += 1
        lease = TeamLease(
            task_id=state.card.task_id,
            members=roster,
            start_round=round_index,
            deadline_round=state.card.deadline_round,
            version=self._lease_version,
        )
        for member in roster:
            self.agent_to_task[member] = state.card.task_id
        state.lease = lease
        state.phase = TaskPhase.LOCKED
        return True, "team.locked"

    def _handle_cancel(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        del round_index
        permitted = sender == state.card.sponsor_id or (
            state.lease is not None and sender in state.lease.members
        )
        if not permitted:
            return False, "authority.cancel_not_permitted"
        self._release(state)
        state.phase = TaskPhase.CANCELLED
        state.close_reason = record.reason
        return True, "task.cancelled"

    def _handle_complete_report(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        del record, round_index
        if state.phase != TaskPhase.LOCKED or state.lease is None:
            return False, "state.task_not_locked"
        if sender not in state.lease.members:
            return False, "authority.sender_not_roster_member"
        state.completion_reports.add(sender)
        return True, "completion.reported"

    def _handle_expire_record(
        self,
        state: TaskState,
        sender: int,
        record: RecruitmentRecord,
        round_index: int,
    ) -> tuple[bool, str]:
        del state, record, round_index
        if sender != SYSTEM_AGENT_ID:
            return False, "authority.expire_system_only"
        return False, "authority.expire_system_only"

    def _validate_roster(self, state: TaskState, roster: tuple[int, ...]) -> tuple[bool, str]:
        if len(roster) != state.card.required_size:
            return False, "roster.size_mismatch"
        if any(member not in self.agent_ids for member in roster):
            return False, "roster.unknown_member"
        busy = [
            member
            for member in roster
            if member in self.agent_to_task and self.agent_to_task[member] != state.card.task_id
        ]
        if busy:
            return False, "state.member_already_teamed"
        return True, "valid"

    def _open_roster(self, state: TaskState) -> tuple[int, ...] | None:
        candidates = sorted(
            (
                bid
                for bid in state.applications.values()
                if bid.agent_id not in state.declines and bid.agent_id not in self.agent_to_task
            ),
            key=lambda bid: (bid.arrival_round, bid.agent_id),
        )
        for candidate_group in combinations(candidates, state.card.required_size):
            if _bids_cover(state.card, candidate_group):
                return tuple(sorted(bid.agent_id for bid in candidate_group))
        return None

    def _expire(self, state: TaskState, round_index: int) -> TransitionResult:
        self._release(state)
        state.phase = TaskPhase.EXPIRED
        state.close_reason = "deadline"
        result = TransitionResult(
            sequence=self._next_sequence(),
            accepted=True,
            code="task.expired",
            sender=SYSTEM_AGENT_ID,
            round_index=round_index,
            task_id=state.card.task_id,
            delivered_recipients=self.agent_ids,
        )
        self._record_audit("transition", result.as_dict())
        return result

    def _release(self, state: TaskState) -> None:
        if state.lease is not None:
            for member in state.lease.members:
                if self.agent_to_task.get(member) == state.card.task_id:
                    del self.agent_to_task[member]
        state.lease = None

    def _transition(
        self,
        envelope: ControlEnvelope,
        round_index: int,
        accepted: bool,
        code: str,
        task_id: str | None,
        recipients: tuple[int, ...] = (),
    ) -> TransitionResult:
        result = TransitionResult(
            sequence=envelope.sequence,
            accepted=accepted,
            code=code,
            sender=envelope.sender,
            round_index=round_index,
            task_id=task_id,
            delivered_recipients=recipients,
        )
        self._record_audit("transition", result.as_dict())
        return result

    def _validate_submission(self, sender: int, sent_round: int) -> None:
        if sender not in self.agent_ids:
            raise ValueError("sender is not a configured agent")
        if sent_round != self.current_round:
            raise ValueError(
                f"sent_round must equal current round {self.current_round}; got {sent_round}"
            )

    def _claim_emission(self, sender: int, sent_round: int) -> None:
        key = (sender, sent_round)
        if key in self._emissions:
            raise ValueError("one emission per agent per round")
        self._emissions.add(key)

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def _record_audit(self, category: str, payload: dict[str, Any]) -> None:
        record = {
            "index": len(self._audit),
            "category": category,
            "payload": payload,
        }
        encoded = _canonical_json(record)
        self._chain_hash = hashlib.sha256(
            f"{self._chain_hash}\n{encoded}".encode("ascii")
        ).hexdigest()
        self._audit.append({**record, "chain_hash": self._chain_hash})


def _validate_agent_id(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_score(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError(f"{name} must be an integer from 0 to 100")


def _validate_vector(value: Any, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != CAPABILITY_DIMENSIONS:
        raise ValueError(f"{name} must be a three-component tuple")
    for component in value:
        _validate_score(component, name)


def _validate_members(members: tuple[int, ...]) -> None:
    if tuple(sorted(set(members))) != members:
        raise ValueError("members must be sorted and unique")
    for member in members:
        _validate_agent_id(member, "member")


def _parse_int(value: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError("noncanonical integer")
    return int(value)


def _parse_score(value: str) -> int:
    parsed = _parse_int(value)
    _validate_score(parsed, "score")
    return parsed


def _parse_vector(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != CAPABILITY_DIMENSIONS:
        raise ValueError("invalid vector length")
    parsed = tuple(_parse_score(part) for part in parts)
    return parsed


def _parse_members(value: str) -> tuple[int, ...]:
    parsed = tuple(_parse_int(part) for part in value.split(","))
    _validate_members(parsed)
    return parsed


def _render_vector(value: tuple[int, int, int] | None) -> str | None:
    return None if value is None else ",".join(str(component) for component in value)


def _bids_cover(card: TaskCard, bids: Any) -> bool:
    bid_list = list(bids)
    if len(bid_list) != card.required_size:
        return False
    for dimension, demand in enumerate(card.demand):
        if demand > 0 and sum(bid.claimed_capabilities[dimension] for bid in bid_list) < demand:
            return False
    return True


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


__all__ = [
    "CAPABILITY_DIMENSIONS",
    "MAX_CONTROL_BYTES",
    "PROTOCOL_VERSION",
    "SYSTEM_AGENT_ID",
    "CandidateBid",
    "ControlEnvelope",
    "DirectorySnapshot",
    "OrdinaryEnvelope",
    "ParseResult",
    "RecordKind",
    "RecruitmentMethod",
    "RecruitmentRecord",
    "RouteDecision",
    "TaskCard",
    "TaskPhase",
    "TeamDirectory",
    "TeamLease",
    "TransitionResult",
    "parse_tfp1",
]
