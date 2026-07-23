"""Embodied squad-command protocol for three-agent Alem teams.

The planner is a second model call made on behalf of an embodied commander.  It
does not add a fourth participant and receives only the commander's legal text
observation, the public rules, its bounded private scratchpad, the active plan,
and validated status reports.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Any

from .agents.base import _ClientProxy
from .prompt_builder import Message

COMMANDER_TOPOLOGIES = frozenset({"embodied_commander_broadcast", "embodied_commander_star"})
STATUS_STATES = frozenset({"ACK", "ACTIVE", "BLOCKED", "COMPLETE", "EMERGENCY", "WAITING"})
EVENT_STATES = frozenset({"BLOCKED", "COMPLETE", "EMERGENCY"})
PLAN_OPERATIONS = frozenset({"KEEP", "REPLACE"})
MAX_STATUS_LENGTH = 400
MAX_EVIDENCE_LENGTH = 160
MAX_REQUEST_LENGTH = 120


def _extract_tagged(text: str | None, tag: str) -> str | None:
    if not text:
        return None
    match = re.search(
        rf"<{tag}\b[^>]*>(.*?)</{tag}\s*>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    return None


def _clean_text(value: Any, *, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    if not cleaned or len(cleaned) > maximum:
        return None
    return cleaned


def _canonical_actions() -> frozenset[str]:
    # Imported lazily so parser-only unit tests do not initialize the simulator.
    from alem.llm.alem_language_wrapper import ACTIONS

    return frozenset(ACTIONS)


@dataclass(frozen=True)
class SquadSpec:
    team_id: str
    members: tuple[int, ...]
    commander_id: int
    review_interval: int = 5
    lease_steps: int = 10
    max_steps: int = 200

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", self.team_id):
            raise ValueError("team_id must be 1-32 identifier characters")
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("team members must be non-empty and unique")
        if any(isinstance(member, bool) or not isinstance(member, int) for member in self.members):
            raise ValueError("team members must be integer agent IDs")
        if self.commander_id not in self.members:
            raise ValueError("commander_id must be a member of the squad")
        if self.review_interval < 1 or self.lease_steps < 1 or self.max_steps < 1:
            raise ValueError("review_interval, lease_steps, and max_steps must be positive")

    @property
    def event_replan_cap(self) -> int:
        return math.ceil(self.max_steps / 10)


@dataclass(frozen=True)
class SyncWindow:
    action: str
    earliest_tick: int
    latest_tick: int


@dataclass(frozen=True)
class SquadAssignment:
    agent_id: int
    task_id: str
    directive: str
    target: str
    completion: str
    dependencies: tuple[str, ...]
    sync: SyncWindow | None = None


@dataclass(frozen=True)
class PlanProposal:
    operation: str
    objective: str | None = None
    assignments: tuple[SquadAssignment, ...] = ()


@dataclass(frozen=True)
class SquadPlan:
    team_id: str
    version: int
    objective: str
    assignments: tuple[SquadAssignment, ...]
    issued_tick: int
    review_tick: int
    expiry_tick: int

    def for_agent(self, agent_id: int) -> SquadAssignment:
        return next(item for item in self.assignments if item.agent_id == agent_id)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SquadStatus:
    team_id: str
    version: int
    task_id: str
    state: str
    evidence: str
    request: str | None = None

    def render(self) -> str:
        fields = [
            "SCP1",
            f"TEAM={self.team_id}",
            f"VERSION={self.version}",
            f"TASK={self.task_id}",
            f"STATE={self.state}",
            f"EVIDENCE={self.evidence}",
        ]
        if self.request:
            fields.append(f"REQUEST={self.request}")
        return "|".join(fields)


@dataclass(frozen=True)
class ReviewRequest:
    trigger: str
    scheduled: bool


@dataclass(frozen=True)
class PlanApplyResult:
    accepted: bool
    reason: str
    plan: SquadPlan | None


def _parse_sync(
    value: Any,
    *,
    step: int,
    lease_steps: int,
    canonical_actions: frozenset[str],
) -> SyncWindow | None | bool:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "action",
        "earliest_tick",
        "latest_tick",
    }:
        return False
    action = value.get("action")
    earliest = value.get("earliest_tick")
    latest = value.get("latest_tick")
    if action not in canonical_actions:
        return False
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (earliest, latest)):
        return False
    if not (step <= earliest <= latest < step + lease_steps):
        return False
    return SyncWindow(action=action, earliest_tick=earliest, latest_tick=latest)


def parse_squad_plan(
    text: str | None,
    *,
    spec: SquadSpec,
    step: int,
    canonical_actions: Iterable[str] | None = None,
) -> PlanProposal | None:
    """Parse and atomically validate one tagged commander proposal."""

    tagged = _extract_tagged(text, "squad_plan")
    if tagged is None:
        return None
    try:
        payload = json.loads(tagged)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    operation = payload.get("operation")
    if operation not in PLAN_OPERATIONS:
        return None
    if operation == "KEEP":
        return PlanProposal(operation="KEEP") if set(payload) == {"operation"} else None
    if set(payload) != {"operation", "objective", "assignments"}:
        return None

    objective = _clean_text(payload.get("objective"), maximum=500)
    assignments = payload.get("assignments")
    if objective is None or not isinstance(assignments, list):
        return None
    if len(assignments) != len(spec.members):
        return None

    allowed_actions = (
        frozenset(canonical_actions) if canonical_actions is not None else _canonical_actions()
    )
    parsed: list[SquadAssignment] = []
    seen_agents: set[int] = set()
    seen_tasks: set[str] = set()
    for item in assignments:
        if not isinstance(item, dict):
            return None
        allowed_keys = {
            "agent_id",
            "task_id",
            "directive",
            "target",
            "completion",
            "dependencies",
            "sync",
        }
        if not set(item).issubset(allowed_keys) or not {
            "agent_id",
            "task_id",
            "directive",
            "target",
            "completion",
            "dependencies",
        }.issubset(item):
            return None
        agent_id = item.get("agent_id")
        if (
            isinstance(agent_id, bool)
            or not isinstance(agent_id, int)
            or agent_id not in spec.members
            or agent_id in seen_agents
        ):
            return None
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", task_id):
            return None
        if task_id in seen_tasks:
            return None
        directive = _clean_text(item.get("directive"), maximum=400)
        target = _clean_text(item.get("target"), maximum=200)
        completion = _clean_text(item.get("completion"), maximum=300)
        dependencies = item.get("dependencies")
        if (
            directive is None
            or target is None
            or completion is None
            or not isinstance(dependencies, list)
            or len(dependencies) > len(spec.members) - 1
            or not all(
                isinstance(dep, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", dep)
                for dep in dependencies
            )
            or len(set(dependencies)) != len(dependencies)
            or task_id in dependencies
        ):
            return None
        sync = _parse_sync(
            item.get("sync"),
            step=step,
            lease_steps=spec.lease_steps,
            canonical_actions=allowed_actions,
        )
        if sync is False:
            return None
        parsed.append(
            SquadAssignment(
                agent_id=agent_id,
                task_id=task_id,
                directive=directive,
                target=target,
                completion=completion,
                dependencies=tuple(dependencies),
                sync=sync,
            )
        )
        seen_agents.add(agent_id)
        seen_tasks.add(task_id)
    if seen_agents != set(spec.members):
        return None
    if any(dep not in seen_tasks for item in parsed for dep in item.dependencies):
        return None

    edges = {item.task_id: item.dependencies for item in parsed}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> bool:
        if task_id in visiting:
            return False
        if task_id in visited:
            return True
        visiting.add(task_id)
        if not all(visit(dep) for dep in edges[task_id]):
            return False
        visiting.remove(task_id)
        visited.add(task_id)
        return True

    if not all(visit(task_id) for task_id in edges):
        return None
    return PlanProposal(
        operation="REPLACE",
        objective=objective,
        assignments=tuple(sorted(parsed, key=lambda value: value.agent_id)),
    )


def parse_squad_status(text: str | None) -> SquadStatus | None:
    """Parse one bounded SCP1 record from a communication payload."""

    if not text or len(text) > MAX_STATUS_LENGTH:
        return None
    pieces = text.strip().split("|")
    if not pieces or pieces[0] != "SCP1":
        return None
    fields: dict[str, str] = {}
    for piece in pieces[1:]:
        key, separator, value = piece.partition("=")
        if (
            not separator
            or key in fields
            or key
            not in {
                "TEAM",
                "VERSION",
                "TASK",
                "STATE",
                "EVIDENCE",
                "REQUEST",
            }
        ):
            return None
        fields[key] = value.strip()
    if not {"TEAM", "VERSION", "TASK", "STATE", "EVIDENCE"}.issubset(fields):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", fields["TEAM"]):
        return None
    if not fields["VERSION"].isdigit():
        return None
    if not re.fullmatch(r"(?:[A-Za-z0-9_.-]{1,48}|NONE)", fields["TASK"]):
        return None
    if fields["STATE"] not in STATUS_STATES:
        return None
    evidence = _clean_text(fields["EVIDENCE"], maximum=MAX_EVIDENCE_LENGTH)
    request = (
        _clean_text(fields["REQUEST"], maximum=MAX_REQUEST_LENGTH) if "REQUEST" in fields else None
    )
    if evidence is None or ("REQUEST" in fields and request is None):
        return None
    return SquadStatus(
        team_id=fields["TEAM"],
        version=int(fields["VERSION"]),
        task_id=fields["TASK"],
        state=fields["STATE"],
        evidence=evidence,
        request=request,
    )


class SquadRuntime:
    """Own leases, authority checks, reports, triggers, and audit metrics."""

    def __init__(self, spec: SquadSpec):
        self.spec = spec
        self.plan: SquadPlan | None = None
        self.pending_event: str | None = None
        self.pending_invalid = False
        self.event_replans = 0
        self.last_status: dict[int, SquadStatus] = {}
        self.reports: list[tuple[int, SquadStatus]] = []
        self._acknowledged: set[tuple[int, int]] = set()
        self._pending_event_tick: int | None = None
        self.metrics: Counter[str] = Counter()
        self.plan_covered_ticks = 0
        self.total_ticks = 0
        self.plan_age_sum = 0
        self.assignment_switches = 0

    def active_plan(self, step: int) -> SquadPlan | None:
        if self.plan is None or step >= self.plan.expiry_tick:
            return None
        return self.plan

    def observe_tick(self, step: int) -> None:
        self.total_ticks += 1
        plan = self.active_plan(step)
        if plan is not None:
            self.plan_covered_ticks += 1
            self.plan_age_sum += step - plan.issued_tick
        elif self.plan is not None and step == self.plan.expiry_tick:
            self.metrics["plan_expiries"] += 1

    def review_due(self, step: int) -> ReviewRequest | None:
        scheduled = step == 0 or step % self.spec.review_interval == 0
        unscheduled = self.pending_invalid or self.pending_event is not None
        if unscheduled and self.event_replans < self.spec.event_replan_cap:
            trigger = "invalid_retry" if self.pending_invalid else f"event:{self.pending_event}"
            if self.pending_event is not None and self._pending_event_tick is not None:
                latency = step - self._pending_event_tick
                state_key = self.pending_event.lower()
                self.metrics[f"{state_key}_to_replan_count"] += 1
                self.metrics[f"{state_key}_to_replan_ticks"] += latency
            self.event_replans += 1
            self.pending_invalid = False
            self.pending_event = None
            self._pending_event_tick = None
            return ReviewRequest(trigger=trigger, scheduled=False)
        if unscheduled and scheduled:
            # A due scheduled call absorbs the event without adding an extra call,
            # but it remains an event/repair review and therefore cannot accept KEEP.
            trigger = "invalid_retry" if self.pending_invalid else f"event:{self.pending_event}"
            if self.pending_event is not None and self._pending_event_tick is not None:
                latency = step - self._pending_event_tick
                state_key = self.pending_event.lower()
                self.metrics[f"{state_key}_to_replan_count"] += 1
                self.metrics[f"{state_key}_to_replan_ticks"] += latency
            self.pending_invalid = False
            self.pending_event = None
            self._pending_event_tick = None
            return ReviewRequest(trigger=trigger, scheduled=False)
        if scheduled:
            return ReviewRequest(trigger="scheduled", scheduled=True)
        return None

    def apply(
        self,
        proposal: PlanProposal | None,
        *,
        step: int,
        review: ReviewRequest,
    ) -> PlanApplyResult:
        self.metrics["plan_calls"] += 1
        if proposal is None:
            self.metrics["invalid_plans"] += 1
            self.pending_invalid = True
            return PlanApplyResult(False, "invalid", self.active_plan(step))
        if proposal.operation == "KEEP":
            active = self.active_plan(step)
            if not review.scheduled or active is None:
                self.metrics["invalid_plans"] += 1
                self.metrics["invalid_keep"] += 1
                self.pending_invalid = True
                return PlanApplyResult(False, "keep_not_allowed", active)
            self.plan = replace(
                active,
                review_tick=step + self.spec.review_interval,
                expiry_tick=step + self.spec.lease_steps,
            )
            self.metrics["valid_plans"] += 1
            self.metrics["keep_plans"] += 1
            return PlanApplyResult(True, "kept", self.plan)

        if proposal.operation != "REPLACE" or proposal.objective is None:
            self.metrics["invalid_plans"] += 1
            self.pending_invalid = True
            return PlanApplyResult(False, "invalid_replace", self.active_plan(step))
        previous = self.active_plan(step)
        version = (self.plan.version if self.plan is not None else 0) + 1
        next_plan = SquadPlan(
            team_id=self.spec.team_id,
            version=version,
            objective=proposal.objective,
            assignments=proposal.assignments,
            issued_tick=step,
            review_tick=step + self.spec.review_interval,
            expiry_tick=step + self.spec.lease_steps,
        )
        if previous is not None:
            for member in self.spec.members:
                if previous.for_agent(member).task_id != next_plan.for_agent(member).task_id:
                    self.assignment_switches += 1
        self.plan = next_plan
        self.metrics["valid_plans"] += 1
        self.metrics["replace_plans"] += 1
        self.metrics["assignments_issued"] += len(self.spec.members)
        self.reports.clear()
        return PlanApplyResult(True, "replaced", self.plan)

    def ingest_status(self, sender: int, text: str | None, *, step: int) -> SquadStatus | None:
        self.metrics["status_attempts"] += 1
        if text and "<squad_plan" in text.lower():
            self.metrics["unauthorized_plan_attempts"] += 1
        status = parse_squad_status(text)
        if status is None:
            self.metrics["invalid_status"] += 1
            return None
        if sender not in self.spec.members or status.team_id != self.spec.team_id:
            self.metrics["wrong_team_status"] += 1
            return None
        active = self.active_plan(step)
        if active is None:
            if status.state != "WAITING" or status.version != 0 or status.task_id != "NONE":
                self.metrics["stale_status"] += 1
                return None
        else:
            assignment = active.for_agent(sender)
            if status.version != active.version:
                self.metrics["stale_status"] += 1
                return None
            if status.task_id != assignment.task_id:
                self.metrics["wrong_assignment_status"] += 1
                return None
        previous_status = self.last_status.get(sender)
        self.metrics["valid_status"] += 1
        self.metrics[f"state_{status.state.lower()}"] += 1
        self.last_status[sender] = status
        self.reports.append((sender, status))
        if active is not None and status.state == "ACK":
            ack_key = (status.version, sender)
            if ack_key not in self._acknowledged:
                self._acknowledged.add(ack_key)
                self.metrics["assignment_acks"] += 1
                self.metrics["ack_latency_ticks"] += step - active.issued_tick
        if status.state in EVENT_STATES and (
            previous_status is None
            or previous_status.version != status.version
            or previous_status.task_id != status.task_id
            or previous_status.state != status.state
        ):
            if self.pending_event is None:
                self.pending_event = status.state
                self._pending_event_tick = step
            self.metrics["event_transitions"] += 1
        return status

    def consume_reports(self) -> list[tuple[int, SquadStatus]]:
        reports = list(self.reports)
        self.reports.clear()
        return reports

    def as_dict(self, *, final_step: int) -> dict[str, Any]:
        calls = self.metrics["plan_calls"]
        status_attempts = self.metrics["status_attempts"]
        status_eligible_turns = self.metrics["status_eligible_turns"]
        active = self.active_plan(final_step)
        return {
            **dict(self.metrics),
            "team_id": self.spec.team_id,
            "commander_id": self.spec.commander_id,
            "members": list(self.spec.members),
            "review_interval": self.spec.review_interval,
            "lease_steps": self.spec.lease_steps,
            "event_replan_cap": self.spec.event_replan_cap,
            "event_replans": self.event_replans,
            "plan_parse_rate": round(self.metrics["valid_plans"] / max(calls, 1), 4),
            "status_parse_rate": round(self.metrics["valid_status"] / max(status_attempts, 1), 4),
            "status_attempt_rate": round(status_attempts / max(status_eligible_turns, 1), 4),
            "valid_status_coverage": round(
                self.metrics["valid_status"] / max(status_eligible_turns, 1), 4
            ),
            "active_plan_coverage": round(self.plan_covered_ticks / max(self.total_ticks, 1), 4),
            "mean_plan_age_ticks": round(self.plan_age_sum / max(self.plan_covered_ticks, 1), 4),
            "assignment_switches": self.assignment_switches,
            "assignment_ack_rate": round(
                self.metrics["assignment_acks"] / max(self.metrics["assignments_issued"], 1),
                4,
            ),
            "mean_ack_latency_ticks": round(
                self.metrics["ack_latency_ticks"] / max(self.metrics["assignment_acks"], 1),
                4,
            ),
            "accepted_unauthorized_plans": 0,
            "accepted_stale_statuses": 0,
            "hidden_state_leak_guard_violations": 0,
            "final_plan": active.as_dict() if active is not None else None,
        }


def commander_planner_system_prompt(worker_prompt: str, spec: SquadSpec) -> str:
    """Extract public rules and add the commander's planning-only contract."""

    _, separator, public_rules = worker_prompt.partition("<game_rules>")
    rules = (separator + public_rules).strip() if separator else worker_prompt.strip()
    member_text = ", ".join(str(member) for member in spec.members)
    return (
        f"You are Agent {spec.commander_id}'s private squad-planning phase for team "
        f"{spec.team_id}. The embodied commander acts later with the other agents. "
        "Plan only from the commander's legal text observation and validated reports "
        "provided by the middleware; never infer hidden world state. Assign exactly one "
        f"bounded task to every member ({member_text}), including the commander. "
        "Use dependencies and optional synchronization only when necessary.\n\n"
        + rules
        + "\n\n<output_format>\n"
        "Return exactly one <squad_plan> JSON object. At a scheduled review with an "
        'active plan you may return {"operation":"KEEP"}. Otherwise return '
        "operation REPLACE with objective and exactly one assignment per member. Each "
        "assignment requires agent_id, unique task_id, directive, target, completion, "
        "dependencies, and may include sync={action,earliest_tick,latest_tick}. "
        "Use sync only for a simultaneous physical environment action; otherwise omit it. "
        "Dependencies must reference task IDs in this plan and be acyclic. You may add "
        "one private <scratchpad> entry after the plan. Never output <action> or "
        "<communication>.\n</output_format>"
    )


class EmbodiedCommanderPlanner:
    """Separate serial planning call owned by an embodied commander."""

    def __init__(
        self,
        client_factory,
        *,
        spec: SquadSpec,
        max_scratchpad_length: int = 1000,
    ):
        self.client = _ClientProxy(client_factory())
        self.spec = spec
        self.max_scratchpad_length = max_scratchpad_length
        self.system_prompt = ""
        self.scratchpad = ""
        self.last_raw_completion = ""

    def reset(self) -> None:
        self.scratchpad = ""
        self.last_raw_completion = ""

    def set_instruction_prompt(self, worker_prompt: str) -> None:
        self.system_prompt = commander_planner_system_prompt(worker_prompt, self.spec)

    def plan(
        self,
        *,
        step: int,
        review: ReviewRequest,
        commander_observation: dict[str, str],
        reports: list[tuple[int, SquadStatus]],
        active_plan: SquadPlan | None,
        canonical_actions: Iterable[str] | None = None,
    ):
        report_payloads = [{"sender": sender, **asdict(report)} for sender, report in reports]
        active_payload = active_plan.as_dict() if active_plan is not None else None
        allowed_actions = sorted(
            canonical_actions if canonical_actions is not None else _canonical_actions()
        )
        user_prompt = (
            f"Tick: {step}\nReview trigger: {review.trigger}\n"
            f"Scheduled review: {str(review.scheduled).lower()}\n"
            "Canonical physical actions allowed in an optional sync.action: "
            f"{json.dumps(allowed_actions, ensure_ascii=False)}\n"
            f"Active plan: {json.dumps(active_payload, ensure_ascii=False)}\n"
            "Validated reports since the last review: "
            f"{json.dumps(report_payloads, ensure_ascii=False)}\n"
            f"Private planner scratchpad: {self.scratchpad or '(empty)'}\n\n"
            "Commander's current legal long-term observation:\n"
            f"{commander_observation.get('obs_long_term', '')}\n\n"
            "Commander's current legal short-term observation:\n"
            f"{commander_observation.get('obs_short_term', '')}\n\n"
            "Return the authorized squad-plan decision."
        )
        response = self.client.generate(
            [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=user_prompt),
            ]
        )
        self.last_raw_completion = response.completion or ""
        proposal = parse_squad_plan(
            self.last_raw_completion,
            spec=self.spec,
            step=step,
            canonical_actions=canonical_actions,
        )
        scratchpad = _extract_tagged(self.last_raw_completion, "scratchpad")
        if scratchpad:
            self.scratchpad = scratchpad[: self.max_scratchpad_length]
        return response, proposal


def format_squad_directive(
    plan: SquadPlan | None,
    *,
    spec: SquadSpec,
    agent_id: int,
) -> str:
    """Render the complete accepted plan plus an agent-specific authority contract."""

    is_commander = agent_id == spec.commander_id
    role = "COMMANDER-EXECUTOR" if is_commander else "WINGMAN-EXECUTOR"
    if plan is None:
        return (
            f"SQUAD COMMAND CONTRACT — team={spec.team_id}; role={role}.\n"
            "There is no valid leased squad plan. Do not invent an objective, assign "
            "work, or create/revise a plan. If an immediately available action is "
            "required to avoid death, take only that survival action; otherwise choose "
            "Noop. Report exactly: "
            f"<communication>SCP1|TEAM={spec.team_id}|VERSION=0|TASK=NONE|"
            "STATE=WAITING|EVIDENCE=no active leased plan</communication>."
        )
    assignment = plan.for_agent(agent_id)
    full_plan = json.dumps(plan.as_dict(), ensure_ascii=False, separators=(",", ":"))
    authority = (
        "This turn's separate commander planning phase is the only authority that may "
        "create or revise team objectives and assignments. In this action phase, execute "
        "your own assignment; do not revise the plan."
        if is_commander
        else "The commander alone creates or revises objectives and assignments. Use "
        "tactical autonomy only for movement, prerequisites, immediate survival, and "
        "local execution of your assignment. Do not create competing plans or reassign work."
    )
    return (
        f"SQUAD COMMAND CONTRACT — team={spec.team_id}; role={role}.\n"
        f"{authority}\nFull accepted plan: {full_plan}\n"
        f"YOUR ASSIGNMENT: task={assignment.task_id}; directive={assignment.directive}; "
        f"target={assignment.target}; completion={assignment.completion}; "
        f"dependencies={list(assignment.dependencies)}; sync="
        f"{asdict(assignment.sync) if assignment.sync else None}.\n"
        "Every action-phase response must include exactly one <communication> payload "
        "containing one SCP1 status record. Choose exactly one state from ACK, ACTIVE, "
        "BLOCKED, COMPLETE, or EMERGENCY. Example: "
        f"<communication>SCP1|TEAM={spec.team_id}|VERSION={plan.version}|"
        f"TASK={assignment.task_id}|STATE=ACTIVE|EVIDENCE=brief observed fact|"
        "REQUEST=optional brief request</communication>. Never place a squad plan in "
        "communication."
    )
