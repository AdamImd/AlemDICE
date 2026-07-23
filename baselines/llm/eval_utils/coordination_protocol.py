"""Auditable peer-to-peer coordination messages for Alem LLM agents.

The protocol deliberately carries no environment state that an agent did not
write itself.  Agent-local ledgers consume only the agent's own messages and
messages delivered through Alem's ordinary one-tick-delayed broadcast channel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

VALID_STRATEGIES = {"free", "consensus", "roles", "cohesion", "integrated"}
VALID_TYPES = {
    "PROPOSE", "VOTE", "COMMIT", "CANCEL",
    "BID", "AWARD", "ACCEPT", "STATUS",
}
VALID_FIELDS = {
    "TYPE", "ID", "TASK", "ROLE", "MEMBERS", "SCORE", "ASSIGNEE",
    "VOTE", "ACTION", "TARGET", "TICK", "STATE", "LEASE",
}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,24}$")


@dataclass(frozen=True)
class ProtocolMessage:
    kind: str
    task_id: str
    fields: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "id": self.task_id, **self.fields}


def parse_protocol_message(text: str | None) -> ProtocolMessage | None:
    """Parse one strict DCP1 record embedded in an ordinary communication."""

    if not text:
        return None
    records = [part.strip() for part in text.splitlines() if "DCP1|" in part]
    if len(records) != 1:
        return None
    line = records[0]
    line = line[line.index("DCP1|") :].strip()
    values: dict[str, str] = {}
    for item in line.split("|")[1:]:
        if "=" not in item:
            return None
        key, value = item.split("=", 1)
        key, value = key.strip().upper(), value.strip()
        if key not in VALID_FIELDS or key in values or not value:
            return None
        values[key] = value
    kind = values.pop("TYPE", "").upper()
    task_id = values.pop("ID", "")
    if kind not in VALID_TYPES or not _ID_RE.fullmatch(task_id):
        return None
    required = {
        "PROPOSE": {"TASK", "MEMBERS"}, "VOTE": {"VOTE"},
        "COMMIT": {"ACTION"}, "CANCEL": {"STATE"},
        "BID": {"ROLE", "SCORE"}, "AWARD": {"ROLE", "ASSIGNEE", "LEASE"},
        "ACCEPT": {"ROLE"}, "STATUS": {"ROLE", "STATE"},
    }[kind]
    if not required.issubset(values):
        return None
    if "VOTE" in values and values["VOTE"].upper() not in {"YES", "NO"}:
        return None
    if "SCORE" in values:
        try:
            score = int(values["SCORE"])
        except ValueError:
            return None
        if not 0 <= score <= 100:
            return None
    for key in ("ASSIGNEE", "TICK", "LEASE"):
        if key in values:
            try:
                if int(values[key]) < 0:
                    return None
            except ValueError:
                return None
    if "MEMBERS" in values:
        try:
            members = [int(value) for value in values["MEMBERS"].split(",")]
        except ValueError:
            return None
        if not members or len(set(members)) != len(members) or any(value < 0 for value in members):
            return None
    return ProtocolMessage(kind=kind, task_id=task_id, fields=values)


@dataclass
class CoordinationLedger:
    """Small, deterministic memory reconstructed from visible communications."""

    agent_id: int
    max_events: int = 12
    events: list[dict[str, Any]] = field(default_factory=list)
    _seen: set[tuple[int, int, str]] = field(default_factory=set)

    def record(self, sender: int, text: str | None, step: int) -> ProtocolMessage | None:
        message = parse_protocol_message(text)
        if message is None:
            return None
        key = (int(sender), int(step), text or "")
        if key in self._seen:
            return message
        self._seen.add(key)
        self.events.append({"sender": int(sender), "step": int(step), **message.as_dict()})
        self.events = self.events[-self.max_events :]
        return message

    def render(self, step: int) -> str:
        if not self.events:
            return "DCP1 local ledger: no valid protocol messages observed yet."
        lines = ["DCP1 local ledger (peer messages are one tick delayed):"]
        for event in self.events[-6:]:
            details = ", ".join(
                f"{key}={value}" for key, value in event.items()
                if key not in {"sender", "step", "type", "id"}
            )
            lines.append(
                f"- tick {event['step']} agent {event['sender']}: "
                f"{event['type']} {event['id']}" + (f" ({details})" if details else "")
            )
        lines.append(f"Current tick: {step}. Ignore awards whose lease has expired.")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "events": list(self.events)}

    def reset(self) -> None:
        self.events.clear()
        self._seen.clear()


class CoordinationMetrics:
    """Evaluator-side observer; never feeds derived state back to an agent."""

    def __init__(self, strategy: str, num_agents: int):
        self.strategy = strategy
        self.num_agents = num_agents
        self.nonempty = 0
        self.parsed = 0
        self.counts = {kind.lower(): 0 for kind in VALID_TYPES}
        self.events: list[dict[str, Any]] = []
        self.proposals: dict[str, dict[str, Any]] = {}
        self.votes: dict[str, dict[int, str]] = {}
        self.bids: dict[str, dict[int, int]] = {}
        self.awards: list[dict[str, Any]] = []
        self.status_windows: dict[int, set[int]] = {}
        self.schedule_valid = 0
        self.schedule_violations = 0

    def _schedule_ok(self, sender: int, message: ProtocolMessage, step: int) -> bool:
        """Check public-time phase compliance without environment information."""
        if self.strategy == "free" or self.strategy == "consensus":
            return True
        if self.strategy == "cohesion":
            return message.kind == "CANCEL" or (
                message.kind == "STATUS" and step % 5 == 0
            )
        if self.strategy == "roles":
            epoch, phase = divmod(step, 5)
            coordinator = epoch % self.num_agents
            expected = ("PROPOSE", "BID", "AWARD", "ACCEPT", "STATUS")[phase]
            if message.kind != expected:
                return False
            if phase in {0, 2}:
                return sender == coordinator
            if phase == 1:
                return sender != coordinator
            return True
        if self.strategy == "integrated":
            epoch, phase = divmod(step, 7)
            coordinator = epoch % self.num_agents
            expected = (
                "PROPOSE", "VOTE", "BID", "AWARD", "ACCEPT", "COMMIT", "STATUS"
            )[phase]
            if message.kind != expected:
                return False
            if phase in {0, 3}:
                return sender == coordinator
            if phase in {1, 2}:
                return sender != coordinator
            return True
        return False

    def observe(self, sender: int, text: str | None, step: int) -> None:
        if not text:
            return
        self.nonempty += 1
        message = parse_protocol_message(text)
        if message is None:
            return
        self.parsed += 1
        if self._schedule_ok(sender, message, step):
            self.schedule_valid += 1
        else:
            self.schedule_violations += 1
        self.counts[message.kind.lower()] += 1
        event = {"sender": sender, "step": step, **message.as_dict()}
        self.events.append(event)
        if message.kind == "PROPOSE":
            members = {int(value) for value in message.fields["MEMBERS"].split(",")}
            self.proposals.setdefault(
                message.task_id,
                {"sender": sender, "step": step, "members": members},
            )
            self.votes.setdefault(message.task_id, {})[sender] = "YES"
        elif message.kind == "VOTE":
            self.votes.setdefault(message.task_id, {})[sender] = message.fields["VOTE"].upper()
        elif message.kind == "BID":
            self.bids.setdefault(message.task_id, {})[sender] = int(message.fields["SCORE"])
        elif message.kind == "AWARD":
            self.awards.append(event)
        elif message.kind == "STATUS":
            self.status_windows.setdefault(step // 5, set()).add(sender)

    def as_dict(self, num_steps: int) -> dict[str, Any]:
        agreements = 0
        agreed_ids: set[str] = set()
        latencies: list[int] = []
        for task_id, proposal in self.proposals.items():
            votes = self.votes.get(task_id, {})
            if proposal["members"] and all(votes.get(member) == "YES" for member in proposal["members"]):
                agreements += 1
                agreed_ids.add(task_id)
                vote_steps = [event["step"] for event in self.events if event["id"] == task_id and event["type"] == "VOTE"]
                latencies.append(max(vote_steps, default=proposal["step"]) - proposal["step"])
        valid_awards = 0
        accepted_awards = 0
        for award in self.awards:
            bids = self.bids.get(award["id"], {})
            if bids:
                winner = min(bids, key=lambda agent: (-bids[agent], agent))
                if winner == int(award["ASSIGNEE"]):
                    valid_awards += 1
            if any(
                event["type"] == "ACCEPT" and event["id"] == award["id"]
                and event["sender"] == int(award["ASSIGNEE"])
                for event in self.events
            ):
                accepted_awards += 1
        commits = [event for event in self.events if event["type"] == "COMMIT"]
        valid_commits = [event for event in commits if event["id"] in agreed_ids]
        committed_actions: dict[str, set[str]] = {}
        for event in valid_commits:
            committed_actions.setdefault(event["id"], set()).add(event["ACTION"].upper())
        coherent_commitments = sum(
            len(actions) == 1 for actions in committed_actions.values()
        )
        accepts = [event for event in self.events if event["type"] == "ACCEPT"]
        awarded_pairs = {
            (event["id"], int(event["ASSIGNEE"])) for event in self.awards
        }
        orphan_accepts = sum(
            (event["id"], event["sender"]) not in awarded_pairs for event in accepts
        )
        expected_windows = max(1, (max(num_steps, 1) + 4) // 5)
        status_coverage = sum(
            len(self.status_windows.get(window, set())) / max(self.num_agents, 1)
            for window in range(expected_windows)
        ) / expected_windows
        return {
            "strategy": self.strategy,
            "nonempty_communications": self.nonempty,
            "protocol_messages": self.parsed,
            "protocol_parse_rate": round(self.parsed / max(self.nonempty, 1), 4),
            "schedule_valid_messages": self.schedule_valid,
            "schedule_violations": self.schedule_violations,
            "schedule_adherence_rate": round(
                self.schedule_valid / max(self.parsed, 1), 4
            ),
            **self.counts,
            "agreements": agreements,
            "proposal_agreement_rate": round(agreements / max(len(self.proposals), 1), 4),
            "mean_agreement_latency_ticks": round(sum(latencies) / max(len(latencies), 1), 3),
            "valid_awards": valid_awards,
            "award_validity_rate": round(valid_awards / max(len(self.awards), 1), 4),
            "accepted_awards": accepted_awards,
            "award_acceptance_rate": round(accepted_awards / max(len(self.awards), 1), 4),
            "valid_commits": len(valid_commits),
            "commit_validity_rate": round(len(valid_commits) / max(len(commits), 1), 4),
            "coherent_commitments": coherent_commitments,
            "commit_action_coherence_rate": round(
                coherent_commitments / max(len(committed_actions), 1), 4
            ),
            "orphan_accepts": orphan_accepts,
            "status_window_coverage": round(status_coverage, 4),
        }
