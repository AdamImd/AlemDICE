"""Bodyless team-leader planning, routing, and communication metrics."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from .agents.base import _ClientProxy
from .prompt_builder import Message

VALID_TOPOLOGIES = frozenset({"baseline", "leader_peer", "leader_no_peer"})
LEADER_ID = 3


@dataclass(frozen=True)
class TeamAssignment:
    agent_id: int
    subgoal: str
    milestones: tuple[str, ...]


@dataclass(frozen=True)
class TeamPlan:
    objective: str
    assignments: tuple[TeamAssignment, ...]

    def for_agent(self, agent_id: int) -> TeamAssignment:
        return next(item for item in self.assignments if item.agent_id == agent_id)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for assignment in payload["assignments"]:
            assignment["milestones"] = list(assignment["milestones"])
        return payload


@dataclass(frozen=True)
class RouteEnvelope:
    sender: int
    recipients: tuple[int, ...]
    channel: str
    content: str
    sent_step: int
    delivered_step: int


def _extract_tagged(text: str | None, tag: str) -> str | None:
    if not text:
        return None
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}\s*>", text, flags=re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return None


def parse_team_plan(text: str | None, *, num_workers: int = 3) -> TeamPlan | None:
    """Parse and atomically validate one leader plan."""

    tagged = _extract_tagged(text, "team_plan")
    if tagged is None:
        return None
    try:
        payload = json.loads(tagged)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    objective = payload.get("objective")
    assignments = payload.get("assignments")
    if not isinstance(objective, str) or not objective.strip():
        return None
    if not isinstance(assignments, list) or len(assignments) != num_workers:
        return None

    parsed: list[TeamAssignment] = []
    seen: set[int] = set()
    for item in assignments:
        if not isinstance(item, dict):
            return None
        agent_id = item.get("agent_id")
        subgoal = item.get("subgoal")
        milestones = item.get("milestones")
        if isinstance(agent_id, bool) or not isinstance(agent_id, int):
            return None
        if agent_id in seen or agent_id not in range(num_workers):
            return None
        if not isinstance(subgoal, str) or not subgoal.strip():
            return None
        if not isinstance(milestones, list) or not 1 <= len(milestones) <= 5:
            return None
        if not all(isinstance(value, str) and value.strip() for value in milestones):
            return None
        seen.add(agent_id)
        parsed.append(
            TeamAssignment(
                agent_id=agent_id,
                subgoal=subgoal.strip(),
                milestones=tuple(value.strip() for value in milestones),
            )
        )
    if seen != set(range(num_workers)):
        return None
    return TeamPlan(
        objective=objective.strip(), assignments=tuple(sorted(parsed, key=lambda x: x.agent_id))
    )


def fallback_plan(num_workers: int = 3) -> TeamPlan:
    fallback = "Use local judgment, survive, maximize achievements, and report blockers."
    return TeamPlan(
        objective="Survive and maximize team achievements.",
        assignments=tuple(
            TeamAssignment(agent_id=i, subgoal=fallback, milestones=("Report blockers.",))
            for i in range(num_workers)
        ),
    )


def format_assignment(
    plan: TeamPlan, agent_id: int, *, version: int, issued_step: int, review_step: int
) -> str:
    assignment = plan.for_agent(agent_id)
    milestones = "\n".join(f"- {value}" for value in assignment.milestones)
    return (
        "Current Team Leader Assignment:\n"
        f"Plan version: {version}; issued step: {issued_step}; next review: {review_step}\n"
        f"Team objective: {plan.objective}\n"
        f"Your subgoal: {assignment.subgoal}\n"
        f"Milestones:\n{milestones}"
    )


def leader_system_prompt(worker_prompt: str) -> str:
    """Reuse public game rules while removing the embodied role introduction."""

    _, separator, public_rules = worker_prompt.partition("<game_rules>")
    if not separator:
        public_rules = worker_prompt
    else:
        public_rules = separator + public_rules
    return (
        "You are the bodyless Team Leader for three embodied workers in Alem. "
        "You do not exist in the world and cannot act. Plan only from the legal text "
        "views and reports supplied below. Assign complementary work to Agents 0, 1, "
        "and 2; account for their warrior, forager, and miner specializations. Never "
        "invent hidden state.\n\n" + public_rules.strip() + "\n\n<output_format>\n"
        "Return exactly one <team_plan> JSON object with a non-empty objective and "
        "exactly one assignment for each agent_id 0, 1, and 2. Each assignment must "
        "contain a non-empty subgoal and 1-5 non-empty milestone strings. You may add "
        "one private <scratchpad> entry after the plan. Do not output an action tag.\n"
        "</output_format>"
    )


class TeamLeaderAgent:
    """Logical planner that owns a model client but no environment body."""

    def __init__(self, client_factory, *, max_scratchpad_length: int = 1000):
        self.client = _ClientProxy(client_factory())
        self.max_scratchpad_length = max_scratchpad_length
        self.system_prompt = ""
        self.scratchpad = ""
        self.last_raw_completion = ""
        self.plan_calls = 0
        self.valid_plans = 0
        self.invalid_plans = 0

    def reset(self) -> None:
        self.scratchpad = ""
        self.last_raw_completion = ""
        self.plan_calls = 0
        self.valid_plans = 0
        self.invalid_plans = 0

    def set_instruction_prompt(self, worker_prompt: str) -> None:
        self.system_prompt = leader_system_prompt(worker_prompt)

    def plan(
        self,
        *,
        step: int,
        observations: list[dict[str, str]],
        reports: dict[int, list[str]],
        previous_plan: TeamPlan,
        previous_version: int,
    ):
        view_sections = []
        for agent_id, observation in enumerate(observations):
            long_text = observation.get("obs_long_term", "")
            short_text = observation.get("obs_short_term", "")
            view_sections.append(
                f"## Agent {agent_id} legal view\n{long_text}\n\n{short_text}".strip()
            )
        report_lines = []
        for agent_id in range(len(observations)):
            for report in reports.get(agent_id, []):
                report_lines.append(f"- Agent {agent_id}: {report}")
        report_text = "\n".join(report_lines) if report_lines else "(no new reports)"
        previous = json.dumps(previous_plan.as_dict(), ensure_ascii=False)
        scratchpad = self.scratchpad or "(empty)"
        user_prompt = (
            f"Scheduled review at step {step}. Previous plan version: {previous_version}.\n"
            f"Previous valid plan: {previous}\n\n"
            f"Private leader scratchpad: {scratchpad}\n\n"
            "Worker reports since the previous review:\n"
            f"{report_text}\n\n"
            + "\n\n".join(view_sections)
            + "\n\nCreate the next whole-team plan."
        )
        response = self.client.generate(
            [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=user_prompt),
            ]
        )
        self.plan_calls += 1
        self.last_raw_completion = response.completion or ""
        parsed = parse_team_plan(self.last_raw_completion, num_workers=len(observations))
        scratchpad_entry = _extract_tagged(self.last_raw_completion, "scratchpad")
        if scratchpad_entry:
            self.scratchpad = scratchpad_entry[: self.max_scratchpad_length]
        if parsed is None:
            self.invalid_plans += 1
        else:
            self.valid_plans += 1
        return response, parsed


def lexical_features(text: str) -> dict[str, Any]:
    """Return compact paper-aligned lexical communication diagnostics."""

    words = re.findall(r"\b[\w'-]+\b", text.lower())
    sentence_count = len(re.findall(r"[.!?]+(?:\s|$)", text)) or bool(text.strip())
    word_set = set(words)
    starts_imperative = bool(
        words
        and words[0]
        in {
            "go",
            "move",
            "dig",
            "mine",
            "craft",
            "build",
            "bring",
            "give",
            "help",
            "meet",
            "wait",
            "attack",
            "follow",
            "report",
        }
    )
    return {
        "characters": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "words": len(words),
        "sentences": int(sentence_count),
        "multi_sentence": int(sentence_count > 1),
        "imperative": int(starts_imperative),
        "teammate_address": int(
            bool(re.search(r"\b(agent\s*[0-2]|team|warrior|forager|miner)\b", text, re.I))
        ),
        "acknowledgement": int(
            bool(word_set & {"ok", "okay", "yes", "thanks", "understood", "copy"})
        ),
        "question": int("?" in text),
        "politeness": int(bool(word_set & {"please", "thanks", "thank"})),
        "urgency": int(
            bool(word_set & {"urgent", "quick", "quickly", "now", "immediately", "hurry"})
        ),
        "coordinate_reference": int(
            bool(re.search(r"\b[xy]\s*=\s*-?\d+|\(-?\d+\s*,\s*-?\d+\)", text, re.I))
        ),
        "future_plan": int(bool(word_set & {"will", "next", "plan", "going", "later"})),
        "past_memory": int(bool(word_set & {"was", "were", "found", "saw", "previous", "before"})),
        "self_reference": int(bool(word_set & {"i", "i'm", "im", "my", "me"})),
    }


class CommunicationTracker:
    """Track emitted payloads separately from fan-out and prompt injection."""

    def __init__(self):
        self.channels: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
        self.envelopes: list[RouteEnvelope] = []
        self._previous: dict[tuple[str, int], str] = {}

    def eligible(self, channel: str, count: int = 1) -> None:
        self.channels[channel]["eligible_sender_turns"] += count

    def route(
        self,
        *,
        sender: int,
        recipients: Iterable[int],
        channel: str,
        content: str,
        sent_step: int,
        delivered_step: int,
    ) -> RouteEnvelope:
        recipient_tuple = tuple(recipients)
        envelope = RouteEnvelope(
            sender, recipient_tuple, channel, content, sent_step, delivered_step
        )
        self.envelopes.append(envelope)
        stats = self.channels[channel]
        payload_bytes = len(content.encode("utf-8"))
        stats["emitted_messages"] += 1
        stats["payload_characters"] += len(content)
        stats["payload_bytes"] += payload_bytes
        stats["delivered_messages"] += len(recipient_tuple)
        stats["delivery_bytes"] += payload_bytes * len(recipient_tuple)
        features = lexical_features(content)
        for key, value in features.items():
            stats[f"lexical_{key}_sum"] += value
        previous = self._previous.get((channel, sender))
        if previous is not None:
            stats["turnover_pairs"] += 1
            stats["turnover_sequence_similarity_sum"] += SequenceMatcher(
                None, previous, content
            ).ratio()
            old_words = set(re.findall(r"\b\w+\b", previous.lower()))
            new_words = set(re.findall(r"\b\w+\b", content.lower()))
            union = old_words | new_words
            stats["turnover_jaccard_sum"] += len(old_words & new_words) / max(len(union), 1)
            stats["turnover_verbatim"] += int(previous == content)
        self._previous[(channel, sender)] = content
        return envelope

    def prompt_injection(self, content: str, *, channel: str = "leader_assignment_prompt") -> None:
        stats = self.channels[channel]
        stats["prompt_injections"] += 1
        stats["payload_characters"] += len(content)
        stats["payload_bytes"] += len(content.encode("utf-8"))

    def parse_failure(self, channel: str) -> None:
        self.channels[channel]["parse_failures"] += 1

    def as_dict(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for channel, raw in self.channels.items():
            stats = dict(raw)
            pairs = int(stats.get("turnover_pairs", 0))
            if pairs:
                stats["turnover_sequence_similarity_mean"] = (
                    stats.get("turnover_sequence_similarity_sum", 0.0) / pairs
                )
                stats["turnover_jaccard_mean"] = stats.get("turnover_jaccard_sum", 0.0) / pairs
            result[channel] = stats
        return result
