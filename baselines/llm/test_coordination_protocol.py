"""Focused tests for the decentralized coordination treatment."""

from eval_utils.coordination_protocol import (
    CoordinationLedger,
    CoordinationMetrics,
    parse_protocol_message,
)


def test_parser_accepts_valid_vote_and_rejects_malformed_records():
    message = parse_protocol_message("DCP1|TYPE=VOTE|ID=wood1|VOTE=NO")
    assert message is not None
    assert message.kind == "VOTE"
    assert message.task_id == "wood1"
    assert parse_protocol_message("DCP1|TYPE=VOTE|ID=wood1|VOTE=MAYBE") is None
    assert parse_protocol_message("DCP1|TYPE=BID|ID=wood1|ROLE=SUPPLY|SCORE=101") is None
    assert parse_protocol_message("ordinary free-form message") is None
    assert (
        parse_protocol_message(
            "DCP1|TYPE=VOTE|ID=x|VOTE=YES\nDCP1|TYPE=VOTE|ID=x|VOTE=NO"
        )
        is None
    )


def test_ledger_is_local_bounded_and_deduplicated():
    ledger = CoordinationLedger(agent_id=1, max_events=2)
    first = "DCP1|TYPE=PROPOSE|ID=a|TASK=get_wood|MEMBERS=0,1,2"
    ledger.record(0, first, 0)
    ledger.record(0, first, 0)
    ledger.record(2, "DCP1|TYPE=VOTE|ID=a|VOTE=NO", 1)
    ledger.record(1, "DCP1|TYPE=CANCEL|ID=a|STATE=blocked", 2)
    assert len(ledger.events) == 2
    assert "VOTE a" in ledger.render(3)
    assert "CANCEL a" in ledger.render(3)


def test_metrics_require_explicit_unanimity_and_validate_awards():
    metrics = CoordinationMetrics("integrated", 3)
    metrics.observe(0, "DCP1|TYPE=PROPOSE|ID=a|TASK=get_wood|MEMBERS=0,1,2", 0)
    metrics.observe(1, "DCP1|TYPE=VOTE|ID=a|VOTE=YES", 1)
    partial = metrics.as_dict(2)
    assert partial["agreements"] == 0
    metrics.observe(2, "DCP1|TYPE=VOTE|ID=a|VOTE=YES", 1)
    metrics.observe(0, "DCP1|TYPE=BID|ID=r|ROLE=SUPPLY|SCORE=70", 2)
    metrics.observe(1, "DCP1|TYPE=BID|ID=r|ROLE=SUPPLY|SCORE=80", 2)
    metrics.observe(2, "DCP1|TYPE=AWARD|ID=r|ROLE=SUPPLY|ASSIGNEE=1|LEASE=5", 3)
    metrics.observe(1, "DCP1|TYPE=ACCEPT|ID=r|ROLE=SUPPLY", 4)
    final = metrics.as_dict(5)
    assert final["agreements"] == 1
    assert final["mean_agreement_latency_ticks"] == 1
    assert final["valid_awards"] == 1
    assert final["accepted_awards"] == 1
    assert final["orphan_accepts"] == 0


def test_metrics_reject_orphan_accept_and_incoherent_commits():
    metrics = CoordinationMetrics("integrated", 3)
    metrics.observe(0, "DCP1|TYPE=PROPOSE|ID=x|TASK=sync|MEMBERS=0,1,2", 0)
    metrics.observe(1, "DCP1|TYPE=VOTE|ID=x|VOTE=YES", 1)
    metrics.observe(2, "DCP1|TYPE=VOTE|ID=x|VOTE=YES", 1)
    metrics.observe(0, "DCP1|TYPE=COMMIT|ID=x|ACTION=Do", 2)
    metrics.observe(1, "DCP1|TYPE=COMMIT|ID=x|ACTION=Move", 2)
    metrics.observe(2, "DCP1|TYPE=ACCEPT|ID=missing|ROLE=SCOUT", 2)
    result = metrics.as_dict(3)
    assert result["valid_commits"] == 2
    assert result["commit_action_coherence_rate"] == 0.0
    assert result["orphan_accepts"] == 1


def test_metrics_report_public_schedule_adherence():
    metrics = CoordinationMetrics("roles", 3)
    metrics.observe(0, "DCP1|TYPE=PROPOSE|ID=R0|TASK=wood|MEMBERS=0,1,2", 0)
    metrics.observe(1, "DCP1|TYPE=BID|ID=R0|ROLE=SUPPLY|SCORE=80", 1)
    metrics.observe(0, "DCP1|TYPE=STATUS|ID=R0|ROLE=SYNC|STATE=early", 1)
    result = metrics.as_dict(2)
    assert result["schedule_valid_messages"] == 2
    assert result["schedule_violations"] == 1
    assert result["schedule_adherence_rate"] == 0.6667
