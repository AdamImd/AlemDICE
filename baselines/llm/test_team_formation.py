"""Focused TFP1 parser, formation, privacy, and replay tests."""

import pytest

from baselines.llm.eval_utils.team_formation import (
    RecordKind,
    RecruitmentMethod,
    RecruitmentRecord,
    TaskCard,
    TaskPhase,
    TeamDirectory,
    parse_tfp1,
)


def _record(kind, task_id="task-a", **kwargs):
    return RecruitmentRecord(kind=kind, task_id=task_id, **kwargs)


@pytest.mark.parametrize(
    "record",
    [
        _record(
            RecordKind.ANNOUNCE,
            demand=(70, 70, 0),
            required_size=2,
            reward=100,
            deadline_round=20,
        ),
        _record(RecordKind.CFP),
        _record(RecordKind.APPLY, capabilities=(80, 20, 20), cost=12),
        _record(RecordKind.NOMINATE, members=(0, 1)),
        _record(RecordKind.BID, capabilities=(20, 80, 20), cost=34),
        _record(RecordKind.AWARD, members=(0, 1)),
        _record(RecordKind.ACCEPT, members=(0, 1)),
        _record(RecordKind.DECLINE, members=(0, 1)),
        _record(RecordKind.LOCK, members=(0, 1)),
        _record(RecordKind.CANCEL, reason="obsolete"),
        _record(RecordKind.COMPLETE),
        _record(RecordKind.EXPIRE),
    ],
)
def test_tfp1_round_trip_and_strict_rejections(record):
    rendered = record.render()
    result = parse_tfp1(rendered)

    assert result.valid is True
    assert result.code == "valid"
    assert result.record == record
    assert result.payload_bytes == len(rendered.encode("utf-8"))

    invalid = {
        "": "parse.empty",
        rendered + "\n": "parse.multiline",
        " " + rendered: "parse.outer_whitespace",
        "TFP1|TYPE=CFP|TASK=t|TASK=t": "schema.duplicate_field",
        "TFP1|TASK=t|TYPE=CFP": "schema.noncanonical_order",
        "TFP1|TYPE=CFP|TASK=t|SENDER=4": "schema.unexpected_field",
        "TFP1|TYPE=LOCK|TASK=t|MEMBERS=1,0": "schema.invalid_value",
        "TFP1|TYPE=CANCEL|TASK=t|REASON=" + "x" * 230: "parse.too_many_bytes",
    }
    for raw, expected_code in invalid.items():
        rejected = parse_tfp1(raw)
        assert rejected.valid is False
        assert rejected.record is None
        assert rejected.code == expected_code


def _make_directory(method):
    directory = TeamDirectory(agent_ids=(0, 1, 2), method=method, seed=20000)
    directory.register_task(
        TaskCard(
            task_id="task-a",
            demand=(70, 70, 0),
            required_size=2,
            reward=100,
            deadline_round=25,
            sponsor_id=0,
        )
    )
    directory.register_task(
        TaskCard(
            task_id="task-b",
            demand=(70, 70, 0),
            required_size=2,
            reward=100,
            deadline_round=30,
            sponsor_id=2,
        )
    )
    directory.submit_control(
        sender=2,
        sent_round=-1,
        raw=_record(
            RecordKind.ANNOUNCE,
            task_id="task-expire",
            demand=(70, 70, 0),
            required_size=2,
            reward=50,
            deadline_round=1,
        ).render(),
    )
    announced = directory.advance(0)
    assert any(result.accepted and result.code == "task.announced" for result in announced)
    return directory


def _submit(directory, sender, record):
    return directory.submit_control(
        sender=sender,
        sent_round=directory.current_round,
        raw=record.render(),
    )


def _advance_with_code(directory, code):
    results = directory.advance(directory.current_round + 1)
    matching = [result for result in results if result.code == code]
    assert matching and all(result.accepted for result in matching)
    return results


def _lock_team(directory, method):
    roster = (0, 1)
    if method == RecruitmentMethod.OPEN_VOLUNTEER:
        _submit(
            directory,
            0,
            _record(
                RecordKind.APPLY,
                capabilities=(80, 20, 20),
                cost=10,
            ),
        )
        _submit(
            directory,
            1,
            _record(
                RecordKind.APPLY,
                capabilities=(20, 80, 20),
                cost=20,
            ),
        )
        _advance_with_code(directory, "volunteer.application_recorded")
        for member in roster:
            _submit(
                directory,
                member,
                _record(RecordKind.ACCEPT, members=roster),
            )
        _advance_with_code(directory, "roster.accepted")
    elif method == RecruitmentMethod.MUTUAL_NOMINATION:
        for member in roster:
            _submit(
                directory,
                member,
                _record(RecordKind.NOMINATE, members=roster),
            )
        _advance_with_code(directory, "nomination.recorded")
    else:
        _submit(directory, 0, _record(RecordKind.CFP))
        _advance_with_code(directory, "contract.cfp_opened")
        _submit(
            directory,
            0,
            _record(
                RecordKind.BID,
                capabilities=(80, 20, 20),
                cost=10,
            ),
        )
        _submit(
            directory,
            1,
            _record(
                RecordKind.BID,
                capabilities=(20, 80, 20),
                cost=20,
            ),
        )
        _advance_with_code(directory, "contract.bid_recorded")
        _submit(directory, 0, _record(RecordKind.AWARD, members=roster))
        _advance_with_code(directory, "contract.award_recorded")
        for member in roster:
            _submit(
                directory,
                member,
                _record(RecordKind.ACCEPT, members=roster),
            )
        _advance_with_code(directory, "roster.accepted")

    _submit(directory, 0, _record(RecordKind.LOCK, members=roster))
    _advance_with_code(directory, "team.locked")
    assert directory.tasks["task-a"].phase == TaskPhase.LOCKED
    assert directory.active_team(0) == directory.active_team(1)
    assert directory.active_team(2) is None


@pytest.mark.parametrize("method", list(RecruitmentMethod))
def test_method_state_machine_membership_privacy_and_replay(method):
    directory = _make_directory(method)
    _lock_team(directory, method)

    queued = directory.submit_ordinary(
        sender=0,
        sent_round=directory.current_round,
        content="team-private",
    )
    rejected = directory.submit_ordinary(
        sender=2,
        sent_round=directory.current_round,
        content="not-on-a-team",
    )
    assert queued.accepted and queued.recipients == (1,)
    assert not rejected.accepted and rejected.code == "route.sender_unteamed"

    directory.advance(directory.current_round + 1)
    delivered = directory.deliver_ordinary(directory.current_round)
    assert len(delivered) == 1
    assert delivered[0].accepted
    assert delivered[0].recipients == (1,)
    assert 2 not in delivered[0].recipients
    assert directory.tasks["task-expire"].phase == TaskPhase.EXPIRED

    _submit(directory, 0, _record(RecordKind.COMPLETE))
    if method == RecruitmentMethod.OPEN_VOLUNTEER:
        _submit(
            directory,
            1,
            _record(
                RecordKind.APPLY,
                task_id="task-b",
                capabilities=(20, 80, 20),
                cost=20,
            ),
        )
        expected_busy_round = directory.current_round + 1
    elif method == RecruitmentMethod.MUTUAL_NOMINATION:
        _submit(
            directory,
            1,
            _record(
                RecordKind.NOMINATE,
                task_id="task-b",
                members=(1, 2),
            ),
        )
        expected_busy_round = directory.current_round + 1
    else:
        _submit(directory, 2, _record(RecordKind.CFP, task_id="task-b"))
        expected_busy_round = directory.current_round + 2
    completion_results = directory.advance(directory.current_round + 1)
    assert any(result.code == "completion.reported" for result in completion_results)

    if method == RecruitmentMethod.CONTRACT_NET:
        _submit(
            directory,
            1,
            _record(
                RecordKind.BID,
                task_id="task-b",
                capabilities=(20, 80, 20),
                cost=20,
            ),
        )
        busy_results = directory.advance(directory.current_round + 1)
    else:
        busy_results = completion_results
    assert directory.current_round == expected_busy_round
    assert any(
        not result.accepted
        and result.code in {"state.agent_already_teamed", "state.member_already_teamed"}
        for result in busy_results
    ), [result.as_dict() for result in busy_results]
    assert directory.agent_to_task == {0: "task-a", 1: "task-a"}

    completed = directory.confirm_complete(
        task_id="task-a",
        round_index=directory.current_round,
        evidence_digest="deadbeef",
    )
    assert completed.accepted and completed.code == "task.completed"
    assert directory.agent_to_task == {}

    _submit(
        directory,
        2,
        _record(RecordKind.CANCEL, task_id="task-b", reason="obsolete"),
    )
    _advance_with_code(directory, "task.cancelled")
    assert directory.tasks["task-b"].phase == TaskPhase.CANCELLED

    replay_payload = directory.export_replay()
    replayed = TeamDirectory.replay(replay_payload)
    assert replayed.state_hash() == directory.state_hash()
    assert replayed.audit_chain_hash == directory.audit_chain_hash
