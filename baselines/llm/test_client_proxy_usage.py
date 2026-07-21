"""Tests for provider-call capture across action-format retries."""

from types import SimpleNamespace

from baselines.llm.eval_utils.agents.base import _ClientProxy
from baselines.llm.eval_utils.agents.robust_all import _extract_safe_action
from baselines.llm.eval_utils.client import ModelResponse
from baselines.llm.eval_utils.prompt_builder import Message


class _SequenceClient:
    def __init__(self, completions):
        self.completions = iter(completions)

    def generate(self, messages):
        return ModelResponse(
            model_id="gpt-test",
            completion=next(self.completions),
            stop_reason="stop",
            input_tokens=10,
            output_tokens=2,
        )


def test_proxy_retains_every_semantic_attempt_then_resets_for_next_call():
    proxy = _ClientProxy(_SequenceClient(["invalid", "<action>Noop</action>", "done"]))
    messages = [Message(role="user", content="choose")]

    _, last, extracted, retries = proxy.generate_with_validation(
        messages,
        validate_fn=lambda response: (
            "Noop" if "<action>Noop</action>" in response.completion else None
        ),
        error_message="retry",
        max_parse_retries=1,
    )

    assert last.completion == "<action>Noop</action>"
    assert extracted == "Noop"
    assert retries == 1
    assert [response.completion for response in proxy.last_call_responses] == [
        "invalid",
        "<action>Noop</action>",
    ]

    proxy.generate([SimpleNamespace(role="user", content="debrief")])
    assert [response.completion for response in proxy.last_call_responses] == ["done"]


def test_filtered_partial_action_is_not_executable():
    response = ModelResponse(
        model_id="gpt-test",
        completion="<action>Move North</action>",
        stop_reason="content_filter",
        input_tokens=10,
        output_tokens=2,
    )

    assert _extract_safe_action(response) is None
