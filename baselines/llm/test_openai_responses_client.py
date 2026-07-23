"""Minimal OpenAI Responses transport contract for Luna experiments."""

from types import SimpleNamespace

import pytest

from baselines.llm.eval_utils import openai_responses
from baselines.llm.eval_utils.agents.robust_all import _extract_safe_action
from baselines.llm.eval_utils.client import create_llm_client
from baselines.llm.eval_utils.openai_responses import OpenAIResponsesWrapper


def _config(*, max_retries=2, **generate_kwargs):
    defaults = {"max_output_tokens": 321, "reasoning_effort": "none"}
    defaults.update(generate_kwargs)
    return SimpleNamespace(
        client_name="openai_responses",
        model_id="gpt-5.6-luna",
        base_url=None,
        timeout=30,
        generate_kwargs=defaults,
        max_retries=max_retries,
        delay=0,
        alternate_roles=False,
        enable_thinking=False,
    )


def _message(role, content):
    return SimpleNamespace(role=role, content=content, attachment=None)


def _response():
    return SimpleNamespace(
        id="resp_test_123",
        model="gpt-5.6-luna-2026-07-01",
        output_text="Plan briefly.\n<action>Move North</action>",
        status="completed",
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=17,
            input_tokens_details=SimpleNamespace(cached_tokens=80, cache_write_tokens=32),
            output_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ),
    )


class _FakeResponses:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _FakeSDKClient:
    def __init__(self, results):
        self.responses = _FakeResponses(results)


class _APIError(RuntimeError):
    def __init__(self, message, status_code, *, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = "req_test"
        self.body = body


def test_maps_tagged_request_cache_fields_and_usage():
    sdk = _FakeSDKClient([_response()])
    client = OpenAIResponsesWrapper(
        _config(
            prompt_cache_key="alem-v1-gpt56",
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
            prompt_cache_retention="24h",
        ),
        sdk_client=sdk,
    )
    result = client.generate(
        [_message("system", "Shared Alem rules"), _message("user", "Observation")]
    )

    request = sdk.responses.calls[0]
    assert request["model"] == "gpt-5.6-luna"
    assert request["max_output_tokens"] == 321
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "none"}
    assert request["prompt_cache_key"] == "alem-v1-gpt56:traffic-0"
    assert request["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert request["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert result.completion.endswith("<action>Move North</action>")
    assert result.response_id == "resp_test_123"
    assert result.input_tokens == 120
    assert result.output_tokens == 17
    assert result.reasoning_tokens == 5
    assert result.cached_tokens == 80
    assert result.cache_write_tokens == 32


def test_policy_invalid_prompt_becomes_non_executable_response():
    for body in ({"code": "invalid_prompt"}, {"error": {"code": "invalid_prompt"}}):
        sdk = _FakeSDKClient([_APIError("provider policy rejection", 400, body=body)])
        result = OpenAIResponsesWrapper(_config(max_retries=3), sdk_client=sdk).generate(
            [_message("user", "Observation")]
        )
        assert result.status == "failed"
        assert result.incomplete_reason == "invalid_prompt"
        assert result.stop_reason == "content_filter"
        assert result.transport_attempt_count == 1
        assert _extract_safe_action(result) is None


def test_exhausted_retries_record_every_transport_attempt():
    sdk = _FakeSDKClient([_APIError("busy", 503) for _ in range(3)])
    client = OpenAIResponsesWrapper(_config(max_retries=2), sdk_client=sdk)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client.generate([_message("user", "Observation")])
    assert len(sdk.responses.calls) == 3
    assert client.last_transport_attempt_count == 3
    assert client.last_transport_error_count == 3


def test_factory_routes_openai_responses(monkeypatch):
    monkeypatch.setattr(openai_responses, "OpenAI", lambda **kwargs: _FakeSDKClient([]))
    assert isinstance(create_llm_client(_config())(), OpenAIResponsesWrapper)
