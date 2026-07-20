"""Focused tests for the OpenAI Responses transport adapter."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from baselines.llm.eval_utils import openai_responses
from baselines.llm.eval_utils.agents.robust_all import _extract_safe_action
from baselines.llm.eval_utils.client import create_llm_client
from baselines.llm.eval_utils.openai_responses import OpenAIResponsesWrapper


def _config(*, model_id="gpt-5.6-luna", max_retries=2, **generate_kwargs):
    defaults = {
        "max_output_tokens": 321,
        "reasoning_effort": "none",
    }
    defaults.update(generate_kwargs)
    return SimpleNamespace(
        client_name="openai_responses",
        model_id=model_id,
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


def _response(
    *,
    output_text="Plan briefly.\n<action>Move North</action>",
    status="completed",
    incomplete_reason=None,
):
    return SimpleNamespace(
        id="resp_test_123",
        model="gpt-5.6-luna-2026-07-01",
        output_text=output_text,
        status=status,
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
        ),
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=17,
            input_tokens_details=SimpleNamespace(
                cached_tokens=80,
                cache_write_tokens=32,
            ),
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


def test_maps_tagged_text_request_and_extracts_usage():
    config = _config(
        prompt_cache_key="alem-v1-gpt56",
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        prompt_cache_retention="24h",
    )
    sdk = _FakeSDKClient([_response()])
    client = OpenAIResponsesWrapper(config, sdk_client=sdk)

    result = client.generate(
        [_message("system", "Shared Alem rules"), _message("user", "Current observation")]
    )

    assert len(sdk.responses.calls) == 1
    request = sdk.responses.calls[0]
    assert request["model"] == "gpt-5.6-luna"
    assert request["max_output_tokens"] == 321
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "none"}
    assert request["prompt_cache_key"] == "alem-v1-gpt56:traffic-0"
    assert request["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert request["prompt_cache_retention"] == "24h"
    assert "prompt_cache_traffic_shards" not in request
    assert request["input"][0] == {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": "Shared Alem rules",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ],
    }
    assert request["input"][1]["content"][0]["text"] == "Current observation"

    assert result.completion == "Plan briefly.\n<action>Move North</action>"
    assert result.stop_reason == "stop"
    assert result.model_id == "gpt-5.6-luna-2026-07-01"
    assert result.response_id == "resp_test_123"
    assert result.status == "completed"
    assert result.incomplete_reason is None
    assert result.input_tokens == 120
    assert result.output_tokens == 17
    assert result.reasoning_tokens == 5
    assert result.cached_tokens == 80
    assert result.cache_write_tokens == 32
    assert result.latency_seconds >= 0
    assert result.transport_attempt_count == 1
    assert result.transport_error_count == 0
    # The model's visible rationale remains in tagged output; private reasoning
    # is neither requested nor exposed as a separate text field.
    assert result.reasoning is None


def test_client_construction_round_robins_stable_cache_traffic_keys():
    config = _config(
        prompt_cache_key="alem-traffic-test:role-warrior",
        prompt_cache_traffic_shards=3,
    )

    def construct_client(_):
        client = OpenAIResponsesWrapper(config, sdk_client=_FakeSDKClient([]))
        return client.effective_prompt_cache_key

    with ThreadPoolExecutor(max_workers=12) as executor:
        keys = list(executor.map(construct_client, range(30)))

    assert Counter(keys) == {
        "alem-traffic-test:role-warrior:traffic-0": 10,
        "alem-traffic-test:role-warrior:traffic-1": 10,
        "alem-traffic-test:role-warrior:traffic-2": 10,
    }


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "many"])
def test_cache_traffic_shards_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="prompt_cache_traffic_shards"):
        OpenAIResponsesWrapper(
            _config(prompt_cache_traffic_shards=value),
            sdk_client=_FakeSDKClient([]),
        )


def test_gpt54_uses_automatic_cache_routing_without_explicit_breakpoints():
    config = _config(
        model_id="gpt-5.4-nano",
        prompt_cache_key="alem-gpt54",
        prompt_cache_traffic_shards=3,
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        prompt_cache_retention="24h",
    )
    sdk = _FakeSDKClient([_response()])
    client = OpenAIResponsesWrapper(config, sdk_client=sdk)

    client.generate([_message("system", "Rules"), _message("user", "Observation")])

    request = sdk.responses.calls[0]
    assert request["prompt_cache_key"] == "alem-gpt54:traffic-0"
    assert "prompt_cache_options" not in request
    assert request["prompt_cache_retention"] == "24h"
    assert "prompt_cache_traffic_shards" not in request
    assert "prompt_cache_breakpoint" not in request["input"][0]["content"][0]


def test_baseline_message_order_is_preserved_without_explicit_breakpoint():
    sdk = _FakeSDKClient([_response()])
    client = OpenAIResponsesWrapper(_config(prompt_cache_key="alem-upstream-main"), sdk_client=sdk)

    client.generate(
        [
            _message("system", "Role-specific rules"),
            _message("user", "Observation zero"),
            _message("assistant", "Previous tagged action"),
            _message("user", "Current observation"),
        ]
    )

    request = sdk.responses.calls[0]
    assert [message["role"] for message in request["input"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message["content"][0]["text"] for message in request["input"]] == [
        "Role-specific rules",
        "Observation zero",
        "Previous tagged action",
        "Current observation",
    ]
    assert [message["content"][0]["type"] for message in request["input"]] == [
        "input_text",
        "input_text",
        "output_text",
        "input_text",
    ]
    assert "prompt_cache_breakpoint" not in request["input"][0]["content"][0]


def test_incomplete_response_preserves_detail_and_legacy_length_reason():
    sdk = _FakeSDKClient(
        [_response(output_text="", status="incomplete", incomplete_reason="max_output_tokens")]
    )
    client = OpenAIResponsesWrapper(_config(), sdk_client=sdk)

    result = client.generate([_message("user", "Observation")])

    assert result.status == "incomplete"
    assert result.incomplete_reason == "max_output_tokens"
    assert result.stop_reason == "length"
    assert result.completion == ""


def test_completed_refusal_is_visible_and_classified():
    response = _response(output_text="")
    response.output = [
        SimpleNamespace(
            content=[SimpleNamespace(type="refusal", refusal="I cannot take that action.")]
        )
    ]
    sdk = _FakeSDKClient([response])
    client = OpenAIResponsesWrapper(_config(), sdk_client=sdk)

    result = client.generate([_message("user", "Observation")])

    assert result.completion == "I cannot take that action."
    assert result.incomplete_reason == "refusal"
    assert result.stop_reason == "content_filter"


def test_retries_transient_error_then_returns_response():
    sdk = _FakeSDKClient([_APIError("rate limited by input_tokens", 429), _response()])
    client = OpenAIResponsesWrapper(_config(max_retries=2), sdk_client=sdk)

    result = client.generate([_message("user", "Observation")])

    assert result.stop_reason == "stop"
    assert len(sdk.responses.calls) == 2
    assert result.transport_attempt_count == 2
    assert result.transport_error_count == 1


def test_does_not_retry_nonretryable_client_error():
    error = _APIError("bad request", 400)
    sdk = _FakeSDKClient([error, _response()])
    client = OpenAIResponsesWrapper(_config(max_retries=3), sdk_client=sdk)

    with pytest.raises(_APIError) as exc_info:
        client.generate([_message("user", "Observation")])

    assert exc_info.value is error
    assert len(sdk.responses.calls) == 1
    assert client.last_transport_attempt_count == 1
    assert client.last_transport_error_count == 1


@pytest.mark.parametrize(
    "body",
    [
        {"code": "invalid_prompt"},
        {"error": {"code": "invalid_prompt"}},
    ],
)
def test_policy_invalid_prompt_becomes_safe_filtered_response(body):
    error = _APIError("provider policy rejection", 400, body=body)
    sdk = _FakeSDKClient([error, _response()])
    client = OpenAIResponsesWrapper(_config(max_retries=3), sdk_client=sdk)

    result = client.generate([_message("user", "Observation")])

    assert len(sdk.responses.calls) == 1
    assert result.model_id == "gpt-5.6-luna"
    assert result.completion == ""
    assert result.stop_reason == "content_filter"
    assert result.status == "failed"
    assert result.incomplete_reason == "invalid_prompt"
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.transport_attempt_count == 1
    assert result.transport_error_count == 1
    assert result.transport_error_types == ("_APIError",)
    assert _extract_safe_action(result) is None


def test_zero_retry_configuration_still_makes_one_attempt():
    sdk = _FakeSDKClient([_response()])
    client = OpenAIResponsesWrapper(_config(max_retries=0), sdk_client=sdk)

    result = client.generate([_message("user", "Observation")])

    assert result.stop_reason == "stop"
    assert len(sdk.responses.calls) == 1


def test_retries_conflict_and_server_errors():
    sdk = _FakeSDKClient([_APIError("conflict", 409), _APIError("unavailable", 503), _response()])
    client = OpenAIResponsesWrapper(_config(max_retries=2), sdk_client=sdk)

    result = client.generate([_message("user", "Observation")])

    assert len(sdk.responses.calls) == 3
    assert result.transport_attempt_count == 3
    assert result.transport_error_count == 2


def test_exhausted_retries_record_every_transport_attempt():
    sdk = _FakeSDKClient([_APIError("busy", 503) for _ in range(3)])
    client = OpenAIResponsesWrapper(_config(max_retries=2), sdk_client=sdk)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client.generate([_message("user", "Observation")])

    assert len(sdk.responses.calls) == 3
    assert client.last_transport_attempt_count == 3
    assert client.last_transport_error_count == 3


def test_programming_error_is_not_retried():
    sdk = _FakeSDKClient([TypeError("bad local request construction"), _response()])
    client = OpenAIResponsesWrapper(_config(max_retries=3), sdk_client=sdk)

    with pytest.raises(TypeError):
        client.generate([_message("user", "Observation")])

    assert len(sdk.responses.calls) == 1


def test_factory_routes_openai_responses_without_changing_legacy_openai():
    factory = create_llm_client(_config())
    assert isinstance(factory(), OpenAIResponsesWrapper)


def test_sdk_initialization_uses_environment_default_for_api_key(monkeypatch):
    captured = {}
    sdk = _FakeSDKClient([_response()])

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return sdk

    monkeypatch.setattr(openai_responses, "OpenAI", fake_openai)
    client = OpenAIResponsesWrapper(_config())
    client.generate([_message("user", "Observation")])

    assert captured == {"timeout": 30, "max_retries": 0}
    assert "api_key" not in captured
