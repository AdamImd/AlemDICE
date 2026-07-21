"""Focused tests for the native Ollama transport adapter."""

from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from PIL import Image

from baselines.llm.eval_utils import ollama_client
from baselines.llm.eval_utils.client import create_llm_client
from baselines.llm.eval_utils.evaluator import _should_early_stop_on_length
from baselines.llm.eval_utils.ollama_client import OllamaWrapper
from baselines.llm.eval_utils.prompt_builder import Message
from baselines.llm.experiment_config import CONFIG_DIR


def _config(*, base_url="http://127.0.0.1:11434", reasoning=True, **generate_kwargs):
    defaults = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "seed": 42,
        "max_tokens": 321,
        "num_ctx": 4096,
    }
    defaults.update(generate_kwargs)
    return SimpleNamespace(
        client_name="ollama",
        model_id="gemma4:31b",
        base_url=base_url,
        timeout=30,
        generate_kwargs=defaults,
        max_retries=2,
        delay=0,
        alternate_roles=False,
        enable_thinking=reasoning,
    )


def _response(*, content="<action>Noop</action>", thinking="plan", done_reason="stop"):
    return SimpleNamespace(
        model="gemma4:31b",
        message=SimpleNamespace(content=content, thinking=thinking),
        done=True,
        done_reason=done_reason,
        prompt_eval_count=120,
        eval_count=17,
        total_duration=1_500_000_000,
    )


class _FakeClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_factory_routes_explicit_ollama_client(monkeypatch):
    monkeypatch.setattr(ollama_client, "Client", lambda **kwargs: _FakeClient([_response()]))

    client = create_llm_client(_config())()

    assert isinstance(client, OllamaWrapper)
    assert client.host == "http://127.0.0.1:11434"


def test_initializes_official_sdk_with_host_and_timeout(monkeypatch):
    captured = {}

    def make_client(**kwargs):
        captured.update(kwargs)
        return _FakeClient([_response()])

    monkeypatch.setattr(ollama_client, "Client", make_client)
    client = OllamaWrapper(_config())

    result = client.generate([Message("user", "Choose")])

    assert captured == {"host": "http://127.0.0.1:11434", "timeout": 30}
    assert result.completion == "<action>Noop</action>"


@pytest.mark.parametrize(
    ("base_url", "message"),
    [
        (None, "base_url must be provided"),
        ("localhost:11434", "must be an HTTP"),
        ("http://localhost:11434/v1", "must not include an API path"),
    ],
)
def test_rejects_invalid_native_host(base_url, message):
    with pytest.raises(ValueError, match=message):
        OllamaWrapper(_config(base_url=base_url), sdk_client=_FakeClient([]))


def test_maps_native_request_and_response_fields():
    sdk = _FakeClient([_response()])
    client = OllamaWrapper(_config(keep_alive="30m"), sdk_client=sdk)

    result = client.generate(
        [Message("system", "Shared rules"), Message("user", "Current observation")]
    )

    request = sdk.calls[0]
    assert request == {
        "model": "gemma4:31b",
        "messages": [
            {"role": "system", "content": "Shared rules"},
            {"role": "user", "content": "Current observation"},
        ],
        "stream": False,
        "think": True,
        "options": {
            "num_predict": 321,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "seed": 42,
            "num_ctx": 4096,
        },
        "keep_alive": "30m",
    }
    assert result.model_id == "gemma4:31b"
    assert result.completion == "<action>Noop</action>"
    assert result.reasoning == "plan"
    assert result.input_tokens == 120
    assert result.output_tokens == 17
    assert result.stop_reason == "stop"
    assert result.status == "completed"


def test_thinking_can_be_disabled_and_length_is_classified():
    sdk = _FakeClient([_response(content="", thinking=None, done_reason="length")])
    client = OllamaWrapper(_config(reasoning=False), sdk_client=sdk)

    result = client.generate([Message("user", "Choose")])

    assert sdk.calls[0]["think"] is False
    assert result.reasoning is None
    assert result.incomplete_reason == "max_completion_tokens"


def test_converts_images_and_merges_alternating_roles():
    config = _config()
    config.alternate_roles = True
    client = OllamaWrapper(config, sdk_client=_FakeClient([]))
    image = Image.new("RGB", (2, 2), "red")

    converted = client.convert_messages(
        [Message("user", "first", image), Message("user", "second")]
    )

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == "first\nsecond"
    assert converted[0]["images"][0].startswith(b"\x89PNG")


def test_retries_transient_native_failures():
    sdk = _FakeClient([ConnectionError("offline"), _response()])
    client = OllamaWrapper(_config(), sdk_client=sdk)

    result = client.generate([Message("user", "Choose")])

    assert result.completion == "<action>Noop</action>"
    assert len(sdk.calls) == 2


def test_ollama_is_treated_as_self_hosted():
    config = SimpleNamespace(client_name="ollama", model_id="gemma4:31b")

    assert _should_early_stop_on_length(config) is False


def test_gemma_provider_preset_composes_three_native_clients():
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR), version_base="1.1", job_name="ollama_preset_test"
    ):
        config = compose(config_name="config", overrides=["provider=ollama_gemma4_31b"])
        without_thinking = compose(
            config_name="config",
            overrides=["provider=ollama_gemma4_31b", "agent.reasoning=false"],
        )

    assert config.agent.reasoning is True
    assert without_thinking.agent.reasoning is False
    assert len(config.clients) == 3
    assert {client.client_name for client in config.clients} == {"ollama"}
    assert {client.model_id for client in config.clients} == {"gemma4:31b"}
    assert {client.base_url for client in config.clients} == {"http://127.0.0.1:11434"}
    assert {client.generate_kwargs.num_ctx for client in config.clients} == {12288}
