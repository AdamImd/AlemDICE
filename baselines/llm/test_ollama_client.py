"""Minimal native Ollama transport contract."""

from types import SimpleNamespace

from baselines.llm.eval_utils import ollama_client
from baselines.llm.eval_utils.client import create_llm_client
from baselines.llm.eval_utils.ollama_client import OllamaWrapper
from baselines.llm.eval_utils.prompt_builder import Message


def _config():
    return SimpleNamespace(
        client_name="ollama",
        model_id="gemma4:31b",
        base_url="http://127.0.0.1:11434",
        timeout=30,
        generate_kwargs={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "seed": 42,
            "max_tokens": 321,
            "num_ctx": 4096,
            "keep_alive": "30m",
        },
        max_retries=2,
        delay=0,
        alternate_roles=False,
        enable_thinking=True,
    )


def _response():
    return SimpleNamespace(
        model="gemma4:31b",
        message=SimpleNamespace(content="<action>Noop</action>", thinking="plan"),
        done=True,
        done_reason="stop",
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


def test_maps_native_request_and_response_fields():
    sdk = _FakeClient([_response()])
    result = OllamaWrapper(_config(), sdk_client=sdk).generate(
        [Message("system", "Shared rules"), Message("user", "Current observation")]
    )

    assert sdk.calls[0] == {
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


def test_retries_transient_native_failures():
    sdk = _FakeClient([ConnectionError("offline"), _response()])
    result = OllamaWrapper(_config(), sdk_client=sdk).generate([Message("user", "Choose")])
    assert result.completion == "<action>Noop</action>"
    assert len(sdk.calls) == 2
