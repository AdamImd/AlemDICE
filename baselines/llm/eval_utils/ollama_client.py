"""Native Ollama transport adapter for the Alem LLM evaluation harness."""

from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from ollama import Client
except ImportError:
    Client = None

from .client import LLMClientWrapper, LLMResponse, _classify_chat_completion_incomplete

_OPTION_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "num_ctx",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
)


def _validate_native_host(base_url: str | None) -> str:
    """Return a normalized Ollama host URL and reject OpenAI-compatible paths."""

    host = str(base_url or "").strip().rstrip("/")
    if not host:
        raise ValueError("base_url must be provided when using the Ollama client")

    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Ollama base_url must be an HTTP(S) host, for example http://127.0.0.1:11434"
        )
    if parsed.path not in {"", "/"}:
        raise ValueError(
            "Ollama native base_url must not include an API path such as /v1; "
            "use the server root, for example http://127.0.0.1:11434"
        )
    return host


def _attachment_bytes(attachment: Any) -> bytes | str | Path:
    """Convert PIL-style attachments while retaining SDK-supported input types."""

    if isinstance(attachment, (bytes, str, Path)):
        return attachment
    if not hasattr(attachment, "save"):
        raise TypeError(
            "Ollama message attachments must be image bytes, a path, or a PIL-style image"
        )

    buffer = BytesIO()
    attachment.save(buffer, format="PNG")
    return buffer.getvalue()


class OllamaWrapper(LLMClientWrapper):
    """Generate responses through Ollama's native ``/api/chat`` endpoint."""

    def __init__(self, client_config: Any, *, sdk_client: Any | None = None):
        if Client is None and sdk_client is None:
            raise ImportError(
                "ollama package is required. Install with: pip install 'ollama>=0.6.2,<0.7'"
            )
        super().__init__(client_config)
        self.host = _validate_native_host(self.base_url)
        self.client = sdk_client
        self._initialized = sdk_client is not None

    def _initialize_client(self) -> None:
        if self._initialized:
            return
        self.client = Client(host=self.host, timeout=self.timeout)
        self._initialized = True

    def convert_messages(self, messages) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            content = str(message.content or "")
            images = []
            attachment = getattr(message, "attachment", None)
            if attachment is not None:
                images.append(_attachment_bytes(attachment))

            if self.alternate_roles and converted and converted[-1]["role"] == message.role:
                if content:
                    previous = converted[-1].get("content", "")
                    converted[-1]["content"] = f"{previous}\n{content}" if previous else content
                if images:
                    converted[-1].setdefault("images", []).extend(images)
                continue

            converted_message: dict[str, Any] = {
                "role": message.role,
                "content": content,
            }
            if images:
                converted_message["images"] = images
            converted.append(converted_message)
        return converted

    def _options(self) -> dict[str, Any]:
        max_tokens = self.client_kwargs.get(
            "max_tokens", self.client_kwargs.get("max_completion_tokens", 2048)
        )
        options: dict[str, Any] = {"num_predict": int(max_tokens)}
        for key in _OPTION_KEYS:
            value = self.client_kwargs.get(key)
            if value is not None:
                options[key] = value
        return options

    def generate(self, messages) -> LLMResponse:
        self._initialize_client()
        converted_messages = self.convert_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": converted_messages,
            "stream": False,
            "think": bool(self.enable_thinking),
            "options": self._options(),
        }
        keep_alive = self.client_kwargs.get("keep_alive")
        if keep_alive is not None:
            request_kwargs["keep_alive"] = keep_alive

        started = time.perf_counter()

        def api_call():
            response = self.client.chat(**request_kwargs)
            if response is None:
                raise RuntimeError("Ollama API returned no response")
            if getattr(response, "message", None) is None:
                raise RuntimeError(f"Ollama response is missing message: {response!r}")
            return response

        response = self.execute_with_retries(api_call)
        latency_seconds = time.perf_counter() - started

        message = response.message
        completion = getattr(message, "content", None)
        stop_reason = getattr(response, "done_reason", None)
        if completion is None:
            if stop_reason in {None, "stop", "length"}:
                completion = ""
            else:
                raise RuntimeError(f"Ollama response is missing content: {response!r}")

        reasoning = getattr(message, "thinking", None)
        input_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
        output_tokens = int(getattr(response, "eval_count", 0) or 0)
        incomplete_reason = _classify_chat_completion_incomplete(
            stop_reason=stop_reason,
            completion_text=completion,
            reasoning_tokens=0,
            output_tokens=output_tokens,
        )

        return LLMResponse(
            model_id=str(getattr(response, "model", None) or self.model_id),
            completion=str(completion).strip(),
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning=str(reasoning).strip() if reasoning else None,
            reasoning_tokens=0,
            status="completed" if getattr(response, "done", True) else "incomplete",
            incomplete_reason=incomplete_reason,
            latency_seconds=latency_seconds,
        )
