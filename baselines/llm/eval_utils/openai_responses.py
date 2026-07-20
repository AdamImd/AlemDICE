"""OpenAI Responses API adapter for the Alem tagged-text LLM baseline.

This module intentionally keeps OpenAI transport details outside the agent
implementations.  Agents continue to pass the same ordered ``Message`` list
and consume the legacy-compatible ``LLMResponse`` tuple.
"""

from __future__ import annotations

import base64
import logging
import random
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from .client import LLMClientWrapper, LLMResponse, OpenAI

logger = logging.getLogger(__name__)

_PROMPT_CACHE_TRAFFIC_LOCK = threading.Lock()
_PROMPT_CACHE_TRAFFIC_COUNTERS: dict[tuple[str, int], int] = {}


@dataclass(frozen=True)
class ModelRequest:
    """Provider-neutral request assembled from the baseline message history."""

    model_id: str
    messages: tuple[dict[str, Any], ...]
    max_output_tokens: int
    reasoning_effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    prompt_cache_key: str | None = None
    prompt_cache_options: Mapping[str, Any] | None = None
    prompt_cache_retention: str | None = None
    store: bool = False


def supports_explicit_prompt_caching(model_id: str) -> bool:
    """Return whether a model supports the GPT-5.6 explicit cache controls.

    Unknown and older model families deliberately return false so the adapter
    cannot send newly introduced request fields to endpoints that reject them.
    Dated GPT-5.6 variants and future GPT versions are accepted.
    """

    match = re.search(r"(?:^|/)gpt-(\d+)(?:\.(\d+))?(?:-|$)", model_id.lower())
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return major > 5 or (major == 5 and minor >= 6)


def _as_plain_mapping(value: Any) -> dict[str, Any] | None:
    """Copy ordinary mappings and OmegaConf-like mapping objects safely."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    items = getattr(value, "items", None)
    if callable(items):
        return {str(key): item for key, item in items()}
    raise TypeError("prompt_cache_options must be a mapping")


def _prompt_cache_traffic_shards(value: Any) -> int:
    """Validate the number of stable cache-key traffic shards."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"prompt_cache_traffic_shards must be a positive integer; got {value!r}")
    return value


def _allocate_prompt_cache_key(base_key: Any, traffic_shards: int) -> tuple[str | None, int | None]:
    """Assign a stable key suffix to a newly constructed client.

    The counter is process-local, but the possible key set depends only on the
    configured base key and shard count. Runtime identifiers are never added.
    """

    if base_key is None or not str(base_key).strip():
        return None, None
    normalized_base = str(base_key).strip()
    counter_key = (normalized_base, traffic_shards)
    with _PROMPT_CACHE_TRAFFIC_LOCK:
        current = _PROMPT_CACHE_TRAFFIC_COUNTERS.get(counter_key, 0)
        _PROMPT_CACHE_TRAFFIC_COUNTERS[counter_key] = current + 1
    shard = current % traffic_shards
    return f"{normalized_base}:traffic-{shard}", shard


def _image_data_url(image: Any) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _nested_int(parent: Any, child_name: str, value_name: str) -> int:
    child = getattr(parent, child_name, None) if parent is not None else None
    return int(getattr(child, value_name, 0) or 0)


def _refusal_text(response: Any) -> str | None:
    """Extract a completed refusal that ``response.output_text`` omits."""

    for item in getattr(response, "output", None) or ():
        for part in getattr(item, "content", None) or ():
            if getattr(part, "type", None) == "refusal":
                refusal = getattr(part, "refusal", None)
                if refusal:
                    return str(refusal)
    return None


class OpenAIResponsesWrapper(LLMClientWrapper):
    """Generate tagged-text decisions through ``client.responses.create``."""

    def __init__(self, client_config: Any, *, sdk_client: Any | None = None):
        if OpenAI is None and sdk_client is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        super().__init__(client_config)
        self.prompt_cache_traffic_shards = _prompt_cache_traffic_shards(
            self.client_kwargs.get("prompt_cache_traffic_shards", 1)
        )
        self.prompt_cache_base_key = self.client_kwargs.get("prompt_cache_key")
        (
            self.effective_prompt_cache_key,
            self.prompt_cache_traffic_shard,
        ) = _allocate_prompt_cache_key(
            self.prompt_cache_base_key,
            self.prompt_cache_traffic_shards,
        )
        self.client = sdk_client
        self._initialized = sdk_client is not None

    def _initialize_client(self) -> None:
        if self._initialized:
            return
        # OPENAI_API_KEY is intentionally left to the official SDK's default
        # environment handling.  Disable SDK retries so this adapter owns the
        # bounded retry policy and its accounting remains predictable.
        self.client = OpenAI(timeout=self.timeout, max_retries=0)
        self._initialized = True

    def convert_messages(self, messages: Sequence[Any]) -> tuple[dict[str, Any], ...]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = []
            if message.content:
                # Responses represents prior assistant turns as model output,
                # while system/developer/user turns are model input. The API
                # rejects ``input_text`` inside an assistant message.
                text_type = "output_text" if message.role == "assistant" else "input_text"
                content.append({"type": text_type, "text": message.content})
            if getattr(message, "attachment", None) is not None:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(message.attachment),
                        "detail": "auto",
                    }
                )

            if self.alternate_roles and converted and converted[-1]["role"] == message.role:
                converted[-1]["content"].extend(content)
            else:
                converted.append({"role": message.role, "content": content})
        return tuple(converted)

    def build_request(self, messages: Sequence[Any]) -> ModelRequest:
        """Map the legacy message/config interface into a transport-neutral request."""

        max_output_tokens = self.client_kwargs.get(
            "max_output_tokens",
            self.client_kwargs.get(
                "max_completion_tokens", self.client_kwargs.get("max_tokens", 2048)
            ),
        )
        options = _as_plain_mapping(self.client_kwargs.get("prompt_cache_options"))
        return ModelRequest(
            model_id=self.model_id,
            messages=self.convert_messages(messages),
            max_output_tokens=int(max_output_tokens),
            reasoning_effort=self.client_kwargs.get("reasoning_effort"),
            temperature=self.client_kwargs.get("temperature"),
            top_p=self.client_kwargs.get("top_p"),
            prompt_cache_key=self.effective_prompt_cache_key,
            prompt_cache_options=options,
            prompt_cache_retention=self.client_kwargs.get("prompt_cache_retention"),
            store=bool(self.client_kwargs.get("store", False)),
        )

    @staticmethod
    def _mark_shared_prefix(messages: list[dict[str, Any]]) -> None:
        """Put one explicit breakpoint at the end of the first system prompt."""

        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content") or []
            if content:
                content[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
            return

    def request_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        """Convert a ModelRequest into supported Responses API parameters."""

        input_messages = [
            {"role": message["role"], "content": [dict(part) for part in message["content"]]}
            for message in request.messages
        ]
        kwargs: dict[str, Any] = {
            "model": request.model_id,
            "input": input_messages,
            "max_output_tokens": request.max_output_tokens,
            "store": request.store,
        }
        if request.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": request.reasoning_effort}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if supports_explicit_prompt_caching(request.model_id):
            if request.prompt_cache_key:
                kwargs["prompt_cache_key"] = request.prompt_cache_key
            if request.prompt_cache_retention:
                kwargs["prompt_cache_retention"] = request.prompt_cache_retention
            if request.prompt_cache_options:
                options = dict(request.prompt_cache_options)
                kwargs["prompt_cache_options"] = options
                if options.get("mode") == "explicit":
                    self._mark_shared_prefix(input_messages)

        return kwargs

    def _safe_retry(self, func):
        """Run a request with bounded backoff without logging payloads or secrets."""

        attempts = 0
        errors = 0
        error_types: list[str] = []
        last_exception: Exception | None = None
        attempt_limit = max(0, int(self.max_retries)) + 1
        while attempts < attempt_limit:
            attempts += 1
            try:
                result = func()
                self.last_transport_attempt_count = attempts
                self.last_transport_error_count = errors
                self.last_transport_error_types = tuple(error_types)
                return result
            except Exception as exc:
                status_code = self._status_code_from_exception(exc)
                request_id = getattr(exc, "request_id", None)
                errors += 1
                error_types.append(type(exc).__name__)
                retryable = (
                    status_code in {408, 409, 429}
                    or isinstance(status_code, int)
                    and status_code >= 500
                    or isinstance(exc, (ConnectionError, TimeoutError))
                    or type(exc).__name__
                    in {"APIConnectionError", "APITimeoutError", "RateLimitError"}
                )
                self.last_transport_attempt_count = attempts
                self.last_transport_error_count = errors
                self.last_transport_error_types = tuple(error_types)
                if not retryable:
                    logger.error(
                        "Non-retryable OpenAI Responses error type=%s status=%s request_id=%s",
                        type(exc).__name__,
                        status_code,
                        request_id,
                    )
                    raise
                last_exception = exc
                logger.warning(
                    "Retryable OpenAI Responses error type=%s status=%s request_id=%s "
                    "attempt=%s/%s",
                    type(exc).__name__,
                    status_code,
                    request_id,
                    attempts,
                    attempt_limit,
                )
                if attempts < attempt_limit:
                    retry_after = None
                    response = getattr(exc, "response", None)
                    headers = getattr(response, "headers", None)
                    if headers is not None:
                        retry_after = headers.get("retry-after")
                    try:
                        retry_after_seconds = float(retry_after)
                    except (TypeError, ValueError):
                        retry_after_seconds = None
                    delay = (
                        retry_after_seconds
                        if retry_after_seconds is not None
                        else self.delay * (2 ** (attempts - 1)) * random.uniform(0.8, 1.2)
                    )
                    if delay > 0:
                        time.sleep(delay)

        raise RuntimeError(
            f"OpenAI Responses request failed after {attempt_limit} attempts"
        ) from last_exception

    @staticmethod
    def _incomplete_reason(response: Any) -> str | None:
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) if details is not None else None
        return str(reason) if reason is not None else None

    @staticmethod
    def _legacy_stop_reason(status: str | None, incomplete_reason: str | None) -> str | None:
        if incomplete_reason == "max_output_tokens":
            return "length"
        if incomplete_reason in {"content_filter", "refusal"}:
            return "content_filter"
        if status == "completed":
            return "stop"
        return incomplete_reason or status

    def generate(self, messages: Sequence[Any]) -> LLMResponse:
        self._initialize_client()
        self.last_transport_attempt_count = 0
        self.last_transport_error_count = 0
        self.last_transport_error_types = ()
        request = self.build_request(messages)
        kwargs = self.request_kwargs(request)
        started = time.perf_counter()

        def api_call():
            response = self.client.responses.create(**kwargs)
            if response is None:
                raise RuntimeError("OpenAI Responses API returned no response")
            status = getattr(response, "status", None)
            if status in {"failed", "cancelled", "in_progress", "queued"}:
                raise RuntimeError(f"OpenAI Responses API returned status {status}")
            return response

        response = self._safe_retry(api_call)
        latency_seconds = time.perf_counter() - started

        status_value = getattr(response, "status", None)
        status = str(status_value) if status_value is not None else None
        refusal = _refusal_text(response)
        incomplete_reason = self._incomplete_reason(response) or ("refusal" if refusal else None)
        completion = getattr(response, "output_text", None) or refusal or ""
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cached_tokens = _nested_int(usage, "input_tokens_details", "cached_tokens")
        cache_write_tokens = _nested_int(usage, "input_tokens_details", "cache_write_tokens")
        reasoning_tokens = _nested_int(usage, "output_tokens_details", "reasoning_tokens")

        return LLMResponse(
            model_id=str(getattr(response, "model", None) or self.model_id),
            completion=str(completion).strip(),
            stop_reason=self._legacy_stop_reason(status, incomplete_reason),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning=None,
            reasoning_tokens=reasoning_tokens,
            response_id=getattr(response, "id", None),
            status=status,
            incomplete_reason=incomplete_reason,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            latency_seconds=latency_seconds,
            transport_attempt_count=self.last_transport_attempt_count,
            transport_error_count=self.last_transport_error_count,
            transport_error_types=self.last_transport_error_types,
        )
