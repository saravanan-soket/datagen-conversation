"""OpenAI-compatible chat-completions client for the Soket / TensorStudio gateway.

A model spec is `provider/model_id` (e.g. `soket/gpt`); a bare id is assumed
to be on the `soket` provider. Only the `soket` provider is wired up — it is
the one every task-bank spec uses.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

DEFAULT_BASE_URL = "https://api.tensorstudio.ai/v1"


class ProviderError(Exception):
    """The model call failed (HTTP error, timeout, gateway error body, empty reply)."""


@dataclass
class Completion:
    message: dict  # the assistant message as returned: content and/or tool_calls
    finish_reason: str
    input_tokens: int
    output_tokens: int
    start: float
    end: float
    reasoning: str = ""
    raw_usage: dict = field(default_factory=dict)


def model_id(spec: str) -> str:
    provider, _, mid = spec.partition("/")
    if not mid:
        return provider
    if provider != "soket":
        raise ValueError(f"unsupported provider {provider!r} in model spec {spec!r} (only soket/...)")
    return mid


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 180.0):
        self.base_url = (base_url or os.environ.get("SOKET_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("SOKET_API_KEY", "")
        self.timeout = timeout
        self.session = requests.Session()
        max_tokens = os.environ.get("CONVGEN_MAX_TOKENS", "").strip()
        self.max_tokens = int(max_tokens) if max_tokens else None
        temperature = os.environ.get("CONVGEN_TEMPERATURE", "").strip()
        self.temperature = float(temperature) if temperature else None

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
             metadata: dict | None = None) -> Completion:
        payload: dict = {"model": model_id(model), "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        if metadata:
            payload["metadata"] = metadata
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        start = time.time()
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise ProviderError(f"request failed: {e}") from e
        end = time.time()

        if resp.status_code >= 400:
            raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(f"non-JSON response: {resp.text[:500]}") from e
        if data.get("error"):
            raise ProviderError(f"gateway error: {data['error']}")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"no choices in response: {str(data)[:500]}")

        choice = choices[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        return Completion(
            message=message,
            finish_reason=choice.get("finish_reason") or "",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            start=start,
            end=end,
            reasoning=reasoning if isinstance(reasoning, str) else "",
            raw_usage=usage,
        )
