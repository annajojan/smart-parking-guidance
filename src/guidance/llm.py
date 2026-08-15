"""LLM-based guidance via any OpenAI-compatible chat-completions endpoint.

Works with hosted APIs (OpenAI, Azure, Groq, ...) and local servers such as
Ollama or llama.cpp. Configuration is read from environment variables or a
``.env`` file:

- ``OPENAI_API_KEY``  (or ``LLM_API_KEY``)
- ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``)
- ``OPENAI_MODEL``    (default ``gpt-4o-mini``)

If the call fails for any reason (no key, offline, timeout) the generator
gracefully falls back to the rule-based generator.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import BaseGuidance, GuidanceResult
from .rule_based import RuleBasedGuidance

_SYSTEM_PROMPT = (
    "You are an embedded parking guidance assistant in a smart-parking system. "
    "You receive structured, real-time parking occupancy data. Reply with clear, "
    "friendly, step-by-step directions to the best available parking spot. Use "
    "the slot id, zone and entrance info. Keep it to 2-4 short sentences. "
    "Never invent slots that are not listed."
)


class LLMGuidance(BaseGuidance):
    """Generate parking directions using an OpenAI-compatible LLM."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 45,
        fallback: BaseGuidance | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.fallback = fallback or RuleBasedGuidance()

    def generate(self, occupancy, parking_map) -> GuidanceResult:
        try:
            text = self._call(occupancy, parking_map)
            return GuidanceResult(text=text, source="llm")
        except Exception as exc:  # noqa: BLE001 - any failure degrades gracefully
            fallback = self.fallback.generate(occupancy, parking_map)
            return GuidanceResult(
                text=fallback.text,
                source=f"rule (LLM unavailable: {type(exc).__name__})",
            )

    def _call(self, occupancy, parking_map) -> str:
        context = {
            "parking_name": parking_map.name,
            "floors": parking_map.floors,
            "entrances": parking_map.entrances,
            "occupancy": occupancy.to_dict(),
        }
        user_prompt = (
            "Here is the real-time parking data as JSON:\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Guide the driver to the best free parking spot with friendly, "
            "clear directions."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
