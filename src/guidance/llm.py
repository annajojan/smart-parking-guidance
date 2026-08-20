"""LLM-based guidance via a LOCAL Ollama server.

Uses Ollama (https://ollama.com) for text generation. No cloud API keys
are required. Configuration is read from environment variables or a ``.env``
file:

- ``OLLAMA_BASE_URL`` (default ``http://localhost:11434/v1``)
- ``OLLAMA_MODEL``    (default ``llama3.1``)

If the call fails for any reason (Ollama not running, model not pulled, timeout)
the generator gracefully falls back to the rule-based generator.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import BaseGuidance, GuidanceResult
from .rule_based import RuleBasedGuidance


class OllamaGuidance(BaseGuidance):
    """Generate parking directions using a local Ollama LLM."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
        fallback: BaseGuidance | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.timeout = timeout
        self.fallback = fallback or RuleBasedGuidance()

    def generate(self, occupancy, parking_map) -> GuidanceResult:
        try:
            text = self._call(occupancy, parking_map)
            return GuidanceResult(text=text, source="llm (Ollama)")
        except Exception as exc:  # noqa: BLE001 - any failure degrades gracefully
            fallback = self.fallback.generate(occupancy, parking_map)
            return GuidanceResult(
                text=fallback.text,
                source=f"rule (LLM unavailable: {type(exc).__name__})",
            )

    def _call(self, occupancy, parking_map) -> str:
        available = [s for s in occupancy.slots if s.status == "available"]

        if not available:
            result = self.fallback.generate(occupancy, parking_map)
            return result.text

        slot_lines = []
        for s in occupancy.slots:
            status_str = "OCCUPIED" if s.status == "occupied" else "AVAILABLE"
            slot_lines.append(f"  {s.slot_id} (Zone {s.zone}): {status_str}")

        occupancy_text = "\n".join(slot_lines)
        recommended = available[0]

        entrance_name = "West" if list(parking_map.entrances.keys())[0] == "west" else "East"

        template = (
            f"SMARTPARK PARKING GUIDANCE\n"
            f"\n"
            f"Parking Lot: {parking_map.name}\n"
            f"Level: {recommended.level}\n"
            f"\n"
            f"PARKING STATUS\n"
            f"Total Slots: {occupancy.total_slots}\n"
            f"Occupied: {occupancy.occupied_slots}\n"
            f"Available: {occupancy.available_slots}\n"
            f"\n"
            f"RECOMMENDED PARKING\n"
            f"Slot: {recommended.slot_id}\n"
            f"Zone: {recommended.zone}\n"
            f"Status: AVAILABLE\n"
            f"\n"
            f"DIRECTIONS\n"
            f"1. Enter through the {entrance_name} entrance.\n"
            f"2. Proceed to {recommended.zone}.\n"
            f"3. Follow the parking signs to {recommended.slot_id}.\n"
            f"4. Park in slot {recommended.slot_id}.\n"
            f"\n"
            f"DESTINATION: {recommended.slot_id}"
        )

        system_prompt = (
            "You are a parking guidance system. You will receive parking occupancy data. "
            "You MUST output EXACTLY the template provided. Do NOT add greetings, "
            "explanations, or any text before or after the template. "
            "Do NOT invent any information. Only use the data provided."
        )

        user_prompt = (
            f"Output EXACTLY this template with the correct values filled in. "
            f"Do NOT add any text before or after.\n\n"
            f"{template}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
