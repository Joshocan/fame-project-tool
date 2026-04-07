from __future__ import annotations

import os
from typing import Optional

from .base import JudgeClient


class AnthropicJudgeClient(JudgeClient):
    """
    Anthropic judge client using Messages API.
    """

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError(f"Missing API key in env var '{self.api_key_env}'")

        base = self.base_url.rstrip("/") or "https://api.anthropic.com"
        url = f"{base}/v1/messages"

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": api_key,
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            "content-type": "application/json",
        }

        r = self._post_with_retries(url, headers=headers, json_payload=payload)
        data = r.json()
        content = data.get("content") or []
        if not content:
            return ""
        # content is a list of blocks; extract text blocks
        parts = [b.get("text", "") for b in content if isinstance(b, dict)]
        return "".join(parts).strip()
