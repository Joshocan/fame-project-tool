from __future__ import annotations

from typing import Optional

from .base import JudgeClient


class OpenAIJudgeClient(JudgeClient):
    """
    OpenAI judge client using Chat Completions.
    """

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError(f"Missing API key in env var '{self.api_key_env}'")

        base = self.base_url.rstrip("/") or "https://api.openai.com"
        url = f"{base}/v1/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        r = self._post_with_retries(url, headers=headers, json_payload=payload)
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()
