from __future__ import annotations

from typing import Optional

from .base import JudgeClient


class GeminiJudgeClient(JudgeClient):
    """
    Google Gemini judge client using generateContent API.
    """

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError(f"Missing API key in env var '{self.api_key_env}'")

        base = self.base_url.rstrip("/") or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{self.model}:generateContent"

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        if system:
            contents = [{"role": "user", "parts": [{"text": system + "\n\n" + prompt}]}]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature if temperature is not None else self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

        headers = {
            "x-goog-api-key": api_key,
            "content-type": "application/json",
        }

        r = self._post_with_retries(url, headers=headers, json_payload=payload)
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        content = (candidates[0].get("content") or {})
        parts = content.get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(texts).strip()
