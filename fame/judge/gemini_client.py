from __future__ import annotations

from typing import Any, Optional

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
        if (self.model or "").strip().lower() == "gemini-3.1-pro-preview":
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": 256,
            }

        headers = {
            "x-goog-api-key": api_key,
            "content-type": "application/json",
        }

        r = self._post_with_retries(url, headers=headers, json_payload=payload)
        data = r.json()
        self.last_response_meta = {
            "usageMetadata": data.get("usageMetadata"),
            "promptFeedback": data.get("promptFeedback"),
        }
        candidates = data.get("candidates") or []
        if not candidates:
            prompt_feedback = data.get("promptFeedback") or {}
            if prompt_feedback:
                print(f"WARN: Gemini returned no candidates. promptFeedback={prompt_feedback}")
            return ""
        first = candidates[0]
        finish_reason = first.get("finishReason")
        safety_ratings = first.get("safetyRatings")
        self.last_response_meta.update(
            {
                "finishReason": finish_reason,
                "safetyRatings": safety_ratings,
            }
        )
        if finish_reason and finish_reason != "STOP":
            usage = data.get("usageMetadata") or {}
            print(
                "WARN: Gemini generation ended with "
                f"finishReason={finish_reason}, usageMetadata={usage}"
            )

        content = (first.get("content") or {})
        parts = content.get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(texts).strip()
