from __future__ import annotations

from typing import Any, Optional

from .base import JudgeClient


class OpenAIJudgeClient(JudgeClient):
    """
    OpenAI judge client using Chat Completions.
    """

    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError(f"Missing API key in env var '{self.api_key_env}'")

        model_name = (self.model or "").strip().lower()
        if model_name == "o3":
            return self._generate_via_responses(
                api_key=api_key,
                prompt=prompt,
                system=system,
            )

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

    def _generate_via_responses(self, *, api_key: str, prompt: str, system: Optional[str]) -> str:
        base = self.base_url.rstrip("/") or "https://api.openai.com"
        url = f"{base}/v1/responses"

        payload = {
            "model": self.model,
            "input": [],
            "max_output_tokens": self.max_tokens,
        }
        if system:
            payload["input"].append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system}],
                }
            )
        payload["input"].append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        r = self._post_with_retries(url, headers=headers, json_payload=payload)
        data = r.json()
        self.last_response_meta = {
            "status": data.get("status"),
            "incomplete_details": data.get("incomplete_details"),
            "usage": data.get("usage"),
        }
        status = data.get("status")
        if status and status != "completed":
            print(
                "WARN: OpenAI Responses generation ended with "
                f"status={status}, incomplete_details={data.get('incomplete_details')}, "
                f"usage={data.get('usage')}"
            )
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        texts = []
        for item in data.get("output") or []:
            for content in item.get("content") or []:
                text = content.get("text")
                if not text and content.get("type") == "output_text":
                    text = content.get("text")
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts).strip()

        # Fallback diagnostics for unexpected response shapes.
        output = data.get("output") or []
        content_types: list[dict[str, Any]] = []
        for item in output:
            for content in item.get("content") or []:
                content_types.append(
                    {
                        "type": content.get("type"),
                        "keys": sorted(content.keys()),
                    }
                )
        if output or data.get("refusal"):
            print(
                "WARN: OpenAI Responses returned no text payload. "
                f"output_content_shapes={content_types}, refusal={data.get('refusal')}, "
                f"status={data.get('status')}"
            )
        return "\n".join(texts).strip()
