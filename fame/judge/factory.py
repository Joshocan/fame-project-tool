from __future__ import annotations

from .base import JudgeClient
from .openai_client import OpenAIJudgeClient
from .anthropic_client import AnthropicJudgeClient
from .gemini_client import GeminiJudgeClient


def create_judge_client(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key_env: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> JudgeClient:
    provider = provider.lower().strip()
    if provider in ("openai", "gpt"):
        return OpenAIJudgeClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    if provider in ("anthropic", "claude", "claude.ai"):
        return AnthropicJudgeClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    if provider in ("gemini", "google"):
        return GeminiJudgeClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    raise ValueError(f"Unsupported judge provider: {provider}")
