from .user_messages import UserMessageError


class LLMTimeoutError(UserMessageError):
    """Raised when an LLM HTTP request times out."""

    def __init__(self, host: str, model: str, timeout_s: int):
        super().__init__(
            f"LLM request timed out after {timeout_s}s (host={host}, model={model}). "
            "Please retry or switch to a closer/faster host."
        )
