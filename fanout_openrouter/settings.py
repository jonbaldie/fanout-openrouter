from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    request_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))
        return cls(
            openrouter_api_key=api_key,
            openrouter_base_url=base_url,
            request_timeout_seconds=timeout,
        )
