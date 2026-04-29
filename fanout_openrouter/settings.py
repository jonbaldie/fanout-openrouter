from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()

        key, separator, value = line.partition("=")
        if not separator or not key:
            continue

        clean_value = value.strip()
        if (
            len(clean_value) >= 2
            and clean_value[0] == clean_value[-1]
            and clean_value[0] in {'"', "'"}
        ):
            clean_value = clean_value[1:-1]
        os.environ.setdefault(key.strip(), clean_value)


def _default_policy_path() -> Path:
    return Path(__file__).resolve().parent.parent / "fanout_policies.json"


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    request_timeout_seconds: float = 60.0
    policy_file: str = str(_default_policy_path())
    rate_limit_rpm: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        dotenv_path = Path(os.getenv("FANOUT_DOTENV_PATH", ".env"))
        _load_dotenv(dotenv_path)

        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))
        policy_file = os.getenv("FANOUT_POLICY_FILE", str(_default_policy_path()))
        rate_limit_rpm = int(os.getenv("FANOUT_RATE_LIMIT_RPM", "60"))
        return cls(
            openrouter_api_key=api_key,
            openrouter_base_url=base_url,
            request_timeout_seconds=timeout,
            policy_file=policy_file,
            rate_limit_rpm=rate_limit_rpm,
        )
