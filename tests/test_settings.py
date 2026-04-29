from __future__ import annotations

from pathlib import Path

from fanout_openrouter.settings import Settings


def test_from_env_loads_dotenv_file(monkeypatch, tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "OPENROUTER_API_KEY=dotenv-key\nFANOUT_POLICY_FILE=/tmp/policies.json\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("FANOUT_DOTENV_PATH", str(dotenv_path))

    settings = Settings.from_env()

    assert settings.openrouter_api_key == "dotenv-key"
    assert settings.policy_file == "/tmp/policies.json"
