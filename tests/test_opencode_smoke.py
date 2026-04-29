"""
Smoke harness: drive the `opencode` CLI against our local facade as an
OpenRouter-equivalent, OpenAI-compatible provider.

The harness:

1. Boots our FastAPI app under a real uvicorn subprocess on a random free
   port. TestClient would not work here because opencode is an external
   process that must hit real HTTP.

2. Writes a throwaway opencode config that registers our facade as a custom
   OpenAI-compatible provider and exposes our `fanout/minimal` virtual model.

3. Invokes `opencode run --model <provider>/<model> "..."` non-interactively
   with OPENCODE_CONFIG pointed at the throwaway config and an isolated
   OPENCODE_CONFIG_DIR / HOME so the user's real opencode state is not
   consulted.

This harness is now a real green-path smoke test. It still uses loose enough
assertions to tolerate normal non-determinism, but it no longer treats a clean
run as merely exploratory wiring.

Run just this file with:

    uv run pytest tests/test_opencode_smoke.py -v -s --tb=short
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest


FACADE_MODULE = "fanout_openrouter.app:app"
VIRTUAL_MODEL = "fanout/minimal"
PROVIDER_ID = "fanoutlocal"


def _log(message: str) -> None:
    print(f"[opencode-smoke] {message}", file=sys.stderr, flush=True)


# ---------- uvicorn lifecycle ----------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class FacadeServer:
    process: subprocess.Popen[bytes]
    base_url: str
    log_path: Path


def _start_facade(port: int, log_path: Path) -> FacadeServer:
    env = os.environ.copy()
    # Surface meaningful logs from our app. We do NOT strip the user's
    # OPENROUTER_API_KEY here; the facade needs to reach real OpenRouter
    # so opencode's request can actually get completed.
    env["PYTHONUNBUFFERED"] = "1"

    log_handle = log_path.open("wb")
    _log(f"spawning uvicorn on 127.0.0.1:{port} (log -> {log_path})")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            FACADE_MODULE,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "info",
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )

    base_url = f"http://127.0.0.1:{port}/api/v1"
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"uvicorn exited early (rc={process.returncode}); see log at {log_path}"
            )
        try:
            response = httpx.get(f"{base_url}/models", timeout=1.0)
            if response.status_code == 200:
                _log(f"facade ready at {base_url}")
                return FacadeServer(
                    process=process, base_url=base_url, log_path=log_path
                )
        except httpx.HTTPError:
            pass
        time.sleep(0.2)

    process.terminate()
    raise RuntimeError(f"facade did not become ready within 20s; see log at {log_path}")


def _stop_facade(server: FacadeServer) -> None:
    _log("stopping facade")
    server.process.terminate()
    try:
        server.process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        _log("facade did not exit, killing")
        server.process.kill()
        server.process.wait(timeout=5.0)


@pytest.fixture(scope="module")
def facade(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FacadeServer]:
    log_dir = tmp_path_factory.mktemp("facade-logs")
    server = _start_facade(_free_port(), log_dir / "uvicorn.log")
    try:
        yield server
    finally:
        _stop_facade(server)
        tail = server.log_path.read_bytes()[-4000:]
        if tail:
            _log("facade log tail:\n" + tail.decode("utf-8", errors="replace"))


# ---------- opencode config ----------


@dataclass
class OpenCodeEnv:
    config_path: Path
    env: dict[str, str]


def _write_opencode_config(
    config_dir: Path,
    home_dir: Path,
    facade_base_url: str,
) -> OpenCodeEnv:
    # opencode resolves api keys for OpenAI-compatible providers via the
    # provider's `options.apiKey`. We use env-var substitution so the actual
    # key (OPENROUTER_API_KEY from the user's .env) flows through without
    # being baked into the file on disk.
    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "fan-out-openrouter (local)",
                "options": {
                    "baseURL": facade_base_url,
                    "apiKey": "{env:FANOUT_API_KEY}",
                },
                "models": {
                    VIRTUAL_MODEL: {
                        "name": "Fanout Minimal (local)",
                    }
                },
            }
        },
        # Keep opencode out of network-dependent defaults. We want a clean
        # surface so the only thing it can talk to is our facade.
        "autoupdate": False,
        "share": "disabled",
        "small_model": f"{PROVIDER_ID}/{VIRTUAL_MODEL}",
        "model": f"{PROVIDER_ID}/{VIRTUAL_MODEL}",
        # Disable all provider discovery except ours so opencode doesn't
        # try to query models.dev / remote .well-known.
        "enabled_providers": [PROVIDER_ID],
    }

    config_path = config_dir / "opencode.json"
    config_path.write_text(json.dumps(opencode_config, indent=2), encoding="utf-8")

    # Load OPENROUTER_API_KEY out of the project .env so the smoke test
    # runs by default without the caller needing to pre-export it. Mirrors
    # the behavior of Settings.from_env() for the facade itself.
    dotenv = Path(".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and dotenv.exists():
        for raw in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if sep and key.strip() == "OPENROUTER_API_KEY":
                api_key = value.strip().strip('"').strip("'")
                break

    if not api_key:
        pytest.skip("OPENROUTER_API_KEY must be available for opencode smoke harness")

    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = str(config_path)
    env["OPENCODE_CONFIG_DIR"] = str(config_dir)
    env["HOME"] = str(home_dir)
    # Opencode reads these paths; point them inside the sandbox so we don't
    # stomp on or inherit from the developer's real opencode state.
    env["XDG_CONFIG_HOME"] = str(home_dir / ".config")
    env["XDG_DATA_HOME"] = str(home_dir / ".local" / "share")
    env["XDG_CACHE_HOME"] = str(home_dir / ".cache")
    env["FANOUT_API_KEY"] = api_key
    # Make sure opencode doesn't try to call out to auth providers.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)

    return OpenCodeEnv(config_path=config_path, env=env)


@pytest.fixture(scope="module")
def opencode_env(
    facade: FacadeServer,
    tmp_path_factory: pytest.TempPathFactory,
) -> OpenCodeEnv:
    config_dir = tmp_path_factory.mktemp("opencode-config")
    home_dir = tmp_path_factory.mktemp("opencode-home")
    (home_dir / ".config").mkdir(parents=True, exist_ok=True)
    (home_dir / ".local" / "share").mkdir(parents=True, exist_ok=True)
    (home_dir / ".cache").mkdir(parents=True, exist_ok=True)
    return _write_opencode_config(config_dir, home_dir, facade.base_url)


# ---------- opencode invocation ----------


def _opencode_available() -> bool:
    return shutil.which("opencode") is not None


@dataclass
class OpenCodeResult:
    returncode: int
    stdout: str
    stderr: str


def _parse_jsonl(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _assistant_text_from_events(events: list[dict[str, Any]]) -> str:
    text_parts = [
        event.get("part", {}).get("text", "")
        for event in events
        if event.get("type") == "text"
    ]
    return "".join(part for part in text_parts if isinstance(part, str)).strip()


def _run_opencode(
    args: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: float = 120.0,
) -> OpenCodeResult:
    cmd = ["opencode", *args]
    _log(f"exec: {' '.join(cmd)} (cwd={cwd})")
    completed = subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd),
        capture_output=True,
        timeout=timeout,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    _log(f"rc={completed.returncode}")
    if stdout.strip():
        _log(f"stdout:\n{stdout.strip()}")
    if stderr.strip():
        _log(f"stderr:\n{stderr.strip()}")
    return OpenCodeResult(returncode=completed.returncode, stdout=stdout, stderr=stderr)


# ---------- tests ----------


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Give opencode its own empty working directory so it doesn't pick up
    # project-level opencode config or files from our repo.
    workdir = tmp_path_factory.mktemp("opencode-workspace")
    (workdir / "README.md").write_text("smoke workspace", encoding="utf-8")
    return workdir


@pytest.mark.skipif(
    not _opencode_available(),
    reason="opencode CLI is not installed on PATH",
)
def test_opencode_lists_our_virtual_model(
    opencode_env: OpenCodeEnv,
    workspace: Path,
) -> None:
    """
    Tracer bullet: does opencode actually load our provider and surface our
    virtual model in its catalog? This is the cheapest step that proves the
    wiring is real before we spend tokens on an actual chat run.
    """
    _log("case: opencode_lists_our_virtual_model")

    result = _run_opencode(
        ["models", PROVIDER_ID],
        env=opencode_env.env,
        cwd=workspace,
        timeout=60.0,
    )

    assert result.returncode == 0, (
        f"opencode models exited rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert VIRTUAL_MODEL in result.stdout or VIRTUAL_MODEL in result.stderr, (
        f"expected {VIRTUAL_MODEL!r} in opencode model catalog output\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.skipif(
    not _opencode_available(),
    reason="opencode CLI is not installed on PATH",
)
def test_opencode_run_round_trip(
    opencode_env: OpenCodeEnv,
    workspace: Path,
) -> None:
    """
    End-to-end tracer bullet: `opencode run` against our facade should
    return a valid JSON event stream containing the assistant's reply.
    """
    _log("case: opencode_run_round_trip")

    result = _run_opencode(
        [
            "run",
            "--model",
            f"{PROVIDER_ID}/{VIRTUAL_MODEL}",
            "--format",
            "json",
            "--dangerously-skip-permissions",
            "Reply with exactly the word ok.",
        ],
        env=opencode_env.env,
        cwd=workspace,
        timeout=180.0,
    )

    assert result.returncode == 0, (
        f"opencode run exited rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    events = _parse_jsonl(result.stdout)
    assert events, "opencode run produced no JSON events"
    assert events[0].get("type") == "step_start"
    assert events[-1].get("type") == "step_finish"
    assert events[-1].get("part", {}).get("reason") == "stop"

    assistant_text = _assistant_text_from_events(events)
    assert assistant_text.lower().rstrip(".!?") == "ok", (
        f"expected assistant text to be 'ok', got {assistant_text!r}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
