"""
Live parity harness: compares our facade against real OpenRouter.ai.

Tracer-bullet slice: ONE case, fully wired, end-to-end, verbose.

No mocks. The case hits both:
  - real OpenRouter at https://openrouter.ai/api/v1
  - our local app via FastAPI TestClient

We diff:
  - HTTP status
  - base content-type
  - JSON body shape

We normalize only fields that are inherently non-deterministic:
  - id
  - created
  - fingerprint-like metadata
  - assistant content text
  - usage values

The goal isn't "same bytes". It's "same wire contract".
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.live

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from fanout_openrouter.app import create_app
from fanout_openrouter.settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ORACLE_MODEL = "google/gemini-3-flash-preview"
LOCAL_VIRTUAL_MODEL = "fanout/minimal"


def _log(message: str) -> None:
    print(f"[parity] {message}", file=sys.stderr, flush=True)


# ---------- normalization ----------


NORMALIZED_SCALAR_KEYS = {
    "id",
    "created",
    "system_fingerprint",
    "provider",
    "fingerprint",
}


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, inner in value.items():
            if key == "user_id":
                continue
            if key in NORMALIZED_SCALAR_KEYS:
                out[key] = "<normalized>"
            elif (
                key == "data"
                and value.get("type") == "reasoning.encrypted"
                and isinstance(inner, str)
            ):
                out[key] = "<normalized>"
            elif key == "content" and isinstance(inner, str):
                out[key] = "<normalized>"
            elif key == "usage" and isinstance(inner, dict):
                out[key] = {uk: "<normalized>" for uk in inner.keys()}
            elif key == "arguments" and isinstance(inner, str):
                # tool_calls[].function.arguments is model-authored JSON text;
                # the exact whitespace/formatting is non-deterministic.
                out[key] = "<normalized>"
            else:
                out[key] = _normalize(inner)
        return out
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _base_content_type(header_value: str) -> str:
    return (header_value or "").split(";")[0].strip().lower()


# ---------- snapshot capture ----------


@dataclass
class Snapshot:
    status: int
    content_type: str
    json_body: Any | None


def _capture_oracle_json(
    api_key: str,
    path: str,
    body: dict[str, Any] | None,
    method: str = "POST",
) -> Snapshot:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    _log(f"oracle {method} {OPENROUTER_BASE_URL}{path}")
    response = httpx.request(
        method,
        f"{OPENROUTER_BASE_URL}{path}",
        headers=headers,
        json=body,
        timeout=60.0,
    )
    _log(f"oracle status={response.status_code}")

    try:
        parsed = response.json()
    except ValueError:
        parsed = None
    return Snapshot(
        status=response.status_code,
        content_type=_base_content_type(response.headers.get("content-type", "")),
        json_body=_normalize(parsed) if parsed is not None else None,
    )


def _capture_local_json(
    client: TestClient,
    path: str,
    body: dict[str, Any] | None,
    timeout: float = 90.0,
    headers: dict[str, str] | None = None,
    method: str = "POST",
) -> Snapshot:
    _log(f"local {method} {path} (timeout={timeout}s)")

    result: dict[str, Any] = {}

    def _do_call() -> None:
        try:
            if method == "GET":
                response = client.get(path, headers=headers)
            else:
                response = client.post(path, json=body, headers=headers)
            result["response"] = response
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_do_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise AssertionError(
            f"local facade did not respond within {timeout}s for POST {path}"
        )

    if "error" in result:
        raise result["error"]

    response = result["response"]
    _log(f"local status={response.status_code}")

    try:
        parsed = response.json()
    except ValueError:
        parsed = None

    _log(f"local body (normalized): {parsed}")

    return Snapshot(
        status=response.status_code,
        content_type=_base_content_type(response.headers.get("content-type", "")),
        json_body=_normalize(parsed) if parsed is not None else None,
    )


def _parse_sse_event(line: str) -> Any:
    if line.startswith(": "):
        return {"comment": line.removeprefix(": ")}
    payload = line.removeprefix("data: ")
    if payload == "[DONE]":
        return payload
    return _normalize_stream_event(json.loads(payload))


def _normalize_stream_event(event: Any) -> Any:
    if not isinstance(event, dict):
        return _normalize(event)

    cloned = json.loads(json.dumps(event))
    choices = cloned.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            delta.pop("reasoning", None)
            delta.pop("reasoning_details", None)

    return _normalize(cloned)


def _capture_oracle_stream(
    api_key: str,
    path: str,
    body: dict[str, Any] | None,
) -> Snapshot:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    _log(f"oracle STREAM POST {OPENROUTER_BASE_URL}{path}")
    with httpx.stream(
        "POST",
        f"{OPENROUTER_BASE_URL}{path}",
        headers=headers,
        json=body,
        timeout=60.0,
    ) as response:
        events = [
            _parse_sse_event(line)
            for line in response.iter_lines()
            if line.startswith("data: ") or line.startswith(": ")
        ]
        _log(f"oracle stream status={response.status_code} events={len(events)}")
        return Snapshot(
            status=response.status_code,
            content_type=_base_content_type(response.headers.get("content-type", "")),
            json_body=events,
        )


def _capture_local_stream(
    client: TestClient,
    path: str,
    body: dict[str, Any] | None,
    timeout: float = 90.0,
) -> Snapshot:
    _log(f"local STREAM POST {path} (timeout={timeout}s)")

    result: dict[str, Any] = {}

    def _do_call() -> None:
        try:
            with client.stream("POST", path, json=body) as response:
                result["status_code"] = response.status_code
                result["content_type"] = response.headers.get("content-type", "")
                result["events"] = [
                    _parse_sse_event(line)
                    for line in response.iter_lines()
                    if line.startswith("data: ")
                ]
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_do_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise AssertionError(
            f"local facade did not respond within {timeout}s for STREAM POST {path}"
        )

    if "error" in result:
        raise result["error"]

    _log(f"local stream status={result['status_code']} events={len(result['events'])}")

    return Snapshot(
        status=result["status_code"],
        content_type=_base_content_type(result["content_type"]),
        json_body=result["events"],
    )


# ---------- diff helpers ----------


def _diff_json(oracle: Any, local: Any, path: str = "$") -> list[str]:
    if type(oracle) is not type(local):
        return [
            f"{path}: type mismatch oracle={type(oracle).__name__} local={type(local).__name__}"
        ]

    if isinstance(oracle, dict):
        problems: list[str] = []
        oracle_keys = set(oracle.keys())
        local_keys = set(local.keys())
        for missing in sorted(oracle_keys - local_keys):
            problems.append(f"{path}.{missing}: present in oracle, missing locally")
        for extra in sorted(local_keys - oracle_keys):
            problems.append(f"{path}.{extra}: present locally, missing in oracle")
        for key in sorted(oracle_keys & local_keys):
            problems.extend(_diff_json(oracle[key], local[key], f"{path}.{key}"))
        return problems

    if isinstance(oracle, list):
        problems = []
        if len(oracle) != len(local):
            problems.append(
                f"{path}: list length mismatch oracle={len(oracle)} local={len(local)}"
            )
        for index, (o_item, l_item) in enumerate(zip(oracle, local)):
            problems.extend(_diff_json(o_item, l_item, f"{path}[{index}]"))
        return problems

    if oracle != local:
        return [f"{path}: value mismatch oracle={oracle!r} local={local!r}"]
    return []


def _diff_snapshots(oracle: Snapshot, local: Snapshot) -> list[str]:
    problems: list[str] = []

    if oracle.status != local.status:
        problems.append(f"status mismatch: oracle={oracle.status} local={local.status}")

    if oracle.content_type != local.content_type:
        problems.append(
            "content-type mismatch: "
            f"oracle={oracle.content_type!r} local={local.content_type!r}"
        )

    # Filter out keep-alive comments from body before shape diffing,
    # because their count and exact placement are non-deterministic.
    def _strip_comments(body: Any) -> Any:
        if isinstance(body, list):
            return [x for x in body if not (isinstance(x, dict) and "comment" in x)]
        return body

    oracle_body = _strip_comments(oracle.json_body)
    local_body = _strip_comments(local.json_body)

    problems.extend(_diff_json(oracle_body, local_body))
    return problems


# ---------- fixtures ----------


@pytest.fixture(scope="module")
def api_key() -> str:
    loaded = Settings.from_env()
    if not loaded.openrouter_api_key:
        raise AssertionError("OPENROUTER_API_KEY must be set for parity tests")
    return loaded.openrouter_api_key


@pytest.fixture(scope="module")
def local_client(api_key: str) -> TestClient:
    loaded = Settings.from_env()
    app = create_app(settings=loaded)
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {api_key}"})
    return client


@pytest.fixture(scope="module")
def local_client_without_embedded_key() -> TestClient:
    loaded = Settings.from_env()
    app = create_app(
        settings=Settings(
            openrouter_api_key=None,
            openrouter_base_url=loaded.openrouter_base_url,
            request_timeout_seconds=loaded.request_timeout_seconds,
            policy_file=loaded.policy_file,
        )
    )
    return TestClient(app)


# ---------- tracer bullet case ----------


def test_parity_chat_non_stream_happy(api_key: str, local_client: TestClient) -> None:
    _log("case: chat_non_stream_happy")

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 10,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter")
    oracle = _capture_oracle_json(api_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade")
    local = _capture_local_json(local_client, "/api/v1/chat/completions", local_body)

    _log("step 3/3: diffing")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_non_stream_happy:\n  - {rendered}")

    _log("PASS")


def test_parity_chat_stream_happy(api_key: str, local_client: TestClient) -> None:
    _log("case: chat_stream_happy")

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 10,
        "stream": True,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter stream")
    oracle = _capture_oracle_stream(api_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade stream")
    local = _capture_local_stream(local_client, "/api/v1/chat/completions", local_body)

    _log("step 3/3: diffing stream snapshots")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_stream_happy:\n  - {rendered}")

    _log("PASS")


def test_parity_chat_missing_auth_error(
    local_client_without_embedded_key: TestClient,
) -> None:
    _log("case: chat_missing_auth_error")

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 10,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter without auth")
    oracle = _capture_oracle_json("", "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade without auth")
    local = _capture_local_json(
        local_client_without_embedded_key,
        "/api/v1/chat/completions",
        local_body,
    )

    _log("step 3/3: diffing")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_missing_auth_error:\n  - {rendered}")

    _log("PASS")


def test_parity_chat_missing_messages_error(
    api_key: str,
    local_client: TestClient,
) -> None:
    _log("case: chat_missing_messages_error")

    oracle_body = {
        "model": ORACLE_MODEL,
        "max_tokens": 10,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter invalid request")
    oracle = _capture_oracle_json(api_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade invalid request")
    local = _capture_local_json(local_client, "/api/v1/chat/completions", local_body)

    _log("step 3/3: diffing")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_missing_messages_error:\n  - {rendered}")

    _log("PASS")


def test_parity_chat_invalid_bearer_error(
    local_client_without_embedded_key: TestClient,
) -> None:
    _log("case: chat_invalid_bearer_error")

    bad_key = "sk-or-v1-definitely-not-a-valid-key-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 10,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter with bad bearer")
    oracle = _capture_oracle_json(bad_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade with bad bearer")
    local = _capture_local_json(
        local_client_without_embedded_key,
        "/api/v1/chat/completions",
        local_body,
        headers={"Authorization": f"Bearer {bad_key}"},
    )

    _log("step 3/3: diffing")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_invalid_bearer_error:\n  - {rendered}")

    _log("PASS")


def test_parity_chat_stream_invalid_bearer_error(
    local_client_without_embedded_key: TestClient,
) -> None:
    """
    When a stream=True request fails auth, OpenRouter doesn't open an SSE
    stream; it returns a plain JSON error at the HTTP level. Our facade must
    behave the same way so clients don't have to special-case streaming
    error handling.
    """
    _log("case: chat_stream_invalid_bearer_error")

    bad_key = "sk-or-v1-definitely-not-a-valid-key-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 10,
        "stream": True,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter stream with bad bearer")
    oracle = _capture_oracle_json(bad_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade stream with bad bearer")
    local = _capture_local_json(
        local_client_without_embedded_key,
        "/api/v1/chat/completions",
        local_body,
        headers={"Authorization": f"Bearer {bad_key}"},
    )

    _log("step 3/3: diffing")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(
            f"parity drift in chat_stream_invalid_bearer_error:\n  - {rendered}"
        )

    _log("PASS")


def test_parity_chat_unknown_submodel_error(
    api_key: str,
    tmp_path: Path,
) -> None:
    """
    When a virtual model's candidate pool is a nonexistent OpenRouter model,
    all candidates fail with the same upstream 400. Our facade should pass
    that error through unchanged so the wire contract matches OpenRouter's
    own 400 for an unknown model ID.
    """
    _log("case: chat_unknown_submodel_error")

    unknown_model = "does-not-exist/nope"
    policy_path = tmp_path / "unknown_model_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policies": [
                    {
                        "virtual_model": "fanout/unknown-submodel",
                        "candidate_models": [unknown_model],
                        "fanout_count": 2,
                        "created": 1710000000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = Settings.from_env()
    settings = Settings(
        openrouter_api_key=loaded.openrouter_api_key,
        openrouter_base_url=loaded.openrouter_base_url,
        request_timeout_seconds=loaded.request_timeout_seconds,
        policy_file=str(policy_path),
    )
    app = create_app(settings=settings)
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {api_key}"})

    oracle_body = {
        "model": unknown_model,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 10,
    }
    local_body = {**oracle_body, "model": "fanout/unknown-submodel"}

    _log("step 1/3: hitting real OpenRouter with unknown model")
    oracle = _capture_oracle_json(api_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade with unknown sub-model policy")
    local = _capture_local_json(client, "/api/v1/chat/completions", local_body)

    _log("step 3/3: diffing")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_unknown_submodel_error:\n  - {rendered}")

    _log("PASS")


def test_parity_chat_empty_messages_error(
    api_key: str,
    local_client: TestClient,
) -> None:
    _log("case: chat_empty_messages_error")

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [],
        "max_tokens": 10,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter with empty messages")
    oracle = _capture_oracle_json(api_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade with empty messages")
    local = _capture_local_json(local_client, "/api/v1/chat/completions", local_body)

    _log("step 3/3: diffing")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_empty_messages_error:\n  - {rendered}")

    _log("PASS")


def test_parity_models_list_shape(local_client: TestClient) -> None:
    """
    OpenRouter's /models catalog is enormous and dynamic; we can't mirror its
    contents. But the wire contract we expose must match: same top-level
    envelope keys, and each model item must carry the same shape (same keys).

    This test diffs the envelope keys and the union of per-item keys.
    """
    _log("case: models_list_shape")

    _log("step 1/3: hitting real OpenRouter /models")
    oracle = _capture_oracle_json("", "/models", None, method="GET")

    _log("step 2/3: hitting local facade /models")
    local = _capture_local_json(
        local_client,
        "/api/v1/models",
        None,
        method="GET",
    )

    _log("step 3/3: diffing envelope and item shape")
    problems: list[str] = []

    if oracle.status != local.status:
        problems.append(f"status mismatch: oracle={oracle.status} local={local.status}")
    if oracle.content_type != local.content_type:
        problems.append(
            "content-type mismatch: "
            f"oracle={oracle.content_type!r} local={local.content_type!r}"
        )

    oracle_body = oracle.json_body
    local_body = local.json_body
    if not isinstance(oracle_body, dict) or not isinstance(local_body, dict):
        pytest.fail(
            f"unexpected /models body types: oracle={type(oracle_body).__name__} "
            f"local={type(local_body).__name__}"
        )

    oracle_keys = set(oracle_body.keys())
    local_keys = set(local_body.keys())
    for missing in sorted(oracle_keys - local_keys):
        problems.append(f"envelope.{missing}: present in oracle, missing locally")
    for extra in sorted(local_keys - oracle_keys):
        problems.append(f"envelope.{extra}: present locally, missing in oracle")

    oracle_items = oracle_body.get("data") or []
    local_items = local_body.get("data") or []
    if not isinstance(oracle_items, list) or not oracle_items:
        pytest.fail("oracle /models returned empty or invalid data list")
    if not isinstance(local_items, list) or not local_items:
        pytest.fail("local /models returned empty or invalid data list")

    oracle_item_keys = set(oracle_items[0].keys())
    for index, item in enumerate(local_items):
        if not isinstance(item, dict):
            problems.append(f"data[{index}]: not an object")
            continue
        item_keys = set(item.keys())
        missing = oracle_item_keys - item_keys
        extra = item_keys - oracle_item_keys
        for key in sorted(missing):
            problems.append(
                f"data[{index}].{key}: present in oracle items, missing locally"
            )
        for key in sorted(extra):
            problems.append(
                f"data[{index}].{key}: present locally, missing in oracle items"
            )

    if problems:
        rendered = "\n  - ".join(problems)
        _log(f"FAIL: {len(problems)} diffs")
        pytest.fail(f"parity drift in models_list_shape:\n  - {rendered}")

    _log("PASS")


# ---------- streaming progressiveness ----------

# OpenRouter streams tokens as they are produced upstream. A conformant facade
# must do the same. Buffering the full response and re-slicing it into fake
# SSE chunks at the end would pass the earlier shape-only parity test because
# content is normalized, but it's not the contract. This harness measures the
# wall-clock arrival times of content-bearing deltas and asserts the stream is
# actually progressive: time-to-first-byte is meaningfully earlier than the
# end of the stream.


@dataclass
class StreamTiming:
    total_events: int
    content_events: int
    first_event_seconds: float
    last_event_seconds: float
    first_content_seconds: float
    last_content_seconds: float


def _capture_stream_timing_oracle(
    api_key: str,
    body: dict[str, Any],
) -> StreamTiming:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    start = time.monotonic()
    event_times: list[float] = []
    content_times: list[float] = []
    _log(f"oracle STREAM POST {OPENROUTER_BASE_URL}/chat/completions (timing)")
    with httpx.stream(
        "POST",
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=body,
        timeout=60.0,
    ) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            elapsed = time.monotonic() - start
            event_times.append(elapsed)
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except ValueError:
                continue
            if _event_has_content(parsed):
                content_times.append(elapsed)
    return _finalize_timing(event_times, content_times)


def _capture_stream_timing_local_http(
    base_url: str,
    body: dict[str, Any],
) -> StreamTiming:
    """
    Measure local stream timing against a real HTTP server. TestClient's
    sync-to-async bridge buffers SSE responses in practice, which masks
    whether the facade is actually streaming. A subprocess-hosted uvicorn
    preserves the real wire behavior so this measurement reflects what
    actual clients experience.
    """
    event_times: list[float] = []
    content_times: list[float] = []
    _log(f"local STREAM POST {base_url}/chat/completions (timing)")
    start = time.monotonic()
    with httpx.stream(
        "POST",
        f"{base_url}/chat/completions",
        json=body,
        timeout=120.0,
    ) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            elapsed = time.monotonic() - start
            event_times.append(elapsed)
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except ValueError:
                continue
            if _event_has_content(parsed):
                content_times.append(elapsed)
    return _finalize_timing(event_times, content_times)


def _event_has_content(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return False
    content = delta.get("content")
    return isinstance(content, str) and len(content) > 0


def _finalize_timing(
    event_times: list[float],
    content_times: list[float],
) -> StreamTiming:
    if not event_times:
        raise AssertionError("stream produced no data events")
    if not content_times:
        raise AssertionError("stream produced no content-bearing deltas")
    return StreamTiming(
        total_events=len(event_times),
        content_events=len(content_times),
        first_event_seconds=event_times[0],
        last_event_seconds=event_times[-1],
        first_content_seconds=content_times[0],
        last_content_seconds=content_times[-1],
    )


def _assert_progressive(label: str, timing: StreamTiming) -> None:
    _log(
        f"{label} timing: events={timing.total_events} "
        f"content_events={timing.content_events} "
        f"first_content={timing.first_content_seconds:.3f}s "
        f"last_content={timing.last_content_seconds:.3f}s "
        f"span={timing.last_content_seconds - timing.first_content_seconds:.3f}s"
    )

    problems: list[str] = []
    if timing.content_events < 3:
        problems.append(
            f"expected at least 3 content-bearing deltas, got {timing.content_events}"
        )

    span = timing.last_content_seconds - timing.first_content_seconds
    if span < 0.2:
        problems.append(
            f"content deltas arrived within {span:.3f}s - looks buffered, not streamed"
        )

    # If the stream is genuinely progressive, the first content delta must
    # arrive meaningfully earlier than the final one. A buffered facade that
    # waits for the full response, then slices it and flushes all chunks at
    # once, produces first~=last which fails this check even when the total
    # upstream work took many seconds.
    total = max(timing.last_content_seconds, 0.001)
    first_fraction = timing.first_content_seconds / total
    if first_fraction > 0.9:
        problems.append(
            f"first content delta arrived at {first_fraction:.0%} of total "
            f"stream duration - upstream tokens are being buffered before emit"
        )

    if problems:
        rendered = "\n  - ".join(problems)
        pytest.fail(f"{label} stream is not progressive:\n  - {rendered}")


@pytest.fixture(scope="module")
def local_http_server(api_key: str) -> Iterator[str]:
    """
    Spawn a real uvicorn subprocess so streaming measurements reflect
    actual wire behavior. TestClient is not adequate for streaming: its
    sync bridge buffers SSE bodies before handing them back.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    _log(f"spawning uvicorn on 127.0.0.1:{port}")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "fanout_openrouter.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    base_url = f"http://127.0.0.1:{port}/api/v1"
    deadline = time.monotonic() + 20.0
    ready = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            response = httpx.get(f"{base_url}/models", timeout=1.0)
            if response.status_code == 200:
                ready = True
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.2)

    if not ready:
        process.terminate()
        process.wait(timeout=5)
        raise AssertionError("local uvicorn did not become ready within 20s")

    try:
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def test_parity_chat_stream_is_progressive(
    api_key: str,
    local_http_server: str,
) -> None:
    """
    Real OpenRouter streams deltas as tokens are produced. Our facade must
    do the same rather than buffer the whole synthesis response and then
    slice it into fake SSE chunks. We prove this by measuring wall-clock
    arrival times of content-bearing deltas and asserting the stream is
    actually progressive, on both sides.
    """
    _log("case: chat_stream_is_progressive")

    # A prompt that guarantees a long-enough response for streaming signal
    # to be measurable, even on fast models.
    prompt = "Write a 200-word paragraph about the history of typewriters."
    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "stream": True,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/2: measuring real OpenRouter stream")
    oracle_timing = _capture_stream_timing_oracle(api_key, oracle_body)
    _assert_progressive("oracle", oracle_timing)

    _log("step 2/2: measuring local facade stream")
    local_timing = _capture_stream_timing_local_http(local_http_server, local_body)
    _assert_progressive("local", local_timing)

    _log("PASS")


# ---------- tool_calls parity ----------

# OpenRouter surfaces tool calls on chat completions by returning a choice with
# `finish_reason == "tool_calls"` and a `message.tool_calls` list. Agents like
# opencode will not function against a provider that strips or synthesizes
# these away. This case proves our facade carries the tool-calls wire shape
# through unchanged for the initial (no prior tool result) request.


TOOL_CALLS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def test_parity_chat_tool_calls_non_stream(
    api_key: str, local_client: TestClient
) -> None:
    """
    Tool-calling contract: when the upstream model decides to invoke a
    function tool, our facade must pass that decision through with
    `finish_reason == "tool_calls"` and `message.tool_calls` intact.
    """
    _log("case: chat_tool_calls_non_stream")

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You must call the get_weather tool when the user asks "
                    "about weather."
                ),
            },
            {"role": "user", "content": "What is the weather in Paris right now?"},
        ],
        "tools": TOOL_CALLS_TOOLS,
        "tool_choice": "auto",
        "max_tokens": 200,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter with tools")
    oracle = _capture_oracle_json(api_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade with tools")
    local = _capture_local_json(local_client, "/api/v1/chat/completions", local_body)

    _log("step 3/3: diffing")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_tool_calls_non_stream:\n  - {rendered}")

    _log("PASS")


def test_parity_chat_tool_calls_stream(api_key: str, local_client: TestClient) -> None:
    """
    Streaming tool-calling contract: upstream tool-call deltas must emit
    as progressive SSE chunks without being dropped or mangled by the
    synthesis path.
    """
    _log("case: chat_tool_calls_stream")

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You must call the get_weather tool when the user asks "
                    "about weather."
                ),
            },
            {"role": "user", "content": "What is the weather in Paris right now?"},
        ],
        "tools": TOOL_CALLS_TOOLS,
        "tool_choice": "auto",
        "max_tokens": 200,
        "stream": True,
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter stream with tools")
    oracle = _capture_oracle_stream(api_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade stream with tools")
    local = _capture_local_stream(local_client, "/api/v1/chat/completions", local_body)

    _log("step 3/3: diffing stream snapshots")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_tool_calls_stream:\n  - {rendered}")

    _log("PASS")


def test_parity_chat_stream_options_include_usage(
    api_key: str, local_client: TestClient
) -> None:
    _log("case: chat_stream_options_include_usage")

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [{"role": "user", "content": "say ok"}],
        "max_tokens": 10,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter stream with options")
    oracle = _capture_oracle_stream(api_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade stream with options")
    local = _capture_local_stream(local_client, "/api/v1/chat/completions", local_body)

    _log("step 3/3: diffing stream snapshots")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(
            f"parity drift in chat_stream_options_include_usage:\n  - {rendered}"
        )

    _log("PASS")


def test_parity_chat_response_format(api_key: str, local_client: TestClient) -> None:
    _log("case: chat_response_format")

    oracle_body = {
        "model": ORACLE_MODEL,
        "messages": [{"role": "user", "content": "Return JSON with a 'status' key."}],
        "max_tokens": 50,
        "response_format": {"type": "json_object"},
    }
    local_body = {**oracle_body, "model": LOCAL_VIRTUAL_MODEL}

    _log("step 1/3: hitting real OpenRouter with response_format")
    oracle = _capture_oracle_json(api_key, "/chat/completions", oracle_body)

    _log("step 2/3: hitting local facade with response_format")
    local = _capture_local_json(local_client, "/api/v1/chat/completions", local_body)

    _log("step 3/3: diffing")
    diffs = _diff_snapshots(oracle, local)

    if diffs:
        rendered = "\n  - ".join(diffs)
        _log(f"FAIL: {len(diffs)} diffs")
        pytest.fail(f"parity drift in chat_response_format:\n  - {rendered}")

    _log("PASS")
