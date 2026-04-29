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
import sys
import threading
from dataclasses import dataclass
from typing import Any

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
            if line.startswith("data: ")
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

    problems.extend(_diff_json(oracle.json_body, local.json_body))
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
