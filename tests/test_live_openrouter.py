from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fanout_openrouter.app import create_app
from fanout_openrouter.settings import Settings

pytestmark = pytest.mark.live


def _settings_without_embedded_key() -> tuple[Settings, str]:
    loaded = Settings.from_env()
    if not loaded.openrouter_api_key:
        raise AssertionError("OPENROUTER_API_KEY must be set for live smoke tests")

    return (
        Settings(
            openrouter_api_key=None,
            openrouter_base_url=loaded.openrouter_base_url,
            request_timeout_seconds=loaded.request_timeout_seconds,
            policy_file=loaded.policy_file,
        ),
        loaded.openrouter_api_key,
    )


def _read_sse_events(response) -> list[str]:
    events: list[str] = []
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        events.append(line[6:])
    return events


def test_live_chat_completion_round_trip() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "fanout/minimal",
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly the word ok.",
                    }
                ],
                "max_tokens": 16,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] != "fanout/minimal"
    assert body["provider"]
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"].strip() == "ok"
    assert isinstance(body["usage"], dict)


def test_live_chat_completion_accepts_bearer_token_fallback() -> None:
    settings, api_key = _settings_without_embedded_key()
    app = create_app(settings=settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "fanout/minimal",
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly the word ok.",
                    }
                ],
                "max_tokens": 16,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"].strip().lower().rstrip(".!") == "ok"


def test_live_chat_completion_streams_openai_style_sse() -> None:
    app = create_app()

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/v1/chat/completions",
            json={
                "model": "fanout/minimal",
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly the word ok.",
                    }
                ],
                "max_tokens": 16,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("text/event-stream")
            events = _read_sse_events(response)

    assert events[-1] == "[DONE]"

    chunks = [json.loads(event) for event in events[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert all(chunk["model"] != "fanout/minimal" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    streamed_content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
    )
    assert streamed_content.strip() == "ok"


def test_live_chat_completion_accepts_multipart_message_content() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "fanout/minimal",
                "messages": [
                    {
                        "role": "developer",
                        "content": "Follow the user's formatting exactly.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Reply with exactly the word"},
                            {"type": "text", "text": " ok."},
                        ],
                    },
                ],
                "max_tokens": 16,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert (
        body["choices"][0]["message"]["content"].strip().lower().rstrip(".!?") == "ok"
    )


def test_models_endpoint_lists_virtual_models() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/api/v1/models")

    assert response.status_code == 200, response.text
    body = response.json()
    # OpenRouter's /models envelope is just {"data": [...]}; no top-level
    # "object" key.
    assert "object" not in body
    assert [entry["id"] for entry in body["data"]] == ["fanout/minimal"]
    entry = body["data"][0]
    # Sanity-check a few OpenRouter-shaped fields we now emit.
    assert entry["canonical_slug"] == "fanout/minimal"
    assert "architecture" in entry
    assert "pricing" in entry
    assert "supported_parameters" in entry


def test_unsupported_virtual_model_returns_structured_error() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "fanout/does-not-exist",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 400, response.text
    assert response.json() == {
        "error": {
            "message": "fanout/does-not-exist is not a valid model ID",
            "code": 400,
        }
    }


def test_missing_messages_returns_structured_validation_error() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={"model": "fanout/minimal"},
        )

    assert response.status_code == 400, response.text
    assert response.json() == {
        "error": {
            "message": 'Input required: specify "prompt" or "messages"',
            "code": 400,
        }
    }


def test_missing_api_key_returns_structured_auth_error() -> None:
    settings, _ = _settings_without_embedded_key()
    app = create_app(settings=settings)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "fanout/minimal",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 401, response.text
    assert response.json() == {
        "error": {
            "message": "No cookie auth credentials found",
            "code": 401,
        }
    }
