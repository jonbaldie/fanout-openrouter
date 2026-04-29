from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from fanout_openrouter.app import create_app
from fanout_openrouter.settings import Settings


async def _no_sleep(_: float) -> None:
    return None


def test_chat_completions_route_returns_openai_style_response() -> None:
    candidate_counter = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content.decode("utf-8"))
        prompt = payload["messages"][0]["content"]

        if "Below are 2 candidate responses." in prompt:
            body = {
                "id": "gen-synthesis",
                "model": payload["model"],
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "final synthesized answer",
                        }
                    }
                ],
            }
            return httpx.Response(status_code=200, json=body)

        candidate_counter["count"] += 1
        body = {
            "id": f"gen-{candidate_counter['count']}",
            "model": payload["model"],
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"candidate {candidate_counter['count']}",
                    }
                }
            ],
        }
        return httpx.Response(status_code=200, json=body)

    app = create_app(
        settings=Settings(
            openrouter_api_key="test-key",
            openrouter_base_url="https://mocked.example/api/v1",
        ),
        transport=httpx.MockTransport(handler),
        sleep_func=_no_sleep,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "fanout/minimal",
                "messages": [{"role": "user", "content": "Explain tracer bullets."}],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fanout/minimal"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "final synthesized answer"
