from fastapi.testclient import TestClient
from fanout_openrouter.app import create_app
from fanout_openrouter.settings import Settings


def test_rate_limiting():
    settings = Settings(openrouter_api_key=None, rate_limit_rpm=2)
    app = create_app(settings=settings)
    client = TestClient(app)

    payload = {
        "model": "fanout/minimal",
        "messages": [{"role": "user", "content": "hi"}],
    }

    resp1 = client.post(
        "/api/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer key1"},
    )
    assert resp1.status_code != 429

    resp2 = client.post(
        "/api/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer key1"},
    )
    assert resp2.status_code != 429

    resp3 = client.post(
        "/api/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer key1"},
    )
    assert resp3.status_code == 429
    assert resp3.json()["error"]["message"] == "Too many requests"

    resp4 = client.post(
        "/api/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer key2"},
    )
    assert resp4.status_code != 429
