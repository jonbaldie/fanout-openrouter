import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fanout_openrouter.app import create_app
from fanout_openrouter.settings import Settings


@pytest.mark.parametrize(
    ("primary_frames", "backup_fails", "expected_model", "expected_attempts"),
    [
        ([], False, "backup", ["primary", "backup"]),
        (["comment"], False, "backup", ["primary", "backup"]),
        (["comment"], True, "first", ["primary", "backup"]),
        (["role", "comment"], False, "primary", ["primary"]),
    ],
)
def test_stream_fallback_boundary(
    tmp_path, primary_frames, backup_fails, expected_model, expected_attempts
):
    policy = {
        "virtual_model": "fanout/test",
        "candidate_models": ["first"],
        "fanout_count": 2,
        "synthesis_model": "primary",
        "default_fallback_model": "backup",
    }
    policy_file = tmp_path / "policies.json"
    policy_file.write_text(json.dumps({"policies": [policy]}))
    comment = ": OPENROUTER PROCESSING"
    role = 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
    error = 'data: {"error":{"message":"interrupted","code":502}}'
    attempts = []

    def upstream(request):
        payload = json.loads(request.content)
        model = payload["model"]
        if not payload["stream"]:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "candidate answer"}}]}
            )
        attempts.append(model)
        if model == "primary":
            frames = [
                comment if frame == "comment" else role for frame in primary_frames
            ]
            frames.append(error)
        elif backup_fails:
            frames = [comment, error]
        else:
            frames = [
                'data: {"choices":[{"delta":{"content":"backup answer"}}]}',
                "data: [DONE]",
            ]
        return httpx.Response(200, text="\n\n".join(frames) + "\n\n")

    app = create_app(
        settings=Settings(openrouter_api_key="test", policy_file=str(policy_file)),
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "fanout/test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert attempts == expected_attempts
    lines = response.text.splitlines()
    assert lines.count(comment) == primary_frames.count("comment") + int(backup_fails)
    events = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    assert {event["model"] for event in events if "choices" in event} == {
        expected_model
    }
    if expected_model == "primary":
        assert events[-1]["error"]["message"] == "interrupted"
        assert "data: [DONE]" not in lines
    else:
        answer = "".join(
            event["choices"][0]["delta"].get("content", "")
            for event in events
            if event.get("choices")
        )
        assert answer == ("candidate answer" if backup_fails else "backup answer")
        assert "data: [DONE]" in lines


@pytest.mark.parametrize("stream", [False, True])
def test_shared_candidate_refusal_reaches_both_response_modes(stream):
    def refuse(request):
        return httpx.Response(
            403, json={"error": {"message": "access denied", "code": 403}}
        )

    app = create_app(
        settings=Settings(openrouter_api_key="test"),
        transport=httpx.MockTransport(refuse),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "first,second",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )

    assert response.status_code == 403
    assert response.json()["error"]["message"] == "access denied"
    assert response.json()["error"]["code"] == 403
