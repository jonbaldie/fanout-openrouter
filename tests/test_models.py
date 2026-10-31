import json
from pathlib import Path

from fastapi.testclient import TestClient

from fanout_openrouter.app import create_app
from fanout_openrouter.settings import Settings


def test_models_endpoint_lists_virtual_models(tmp_path: Path) -> None:
    policy_file = tmp_path / "policies.json"
    policy_file.write_text(
        json.dumps(
            {
                "policies": [
                    {"virtual_model": "fanout/first", "candidate_models": ["a/model"]},
                    {"virtual_model": "fanout/second", "candidate_models": ["b/model"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        settings=Settings(openrouter_api_key=None, policy_file=str(policy_file))
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/models")

    assert response.status_code == 200, response.text
    body = response.json()
    assert "object" not in body
    assert [entry["id"] for entry in body["data"]] == ["fanout/first", "fanout/second"]
    entry = body["data"][0]
    assert entry["canonical_slug"] == "fanout/first"
    assert "architecture" in entry
    assert "pricing" in entry
    assert "supported_parameters" in entry
