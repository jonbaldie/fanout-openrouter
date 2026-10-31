from fastapi.testclient import TestClient

from fanout_openrouter.app import create_app


def test_models_endpoint_lists_virtual_models() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/api/v1/models")

    assert response.status_code == 200, response.text
    body = response.json()
    assert "object" not in body
    assert [entry["id"] for entry in body["data"]] == ["fanout/minimal"]
    entry = body["data"][0]
    assert entry["canonical_slug"] == "fanout/minimal"
    assert "architecture" in entry
    assert "pricing" in entry
    assert "supported_parameters" in entry
