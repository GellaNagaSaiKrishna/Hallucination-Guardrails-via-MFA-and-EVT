import pytest
from fastapi.testclient import TestClient

from service.api import create_app


@pytest.fixture
def client(components):
    with TestClient(create_app(components)) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models_loaded"] and body["vector_store_ok"]
    assert body["generator"] == "fake"


def test_ask_answers(client):
    r = client.post("/ask", json={"question": "What drove the increase in Costco's net sales in 2021?"})
    assert r.status_code == 200
    b = r.json()
    assert b["refused"] is False and b["answer"] == "ANSWER(18%)"
    assert len(b["sources"]) == 3 and b["sources"][0]["chunk_id"] == "C_0"
    assert b["gate"]["ood"] is False and b["premise"]["supported"] is True
    assert "generate" in b["timings_ms"] and b["total_ms"] >= 0


def test_ask_refuses_ood(client):
    b = client.post("/ask", json={"question": "Who painted the Mona Lisa?"}).json()
    assert b["refused"] and b["refusal_reason"] == "out_of_distribution" and b["sources"] == []


def test_ask_refuses_false_premise(client):
    b = client.post("/ask", json={"question": "Why did Costco's net sales decline in 2021?", "top_k": 2}).json()
    assert b["refused"] and b["refusal_reason"] == "unsupported_premise"
    assert len(b["sources"]) == 2               # sources are still returned for transparency


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "ok?", "top_k": 0},
                                     {"question": "x" * 2000}])
def test_ask_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_unready_service_returns_503(monkeypatch):
    import service.api as api

    def boom(settings):
        raise RuntimeError("no database")
    monkeypatch.setattr(api, "load_components", boom)
    monkeypatch.setenv("EAGER_LOAD", "1")
    with TestClient(api.create_app()) as c:
        h = c.get("/healthz").json()
        assert h["status"] == "degraded" and "no database" in h["error"]
        assert c.post("/ask", json={"question": "anything at all?"}).status_code == 503
