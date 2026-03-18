from __future__ import annotations


class TestQAApi:
    def test_ask_endpoint(self, client):
        resp = client.post(
            "/api/v1/qa/ask",
            json={"question": "What is RAG?", "collection_name": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "elapsed_ms" in data

    def test_ask_empty_question(self, client):
        resp = client.post(
            "/api/v1/qa/ask",
            json={"question": "", "collection_name": "default"},
        )
        assert resp.status_code == 422

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
