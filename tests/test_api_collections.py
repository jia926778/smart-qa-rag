from __future__ import annotations


class TestCollectionsApi:
    def test_create_collection(self, client):
        resp = client.post(
            "/api/v1/collections/",
            json={"name": "test", "description": "Test collection"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test"

    def test_list_collections(self, client):
        resp = client.get("/api/v1/collections/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_delete_collection(self, client):
        resp = client.delete("/api/v1/collections/default")
        assert resp.status_code == 200

    def test_collection_stats(self, client):
        resp = client.get("/api/v1/collections/default/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents_count" in data
