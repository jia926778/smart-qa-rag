from __future__ import annotations

import io


class TestDocumentsApi:
    def test_upload_document(self, client):
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
            data={"collection_name": "default"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.txt"
        assert data["chunks_count"] == 3

    def test_list_documents(self, client):
        resp = client.get("/api/v1/documents/default")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_delete_document(self, client):
        resp = client.delete("/api/v1/documents/default/test.txt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted_chunks"] == 3
