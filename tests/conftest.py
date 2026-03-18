from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def test_settings(tmp_path):
    """Return a Settings object pointing to a temporary ChromaDB directory."""
    return Settings(
        OPENAI_API_KEY="test-key",
        OPENAI_API_BASE=None,
        LLM_MODEL="gpt-3.5-turbo",
        EMBEDDING_PROVIDER="openai",
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
        RETRIEVAL_TOP_K=3,
        RETRIEVAL_SCORE_THRESHOLD=0.3,
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=40,
        MAX_UPLOAD_SIZE_MB=10,
        LOG_LEVEL="DEBUG",
    )


@pytest.fixture()
def mock_embedding():
    """Return a mocked Embeddings object."""
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1] * 384]
    emb.embed_query.return_value = [0.1] * 384
    return emb


@pytest.fixture()
def mock_qa_service():
    """Return a mocked QAService."""
    from app.models.schemas import AskResponse

    svc = MagicMock()
    svc.ask = AsyncMock(
        return_value=AskResponse(answer="Test answer", sources=[], elapsed_ms=42.0)
    )
    return svc


@pytest.fixture()
def mock_document_service():
    """Return a mocked DocumentService."""
    from app.models.schemas import UploadResponse

    svc = MagicMock()
    svc.ingest = AsyncMock(
        return_value=UploadResponse(
            filename="test.txt",
            collection_name="default",
            chunks_count=3,
            message="ok",
        )
    )
    svc.list_documents.return_value = [
        {"source": "test.txt", "id": "abc", "chunks": 3}
    ]
    svc.delete_document.return_value = 3
    return svc


@pytest.fixture()
def mock_collection_service():
    """Return a mocked CollectionService."""
    from app.models.schemas import CollectionInfo, CollectionStats

    svc = MagicMock()
    svc.create.return_value = CollectionInfo(
        name="test", description="desc", documents_count=0
    )
    svc.list_all.return_value = [
        CollectionInfo(name="default", description="", documents_count=5)
    ]
    svc.delete.return_value = None
    svc.stats.return_value = CollectionStats(
        name="default", documents_count=5, description=""
    )
    return svc


@pytest.fixture()
def client(mock_qa_service, mock_document_service, mock_collection_service):
    """TestClient with all services mocked."""
    from app import dependencies

    app = create_app()

    app.dependency_overrides[dependencies.get_qa_service] = lambda: mock_qa_service
    app.dependency_overrides[dependencies.get_document_service] = (
        lambda: mock_document_service
    )
    app.dependency_overrides[dependencies.get_collection_service] = (
        lambda: mock_collection_service
    )

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()
