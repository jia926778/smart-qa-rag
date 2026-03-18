from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.services.collection_service import CollectionService
from app.services.document_loader import DocumentLoaderFactory
from app.services.text_splitter import TextSplitterService
from app.models.schemas import UploadResponse
from app.utils.exceptions import FileTooLargeError
from app.utils.logger import get_logger

logger = get_logger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")


class DocumentService:
    """Handle the full pipeline: save -> load -> split -> embed -> store."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings,
        text_splitter: TextSplitterService,
        collection_service: CollectionService,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._splitter = text_splitter
        self._collection_service = collection_service

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        collection_name: str = "default",
    ) -> UploadResponse:
        # Validate size
        max_bytes = self._settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise FileTooLargeError(self._settings.MAX_UPLOAD_SIZE_MB)

        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Save to temp file with unique name
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        try:
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            # Ensure collection exists
            self._collection_service.ensure_exists(collection_name)

            # Load documents
            documents = DocumentLoaderFactory.load(file_path)

            # Inject original filename into metadata
            for doc in documents:
                doc.metadata["source"] = filename

            # Split
            chunks = self._splitter.split(documents)

            # Store in ChromaDB
            vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=self._settings.CHROMA_PERSIST_DIR,
                embedding_function=self._embeddings,
            )
            vectorstore.add_documents(chunks)

            logger.info(
                "Ingested '%s' -> %d chunks into collection '%s'",
                filename,
                len(chunks),
                collection_name,
            )

            return UploadResponse(
                filename=filename,
                collection_name=collection_name,
                chunks_count=len(chunks),
                message=f"Successfully ingested {filename} ({len(chunks)} chunks)",
            )
        finally:
            # Cleanup temp file
            if os.path.exists(file_path):
                os.remove(file_path)

    def list_documents(self, collection_name: str):
        """Return a list of unique source documents in a collection."""
        self._collection_service.get(collection_name)  # will raise if not found
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=self._settings.CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
        )
        result = vectorstore.get()
        sources: dict = {}
        metadatas = result.get("metadatas") or []
        ids = result.get("ids") or []
        for doc_id, meta in zip(ids, metadatas):
            src = (meta or {}).get("source", "unknown")
            if src not in sources:
                sources[src] = {"source": src, "id": doc_id, "chunks": 0}
            sources[src]["chunks"] += 1
        return list(sources.values())

    def delete_document(self, collection_name: str, doc_source: str) -> int:
        """Delete all chunks belonging to a specific source document."""
        self._collection_service.get(collection_name)  # will raise if not found
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=self._settings.CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
        )
        result = vectorstore.get(where={"source": doc_source})
        ids_to_delete = result.get("ids") or []
        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d chunk(s) for source '%s' from '%s'",
            len(ids_to_delete),
            doc_source,
            collection_name,
        )
        return len(ids_to_delete)
