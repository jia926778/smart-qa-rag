from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import chromadb

from app.models.schemas import CollectionInfo, CollectionStats
from app.utils.exceptions import CollectionAlreadyExistsError, CollectionNotFoundError
from app.utils.logger import get_logger

logger = get_logger(__name__)

_METADATA_KEY_DESC = "description"
_METADATA_KEY_CREATED = "created_at"


class CollectionService:
    """CRUD operations for ChromaDB collections."""

    def __init__(self, chroma_client: chromadb.ClientAPI) -> None:
        self._client = chroma_client

    def create(self, name: str, description: str = "") -> CollectionInfo:
        existing = [c.name for c in self._client.list_collections()]
        if name in existing:
            raise CollectionAlreadyExistsError(name)
        now = datetime.now(timezone.utc).isoformat()
        metadata: Dict[str, str] = {
            _METADATA_KEY_DESC: description,
            _METADATA_KEY_CREATED: now,
        }
        self._client.get_or_create_collection(name=name, metadata=metadata)
        logger.info("Created collection '%s'", name)
        return CollectionInfo(
            name=name,
            description=description,
            documents_count=0,
            created_at=now,
        )

    def list_all(self) -> List[CollectionInfo]:
        collections = self._client.list_collections()
        result: List[CollectionInfo] = []
        for col_name in collections:
            name = col_name if isinstance(col_name, str) else col_name.name
            try:
                col = self._client.get_collection(name)
            except Exception:
                continue
            meta = col.metadata or {}
            result.append(
                CollectionInfo(
                    name=name,
                    description=meta.get(_METADATA_KEY_DESC, ""),
                    documents_count=col.count(),
                    created_at=meta.get(_METADATA_KEY_CREATED),
                )
            )
        return result

    def get(self, name: str) -> CollectionInfo:
        try:
            col = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(name) from exc
        meta = col.metadata or {}
        return CollectionInfo(
            name=name,
            description=meta.get(_METADATA_KEY_DESC, ""),
            documents_count=col.count(),
            created_at=meta.get(_METADATA_KEY_CREATED),
        )

    def delete(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
            logger.info("Deleted collection '%s'", name)
        except Exception as exc:
            raise CollectionNotFoundError(name) from exc

    def stats(self, name: str) -> CollectionStats:
        info = self.get(name)
        return CollectionStats(
            name=info.name,
            documents_count=info.documents_count,
            description=info.description,
        )

    def ensure_exists(self, name: str) -> None:
        """Make sure a collection exists; create it if not."""
        existing = [c.name if isinstance(c, str) else c.name if hasattr(c, "name") else c for c in self._client.list_collections()]
        # handle both str and Collection objects
        names = []
        for c in existing:
            if isinstance(c, str):
                names.append(c)
            elif hasattr(c, "name"):
                names.append(c.name)
        if name not in names:
            self.create(name)
