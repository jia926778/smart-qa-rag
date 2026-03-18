"""BM25 keyword retriever with persistent index.

Uses ``rank_bm25`` for Okapi BM25 scoring.  The index is built from
all child chunks in a collection and persisted to disk as a pickle file
so that it survives process restarts without re-indexing.

For Chinese text, ``jieba`` is used for word segmentation before
feeding into BM25.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25.  Uses jieba for Chinese, whitespace for other."""
    try:
        import jieba  # type: ignore
        # Cut with search mode for better recall
        tokens = list(jieba.cut_for_search(text))
    except ImportError:
        # Fallback: simple char-level + whitespace splitting
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
    # Remove very short tokens
    return [t.strip().lower() for t in tokens if len(t.strip()) > 1]


class BM25Index:
    """In-memory BM25 index over a set of Documents."""

    def __init__(self, documents: List[Document]) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore

        self._documents = documents
        corpus = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._corpus_size = len(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Return (document, score) pairs sorted by BM25 score descending."""
        if not self._bm25 or self._corpus_size == 0:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        # Get top-k indices
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[Document, float]] = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                doc = self._documents[idx]
                results.append((doc, float(score)))
        return results


class BM25RetrieverService:
    """Manage BM25 indices per collection with disk persistence.

    Index files are stored under ``{CHROMA_PERSIST_DIR}/../bm25_indices/``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._index_dir = os.path.join(
            os.path.dirname(settings.CHROMA_PERSIST_DIR), "bm25_indices"
        )
        os.makedirs(self._index_dir, exist_ok=True)
        # In-memory cache: collection_name -> BM25Index
        self._cache: Dict[str, BM25Index] = {}

    def _index_path(self, collection_name: str) -> str:
        return os.path.join(self._index_dir, f"{collection_name}.bm25.pkl")

    def build_index(self, collection_name: str, documents: List[Document]) -> None:
        """Build and persist a BM25 index for the given documents."""
        index = BM25Index(documents)
        self._cache[collection_name] = index

        # Persist to disk
        path = self._index_path(collection_name)
        try:
            with open(path, "wb") as f:
                pickle.dump(documents, f)
            logger.info(
                "BM25 index built and saved: %s (%d docs)",
                collection_name,
                len(documents),
            )
        except Exception as exc:
            logger.warning("Failed to persist BM25 index: %s", exc)

    def add_documents(self, collection_name: str, documents: List[Document]) -> None:
        """Add documents to an existing index (rebuild incrementally)."""
        existing = self._load_docs(collection_name)
        all_docs = existing + documents
        self.build_index(collection_name, all_docs)

    def _load_docs(self, collection_name: str) -> List[Document]:
        """Load persisted documents for a collection."""
        path = self._index_path(collection_name)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Failed to load BM25 docs: %s", exc)
            return []

    def _get_index(self, collection_name: str) -> Optional[BM25Index]:
        """Get or load index from cache/disk."""
        if collection_name in self._cache:
            return self._cache[collection_name]

        docs = self._load_docs(collection_name)
        if docs:
            index = BM25Index(docs)
            self._cache[collection_name] = index
            return index
        return None

    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
    ) -> List[Document]:
        """Search using BM25 keyword matching."""
        index = self._get_index(collection_name)
        if not index:
            logger.debug("No BM25 index for collection '%s'", collection_name)
            return []

        results = index.search(query, top_k=top_k)
        docs = []
        for doc, score in results:
            doc.metadata["bm25_score"] = score
            doc.metadata["retrieval_method"] = "bm25"
            docs.append(doc)

        logger.info(
            "BM25 search: %d results for collection '%s'",
            len(docs),
            collection_name,
        )
        return docs

    def delete_index(self, collection_name: str) -> None:
        """Remove index for a collection."""
        self._cache.pop(collection_name, None)
        path = self._index_path(collection_name)
        if os.path.exists(path):
            os.remove(path)

    def delete_source(self, collection_name: str, source: str) -> None:
        """Remove docs for a specific source and rebuild index."""
        docs = self._load_docs(collection_name)
        filtered = [d for d in docs if d.metadata.get("source") != source]
        if len(filtered) != len(docs):
            self.build_index(collection_name, filtered)
