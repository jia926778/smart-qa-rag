from __future__ import annotations

from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.services.reranker import BaseReranker
from app.utils.exceptions import RetrievalError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SmartRetriever:
    """Multi-stage retrieval pipeline:

    1. **Child-chunk retrieval** — search the vector store for small,
       precise child chunks (high recall via over-retrieval).
    2. **Parent-chunk expansion** — for each matched child, look up its
       parent chunk to obtain broader context.  Deduplicate parents so
       that the same parent is not included multiple times.
    3. **Reranking** — pass the parent-level documents through a
       reranker (cross-encoder / LLM / cosine) to surface the most
       relevant results.
    4. **Return** the top-N reranked parent chunks with enriched
       metadata.

    Falls back to flat retrieval when documents lack parent-child
    metadata (backward compatible).
    """

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings,
        reranker: Optional[BaseReranker] = None,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._reranker = reranker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> List[Document]:
        top_k = top_k or self._settings.RETRIEVAL_TOP_K
        score_threshold = score_threshold or self._settings.RETRIEVAL_SCORE_THRESHOLD

        try:
            # Stage 1: over-retrieve child chunks
            child_docs = self._retrieve_children(
                query, collection_name, score_threshold
            )

            if not child_docs:
                logger.info("No child docs found, returning empty result.")
                return []

            # Stage 2: expand to parent chunks
            parent_docs = self._expand_to_parents(child_docs, collection_name)

            # Stage 3: rerank
            if self._reranker and self._settings.RERANKER_ENABLED:
                final_docs = self._reranker.rerank(
                    query=query,
                    documents=parent_docs,
                    top_n=self._settings.RERANKER_TOP_N,
                )
            else:
                final_docs = parent_docs[: top_k]

            logger.info(
                "Retrieval pipeline: %d children -> %d parents -> %d final "
                "(collection=%s)",
                len(child_docs),
                len(parent_docs),
                len(final_docs),
                collection_name,
            )
            return final_docs

        except RetrievalError:
            raise
        except Exception as exc:
            logger.error("Retrieval error: %s", exc)
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Stage 1 — child chunk retrieval
    # ------------------------------------------------------------------

    def _retrieve_children(
        self,
        query: str,
        collection_name: str,
        score_threshold: float,
    ) -> List[Document]:
        """Retrieve small child chunks via embedding similarity."""
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=self._settings.CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
        )

        initial_k = self._settings.RETRIEVAL_INITIAL_K
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": initial_k,
                "score_threshold": score_threshold,
            },
        )
        docs = retriever.invoke(query)

        # If using parent-child, filter to child-type only
        child_docs = [
            d for d in docs if d.metadata.get("chunk_type") == "child"
        ]

        # Backward compat: if none are tagged, treat all as children
        if not child_docs:
            child_docs = docs

        return child_docs

    # ------------------------------------------------------------------
    # Stage 2 — parent chunk expansion
    # ------------------------------------------------------------------

    def _expand_to_parents(
        self,
        child_docs: List[Document],
        collection_name: str,
    ) -> List[Document]:
        """For each child, fetch its parent chunk.  Deduplicate by parent_id."""

        # Collect unique parent IDs
        parent_ids: List[str] = []
        seen: set = set()
        child_by_parent: Dict[str, List[Document]] = {}
        has_parent_meta = False

        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if pid:
                has_parent_meta = True
                if pid not in seen:
                    parent_ids.append(pid)
                    seen.add(pid)
                child_by_parent.setdefault(pid, []).append(child)

        # If no parent metadata exists (flat-chunked docs), return as-is
        if not has_parent_meta:
            logger.debug("No parent metadata found; using child docs directly.")
            return child_docs

        if not parent_ids:
            return child_docs

        # Fetch parent documents from the PARENT collection
        parent_collection = f"{collection_name}_parents"
        try:
            parent_store = Chroma(
                collection_name=parent_collection,
                persist_directory=self._settings.CHROMA_PERSIST_DIR,
                embedding_function=self._embeddings,
            )
            result = parent_store.get(
                where={"parent_id": {"$in": parent_ids}},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning(
                "Could not fetch parents from '%s': %s. "
                "Falling back to child docs.",
                parent_collection,
                exc,
            )
            return child_docs

        # Build parent_id -> parent Document map
        fetched_parents: Dict[str, Document] = {}
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for doc_id, text, meta in zip(ids, documents, metadatas):
            pid = (meta or {}).get("parent_id")
            if pid and pid not in fetched_parents:
                parent_doc = Document(page_content=text or "", metadata=meta or {})
                # Enrich metadata with matched child count
                matched_children = child_by_parent.get(pid, [])
                parent_doc.metadata["matched_children_count"] = len(matched_children)
                fetched_parents[pid] = parent_doc

        # Build ordered result, preserving the child-retrieval order
        parent_docs: List[Document] = []
        for pid in parent_ids:
            if pid in fetched_parents:
                parent_docs.append(fetched_parents[pid])
            else:
                # Fallback: use the first matched child if parent not found
                fallback = child_by_parent.get(pid, [None])[0]
                if fallback:
                    parent_docs.append(fallback)

        return parent_docs

    # ------------------------------------------------------------------
    # Legacy flat retrieval (no parent-child, no reranking)
    # ------------------------------------------------------------------

    def retrieve_flat(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> List[Document]:
        """Simple single-stage retrieval without parent expansion or reranking."""
        top_k = top_k or self._settings.RETRIEVAL_TOP_K
        score_threshold = score_threshold or self._settings.RETRIEVAL_SCORE_THRESHOLD

        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=self._settings.CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold},
        )
        return retriever.invoke(query)
