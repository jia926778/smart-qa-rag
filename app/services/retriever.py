from __future__ import annotations

from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.services.bm25_retriever import BM25RetrieverService
from app.services.reranker import BaseReranker
from app.utils.exceptions import RetrievalError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _rrf_merge(
    doc_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            key = _doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
                scores[key] = 0.0
            scores[key] += 1.0 / (k + rank)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = scores[key]
        result.append(doc)
    return result


def _doc_key(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    parent_id = doc.metadata.get("parent_id", "")
    if parent_id:
        return f"{source}::{parent_id}"
    return f"{source}::{hash(doc.page_content[:200])}"


class SmartRetriever:
    """Multi-stage hybrid retrieval pipeline:

    1. **Vector retrieval** — semantic child-chunk search via ChromaDB.
    2. **BM25 retrieval** — keyword search via rank_bm25 index.
    3. **Hybrid fusion** — merge vector + BM25 results via RRF.
    4. **Parent expansion** — look up parent chunks for richer context.
    5. **Reranking** — cross-encoder / LLM / cosine rerank.

    The retrieval strategy (vector / bm25 / hybrid) can be controlled
    per-query via the ``strategy`` parameter.
    """

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings,
        reranker: Optional[BaseReranker] = None,
        bm25_service: Optional[BM25RetrieverService] = None,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._reranker = reranker
        self._bm25 = bm25_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int | None = None,
        score_threshold: float | None = None,
        strategy: str = "hybrid",
    ) -> List[Document]:
        """Retrieve documents using the specified strategy.

        Args:
            strategy: "vector", "bm25", or "hybrid" (default).
        """
        top_k = top_k or self._settings.RETRIEVAL_TOP_K
        score_threshold = score_threshold or self._settings.RETRIEVAL_SCORE_THRESHOLD

        try:
            doc_lists: List[List[Document]] = []

            # Vector retrieval
            if strategy in ("vector", "hybrid"):
                vector_docs = self._retrieve_children(query, collection_name, score_threshold)
                if vector_docs:
                    for d in vector_docs:
                        d.metadata["retrieval_method"] = "vector"
                    doc_lists.append(vector_docs)

            # BM25 retrieval
            if strategy in ("bm25", "hybrid") and self._bm25 and self._settings.BM25_ENABLED:
                bm25_docs = self._bm25.search(
                    query=query,
                    collection_name=collection_name,
                    top_k=self._settings.BM25_TOP_K,
                )
                if bm25_docs:
                    doc_lists.append(bm25_docs)

            if not doc_lists:
                logger.info("No docs found from any retrieval method.")
                return []

            # Merge via RRF if multiple sources
            if len(doc_lists) == 1:
                candidates = doc_lists[0]
            else:
                candidates = _rrf_merge(doc_lists)
                logger.info(
                    "Hybrid fusion: vector(%d) + bm25(%d) -> %d merged",
                    len(doc_lists[0]) if doc_lists else 0,
                    len(doc_lists[1]) if len(doc_lists) > 1 else 0,
                    len(candidates),
                )

            # Parent expansion
            parent_docs = self._expand_to_parents(candidates, collection_name)

            # Reranking
            if self._reranker and self._settings.RERANKER_ENABLED:
                final_docs = self._reranker.rerank(
                    query=query,
                    documents=parent_docs,
                    top_n=self._settings.RERANKER_TOP_N,
                )
            else:
                final_docs = parent_docs[:top_k]

            logger.info(
                "Retrieval pipeline [%s]: %d candidates -> %d parents -> %d final "
                "(collection=%s)",
                strategy,
                len(candidates),
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
    # Stage 1a — vector child chunk retrieval
    # ------------------------------------------------------------------

    def _retrieve_children(
        self, query: str, collection_name: str, score_threshold: float,
    ) -> List[Document]:
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=self._settings.CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
        )
        initial_k = self._settings.RETRIEVAL_INITIAL_K
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": initial_k, "score_threshold": score_threshold},
        )
        docs = retriever.invoke(query)
        child_docs = [d for d in docs if d.metadata.get("chunk_type") == "child"]
        if not child_docs:
            child_docs = docs
        return child_docs

    # ------------------------------------------------------------------
    # Stage 2 — parent chunk expansion
    # ------------------------------------------------------------------

    def _expand_to_parents(
        self, child_docs: List[Document], collection_name: str,
    ) -> List[Document]:
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

        if not has_parent_meta or not parent_ids:
            return child_docs

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
            logger.warning("Parent fetch failed: %s. Using children.", exc)
            return child_docs

        fetched_parents: Dict[str, Document] = {}
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for doc_id, text, meta in zip(ids, documents, metadatas):
            pid = (meta or {}).get("parent_id")
            if pid and pid not in fetched_parents:
                parent_doc = Document(page_content=text or "", metadata=meta or {})
                parent_doc.metadata["matched_children_count"] = len(child_by_parent.get(pid, []))
                fetched_parents[pid] = parent_doc

        parent_docs: List[Document] = []
        for pid in parent_ids:
            if pid in fetched_parents:
                parent_docs.append(fetched_parents[pid])
            else:
                fallback = child_by_parent.get(pid, [None])[0]
                if fallback:
                    parent_docs.append(fallback)
        return parent_docs

    # ------------------------------------------------------------------
    # Legacy flat retrieval
    # ------------------------------------------------------------------

    def retrieve_flat(
        self, query: str, collection_name: str = "default",
        top_k: int | None = None, score_threshold: float | None = None,
    ) -> List[Document]:
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
