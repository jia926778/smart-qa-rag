from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain_core.documents import Document

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Interface that all reranker implementations must follow."""

    @abstractmethod
    def rerank(
        self, query: str, documents: List[Document], top_n: int
    ) -> List[Document]:
        """Return the *top_n* most relevant documents for *query*."""


# ---------------------------------------------------------------------------
# Cross-Encoder reranker  (sentence-transformers)
# ---------------------------------------------------------------------------

class CrossEncoderReranker(BaseReranker):
    """Rerank using a cross-encoder model from sentence-transformers.

    Cross-encoders jointly encode (query, document) pairs and produce a
    single relevance score.  This is significantly more accurate than
    bi-encoder cosine similarity but slower — which is fine because we
    only run it on the small candidate set returned by the first-stage
    retrieval.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        logger.info("Loading cross-encoder model: %s", model_name)
        self._model = CrossEncoder(model_name, trust_remote_code=True)
        self._model_name = model_name

    def rerank(
        self, query: str, documents: List[Document], top_n: int
    ) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        # Attach scores to docs, sort descending, take top_n
        scored: List[Tuple[float, Document]] = []
        for score, doc in zip(scores, documents):
            doc.metadata["rerank_score"] = float(score)
            scored.append((float(score), doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [doc for _, doc in scored[:top_n]]
        logger.info(
            "CrossEncoder reranked %d -> %d docs (model=%s)",
            len(documents),
            len(result),
            self._model_name,
        )
        return result


# ---------------------------------------------------------------------------
# LLM-based reranker  (uses existing OpenAI connection)
# ---------------------------------------------------------------------------

class LLMReranker(BaseReranker):
    """Rerank by asking the LLM to score each document's relevance.

    This is a fallback for environments where loading a local
    cross-encoder model is impractical.  Each candidate document is
    scored with a lightweight prompt asking the LLM to rate relevance
    from 0 to 10.
    """

    _SCORE_PROMPT = (
        "Rate the relevance of the following document to the query on a scale "
        "of 0 to 10, where 0 means completely irrelevant and 10 means "
        "perfectly relevant.  Reply with ONLY a single integer.\n\n"
        "Query: {query}\n\n"
        "Document:\n{document}\n\n"
        "Relevance score:"
    )

    def __init__(self, settings: Settings) -> None:
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": settings.LLM_MODEL,
            "temperature": 0,
            "max_tokens": 4,
            "openai_api_key": settings.OPENAI_API_KEY,
        }
        if settings.OPENAI_API_BASE:
            kwargs["openai_api_base"] = settings.OPENAI_API_BASE
        self._llm = ChatOpenAI(**kwargs)

    def rerank(
        self, query: str, documents: List[Document], top_n: int
    ) -> List[Document]:
        if not documents:
            return []

        scored: List[Tuple[float, Document]] = []
        for doc in documents:
            prompt = self._SCORE_PROMPT.format(
                query=query, document=doc.page_content[:800]
            )
            try:
                resp = self._llm.invoke(prompt)
                raw = resp.content.strip()
                score = float(raw)
            except (ValueError, TypeError):
                score = 0.0
            except Exception as exc:
                logger.warning("LLM rerank scoring failed: %s", exc)
                score = 0.0
            doc.metadata["rerank_score"] = score
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [doc for _, doc in scored[:top_n]]
        logger.info(
            "LLM reranked %d -> %d docs", len(documents), len(result)
        )
        return result


# ---------------------------------------------------------------------------
# Simple cosine similarity reranker  (no extra model needed)
# ---------------------------------------------------------------------------

class CosineReranker(BaseReranker):
    """Lightweight fallback: re-score using embedding cosine similarity.

    Useful when neither a cross-encoder model nor LLM calls are
    desirable.  Uses the same embedding model already loaded by the
    application.
    """

    def __init__(self, embeddings) -> None:
        self._embeddings = embeddings

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def rerank(
        self, query: str, documents: List[Document], top_n: int
    ) -> List[Document]:
        if not documents:
            return []

        query_vec = self._embeddings.embed_query(query)
        doc_texts = [d.page_content for d in documents]
        doc_vecs = self._embeddings.embed_documents(doc_texts)

        scored: List[Tuple[float, Document]] = []
        for vec, doc in zip(doc_vecs, documents):
            sim = self._cosine_sim(query_vec, vec)
            doc.metadata["rerank_score"] = sim
            scored.append((sim, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [doc for _, doc in scored[:top_n]]
        logger.info(
            "Cosine reranked %d -> %d docs", len(documents), len(result)
        )
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_reranker(settings: Settings, embeddings=None) -> BaseReranker:
    """Build the appropriate reranker based on configuration."""
    provider = settings.RERANKER_PROVIDER.lower()
    if provider == "cross-encoder":
        return CrossEncoderReranker(model_name=settings.RERANKER_MODEL)
    elif provider == "llm":
        return LLMReranker(settings)
    elif provider == "cosine":
        if embeddings is None:
            raise ValueError("CosineReranker requires an embeddings instance")
        return CosineReranker(embeddings)
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
