from __future__ import annotations

from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.utils.exceptions import RetrievalError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SmartRetriever:
    """Retrieve relevant documents from ChromaDB using similarity search with score threshold."""

    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self._settings = settings
        self._embeddings = embeddings

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
            vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=self._settings.CHROMA_PERSIST_DIR,
                embedding_function=self._embeddings,
            )

            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": top_k,
                    "score_threshold": score_threshold,
                },
            )

            docs = retriever.invoke(query)
            logger.info(
                "Retrieved %d doc(s) for query (collection=%s)",
                len(docs),
                collection_name,
            )
            return docs
        except Exception as exc:
            logger.error("Retrieval error: %s", exc)
            raise RetrievalError(f"Retrieval failed: {exc}") from exc
