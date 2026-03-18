from __future__ import annotations

from functools import lru_cache

import chromadb

from app.config import Settings, settings
from app.services.collection_service import CollectionService
from app.services.document_service import DocumentService
from app.services.embedding_engine import EmbeddingEngine
from app.services.prompt_builder import PromptBuilder
from app.services.qa_service import QAService
from app.services.reranker import BaseReranker, create_reranker
from app.services.retriever import SmartRetriever
from app.services.text_splitter import ParentChildTextSplitter, TextSplitterService
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Singleton-style providers
# ---------------------------------------------------------------------------

@lru_cache()
def get_settings() -> Settings:
    return settings


@lru_cache()
def get_chroma_client() -> chromadb.ClientAPI:
    s = get_settings()
    return chromadb.PersistentClient(path=s.CHROMA_PERSIST_DIR)


@lru_cache()
def get_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine(get_settings())


@lru_cache()
def get_collection_service() -> CollectionService:
    return CollectionService(get_chroma_client())


@lru_cache()
def get_text_splitter() -> TextSplitterService:
    return TextSplitterService(get_settings())


@lru_cache()
def get_parent_child_splitter() -> ParentChildTextSplitter:
    return ParentChildTextSplitter(get_settings())


@lru_cache()
def get_reranker() -> BaseReranker | None:
    """Build the reranker.  Returns None if reranking is disabled."""
    s = get_settings()
    if not s.RERANKER_ENABLED:
        return None
    try:
        return create_reranker(
            settings=s,
            embeddings=get_embedding_engine().embeddings,
        )
    except Exception as exc:
        logger.warning(
            "Failed to initialise reranker (%s), disabling: %s",
            s.RERANKER_PROVIDER,
            exc,
        )
        return None


@lru_cache()
def get_retriever() -> SmartRetriever:
    return SmartRetriever(
        settings=get_settings(),
        embeddings=get_embedding_engine().embeddings,
        reranker=get_reranker(),
    )


@lru_cache()
def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@lru_cache()
def get_qa_service() -> QAService:
    return QAService(
        settings=get_settings(),
        retriever=get_retriever(),
        prompt_builder=get_prompt_builder(),
    )


@lru_cache()
def get_document_service() -> DocumentService:
    return DocumentService(
        settings=get_settings(),
        embeddings=get_embedding_engine().embeddings,
        text_splitter=get_text_splitter(),
        parent_child_splitter=get_parent_child_splitter(),
        collection_service=get_collection_service(),
    )
