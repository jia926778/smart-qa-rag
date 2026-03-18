from __future__ import annotations

from functools import lru_cache

import chromadb

from app.config import Settings, settings
from app.services.collection_service import CollectionService
from app.services.document_service import DocumentService
from app.services.embedding_engine import EmbeddingEngine
from app.services.prompt_builder import PromptBuilder
from app.services.qa_service import QAService
from app.services.retriever import SmartRetriever
from app.services.text_splitter import TextSplitterService


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
def get_retriever() -> SmartRetriever:
    return SmartRetriever(get_settings(), get_embedding_engine().embeddings)


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
        collection_service=get_collection_service(),
    )
