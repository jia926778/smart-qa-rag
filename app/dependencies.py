"""
依赖注入模块

本模块提供各服务组件的单例式提供者，使用 lru_cache 装饰器实现单例模式。
主要服务包括：配置、向量数据库客户端、嵌入引擎、检索器、问答服务等。
"""

from __future__ import annotations

from functools import lru_cache

import chromadb

from app.config import Settings, settings
from app.services.bm25_retriever import BM25RetrieverService
from app.services.collection_service import CollectionService
from app.services.document_service import DocumentService
from app.services.embedding_engine import EmbeddingEngine
from app.services.qa_service import QAService
from app.services.reranker import BaseReranker, create_reranker
from app.services.retriever import SmartRetriever
from app.services.sql_store import SQLStore
from app.services.text_splitter import ParentChildTextSplitter, TextSplitterService
from app.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 单例式提供者
# ---------------------------------------------------------------------------

@lru_cache()
def get_settings() -> Settings:
    """
    获取全局配置实例

    Returns:
        Settings: 应用配置实例
    """
    return settings


@lru_cache()
def get_chroma_client() -> chromadb.ClientAPI:
    """
    获取 ChromaDB 客户端实例

    Returns:
        chromadb.ClientAPI: ChromaDB 客户端实例
    """
    s = get_settings()
    return chromadb.PersistentClient(path=s.CHROMA_PERSIST_DIR)


@lru_cache()
def get_embedding_engine() -> EmbeddingEngine:
    """
    获取嵌入引擎实例

    Returns:
        EmbeddingEngine: 嵌入引擎实例
    """
    return EmbeddingEngine(get_settings())


@lru_cache()
def get_collection_service() -> CollectionService:
    """
    获取集合服务实例

    Returns:
        CollectionService: 集合服务实例
    """
    return CollectionService(get_chroma_client())


@lru_cache()
def get_text_splitter() -> TextSplitterService:
    """
    获取文本分割服务实例

    Returns:
        TextSplitterService: 文本分割服务实例
    """
    return TextSplitterService(get_settings())


@lru_cache()
def get_parent_child_splitter() -> ParentChildTextSplitter:
    """
    获取父子文本分割器实例

    Returns:
        ParentChildTextSplitter: 父子文本分割器实例
    """
    return ParentChildTextSplitter(get_settings())


@lru_cache()
def get_reranker() -> BaseReranker | None:
    """
    构建重排序器实例

    Returns:
        BaseReranker | None: 重排序器实例，如果禁用则返回 None
    """
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
def get_bm25_service() -> BM25RetrieverService | None:
    """
    构建 BM25 检索服务实例

    Returns:
        BM25RetrieverService | None: BM25 检索服务实例，如果禁用则返回 None
    """
    s = get_settings()
    if not s.BM25_ENABLED:
        return None
    return BM25RetrieverService(s)


@lru_cache()
def get_sql_store() -> SQLStore | None:
    """
    构建 SQL 存储实例

    Returns:
        SQLStore | None: SQL 存储实例，如果禁用则返回 None
    """
    s = get_settings()
    if not s.TEXT_TO_SQL_ENABLED:
        return None
    return SQLStore(s)


@lru_cache()
def get_retriever() -> SmartRetriever:
    """
    获取智能检索器实例

    Returns:
        SmartRetriever: 智能检索器实例
    """
    return SmartRetriever(
        settings=get_settings(),
        embeddings=get_embedding_engine().embeddings,
        reranker=get_reranker(),
        bm25_service=get_bm25_service(),
    )


@lru_cache()
def get_qa_service() -> QAService:
    """
    构建问答服务实例，内部会构建 LangGraph 流水线

    Returns:
        QAService: 问答服务实例
    """
    return QAService(
        settings=get_settings(),
        retriever=get_retriever(),
        sql_store=get_sql_store(),
    )


@lru_cache()
def get_document_service() -> DocumentService:
    """
    获取文档服务实例

    Returns:
        DocumentService: 文档服务实例
    """
    return DocumentService(
        settings=get_settings(),
        embeddings=get_embedding_engine().embeddings,
        text_splitter=get_text_splitter(),
        parent_child_splitter=get_parent_child_splitter(),
        collection_service=get_collection_service(),
        bm25_service=get_bm25_service(),
        sql_store=get_sql_store(),
    )
