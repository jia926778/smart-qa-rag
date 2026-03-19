"""
pytest 配置模块

本模块提供测试所需的共享 fixture，包括：
- 测试配置对象
- 模拟服务对象（QA服务、文档服务、集合服务）
- 测试客户端

这些 fixture 被所有测试文件共享，用于隔离测试环境并提供统一的测试依赖。
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def test_settings(tmp_path):
    """
    创建测试用的配置对象。

    使用临时目录作为 ChromaDB 持久化目录，避免污染真实数据。
    所有配置项使用测试友好的默认值。

    Args:
        tmp_path: pytest 提供的临时目录 fixture

    Returns:
        Settings: 配置好的测试配置对象
    """
    return Settings(
        OPENAI_API_KEY="test-key",  # 测试用的 API 密钥
        OPENAI_API_BASE=None,  # 使用默认 API 地址
        LLM_MODEL="gpt-3.5-turbo",  # 测试用的小型模型
        EMBEDDING_PROVIDER="openai",  # 使用 OpenAI 嵌入
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),  # 临时 ChromaDB 目录
        RETRIEVAL_TOP_K=3,  # 检索返回前 3 个结果
        RETRIEVAL_SCORE_THRESHOLD=0.3,  # 相似度阈值
        CHUNK_SIZE=200,  # 文本分块大小
        CHUNK_OVERLAP=40,  # 分块重叠字符数
        MAX_UPLOAD_SIZE_MB=10,  # 最大上传文件大小
        LOG_LEVEL="DEBUG",  # 详细日志级别便于调试
    )


@pytest.fixture()
def mock_embedding():
    """
    创建模拟的嵌入模型对象。

    返回一个 MagicMock 对象，模拟 LangChain Embeddings 接口，
    提供固定的嵌入向量用于测试。

    Returns:
        MagicMock: 模拟的嵌入对象，返回固定 384 维向量
    """
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1] * 384]  # 批量嵌入返回固定向量
    emb.embed_query.return_value = [0.1] * 384  # 单条查询嵌入返回固定向量
    return emb


@pytest.fixture()
def mock_qa_service():
    """
    创建模拟的问答服务对象。

    返回一个模拟的 QAService，提供预定义的问答响应，
    用于测试 API 端点而无需调用真实的 LLM。

    Returns:
        MagicMock: 模拟的问答服务对象
    """
    from app.models.schemas import AskResponse

    svc = MagicMock()
    # 配置异步 ask 方法返回固定响应
    svc.ask = AsyncMock(
        return_value=AskResponse(answer="Test answer", sources=[], elapsed_ms=42.0)
    )
    return svc


@pytest.fixture()
def mock_document_service():
    """
    创建模拟的文档服务对象。

    返回一个模拟的 DocumentService，提供预定义的文档操作响应，
    用于测试文档上传、列表和删除 API。

    Returns:
        MagicMock: 模拟的文档服务对象
    """
    from app.models.schemas import UploadResponse

    svc = MagicMock()
    # 模拟文档上传（摄取）操作
    svc.ingest = AsyncMock(
        return_value=UploadResponse(
            filename="test.txt",
            collection_name="default",
            chunks_count=3,
            message="ok",
        )
    )
    # 模拟列出文档操作
    svc.list_documents.return_value = [
        {"source": "test.txt", "id": "abc", "chunks": 3}
    ]
    # 模拟删除文档操作，返回删除的分块数
    svc.delete_document.return_value = 3
    return svc


@pytest.fixture()
def mock_collection_service():
    """
    创建模拟的集合服务对象。

    返回一个模拟的 CollectionService，提供预定义的集合操作响应，
    用于测试集合创建、列表、删除和统计 API。

    Returns:
        MagicMock: 模拟的集合服务对象
    """
    from app.models.schemas import CollectionInfo, CollectionStats

    svc = MagicMock()
    # 模拟创建集合操作
    svc.create.return_value = CollectionInfo(
        name="test", description="desc", documents_count=0
    )
    # 模拟列出所有集合操作
    svc.list_all.return_value = [
        CollectionInfo(name="default", description="", documents_count=5)
    ]
    # 模拟删除集合操作
    svc.delete.return_value = None
    # 模拟获取集合统计信息操作
    svc.stats.return_value = CollectionStats(
        name="default", documents_count=5, description=""
    )
    return svc


@pytest.fixture()
def client(mock_qa_service, mock_document_service, mock_collection_service):
    """
    创建配置了所有模拟服务的测试客户端。

    通过 FastAPI 的依赖注入覆盖机制，将真实服务替换为模拟服务，
    实现测试隔离。测试结束后自动清理依赖覆盖。

    Args:
        mock_qa_service: 模拟的问答服务
        mock_document_service: 模拟的文档服务
        mock_collection_service: 模拟的集合服务

    Yields:
        TestClient: 配置好的 FastAPI 测试客户端
    """
    from app import dependencies

    app = create_app()

    # 覆盖依赖注入，使用模拟服务替代真实服务
    app.dependency_overrides[dependencies.get_qa_service] = lambda: mock_qa_service
    app.dependency_overrides[dependencies.get_document_service] = (
        lambda: mock_document_service
    )
    app.dependency_overrides[dependencies.get_collection_service] = (
        lambda: mock_collection_service
    )

    # 创建测试客户端，禁用服务器异常自动抛出以便测试错误响应
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # 清理依赖覆盖，避免影响其他测试
    app.dependency_overrides.clear()
