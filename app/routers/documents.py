"""
文档管理路由模块

本模块提供文档上传和管理的 REST API 接口，包括：
- 上传文档到指定知识库集合
- 列出集合中的所有文档
- 删除指定文档

文档上传后会自动进行分块、向量化处理并存储到向量数据库中。
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.dependencies import get_document_service
from app.models.schemas import UploadResponse
from app.services.document_service import DocumentService

# 创建路由器实例，用于注册文档管理相关的 API 端点
router = APIRouter()


@router.post("/upload", response_model=UploadResponse, summary="Upload a document")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
    doc_service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    """
    上传文档到知识库集合

    文档上传后会自动进行以下处理：
    1. 文档解析和分块
    2. 文本向量化
    3. 存储到向量数据库

    Args:
        file: 上传的文件对象，支持多种文档格式
        collection_name: 目标集合名称，默认为 "default"
        doc_service: 文档服务实例，通过依赖注入获取

    Returns:
        UploadResponse: 上传响应，包含处理结果和文档信息

    Raises:
        HTTPException: 当文件格式不支持或处理失败时抛出
    """
    # 读取上传文件的二进制内容
    file_bytes = await file.read()
    
    # 调用服务层进行文档摄取和向量化处理
    return await doc_service.ingest(
        file_bytes=file_bytes,
        filename=file.filename or "unknown",
        collection_name=collection_name,
    )


@router.get("/{collection_name}", summary="List documents in a collection")
async def list_documents(
    collection_name: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> List[dict]:
    """
    获取指定集合中的所有文档列表

    Args:
        collection_name: 集合名称
        doc_service: 文档服务实例，通过依赖注入获取

    Returns:
        List[dict]: 文档信息列表，每个字典包含文档的元数据信息

    Raises:
        HTTPException: 当集合不存在时抛出
    """
    # 调用服务层获取集合中的文档列表
    return doc_service.list_documents(collection_name)


@router.delete(
    "/{collection_name}/{doc_source}",
    summary="Delete a document by source name",
)
async def delete_document(
    collection_name: str,
    doc_source: str,
    doc_service: DocumentService = Depends(get_document_service),
) -> dict:
    """
    从集合中删除指定文档

    Args:
        collection_name: 集合名称
        doc_source: 文档源名称（通常是文件名）
        doc_service: 文档服务实例，通过依赖注入获取

    Returns:
        dict: 包含删除结果的字典，包括删除的分块数量

    Raises:
        HTTPException: 当集合或文档不存在时抛出
    """
    # 调用服务层删除文档，返回删除的分块数量
    count = doc_service.delete_document(collection_name, doc_source)
    
    # 返回删除结果信息
    return {"deleted_chunks": count, "source": doc_source, "collection": collection_name}
