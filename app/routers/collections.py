"""
知识库集合管理路由模块

本模块提供知识库集合（Collection）的 REST API 接口，包括：
- 创建知识库集合
- 列出所有集合
- 删除指定集合
- 获取集合统计信息

这些接口是 RAG 系统中知识库管理的核心入口。
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends

from app.dependencies import get_collection_service
from app.models.schemas import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionStats,
)
from app.services.collection_service import CollectionService

# 创建路由器实例，用于注册集合管理相关的 API 端点
router = APIRouter()


@router.post("/", response_model=CollectionInfo, summary="Create a collection")
async def create_collection(
    body: CollectionCreateRequest,
    svc: CollectionService = Depends(get_collection_service),
) -> CollectionInfo:
    """
    创建新的知识库集合

    Args:
        body: 集合创建请求体，包含集合名称和描述
        svc: 集合服务实例，通过依赖注入获取

    Returns:
        CollectionInfo: 创建成功的集合信息对象

    Raises:
        HTTPException: 当集合名称已存在或创建失败时抛出
    """
    # 调用服务层创建集合
    return svc.create(name=body.name, description=body.description)


@router.get("/", response_model=List[CollectionInfo], summary="List all collections")
async def list_collections(
    svc: CollectionService = Depends(get_collection_service),
) -> List[CollectionInfo]:
    """
    获取所有知识库集合列表

    Args:
        svc: 集合服务实例，通过依赖注入获取

    Returns:
        List[CollectionInfo]: 所有集合的信息列表
    """
    # 调用服务层获取所有集合信息
    return svc.list_all()


@router.delete("/{name}", summary="Delete a collection")
async def delete_collection(
    name: str,
    svc: CollectionService = Depends(get_collection_service),
) -> dict:
    """
    删除指定的知识库集合

    Args:
        name: 要删除的集合名称
        svc: 集合服务实例，通过依赖注入获取

    Returns:
        dict: 包含删除成功消息的字典

    Raises:
        HTTPException: 当集合不存在或删除失败时抛出
    """
    # 调用服务层删除指定集合
    svc.delete(name)
    return {"message": f"Collection '{name}' deleted"}


@router.get("/{name}/stats", response_model=CollectionStats, summary="Collection stats")
async def collection_stats(
    name: str,
    svc: CollectionService = Depends(get_collection_service),
) -> CollectionStats:
    """
    获取指定集合的统计信息

    Args:
        name: 集合名称
        svc: 集合服务实例，通过依赖注入获取

    Returns:
        CollectionStats: 集合的统计信息，包括文档数量、向量数量等

    Raises:
        HTTPException: 当集合不存在时抛出
    """
    # 调用服务层获取集合统计信息
    return svc.stats(name)
