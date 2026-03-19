"""
问答交互路由模块

本模块提供基于 RAG（检索增强生成）的问答 API 接口。
用户提出问题后，系统会：
1. 在知识库中检索相关文档片段
2. 将检索结果作为上下文
3. 使用大语言模型生成答案

这是智能问答系统的核心交互入口。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_qa_service
from app.models.schemas import AskRequest, AskResponse
from app.services.qa_service import QAService

# 创建路由器实例，用于注册问答相关的 API 端点
router = APIRouter()


@router.post("/ask", response_model=AskResponse, summary="Ask a question")
async def ask_question(
    body: AskRequest,
    qa_service: QAService = Depends(get_qa_service),
) -> AskResponse:
    """
    提交问题并获取答案

    该接口实现了完整的 RAG 流程：
    1. 问题理解和改写
    2. 向量检索获取相关文档
    3. 上下文构建
    4. 大模型生成答案

    Args:
        body: 问答请求体，包含用户问题和相关参数
        qa_service: 问答服务实例，通过依赖注入获取

    Returns:
        AskResponse: 问答响应，包含生成的答案和来源信息

    Raises:
        HTTPException: 当问题处理失败或服务不可用时抛出
    """
    # 调用问答服务处理问题
    return await qa_service.ask(body)
