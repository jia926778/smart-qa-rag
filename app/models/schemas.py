"""
数据模型模块

本模块定义了智能问答系统的所有数据传输对象（DTO）和请求/响应模型。
使用 Pydantic 进行数据验证和序列化，确保 API 接口的数据完整性。

主要包含以下模型类别：
- 问答相关：聊天消息、问答请求/响应、来源信息
- 文档上传：上传响应模型
- 知识库管理：集合创建、信息查询、统计信息
- 错误处理：统一错误响应格式
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat / QA - 问答相关模型
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """
    聊天消息模型
    
    表示对话中的单条消息，用于维护对话上下文和历史记录。
    """
    role: str = Field(..., description="角色：'user'（用户）或 'assistant'（助手）")
    content: str = Field(..., description="消息内容")


class AskRequest(BaseModel):
    """
    问答请求模型
    
    封装用户提问的所有必要信息，包括问题内容、目标知识库和历史对话。
    """
    question: str = Field(..., min_length=1, description="用户提出的问题")
    collection_name: str = Field(
        default="default", description="目标知识库集合名称"
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list, description="最近的对话历史记录"
    )


class SourceInfo(BaseModel):
    """
    来源信息模型
    
    表示答案引用的文档来源，包含文档路径、页码和相关内容片段。
    """
    source: str = Field(..., description="源文档名称或路径")
    page: Optional[int] = Field(None, description="页码（如果适用）")
    content: str = Field(..., description="相关内容片段")


class AskResponse(BaseModel):
    """
    问答响应模型
    
    封装问答系统的完整响应，包括生成的答案、引用来源和处理耗时。
    """
    answer: str  # 生成的答案内容
    sources: List[SourceInfo] = Field(default_factory=list)  # 引用的来源列表
    elapsed_ms: float = Field(..., description="处理耗时（毫秒）")


# ---------------------------------------------------------------------------
# Document upload - 文档上传相关模型
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """
    文档上传响应模型
    
    返回文档上传处理的结果信息，包括文件名、目标集合、分块数量和状态消息。
    """
    filename: str  # 上传的文件名
    collection_name: str  # 存入的知识库集合名称
    chunks_count: int  # 文档分块数量
    message: str  # 处理结果消息


# ---------------------------------------------------------------------------
# Collection management - 知识库管理相关模型
# ---------------------------------------------------------------------------

class CollectionCreateRequest(BaseModel):
    """
    知识库创建请求模型
    
    定义创建新知识库集合所需的参数。
    """
    name: str = Field(..., min_length=1, max_length=128)  # 集合名称，长度限制1-128字符
    description: str = Field(default="")  # 集合描述，可选


class CollectionInfo(BaseModel):
    """
    知识库信息模型
    
    表示知识库集合的详细信息，包括名称、描述、文档数量和创建时间。
    """
    name: str  # 集合名称
    description: str = ""  # 集合描述
    documents_count: int = 0  # 包含的文档数量
    created_at: Optional[datetime] = None  # 创建时间


class CollectionStats(BaseModel):
    """
    知识库统计信息模型
    
    提供知识库集合的统计概览，用于列表展示和监控。
    """
    name: str  # 集合名称
    documents_count: int = 0  # 文档数量
    description: str = ""  # 集合描述


# ---------------------------------------------------------------------------
# Errors - 错误处理相关模型
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """
    错误响应模型
    
    统一的 API 错误响应格式，包含错误详情和可选的错误代码。
    """
    detail: str  # 错误详细信息
    error_code: Optional[str] = None  # 错误代码，用于程序化错误识别
