"""
数据模型包

本包定义了智能问答系统的所有数据模型，包括：
- schemas: API 请求/响应模型和数据传输对象（DTO）

这些模型使用 Pydantic 进行数据验证和序列化，确保类型安全和数据完整性。
"""

# 导出所有公共模型，方便外部模块导入使用
from .schemas import (
    AskRequest,
    AskResponse,
    ChatMessage,
    CollectionCreateRequest,
    CollectionInfo,
    CollectionStats,
    ErrorResponse,
    SourceInfo,
    UploadResponse,
)

__all__ = [
    # 问答相关模型
    "ChatMessage",
    "AskRequest",
    "AskResponse",
    "SourceInfo",
    # 文档上传相关模型
    "UploadResponse",
    # 知识库管理相关模型
    "CollectionCreateRequest",
    "CollectionInfo",
    "CollectionStats",
    # 错误处理相关模型
    "ErrorResponse",
]
