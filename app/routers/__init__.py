"""
路由模块包

本包包含 Smart-QA-RAG 系统的所有 API 路由模块：
- collections: 知识库集合管理路由
- documents: 文档上传和管理路由
- qa: 问答交互路由

这些路由模块共同构成了系统的 REST API 层，负责：
1. 接收 HTTP 请求
2. 参数验证和转换
3. 调用业务服务层处理请求
4. 返回格式化的响应

所有路由都通过 FastAPI 的 APIRouter 进行组织，最终在主应用中注册。
"""

from app.routers.collections import router as collections_router
from app.routers.documents import router as documents_router
from app.routers.qa import router as qa_router

__all__ = [
    "collections_router",
    "documents_router",
    "qa_router",
]
