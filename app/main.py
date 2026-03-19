"""
Smart QA RAG 应用主入口模块

本模块负责创建和配置 FastAPI 应用实例，包括：
- CORS 中间件配置
- API 路由注册
- 异常处理器设置
- 静态文件服务挂载
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.models.schemas import ErrorResponse
from app.routers import collections, documents, qa
from app.utils.exceptions import AppException
from app.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)

# 静态文件目录路径
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例

    Returns:
        FastAPI: 配置完成的应用实例
    """
    # 创建 FastAPI 应用实例
    app = FastAPI(
        title="Smart QA RAG",
        description="Intelligent Q&A system powered by RAG (Retrieval-Augmented Generation)",
        version="1.0.0",
    )

    # --- CORS 中间件配置 -------------------------------------------------------
    # 允许跨域请求，支持所有来源、方法和头部
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- 路由注册 --------------------------------------------------------------
    # 注册各业务模块的 API 路由
    app.include_router(qa.router, prefix="/api/v1/qa", tags=["QA"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(collections.router, prefix="/api/v1/collections", tags=["Collections"])

    # --- 健康检查端点 ----------------------------------------------------------
    @app.get("/health", tags=["System"])
    async def health_check():
        """
        健康检查端点

        Returns:
            dict: 包含服务状态的字典
        """
        return {"status": "ok"}

    @app.get("/api/v1/supported-formats", tags=["System"])
    async def supported_formats():
        """
        获取系统支持的文档格式列表

        Returns:
            dict: 包含支持的文件扩展名和分类的字典
        """
        from app.services.document_loader import DocumentLoaderFactory
        return {
            "extensions": DocumentLoaderFactory.supported_extensions(),
            "categories": DocumentLoaderFactory.supported_categories(),
        }

    # --- 异常处理器 ------------------------------------------------------------
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        """
        应用自定义异常处理器

        Args:
            request: 请求对象
            exc: 应用异常实例

        Returns:
            JSONResponse: 包含错误详情的 JSON 响应
        """
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=exc.detail, error_code=exc.error_code).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """
        通用异常处理器，捕获未处理的异常

        Args:
            request: 请求对象
            exc: 异常实例

        Returns:
            JSONResponse: 包含通用错误信息的 JSON 响应
        """
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="Internal server error", error_code="INTERNAL_ERROR").model_dump(),
        )

    # --- 静态文件服务 ----------------------------------------------------------
    # 如果静态文件目录存在，则挂载静态文件服务
    if os.path.isdir(STATIC_DIR):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


# 创建应用实例
app = create_app()
