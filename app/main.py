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

logger = get_logger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart QA RAG",
        description="Intelligent Q&A system powered by RAG (Retrieval-Augmented Generation)",
        version="1.0.0",
    )

    # --- CORS -----------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers --------------------------------------------------------------
    app.include_router(qa.router, prefix="/api/v1/qa", tags=["QA"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(collections.router, prefix="/api/v1/collections", tags=["Collections"])

    # --- Health check ---------------------------------------------------------
    @app.get("/health", tags=["System"])
    async def health_check():
        return {"status": "ok"}

    @app.get("/api/v1/supported-formats", tags=["System"])
    async def supported_formats():
        from app.services.document_loader import DocumentLoaderFactory
        return {
            "extensions": DocumentLoaderFactory.supported_extensions(),
            "categories": DocumentLoaderFactory.supported_categories(),
        }

    # --- Exception handlers ---------------------------------------------------
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=exc.detail, error_code=exc.error_code).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="Internal server error", error_code="INTERNAL_ERROR").model_dump(),
        )

    # --- Static files ---------------------------------------------------------
    if os.path.isdir(STATIC_DIR):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
