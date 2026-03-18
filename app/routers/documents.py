from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.dependencies import get_document_service
from app.models.schemas import UploadResponse
from app.services.document_service import DocumentService

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, summary="Upload a document")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
    doc_service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    file_bytes = await file.read()
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
    count = doc_service.delete_document(collection_name, doc_source)
    return {"deleted_chunks": count, "source": doc_source, "collection": collection_name}
