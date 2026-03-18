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

router = APIRouter()


@router.post("/", response_model=CollectionInfo, summary="Create a collection")
async def create_collection(
    body: CollectionCreateRequest,
    svc: CollectionService = Depends(get_collection_service),
) -> CollectionInfo:
    return svc.create(name=body.name, description=body.description)


@router.get("/", response_model=List[CollectionInfo], summary="List all collections")
async def list_collections(
    svc: CollectionService = Depends(get_collection_service),
) -> List[CollectionInfo]:
    return svc.list_all()


@router.delete("/{name}", summary="Delete a collection")
async def delete_collection(
    name: str,
    svc: CollectionService = Depends(get_collection_service),
) -> dict:
    svc.delete(name)
    return {"message": f"Collection '{name}' deleted"}


@router.get("/{name}/stats", response_model=CollectionStats, summary="Collection stats")
async def collection_stats(
    name: str,
    svc: CollectionService = Depends(get_collection_service),
) -> CollectionStats:
    return svc.stats(name)
