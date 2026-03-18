from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat / QA
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    collection_name: str = Field(
        default="default", description="Target knowledge-base collection"
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list, description="Recent conversation history"
    )


class SourceInfo(BaseModel):
    source: str = Field(..., description="Source document name or path")
    page: Optional[int] = Field(None, description="Page number if applicable")
    content: str = Field(..., description="Relevant content snippet")


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = Field(default_factory=list)
    elapsed_ms: float = Field(..., description="Processing time in milliseconds")


# ---------------------------------------------------------------------------
# Document upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    filename: str
    collection_name: str
    chunks_count: int
    message: str


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="")


class CollectionInfo(BaseModel):
    name: str
    description: str = ""
    documents_count: int = 0
    created_at: Optional[datetime] = None


class CollectionStats(BaseModel):
    name: str
    documents_count: int = 0
    description: str = ""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
