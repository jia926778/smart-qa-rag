from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_qa_service
from app.models.schemas import AskRequest, AskResponse
from app.services.qa_service import QAService

router = APIRouter()


@router.post("/ask", response_model=AskResponse, summary="Ask a question")
async def ask_question(
    body: AskRequest,
    qa_service: QAService = Depends(get_qa_service),
) -> AskResponse:
    return await qa_service.ask(body)
