from __future__ import annotations

import time
from typing import List

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.models.schemas import AskRequest, AskResponse, SourceInfo
from app.services.prompt_builder import PromptBuilder
from app.services.retriever import SmartRetriever
from app.utils.exceptions import LLMError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QAService:
    """Orchestrate retrieval, prompt building, and LLM call."""

    def __init__(
        self,
        settings: Settings,
        retriever: SmartRetriever,
        prompt_builder: PromptBuilder,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = self._build_llm()

    def _build_llm(self) -> ChatOpenAI:
        kwargs: dict = {
            "model": self._settings.LLM_MODEL,
            "temperature": self._settings.LLM_TEMPERATURE,
            "max_tokens": self._settings.LLM_MAX_TOKENS,
            "openai_api_key": self._settings.OPENAI_API_KEY,
        }
        if self._settings.OPENAI_API_BASE:
            kwargs["openai_api_base"] = self._settings.OPENAI_API_BASE
        return ChatOpenAI(**kwargs)

    async def ask(self, request: AskRequest) -> AskResponse:
        start = time.perf_counter()

        # 1. Retrieve relevant documents
        docs = self._retriever.retrieve(
            query=request.question,
            collection_name=request.collection_name,
        )

        # 2. Build prompt
        messages = self._prompt_builder.build(
            question=request.question,
            context_docs=docs,
            chat_history=request.chat_history,
        )

        # 3. Call LLM
        try:
            response = await self._llm.ainvoke(messages)
            answer = response.content
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise LLMError(f"LLM call failed: {exc}") from exc

        # 4. Assemble sources
        sources: List[SourceInfo] = []
        for doc in docs:
            sources.append(
                SourceInfo(
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    content=doc.page_content[:300],
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("QA completed in %.1f ms", elapsed_ms)

        return AskResponse(answer=answer, sources=sources, elapsed_ms=round(elapsed_ms, 1))
