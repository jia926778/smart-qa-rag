"""QA Service — delegates to the LangGraph multi-agent pipeline.

The service acts as the bridge between the FastAPI layer and the
LangGraph graph.  It initializes the graph state from the incoming
request, invokes the compiled graph, and maps the final state back
to the API response model.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from app.agents.graph import build_rag_graph
from app.agents.state import GraphState
from app.config import Settings
from app.models.schemas import AskRequest, AskResponse, SourceInfo
from app.services.retriever import SmartRetriever
from app.utils.exceptions import LLMError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QAService:
    """Orchestrate the multi-agent RAG pipeline via LangGraph."""

    def __init__(
        self,
        settings: Settings,
        retriever: SmartRetriever,
    ) -> None:
        self._settings = settings
        self._graph = build_rag_graph(settings=settings, retriever=retriever)

    async def ask(self, request: AskRequest) -> AskResponse:
        start = time.perf_counter()

        # Build initial graph state
        initial_state: Dict[str, Any] = {
            "question": request.question,
            "collection_name": request.collection_name,
            "chat_history": [
                {"role": msg.role, "content": msg.content}
                for msg in request.chat_history
            ],
            "retry_count": 0,
            "max_retries": self._settings.AGENT_MAX_RETRIES,
        }

        try:
            # Run the full agent graph
            final_state = await self._graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("LangGraph execution failed: %s", exc)
            raise LLMError(f"Agent pipeline failed: {exc}") from exc

        # Extract results from final state
        answer = final_state.get("answer", "抱歉，无法生成回答。")
        raw_sources = final_state.get("sources", [])
        evaluation = final_state.get("evaluation", {})
        query_analysis = final_state.get("query_analysis", {})
        retry_count = final_state.get("retry_count", 0)

        # Map to response model
        sources: List[SourceInfo] = []
        for src in raw_sources:
            sources.append(
                SourceInfo(
                    source=src.get("source", "unknown"),
                    page=src.get("page"),
                    content=src.get("content", "")[:300],
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "QA completed in %.1f ms | intent=%s | retries=%d | "
            "confidence=%.2f | docs=%d",
            elapsed_ms,
            query_analysis.get("intent", "?"),
            retry_count,
            evaluation.get("confidence", 0),
            len(sources),
        )

        return AskResponse(
            answer=answer,
            sources=sources,
            elapsed_ms=round(elapsed_ms, 1),
        )
