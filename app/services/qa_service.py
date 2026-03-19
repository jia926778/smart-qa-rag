"""问答服务模块，委托给 LangGraph 多智能体流水线。

本服务作为 FastAPI 层与 LangGraph 图之间的桥梁：
- 从传入请求初始化图状态
- 调用编译好的图执行
- 将最终状态映射回 API 响应模型
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from app.agents.graph import build_rag_graph
from app.agents.state import GraphState
from app.config import Settings
from app.models.schemas import AskRequest, AskResponse, SourceInfo
from app.services.retriever import SmartRetriever
from app.services.sql_store import SQLStore
from app.utils.exceptions import LLMError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QAService:
    """问答服务类，通过 LangGraph 编排多智能体 RAG 流水线。
    
    Attributes:
        _settings: 应用配置对象。
        _graph: 编译好的 LangGraph 图。
    """

    def __init__(
        self,
        settings: Settings,
        retriever: SmartRetriever,
        sql_store: SQLStore | None = None,
    ) -> None:
        """初始化问答服务。
        
        Args:
            settings: 应用配置对象。
            retriever: 智能检索器实例。
            sql_store: SQL 存储服务，可选。
        """
        self._settings = settings
        self._graph = build_rag_graph(
            settings=settings, retriever=retriever, sql_store=sql_store,
        )

    async def ask(self, request: AskRequest) -> AskResponse:
        """处理问答请求。
        
        Args:
            request: 问答请求对象。
        
        Returns:
            问答响应对象，包含答案、来源和耗时。
        
        Raises:
            LLMError: 智能体流水线执行失败时抛出。
        """
        start = time.perf_counter()

        # 构建初始图状态
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
            # 运行完整的智能体图
            final_state = await self._graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("LangGraph execution failed: %s", exc)
            raise LLMError(f"Agent pipeline failed: {exc}") from exc

        # 从最终状态提取结果
        answer = final_state.get("answer", "抱歉，无法生成回答。")
        raw_sources = final_state.get("sources", [])
        evaluation = final_state.get("evaluation", {})
        query_analysis = final_state.get("query_analysis", {})
        retry_count = final_state.get("retry_count", 0)

        # 映射到响应模型
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
