"""Answer Generator Agent — third node in the RAG graph.

Responsibilities:
  1. Build a context-aware prompt using retrieved documents.
  2. Inject chat history for multi-turn coherence.
  3. Adapt generation strategy based on query intent/complexity.
  4. Call the LLM and assemble source citations.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.agents.state import GraphState
from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Intent-specific system prompt suffixes
_INTENT_INSTRUCTIONS = {
    "factual": "请直接给出准确的事实性回答。",
    "comparison": "请从多个维度进行对比分析，使用表格或列表呈现。",
    "summary": "请提供全面但简明的概要，覆盖要点。",
    "how-to": "请给出清晰的步骤说明，使用编号列表。",
    "definition": "请先给出简明定义，再展开解释。",
    "opinion": "请基于参考资料给出客观分析，说明不同观点。",
    "data_query": "请基于查询到的数据给出准确的数据分析回答，突出关键数字。",
}

_SYSTEM_PROMPT = """\
你是一个专业的智能问答助手。请根据提供的参考资料回答用户的问题。

回答要求：
1. 仅基于参考资料中的内容进行回答，不要编造信息。
2. 如果参考资料中没有相关信息，请诚实地告诉用户"根据现有资料，我无法找到相关信息"。
3. 回答时请引用来源，例如【来源: 文档名, 第X页】。
4. 使用清晰、简洁的中文进行回答。
5. {intent_instruction}

问题复杂度: {complexity}
"""

_RETRY_SUPPLEMENT = """
注意：这是第 {retry} 次生成回答。上一次回答的反馈是：
{feedback}
请改进你的回答以解决上述问题。
"""


def build_generator_agent(settings: Settings):
    """Return a LangGraph node function."""

    kwargs: Dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "openai_api_key": settings.OPENAI_API_KEY,
    }
    if settings.OPENAI_API_BASE:
        kwargs["openai_api_base"] = settings.OPENAI_API_BASE
    llm = ChatOpenAI(**kwargs)

    async def generator_agent_node(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        analysis = state.get("query_analysis", {})
        docs: List[Document] = state.get("retrieved_docs", [])
        chat_history = state.get("chat_history", [])
        retry_count = state.get("retry_count", 0)
        evaluation = state.get("evaluation", {})
        sql_result = state.get("sql_result")

        intent = analysis.get("intent", "factual")
        complexity = analysis.get("complexity", "simple")
        intent_instruction = _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["factual"])

        # Build system message
        system_text = _SYSTEM_PROMPT.format(
            intent_instruction=intent_instruction,
            complexity=complexity,
        )

        # If retrying, append feedback
        if retry_count > 0 and evaluation.get("feedback"):
            system_text += _RETRY_SUPPLEMENT.format(
                retry=retry_count,
                feedback=evaluation["feedback"],
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_text}]

        # Build context block
        if docs:
            context_parts: List[str] = []
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page")
                header = f"【参考资料 {idx} | 来源: {source}"
                if page is not None:
                    header += f", 第{page}页"
                rrf = doc.metadata.get("rrf_score")
                rerank = doc.metadata.get("rerank_score")
                if rrf is not None:
                    header += f", RRF={rrf:.4f}"
                if rerank is not None:
                    header += f", Rerank={rerank:.2f}"
                header += "】"
                context_parts.append(f"{header}\n{doc.page_content}")
            context_text = "\n\n".join(context_parts)
            messages.append({
                "role": "system",
                "content": f"以下是检索到的 {len(docs)} 条参考资料：\n\n{context_text}",
            })

        # Inject SQL query results if available
        if sql_result and sql_result.get("formatted_answer"):
            sql_context = (
                f"以下是通过数据库查询获得的结构化数据结果：\n\n"
                f"SQL查询: {sql_result.get('query', 'N/A')}\n"
                f"查询结果 ({sql_result.get('row_count', 0)} 行):\n"
                f"{sql_result['formatted_answer']}"
            )
            messages.append({
                "role": "system",
                "content": sql_context,
            })

        # Chat history
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        # Use rewritten query if available
        effective_question = analysis.get("rewritten_query", question)
        messages.append({"role": "user", "content": effective_question})

        # Call LLM
        try:
            response = await llm.ainvoke(messages)
            answer = response.content
        except Exception as exc:
            logger.error("Generator LLM call failed: %s", exc)
            return {"answer": f"抱歉，生成回答时出现错误: {str(exc)}", "sources": [], "error": str(exc)}

        # Assemble sources
        sources: List[Dict[str, Any]] = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:300],
            })

        logger.info(
            "Generator: produced answer (%d chars) with %d sources",
            len(answer),
            len(sources),
        )

        return {"answer": answer, "sources": sources}

    return generator_agent_node
