"""Query Analyzer Agent — first node in the RAG graph.

Responsibilities:
  1. Classify user intent (factual / comparison / summary / how-to).
  2. Rewrite the question for better retrieval.
  3. Generate 2-3 sub-queries from different angles to improve recall.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from app.agents.state import GraphState, QueryAnalysis
from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ANALYSIS_PROMPT = """\
You are a query analysis expert for a RAG system.  Analyze the user's \
question and return a JSON object with exactly these fields:

{{
  "intent": "<one of: factual, comparison, summary, how-to, definition, opinion, data_query>",
  "rewritten_query": "<improved, self-contained version of the question for search>",
  "sub_queries": ["<query variant 1>", "<query variant 2>", "<query variant 3>"],
  "language": "<zh | en | mixed>",
  "complexity": "<simple | moderate | complex>",
  "retrieval_strategy": "<one of: hybrid, vector, bm25, sql, hybrid+sql>"
}}

Intent guidelines:
- "data_query": questions asking for specific numbers, rankings, statistics, \
  aggregations, or comparisons from structured/tabular data (e.g., "销售额最高的产品", \
  "2024年Q1的平均收入", "哪个部门人数最多").
- Other intents: for general knowledge, explanations, summaries, etc.

Retrieval strategy guidelines:
- "hybrid": default for most questions — combines vector semantic search + BM25 keyword search.
- "sql": use when the question is clearly about structured data (numbers, rankings, aggregations).
- "hybrid+sql": use when the question might benefit from both text context AND structured data.
- "vector": use when the question is purely semantic and BM25 won't help.
- "bm25": use when the question contains very specific terms/names/codes.

Rules:
- The rewritten query must be self-contained (resolve pronouns using chat history).
- Sub-queries should approach the question from different angles to maximize recall.
- Generate exactly 3 sub-queries.
- Reply with ONLY valid JSON, no markdown fences.

Chat history (for reference resolution):
{history}

User question: {question}
"""


def build_query_analyzer(settings: Settings):
    """Return a LangGraph node function."""

    kwargs: Dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "temperature": 0,
        "max_tokens": 512,
        "openai_api_key": settings.OPENAI_API_KEY,
    }
    if settings.OPENAI_API_BASE:
        kwargs["openai_api_base"] = settings.OPENAI_API_BASE
    llm = ChatOpenAI(**kwargs)

    async def query_analyzer_node(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        history = state.get("chat_history", [])

        history_text = ""
        if history:
            for msg in history[-6:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_text += f"{role}: {content}\n"
        if not history_text:
            history_text = "(no prior conversation)"

        prompt = _ANALYSIS_PROMPT.format(question=question, history=history_text)

        try:
            response = await llm.ainvoke(prompt)
            raw = response.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            analysis: QueryAnalysis = json.loads(raw)

            # Ensure sub_queries always includes the rewritten query
            sub_queries = analysis.get("sub_queries", [])
            rewritten = analysis.get("rewritten_query", question)
            if rewritten not in sub_queries:
                sub_queries.insert(0, rewritten)
            analysis["sub_queries"] = sub_queries[:4]  # cap at 4
            analysis["rewritten_query"] = rewritten

            logger.info(
                "QueryAnalyzer: intent=%s, complexity=%s, %d sub-queries",
                analysis.get("intent"),
                analysis.get("complexity"),
                len(analysis.get("sub_queries", [])),
            )

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("QueryAnalyzer parse error: %s. Using fallback.", exc)
            analysis = QueryAnalysis(
                intent="factual",
                rewritten_query=question,
                sub_queries=[question],
                language="zh",
                complexity="simple",
            )

        return {"query_analysis": analysis}

    return query_analyzer_node
