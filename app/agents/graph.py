"""LangGraph workflow — assembles the multi-agent RAG graph.

Graph topology:
    START → query_analyzer → retriever → generator → evaluator → router
                  ↑                                                  │
                  └──────────── retry (max N) ────────────────────────┘
                                                                     │
                                                                  accept → END
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from langgraph.graph import END, START, StateGraph

from app.agents.evaluator import build_evaluator_agent
from app.agents.generator import build_generator_agent
from app.agents.query_analyzer import build_query_analyzer
from app.agents.retriever_agent import build_retriever_agent
from app.agents.state import GraphState
from app.config import Settings
from app.services.retriever import SmartRetriever
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _should_retry(state: GraphState) -> Literal["query_analyzer", "__end__"]:
    """Conditional edge: route based on evaluator decision."""
    evaluation = state.get("evaluation", {})
    decision = evaluation.get("decision", "accept")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if decision == "retry" and retry_count < max_retries:
        logger.info(
            "Router: retrying (attempt %d/%d)", retry_count + 1, max_retries
        )
        return "query_analyzer"
    return END


def build_rag_graph(
    settings: Settings,
    retriever: SmartRetriever,
) -> Any:
    """Build and compile the LangGraph state graph.

    Returns a compiled graph that can be invoked with:
        result = await graph.ainvoke(initial_state)
    """

    # --- Build agent nodes ---
    query_analyzer = build_query_analyzer(settings)
    retriever_agent = build_retriever_agent(retriever)
    generator = build_generator_agent(settings)
    evaluator = build_evaluator_agent(settings)

    # --- Assemble graph ---
    workflow = StateGraph(GraphState)

    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("generator", generator)
    workflow.add_node("evaluator", evaluator)

    # Linear flow: start → analyze → retrieve → generate → evaluate
    workflow.add_edge(START, "query_analyzer")
    workflow.add_edge("query_analyzer", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "evaluator")

    # Conditional: evaluate → retry or finish
    workflow.add_conditional_edges(
        "evaluator",
        _should_retry,
        {
            "query_analyzer": "query_analyzer",
            END: END,
        },
    )

    graph = workflow.compile()
    logger.info("RAG graph compiled successfully with 4 agent nodes.")
    return graph
