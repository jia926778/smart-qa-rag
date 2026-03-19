"""LangGraph 工作流 — 组装多智能体 RAG 图。

本模块负责构建和组装完整的 RAG（检索增强生成）工作流图，
协调各个智能体节点之间的数据流转和条件分支。

图拓扑结构：
    START → query_analyzer → router_after_analysis
                              ├─ (needs_sql) → sql_agent ──┐
                              └─ (no_sql) ─────────────────┤
                                                           ↓
                             retriever → generator → evaluator → retry_router
                                  ↑                                      │
                                  └──────── retry (max N) ───────────────┘
                                                                         │
                                                                      accept → END

主要组件：
    - query_analyzer: 查询分析器，分析用户问题的意图和复杂度
    - sql_agent: SQL 智能体，处理结构化数据查询（可选）
    - retriever: 检索器，从知识库检索相关文档
    - generator: 生成器，基于检索结果生成答案
    - evaluator: 评估器，评估答案质量并决定是否重试
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph

from app.agents.evaluator import build_evaluator_agent
from app.agents.generator import build_generator_agent
from app.agents.query_analyzer import build_query_analyzer
from app.agents.retriever_agent import build_retriever_agent
from app.agents.sql_agent import build_sql_agent
from app.agents.state import GraphState
from app.config import Settings
from app.services.retriever import SmartRetriever
from app.services.sql_store import SQLStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _should_retry(state: GraphState) -> Literal["query_analyzer", "__end__"]:
    """条件边：根据评估结果决定是否重试。

    检查评估器的决策和当前重试次数，决定是重新分析查询还是结束流程。

    Args:
        state (GraphState): RAG 图的当前状态。

    Returns:
        Literal["query_analyzer", "__end__"]: 返回 "query_analyzer" 进行重试，
            或返回 END 结束流程。

    Note:
        只有当决策为 "retry" 且未达到最大重试次数时才会重试。
    """
    evaluation = state.get("evaluation", {})
    decision = evaluation.get("decision", "accept")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # 如果决策为重试且未达到最大重试次数，返回查询分析器节点
    if decision == "retry" and retry_count < max_retries:
        logger.info(
            "Router: retrying (attempt %d/%d)", retry_count + 1, max_retries
        )
        return "query_analyzer"
    return END


def _route_after_analysis(state: GraphState) -> Literal["sql_agent", "retriever"]:
    """条件边：根据查询分析结果决定是否执行 SQL 智能体。

    检查查询分析结果中的检索策略，决定是否需要执行结构化数据查询。

    Args:
        state (GraphState): RAG 图的当前状态。

    Returns:
        Literal["sql_agent", "retriever"]: 返回 "sql_agent" 执行 SQL 查询，
            或返回 "retriever" 直接进行文档检索。

    Note:
        当检索策略为 "sql" 或 "hybrid+sql" 时，会路由到 SQL 智能体。
    """
    analysis = state.get("query_analysis", {})
    strategy = analysis.get("retrieval_strategy", "hybrid")
    # 如果策略包含 SQL，则路由到 SQL 智能体
    if strategy in ("sql", "hybrid+sql"):
        return "sql_agent"
    return "retriever"


def build_rag_graph(
    settings: Settings,
    retriever: SmartRetriever,
    sql_store: Optional[SQLStore] = None,
) -> Any:
    """构建并编译 LangGraph 状态图。

    组装所有智能体节点，定义节点之间的边和条件路由，
    构建完整的 RAG 工作流图。

    Args:
        settings (Settings): 应用配置对象，包含各种配置参数。
        retriever (SmartRetriever): 智能检索器实例，用于文档检索。
        sql_store (Optional[SQLStore]): SQL 存储服务实例，用于结构化数据查询。
            默认为 None，表示不启用 SQL 查询功能。

    Returns:
        Any: 编译后的 LangGraph 图对象，可以通过以下方式调用：
            result = await graph.ainvoke(initial_state)

    Note:
        图的节点数量取决于是否启用了 SQL 智能体功能。
    """

    # --- 构建智能体节点 ---
    query_analyzer = build_query_analyzer(settings)  # 查询分析器
    retriever_agent = build_retriever_agent(retriever)  # 检索器
    generator = build_generator_agent(settings)  # 答案生成器
    evaluator = build_evaluator_agent(settings)  # 质量评估器

    # --- 组装工作流图 ---
    workflow = StateGraph(GraphState)

    # 添加节点到图中
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("generator", generator)
    workflow.add_node("evaluator", evaluator)

    # 线性流程：起点 → 查询分析
    workflow.add_edge(START, "query_analyzer")

    # 条件分支：分析后可选执行 SQL 智能体
    if sql_store and settings.TEXT_TO_SQL_ENABLED:
        # 如果启用了 SQL 功能，添加 SQL 智能体节点
        sql_agent = build_sql_agent(settings, sql_store)
        workflow.add_node("sql_agent", sql_agent)

        # 添加条件边：根据分析结果路由
        workflow.add_conditional_edges(
            "query_analyzer",
            _route_after_analysis,
            {
                "sql_agent": "sql_agent",
                "retriever": "retriever",
            },
        )
        # SQL 智能体执行后，继续到检索器获取文本上下文
        workflow.add_edge("sql_agent", "retriever")
    else:
        # 未启用 SQL 功能，直接连接到检索器
        workflow.add_edge("query_analyzer", "retriever")

    # 添加线性边：检索器 → 生成器 → 评估器
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "evaluator")

    # 条件分支：评估后重试或结束
    workflow.add_conditional_edges(
        "evaluator",
        _should_retry,
        {
            "query_analyzer": "query_analyzer",  # 重试时返回查询分析器
            END: END,  # 接受答案时结束
        },
    )

    # 计算节点数量并编译图
    node_count = 4 + (1 if sql_store and settings.TEXT_TO_SQL_ENABLED else 0)
    graph = workflow.compile()
    logger.info("RAG graph compiled successfully with %d agent nodes.", node_count)
    return graph
