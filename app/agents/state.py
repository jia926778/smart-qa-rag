"""
多智能体 RAG 流水线的共享图状态定义

LangGraph 中的每个节点都从这个类型化字典中读取和写入数据。
使用 ``TypedDict`` + ``Annotated`` 配合归约器函数，允许 LangGraph
自动合并来自每个智能体的部分更新。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# 辅助归约器：替换值（后写入者胜出）
# ---------------------------------------------------------------------------
def _replace(a: Any, b: Any) -> Any:
    """
    归约器函数，总是取较新的值

    Args:
        a: 旧值
        b: 新值

    Returns:
        Any: 返回新值
    """
    return b


# ---------------------------------------------------------------------------
# 子结构定义
# ---------------------------------------------------------------------------

class QueryAnalysis(TypedDict, total=False):
    """
    查询分析器智能体的输出

    包含查询意图、重写后的查询、子查询、语言、复杂度和检索策略等信息。
    """
    intent: str                         # 意图类型，如 "factual", "comparison", "summary", "how-to", "data_query"
    rewritten_query: str                # 原始问题的改进版本
    sub_queries: List[str]              # 从不同角度生成的多个搜索查询
    language: str                       # 检测到的语言（"zh", "en", "mixed"）
    complexity: str                     # 复杂度级别（"simple", "moderate", "complex"）
    retrieval_strategy: str             # 检索策略（"hybrid", "vector", "bm25", "sql", "hybrid+sql"）


class EvaluationResult(TypedDict, total=False):
    """
    质量评估器智能体的输出

    包含答案是否基于事实、是否充分、置信度、反馈和决策等信息。
    """
    is_grounded: bool                   # 答案是否有来源支持
    is_sufficient: bool                 # 答案是否充分回答了问题
    confidence: float                   # 置信度分数（0.0 – 1.0）
    feedback: str                       # 自由文本解释
    decision: Literal["accept", "retry"]  # 决策：接受或重试


# ---------------------------------------------------------------------------
# 主图状态
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    在 LangGraph 中流动的共享状态

    包含输入问题、查询分析结果、检索文档、SQL 结果、生成的答案、评估结果和控制流信息。
    """

    # --- 输入（在开始时设置一次）---
    question: str                       # 用户问题
    collection_name: str                # 集合名称
    chat_history: List[Dict[str, str]]  # 聊天历史

    # --- 查询分析器输出 ---
    query_analysis: Annotated[QueryAnalysis, _replace]

    # --- 检索器输出 ---
    retrieved_docs: Annotated[List[Document], _replace]  # 检索到的文档列表
    retrieval_queries_used: Annotated[List[str], _replace]  # 使用的检索查询列表

    # --- SQL 智能体输出 ---
    sql_result: Annotated[Optional[Dict[str, Any]], _replace]  # SQL 查询结果
    sql_query: Annotated[Optional[str], _replace]  # 执行的 SQL 查询

    # --- 生成器输出 ---
    answer: Annotated[str, _replace]  # 生成的答案
    sources: Annotated[List[Dict[str, Any]], _replace]  # 来源信息列表

    # --- 评估器输出 ---
    evaluation: Annotated[EvaluationResult, _replace]

    # --- 控制流 ---
    retry_count: Annotated[int, _replace]  # 当前重试次数
    max_retries: Annotated[int, _replace]  # 最大重试次数
    error: Annotated[Optional[str], _replace]  # 错误信息
