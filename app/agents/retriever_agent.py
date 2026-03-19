"""
检索智能体 — RAG 图的第二个节点

职责：
  1. 使用查询分析器生成的子查询执行多查询检索。
  2. 使用推荐的检索策略（向量 / BM25 / 混合）。
  3. 通过倒数排名融合（RRF）合并和去重结果。
  4. 利用现有的 SmartRetriever（混合 + 父子分块 + 重排序）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from langchain_core.documents import Document

from app.agents.state import GraphState
from app.services.retriever import SmartRetriever
from app.utils.logger import get_logger

# 初始化日志记录器
logger = get_logger(__name__)


def _reciprocal_rank_fusion(
    doc_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    使用倒数排名融合（RRF）算法合并多个排序列表

    RRF 公式：score(d) = Σ 1/(k + rank(d))

    Args:
        doc_lists: 多个文档列表，每个列表都是一个排序结果
        k: RRF 参数，默认为 60

    Returns:
        List[Document]: 合并后的文档列表，按 RRF 分数降序排列
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # 计算每个文档的 RRF 分数
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            key = _doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
                scores[key] = 0.0
            scores[key] += 1.0 / (k + rank)

    # 按分数降序排序
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = scores[key]
        result.append(doc)
    return result


def _doc_key(doc: Document) -> str:
    """
    生成文档的唯一标识键

    Args:
        doc: 文档对象

    Returns:
        str: 文档的唯一标识键
    """
    source = doc.metadata.get("source", "")
    parent_id = doc.metadata.get("parent_id", "")
    if parent_id:
        return f"{source}::{parent_id}"
    return f"{source}::{hash(doc.page_content[:200])}"


def build_retriever_agent(retriever: SmartRetriever):
    """
    构建检索智能体节点函数

    Args:
        retriever: 智能检索器实例

    Returns:
        Callable: LangGraph 节点函数
    """

    async def retriever_agent_node(state: GraphState) -> Dict[str, Any]:
        """
        检索智能体节点函数

        Args:
            state: 图状态

        Returns:
            Dict[str, Any]: 包含检索到的文档和使用的查询列表
        """
        analysis = state.get("query_analysis", {})
        collection_name = state.get("collection_name", "default")

        # 获取子查询列表
        sub_queries = analysis.get("sub_queries", [])
        if not sub_queries:
            sub_queries = [state["question"]]

        # 从分析器确定检索策略
        strategy = analysis.get("retrieval_strategy", "hybrid")
        # 如果策略涉及仅 SQL，检索器仍运行以获取文本上下文
        if strategy == "sql":
            strategy = "hybrid"  # 仍获取文本文档作为后备上下文
        elif strategy == "hybrid+sql":
            strategy = "hybrid"

        # 为每个子查询执行检索
        all_doc_lists: List[List[Document]] = []
        for query in sub_queries:
            try:
                docs = retriever.retrieve(
                    query=query,
                    collection_name=collection_name,
                    strategy=strategy,
                )
                all_doc_lists.append(docs)
            except Exception as exc:
                logger.warning("Retrieval failed for sub-query '%s': %s", query, exc)
                all_doc_lists.append([])

        # 通过倒数排名融合合并结果
        if len(all_doc_lists) == 1:
            merged = all_doc_lists[0]
        else:
            merged = _reciprocal_rank_fusion(all_doc_lists)

        logger.info(
            "RetrieverAgent [%s]: %d sub-queries -> %d unique docs after RRF",
            strategy,
            len(sub_queries),
            len(merged),
        )

        return {
            "retrieved_docs": merged,
            "retrieval_queries_used": sub_queries,
        }

    return retriever_agent_node
