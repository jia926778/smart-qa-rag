"""Retriever Agent — second node in the RAG graph.

Responsibilities:
  1. Execute multi-query retrieval using the sub-queries from QueryAnalyzer.
  2. Use the recommended retrieval_strategy (vector / bm25 / hybrid).
  3. Merge and deduplicate results via Reciprocal Rank Fusion (RRF).
  4. Leverage the existing SmartRetriever (hybrid + parent-child + reranking).
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from langchain_core.documents import Document

from app.agents.state import GraphState
from app.services.retriever import SmartRetriever
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _reciprocal_rank_fusion(
    doc_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """Merge multiple ranked lists using RRF."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            key = _doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
                scores[key] = 0.0
            scores[key] += 1.0 / (k + rank)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = scores[key]
        result.append(doc)
    return result


def _doc_key(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    parent_id = doc.metadata.get("parent_id", "")
    if parent_id:
        return f"{source}::{parent_id}"
    return f"{source}::{hash(doc.page_content[:200])}"


def build_retriever_agent(retriever: SmartRetriever):
    """Return a LangGraph node function."""

    async def retriever_agent_node(state: GraphState) -> Dict[str, Any]:
        analysis = state.get("query_analysis", {})
        collection_name = state.get("collection_name", "default")

        sub_queries = analysis.get("sub_queries", [])
        if not sub_queries:
            sub_queries = [state["question"]]

        # Determine retrieval strategy from analyzer
        strategy = analysis.get("retrieval_strategy", "hybrid")
        # If strategy involves sql-only, retriever still runs for text context
        if strategy == "sql":
            strategy = "hybrid"  # still fetch text docs as fallback context
        elif strategy == "hybrid+sql":
            strategy = "hybrid"

        # Execute retrieval for each sub-query
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

        # Merge via Reciprocal Rank Fusion
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
