"""Shared graph state definition for the multi-agent RAG pipeline.

Every node in the LangGraph reads from and writes to this typed dictionary.
Using ``TypedDict`` + ``Annotated`` with reducer functions allows LangGraph
to merge partial updates from each agent automatically.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helper reducer: replace value (last write wins)
# ---------------------------------------------------------------------------
def _replace(a: Any, b: Any) -> Any:
    """Reducer that always takes the newer value."""
    return b


# ---------------------------------------------------------------------------
# Sub-structures
# ---------------------------------------------------------------------------

class QueryAnalysis(TypedDict, total=False):
    """Output of the Query Analyzer agent."""
    intent: str                         # e.g. "factual", "comparison", "summary", "how-to"
    rewritten_query: str                # Improved version of the original question
    sub_queries: List[str]              # Multiple search queries from different angles
    language: str                       # Detected language ("zh", "en", "mixed")
    complexity: str                     # "simple", "moderate", "complex"


class EvaluationResult(TypedDict, total=False):
    """Output of the Quality Evaluator agent."""
    is_grounded: bool                   # Answer is supported by sources?
    is_sufficient: bool                 # Answer adequately addresses the question?
    confidence: float                   # 0.0 – 1.0
    feedback: str                       # Free-text explanation
    decision: Literal["accept", "retry"]


# ---------------------------------------------------------------------------
# Main graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """The shared state that flows through the LangGraph."""

    # --- Input (set once at the start) ---
    question: str
    collection_name: str
    chat_history: List[Dict[str, str]]

    # --- Query Analyzer output ---
    query_analysis: Annotated[QueryAnalysis, _replace]

    # --- Retriever output ---
    retrieved_docs: Annotated[List[Document], _replace]
    retrieval_queries_used: Annotated[List[str], _replace]

    # --- Generator output ---
    answer: Annotated[str, _replace]
    sources: Annotated[List[Dict[str, Any]], _replace]

    # --- Evaluator output ---
    evaluation: Annotated[EvaluationResult, _replace]

    # --- Control flow ---
    retry_count: Annotated[int, _replace]
    max_retries: Annotated[int, _replace]
    error: Annotated[Optional[str], _replace]
