"""Quality Evaluator Agent — fourth node in the RAG graph.

Responsibilities:
  1. Check if the answer is grounded in the provided sources (hallucination detection).
  2. Check if the answer sufficiently addresses the user's question.
  3. Decide: "accept" the answer or "retry" with feedback.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from app.agents.state import EvaluationResult, GraphState
from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_EVALUATION_PROMPT = """\
You are a strict quality evaluator for a RAG (Retrieval-Augmented Generation) system.

Evaluate the generated answer against the original question and source documents.

Return a JSON object with exactly these fields:
{{
  "is_grounded": <true if the answer is fully supported by the sources, false if it contains fabricated info>,
  "is_sufficient": <true if the answer adequately addresses the question, false otherwise>,
  "confidence": <float 0.0 to 1.0, your confidence in the answer quality>,
  "feedback": "<specific feedback on what's wrong or could be improved, or 'Good quality answer' if acceptable>",
  "decision": "<'accept' if is_grounded AND is_sufficient AND confidence >= 0.6, otherwise 'retry'>"
}}

Rules:
- Be strict about grounding: any claim not supported by sources means is_grounded=false.
- An answer saying "I don't have enough information" when sources ARE relevant is is_sufficient=false.
- Reply with ONLY valid JSON, no markdown fences.

Question: {question}

Source documents:
{sources}

Generated answer:
{answer}
"""


def build_evaluator_agent(settings: Settings):
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

    async def evaluator_agent_node(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        answer = state.get("answer", "")
        sources = state.get("sources", [])
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        # If already at max retries, auto-accept
        if retry_count >= max_retries:
            logger.info("Evaluator: max retries (%d) reached, auto-accepting.", max_retries)
            return {
                "evaluation": EvaluationResult(
                    is_grounded=True,
                    is_sufficient=True,
                    confidence=0.5,
                    feedback="Max retries reached, accepting current answer.",
                    decision="accept",
                ),
            }

        # Build source text for evaluation
        source_parts = []
        for idx, src in enumerate(sources, 1):
            name = src.get("source", "unknown")
            page = src.get("page")
            content = src.get("content", "")
            header = f"[{idx}] {name}"
            if page:
                header += f" (p.{page})"
            source_parts.append(f"{header}: {content}")
        source_text = "\n".join(source_parts) if source_parts else "(no sources retrieved)"

        prompt = _EVALUATION_PROMPT.format(
            question=question,
            sources=source_text,
            answer=answer,
        )

        try:
            response = await llm.ainvoke(prompt)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            evaluation: EvaluationResult = json.loads(raw)

            # Enforce decision logic
            is_grounded = evaluation.get("is_grounded", False)
            is_sufficient = evaluation.get("is_sufficient", False)
            confidence = evaluation.get("confidence", 0.0)
            if is_grounded and is_sufficient and confidence >= 0.6:
                evaluation["decision"] = "accept"
            else:
                evaluation["decision"] = "retry"

            logger.info(
                "Evaluator: grounded=%s, sufficient=%s, confidence=%.2f, decision=%s",
                is_grounded,
                is_sufficient,
                confidence,
                evaluation["decision"],
            )

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Evaluator parse error: %s. Auto-accepting.", exc)
            evaluation = EvaluationResult(
                is_grounded=True,
                is_sufficient=True,
                confidence=0.7,
                feedback="Evaluation failed, accepting answer.",
                decision="accept",
            )

        # Increment retry count if retrying
        new_retry = retry_count
        if evaluation.get("decision") == "retry":
            new_retry = retry_count + 1

        return {
            "evaluation": evaluation,
            "retry_count": new_retry,
        }

    return evaluator_agent_node
