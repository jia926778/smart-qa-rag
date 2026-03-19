"""质量评估智能体 — RAG 图中的第四个节点。

本模块负责对生成的答案进行质量评估，确保答案的可靠性和完整性。

主要职责：
    1. 检查答案是否基于提供的来源（幻觉检测）。
    2. 检查答案是否充分回答了用户的问题。
    3. 做出决策："接受"答案或"重试"并附带反馈意见。

评估流程：
    接收答案 → 检查依据性 → 检查充分性 → 返回评估结果和决策
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from app.agents.state import EvaluationResult, GraphState
from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 评估提示词模板
# 指导 LLM 对生成的答案进行质量评估，返回结构化的评估结果
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
    """构建并返回质量评估智能体的 LangGraph 节点函数。

    Args:
        settings (Settings): 应用配置对象，包含 LLM 模型参数、API 密钥等配置。

    Returns:
        Callable: 异步的 LangGraph 节点函数，接收 GraphState 作为输入，
            返回包含 evaluation 和 retry_count 的状态更新字典。

    Note:
        该函数内部定义了 evaluator_agent_node 异步函数，
        该函数会对生成的答案进行依据性和充分性评估，并决定是否需要重试。
    """

    # 构建 LLM 调用参数，使用 temperature=0 确保评估结果稳定
    kwargs: Dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "temperature": 0,
        "max_tokens": 512,
        "openai_api_key": settings.OPENAI_API_KEY,
    }
    # 如果配置了自定义 API 端点，则添加到参数中
    if settings.OPENAI_API_BASE:
        kwargs["openai_api_base"] = settings.OPENAI_API_BASE
    # 初始化 ChatOpenAI 实例
    llm = ChatOpenAI(**kwargs)

    async def evaluator_agent_node(state: GraphState) -> Dict[str, Any]:
        """质量评估节点的核心逻辑。

        Args:
            state (GraphState): RAG 图的当前状态，包含问题、答案、来源等信息。

        Returns:
            Dict[str, Any]: 状态更新字典，包含以下键：
                - evaluation (EvaluationResult): 评估结果对象
                - retry_count (int): 更新后的重试次数

        Note:
            如果已达到最大重试次数，将自动接受当前答案。
        """
        # 从状态中提取必要信息
        question = state["question"]  # 用户原始问题
        answer = state.get("answer", "")  # 生成的答案
        sources = state.get("sources", [])  # 来源引用列表
        retry_count = state.get("retry_count", 0)  # 当前重试次数
        max_retries = state.get("max_retries", 2)  # 最大重试次数

        # 如果已达到最大重试次数，自动接受当前答案
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

        # 构建来源文本，用于评估提示词
        source_parts = []
        for idx, src in enumerate(sources, 1):
            name = src.get("source", "unknown")
            page = src.get("page")
            content = src.get("content", "")
            # 格式化每个来源条目
            header = f"[{idx}] {name}"
            if page:
                header += f" (p.{page})"
            source_parts.append(f"{header}: {content}")
        source_text = "\n".join(source_parts) if source_parts else "(no sources retrieved)"

        # 构建评估提示词
        prompt = _EVALUATION_PROMPT.format(
            question=question,
            sources=source_text,
            answer=answer,
        )

        try:
            # 调用 LLM 进行评估
            response = await llm.ainvoke(prompt)
            raw = response.content.strip()
            # 清理可能的 markdown 代码块标记
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            # 解析 JSON 格式的评估结果
            evaluation: EvaluationResult = json.loads(raw)

            # 强制执行决策逻辑：只有当依据性、充分性和置信度都满足时才接受
            is_grounded = evaluation.get("is_grounded", False)
            is_sufficient = evaluation.get("is_sufficient", False)
            confidence = evaluation.get("confidence", 0.0)
            if is_grounded and is_sufficient and confidence >= 0.6:
                evaluation["decision"] = "accept"
            else:
                evaluation["decision"] = "retry"

            # 记录评估结果日志
            logger.info(
                "Evaluator: grounded=%s, sufficient=%s, confidence=%.2f, decision=%s",
                is_grounded,
                is_sufficient,
                confidence,
                evaluation["decision"],
            )

        except (json.JSONDecodeError, Exception) as exc:
            # 评估失败时，记录警告并自动接受答案
            logger.warning("Evaluator parse error: %s. Auto-accepting.", exc)
            evaluation = EvaluationResult(
                is_grounded=True,
                is_sufficient=True,
                confidence=0.7,
                feedback="Evaluation failed, accepting answer.",
                decision="accept",
            )

        # 如果决策为重试，增加重试计数
        new_retry = retry_count
        if evaluation.get("decision") == "retry":
            new_retry = retry_count + 1

        # 返回状态更新
        return {
            "evaluation": evaluation,
            "retry_count": new_retry,
        }

    return evaluator_agent_node
