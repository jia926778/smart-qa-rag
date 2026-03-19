"""答案生成智能体 — RAG 图中的第三个节点。

本模块负责基于检索到的文档生成最终答案，是 RAG 流程的核心生成组件。

主要职责：
    1. 构建上下文感知的提示词，整合检索到的文档内容。
    2. 注入聊天历史，实现多轮对话的连贯性。
    3. 根据查询意图和复杂度自适应调整生成策略。
    4. 调用大语言模型生成回答，并组装来源引用信息。

工作流程：
    检索文档 → 构建提示词 → 注入历史 → 调用LLM → 返回答案和来源
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.agents.state import GraphState
from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 意图类型与对应的系统提示指令映射
# 根据不同的查询意图，提供针对性的回答风格指导
_INTENT_INSTRUCTIONS = {
    "factual": "请直接给出准确的事实性回答。",
    "comparison": "请从多个维度进行对比分析，使用表格或列表呈现。",
    "summary": "请提供全面但简明的概要，覆盖要点。",
    "how-to": "请给出清晰的步骤说明，使用编号列表。",
    "definition": "请先给出简明定义，再展开解释。",
    "opinion": "请基于参考资料给出客观分析，说明不同观点。",
    "data_query": "请基于查询到的数据给出准确的数据分析回答，突出关键数字。",
}

# 系统提示词模板
# 定义智能问答助手的基本行为规范和回答要求
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

# 重试补充提示词模板
# 当回答质量评估未通过时，附加此提示词指导模型改进
_RETRY_SUPPLEMENT = """
注意：这是第 {retry} 次生成回答。上一次回答的反馈是：
{feedback}
请改进你的回答以解决上述问题。
"""


def build_generator_agent(settings: Settings):
    """构建并返回答案生成智能体的 LangGraph 节点函数。

    Args:
        settings (Settings): 应用配置对象，包含 LLM 模型参数、API 密钥等配置。

    Returns:
        Callable: 异步的 LangGraph 节点函数，接收 GraphState 作为输入，
            返回包含 answer 和 sources 的状态更新字典。

    Note:
        该函数内部定义了 generator_agent_node 异步函数，
        该函数会根据检索文档、聊天历史和查询分析结果生成最终答案。
    """

    # 构建 LLM 调用参数
    kwargs: Dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "openai_api_key": settings.OPENAI_API_KEY,
    }
    # 如果配置了自定义 API 端点，则添加到参数中
    if settings.OPENAI_API_BASE:
        kwargs["openai_api_base"] = settings.OPENAI_API_BASE
    # 初始化 ChatOpenAI 实例
    llm = ChatOpenAI(**kwargs)

    async def generator_agent_node(state: GraphState) -> Dict[str, Any]:
        """答案生成节点的核心逻辑。

        Args:
            state (GraphState): RAG 图的当前状态，包含问题、检索文档、聊天历史等信息。

        Returns:
            Dict[str, Any]: 状态更新字典，包含以下键：
                - answer (str): 生成的答案文本
                - sources (List[Dict]): 来源引用列表
                - error (str, optional): 如果生成失败，包含错误信息

        Raises:
            无显式异常抛出，错误会被捕获并返回包含错误信息的字典。
        """
        # 从状态中提取必要信息
        question = state["question"]  # 用户原始问题
        analysis = state.get("query_analysis", {})  # 查询分析结果
        docs: List[Document] = state.get("retrieved_docs", [])  # 检索到的文档列表
        chat_history = state.get("chat_history", [])  # 聊天历史
        retry_count = state.get("retry_count", 0)  # 当前重试次数
        evaluation = state.get("evaluation", {})  # 上一次评估结果
        sql_result = state.get("sql_result")  # SQL 查询结果（如果有）

        # 提取查询意图和复杂度，用于选择合适的生成策略
        intent = analysis.get("intent", "factual")
        complexity = analysis.get("complexity", "simple")
        intent_instruction = _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["factual"])

        # 构建系统消息，包含基本指令和意图特定指令
        system_text = _SYSTEM_PROMPT.format(
            intent_instruction=intent_instruction,
            complexity=complexity,
        )

        # 如果是重试生成，附加评估反馈以指导改进
        if retry_count > 0 and evaluation.get("feedback"):
            system_text += _RETRY_SUPPLEMENT.format(
                retry=retry_count,
                feedback=evaluation["feedback"],
            )

        # 初始化消息列表，首先添加系统消息
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_text}]

        # 构建参考资料上下文块
        # 将每个检索到的文档格式化为带元数据的参考条目
        if docs:
            context_parts: List[str] = []
            for idx, doc in enumerate(docs, 1):
                # 提取文档元数据
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page")
                # 构建文档头部信息，包含来源、页码和相关性分数
                header = f"【参考资料 {idx} | 来源: {source}"
                if page is not None:
                    header += f", 第{page}页"
                # 添加 RRF（倒数排名融合）分数
                rrf = doc.metadata.get("rrf_score")
                # 添加重排序分数
                rerank = doc.metadata.get("rerank_score")
                if rrf is not None:
                    header += f", RRF={rrf:.4f}"
                if rerank is not None:
                    header += f", Rerank={rerank:.2f}"
                header += "】"
                context_parts.append(f"{header}\n{doc.page_content}")
            # 将所有参考资料合并为上下文文本
            context_text = "\n\n".join(context_parts)
            messages.append({
                "role": "system",
                "content": f"以下是检索到的 {len(docs)} 条参考资料：\n\n{context_text}",
            })

        # 如果存在 SQL 查询结果，将其作为结构化数据上下文注入
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

        # 注入聊天历史，保留最近 6 条消息以维持对话连贯性
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        # 使用重写后的查询（如果有），否则使用原始问题
        effective_question = analysis.get("rewritten_query", question)
        messages.append({"role": "user", "content": effective_question})

        # 调用大语言模型生成回答
        try:
            response = await llm.ainvoke(messages)
            answer = response.content
        except Exception as exc:
            logger.error("Generator LLM call failed: %s", exc)
            return {"answer": f"抱歉，生成回答时出现错误: {str(exc)}", "sources": [], "error": str(exc)}

        # 组装来源引用信息，用于在回答中展示参考来源
        sources: List[Dict[str, Any]] = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "unknown"),  # 来源文件名
                "page": doc.metadata.get("page"),  # 页码
                "content": doc.page_content[:300],  # 内容摘要（截取前300字符）
            })

        # 记录生成结果日志
        logger.info(
            "Generator: produced answer (%d chars) with %d sources",
            len(answer),
            len(sources),
        )

        return {"answer": answer, "sources": sources}

    return generator_agent_node
