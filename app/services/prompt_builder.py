"""提示词构建模块，用于构建 LLM 调用的消息列表。

本模块负责将用户问题、检索到的上下文文档和聊天历史，
组装成适合 LLM 调用的消息格式。

主要组件：
- PromptBuilder: 提示词构建器类
- SYSTEM_PROMPT: 系统提示词模板
"""

from typing import Dict, List

from langchain_core.documents import Document

from app.models.schemas import ChatMessage
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 系统提示词，定义 AI 助手的行为规范
SYSTEM_PROMPT = (
    "你是一个专业的智能问答助手。请根据提供的参考资料回答用户的问题。\n"
    "回答要求：\n"
    "1. 仅基于参考资料中的内容进行回答，不要编造信息。\n"
    "2. 如果参考资料中没有相关信息，请诚实地告诉用户'根据现有资料，我无法找到相关信息'。\n"
    "3. 回答时请引用来源，例如【来源: 文档名, 第X页】。\n"
    "4. 使用清晰、简洁的中文进行回答。\n"
    "5. 如果问题需要分步骤回答，请使用编号列表。"
)

# 聊天历史的最大消息数
MAX_HISTORY_MESSAGES = 6


class PromptBuilder:
    """提示词构建器，用于构建 LLM 调用的消息列表。"""

    @staticmethod
    def build(
        question: str,
        context_docs: List[Document],
        chat_history: List[ChatMessage] | None = None,
    ) -> List[Dict[str, str]]:
        """构建 LLM 调用的消息列表。
        
        Args:
            question: 用户当前问题。
            context_docs: 检索到的上下文文档列表。
            chat_history: 聊天历史消息列表，可选。
        
        Returns:
            消息列表，每个消息包含 role 和 content 字段。
        """
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # --- 上下文部分 ---
        if context_docs:
            context_parts: List[str] = []
            for idx, doc in enumerate(context_docs, 1):
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page")
                header = f"【参考资料 {idx} | 来源: {source}"
                if page is not None:
                    header += f", 第{page}页"
                header += "】"
                context_parts.append(f"{header}\n{doc.page_content}")
            context_text = "\n\n".join(context_parts)
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是与用户问题相关的参考资料：\n\n{context_text}",
                }
            )

        # --- 聊天历史（最近 N 条消息）---
        if chat_history:
            recent = chat_history[-MAX_HISTORY_MESSAGES:]
            for msg in recent:
                messages.append({"role": msg.role, "content": msg.content})

        # --- 当前问题 ---
        messages.append({"role": "user", "content": question})

        logger.debug("Prompt built with %d message(s)", len(messages))
        return messages
