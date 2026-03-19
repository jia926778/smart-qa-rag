"""
问答服务测试模块

本模块测试问答服务和提示构建器的功能，包括：
- 问答服务正常响应
- LLM 调用失败处理
- 提示构建器系统消息
- 提示构建器历史消息
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import AskRequest, ChatMessage
from app.services.prompt_builder import PromptBuilder
from app.services.qa_service import QAService
from app.utils.exceptions import LLMError


class TestQAService:
    """问答服务测试类"""

    @pytest.fixture()
    def qa_service(self, test_settings, mock_embedding):
        """
        创建配置好模拟依赖的问答服务实例。

        Args:
            test_settings: 测试配置
            mock_embedding: 模拟嵌入对象

        Returns:
            QAService: 配置好的问答服务实例
        """
        # 创建模拟检索器
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        prompt_builder = PromptBuilder()
        # 模拟 ChatOpenAI 避免真实 API 调用
        with patch("app.services.qa_service.ChatOpenAI"):
            svc = QAService(
                settings=test_settings,
                retriever=retriever,
                prompt_builder=prompt_builder,
            )
        return svc

    @pytest.mark.asyncio
    async def test_ask_returns_response(self, qa_service):
        """
        测试问答服务正常返回响应。

        测试场景：调用 ask 方法处理问题
        预期结果：返回包含答案和耗时的响应对象
        """
        # 配置模拟 LLM 响应
        mock_resp = MagicMock()
        mock_resp.content = "This is the answer."
        qa_service._llm.ainvoke = AsyncMock(return_value=mock_resp)

        # 创建请求并调用服务
        req = AskRequest(question="What is RAG?", collection_name="default")
        result = await qa_service.ask(req)

        assert result.answer == "This is the answer."  # 验证答案正确
        assert result.elapsed_ms > 0  # 验证耗时为正数

    @pytest.mark.asyncio
    async def test_ask_llm_failure_raises(self, qa_service):
        """
        测试 LLM 调用失败时抛出异常。

        测试场景：模拟 LLM API 调用失败
        预期结果：抛出 LLMError 异常
        """
        # 配置模拟 LLM 抛出异常
        qa_service._llm.ainvoke = AsyncMock(side_effect=RuntimeError("api down"))

        req = AskRequest(question="test", collection_name="default")
        with pytest.raises(LLMError):
            await qa_service.ask(req)

    def test_prompt_builder_includes_system(self):
        """
        测试提示构建器包含系统消息。

        测试场景：构建提示消息
        预期结果：第一条消息为系统角色，包含智能问答助手标识
        """
        builder = PromptBuilder()
        messages = builder.build("question", [])
        assert messages[0]["role"] == "system"  # 验证第一条是系统消息
        assert "智能问答助手" in messages[0]["content"]  # 验证包含系统提示词

    def test_prompt_builder_includes_history(self):
        """
        测试提示构建器包含历史消息。

        测试场景：构建包含历史对话的提示消息
        预期结果：消息列表包含历史角色，最后一条为当前问题
        """
        builder = PromptBuilder()
        # 构建历史对话
        history = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
        messages = builder.build("follow up", [], history)
        roles = [m["role"] for m in messages]
        assert "user" in roles  # 验证包含用户消息
        # 最后一条消息应该是当前问题
        assert messages[-1]["content"] == "follow up"
