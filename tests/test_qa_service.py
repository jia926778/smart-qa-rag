from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import AskRequest, ChatMessage
from app.services.prompt_builder import PromptBuilder
from app.services.qa_service import QAService
from app.utils.exceptions import LLMError


class TestQAService:
    @pytest.fixture()
    def qa_service(self, test_settings, mock_embedding):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        prompt_builder = PromptBuilder()
        with patch("app.services.qa_service.ChatOpenAI"):
            svc = QAService(
                settings=test_settings,
                retriever=retriever,
                prompt_builder=prompt_builder,
            )
        return svc

    @pytest.mark.asyncio
    async def test_ask_returns_response(self, qa_service):
        mock_resp = MagicMock()
        mock_resp.content = "This is the answer."
        qa_service._llm.ainvoke = AsyncMock(return_value=mock_resp)

        req = AskRequest(question="What is RAG?", collection_name="default")
        result = await qa_service.ask(req)

        assert result.answer == "This is the answer."
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_ask_llm_failure_raises(self, qa_service):
        qa_service._llm.ainvoke = AsyncMock(side_effect=RuntimeError("api down"))

        req = AskRequest(question="test", collection_name="default")
        with pytest.raises(LLMError):
            await qa_service.ask(req)

    def test_prompt_builder_includes_system(self):
        builder = PromptBuilder()
        messages = builder.build("question", [])
        assert messages[0]["role"] == "system"
        assert "智能问答助手" in messages[0]["content"]

    def test_prompt_builder_includes_history(self):
        builder = PromptBuilder()
        history = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
        messages = builder.build("follow up", [], history)
        roles = [m["role"] for m in messages]
        assert "user" in roles
        # last message should be the current question
        assert messages[-1]["content"] == "follow up"
