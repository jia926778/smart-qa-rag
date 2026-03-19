"""
检索器测试模块

本模块测试智能检索器的功能，包括：
- 正常检索返回文档
- 检索错误处理
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.services.retriever import SmartRetriever
from app.utils.exceptions import RetrievalError


class TestSmartRetriever:
    """智能检索器测试类"""

    def test_retrieve_returns_documents(self, test_settings, mock_embedding):
        """
        测试检索器正常返回文档。

        测试场景：调用 retrieve 方法检索问题相关文档
        预期结果：返回包含相关内容的文档列表
        """
        retriever = SmartRetriever(test_settings, mock_embedding)

        # 准备模拟的检索结果文档
        fake_docs = [Document(page_content="answer", metadata={"source": "a.txt"})]

        # 模拟 Chroma 向量数据库
        with patch("app.services.retriever.Chroma") as MockChroma:
            mock_vs = MagicMock()
            MockChroma.return_value = mock_vs
            mock_ret = MagicMock()
            mock_vs.as_retriever.return_value = mock_ret
            mock_ret.invoke.return_value = fake_docs

            # 执行检索
            result = retriever.retrieve("test question", "default")
            assert len(result) == 1  # 验证返回一个文档
            assert result[0].page_content == "answer"  # 验证内容正确

    def test_retrieve_error_raises(self, test_settings, mock_embedding):
        """
        测试检索器错误时抛出异常。

        测试场景：模拟向量数据库连接错误
        预期结果：抛出 RetrievalError 异常
        """
        retriever = SmartRetriever(test_settings, mock_embedding)

        # 模拟 Chroma 抛出异常
        with patch("app.services.retriever.Chroma") as MockChroma:
            MockChroma.side_effect = RuntimeError("db error")
            with pytest.raises(RetrievalError):
                retriever.retrieve("question", "default")
