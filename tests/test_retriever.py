from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.services.retriever import SmartRetriever
from app.utils.exceptions import RetrievalError


class TestSmartRetriever:
    def test_retrieve_returns_documents(self, test_settings, mock_embedding):
        retriever = SmartRetriever(test_settings, mock_embedding)

        fake_docs = [Document(page_content="answer", metadata={"source": "a.txt"})]

        with patch("app.services.retriever.Chroma") as MockChroma:
            mock_vs = MagicMock()
            MockChroma.return_value = mock_vs
            mock_ret = MagicMock()
            mock_vs.as_retriever.return_value = mock_ret
            mock_ret.invoke.return_value = fake_docs

            result = retriever.retrieve("test question", "default")
            assert len(result) == 1
            assert result[0].page_content == "answer"

    def test_retrieve_error_raises(self, test_settings, mock_embedding):
        retriever = SmartRetriever(test_settings, mock_embedding)

        with patch("app.services.retriever.Chroma") as MockChroma:
            MockChroma.side_effect = RuntimeError("db error")
            with pytest.raises(RetrievalError):
                retriever.retrieve("question", "default")
