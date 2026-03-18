from __future__ import annotations

from langchain_core.documents import Document

from app.services.text_splitter import TextSplitterService


class TestTextSplitter:
    def test_split_basic(self, test_settings):
        splitter = TextSplitterService(test_settings)
        long_text = "这是一段很长的测试文本。" * 100
        docs = [Document(page_content=long_text, metadata={"source": "test"})]
        chunks = splitter.split(docs)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self, test_settings):
        splitter = TextSplitterService(test_settings)
        docs = [Document(page_content="Short text.", metadata={"source": "test"})]
        chunks = splitter.split(docs)
        assert len(chunks) == 1

    def test_metadata_preserved(self, test_settings):
        splitter = TextSplitterService(test_settings)
        long_text = "内容。" * 200
        docs = [Document(page_content=long_text, metadata={"source": "doc.pdf", "page": 1})]
        chunks = splitter.split(docs)
        for chunk in chunks:
            assert chunk.metadata["source"] == "doc.pdf"
