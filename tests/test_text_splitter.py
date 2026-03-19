"""
文本分割器测试模块

本模块测试文本分割服务的功能，包括：
- 长文本分割
- 短文本单块处理
- 元数据保留
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.services.text_splitter import TextSplitterService


class TestTextSplitter:
    """文本分割器测试类"""

    def test_split_basic(self, test_settings):
        """
        测试长文本分割功能。

        测试场景：输入超长文本进行分割
        预期结果：文本被分割成多个块
        """
        splitter = TextSplitterService(test_settings)
        # 创建超长测试文本
        long_text = "这是一段很长的测试文本。" * 100
        docs = [Document(page_content=long_text, metadata={"source": "test"})]
        chunks = splitter.split(docs)
        assert len(chunks) > 1  # 验证文本被分割成多个块

    def test_short_text_single_chunk(self, test_settings):
        """
        测试短文本保持单块。

        测试场景：输入短文本进行分割
        预期结果：文本保持为单个块，不进行分割
        """
        splitter = TextSplitterService(test_settings)
        docs = [Document(page_content="Short text.", metadata={"source": "test"})]
        chunks = splitter.split(docs)
        assert len(chunks) == 1  # 验证短文本保持单块

    def test_metadata_preserved(self, test_settings):
        """
        测试分割后元数据保留。

        测试场景：分割带有元数据的文档
        预期结果：所有分割块都保留原始元数据
        """
        splitter = TextSplitterService(test_settings)
        # 创建带元数据的长文本
        long_text = "内容。" * 200
        docs = [Document(page_content=long_text, metadata={"source": "doc.pdf", "page": 1})]
        chunks = splitter.split(docs)
        # 验证每个块都保留了源文件元数据
        for chunk in chunks:
            assert chunk.metadata["source"] == "doc.pdf"
