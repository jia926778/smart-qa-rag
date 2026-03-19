"""
文档加载器测试模块

本模块测试文档加载工厂类的功能，包括：
- 支持的文件扩展名检查
- 文本文件加载
- 不支持的文件类型错误处理
- 文件不存在错误处理
"""

from __future__ import annotations

import os
import tempfile

import pytest

from app.services.document_loader import DocumentLoaderFactory
from app.utils.exceptions import UnsupportedFileTypeError


class TestDocumentLoaderFactory:
    """文档加载工厂测试类"""

    def test_supported_extensions(self):
        """
        测试获取支持的文件扩展名列表。

        测试场景：调用 supported_extensions 方法
        预期结果：返回的列表包含 pdf、txt、docx、md 扩展名
        """
        exts = DocumentLoaderFactory.supported_extensions()
        assert ".pdf" in exts  # 验证支持 PDF 格式
        assert ".txt" in exts  # 验证支持纯文本格式
        assert ".docx" in exts  # 验证支持 Word 文档格式
        assert ".md" in exts  # 验证支持 Markdown 格式

    def test_load_txt_file(self, tmp_path):
        """
        测试加载文本文件。

        测试场景：创建临时文本文件并加载
        预期结果：成功加载文档，内容包含预期文本
        """
        # 创建临时测试文件
        p = tmp_path / "sample.txt"
        p.write_text("Hello world. This is a test document.", encoding="utf-8")
        # 加载文档
        docs = DocumentLoaderFactory.load(str(p))
        assert len(docs) >= 1  # 验证至少加载了一个文档
        assert "Hello world" in docs[0].page_content  # 验证内容正确

    def test_unsupported_extension_raises(self):
        """
        测试不支持的文件扩展名抛出异常。

        测试场景：尝试加载 .xyz 扩展名的文件
        预期结果：抛出 UnsupportedFileTypeError 异常
        """
        with pytest.raises(UnsupportedFileTypeError):
            DocumentLoaderFactory.load("/tmp/file.xyz")

    def test_load_nonexistent_raises(self):
        """
        测试加载不存在的文件抛出异常。

        测试场景：尝试加载不存在的文件
        预期结果：抛出 DocumentLoadError 异常
        """
        from app.utils.exceptions import DocumentLoadError

        with pytest.raises(DocumentLoadError):
            DocumentLoaderFactory.load("/tmp/nonexistent_file_abc123.txt")
