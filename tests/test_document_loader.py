from __future__ import annotations

import os
import tempfile

import pytest

from app.services.document_loader import DocumentLoaderFactory
from app.utils.exceptions import UnsupportedFileTypeError


class TestDocumentLoaderFactory:
    def test_supported_extensions(self):
        exts = DocumentLoaderFactory.supported_extensions()
        assert ".pdf" in exts
        assert ".txt" in exts
        assert ".docx" in exts
        assert ".md" in exts

    def test_load_txt_file(self, tmp_path):
        p = tmp_path / "sample.txt"
        p.write_text("Hello world. This is a test document.", encoding="utf-8")
        docs = DocumentLoaderFactory.load(str(p))
        assert len(docs) >= 1
        assert "Hello world" in docs[0].page_content

    def test_unsupported_extension_raises(self):
        with pytest.raises(UnsupportedFileTypeError):
            DocumentLoaderFactory.load("/tmp/file.xyz")

    def test_load_nonexistent_raises(self):
        from app.utils.exceptions import DocumentLoadError

        with pytest.raises(DocumentLoadError):
            DocumentLoaderFactory.load("/tmp/nonexistent_file_abc123.txt")
