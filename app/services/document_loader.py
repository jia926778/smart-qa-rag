from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Type

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from app.utils.exceptions import UnsupportedFileTypeError, DocumentLoadError
from app.utils.logger import get_logger

logger = get_logger(__name__)

LOADER_MAP: Dict[str, Type[BaseLoader]] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}


class DocumentLoaderFactory:
    """Create the appropriate document loader based on file extension."""

    @staticmethod
    def supported_extensions() -> List[str]:
        return list(LOADER_MAP.keys())

    @staticmethod
    def load(file_path: str) -> List[Document]:
        ext = Path(file_path).suffix.lower()
        loader_cls = LOADER_MAP.get(ext)
        if loader_cls is None:
            raise UnsupportedFileTypeError(ext)
        try:
            loader = loader_cls(file_path)
            documents = loader.load()
            logger.info("Loaded %d document(s) from %s", len(documents), file_path)
            return documents
        except UnsupportedFileTypeError:
            raise
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path, exc)
            raise DocumentLoadError(f"Failed to load {file_path}: {exc}") from exc
