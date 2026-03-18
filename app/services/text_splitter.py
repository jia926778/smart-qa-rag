from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Separators optimised for Chinese + English mixed text
CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "……",
    "…",
    ". ",
    "! ",
    "? ",
    ";",
    " ",
    "",
]


class TextSplitterService:
    """Wraps RecursiveCharacterTextSplitter with sensible defaults."""

    def __init__(self, settings: Settings) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=CHINESE_SEPARATORS,
            length_function=len,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self._splitter.split_documents(documents)
        logger.info(
            "Split %d document(s) into %d chunk(s)", len(documents), len(chunks)
        )
        return chunks
