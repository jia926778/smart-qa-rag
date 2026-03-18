from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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
    """Wraps RecursiveCharacterTextSplitter with sensible defaults (flat mode)."""

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


@dataclass
class ParentChildResult:
    """Container for parent-child splitting output."""

    parent_docs: List[Document] = field(default_factory=list)
    child_docs: List[Document] = field(default_factory=list)
    parent_map: Dict[str, Document] = field(default_factory=dict)  # parent_id -> doc


class ParentChildTextSplitter:
    """Two-level hierarchical splitter.

    1. First split into large *parent* chunks that preserve broad context.
    2. Each parent is further split into smaller *child* chunks used for
       precise vector retrieval.
    3. Child metadata contains ``parent_id`` so that the retriever can look
       up the richer parent chunk after matching a child.
    """

    def __init__(self, settings: Settings) -> None:
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.PARENT_CHUNK_SIZE,
            chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
            separators=CHINESE_SEPARATORS,
            length_function=len,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHILD_CHUNK_SIZE,
            chunk_overlap=settings.CHILD_CHUNK_OVERLAP,
            separators=CHINESE_SEPARATORS,
            length_function=len,
        )

    def split(self, documents: List[Document]) -> ParentChildResult:
        """Split *documents* into parent and child chunks.

        Returns a ``ParentChildResult`` containing both layers and a lookup
        map from ``parent_id`` to parent ``Document``.
        """
        result = ParentChildResult()

        # --- Stage 1: create parent chunks ------------------------------------
        parent_chunks = self._parent_splitter.split_documents(documents)

        for idx, parent in enumerate(parent_chunks):
            parent_id = uuid.uuid4().hex
            parent.metadata = {
                **parent.metadata,
                "parent_id": parent_id,
                "chunk_type": "parent",
                "parent_index": idx,
            }
            result.parent_docs.append(parent)
            result.parent_map[parent_id] = parent

            # --- Stage 2: split each parent into children ---------------------
            child_chunks = self._child_splitter.split_documents([parent])
            for child_idx, child in enumerate(child_chunks):
                child.metadata = {
                    **child.metadata,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "child_index": child_idx,
                }
                result.child_docs.append(child)

        logger.info(
            "Parent-child split: %d doc(s) -> %d parent(s) + %d child(ren)",
            len(documents),
            len(result.parent_docs),
            len(result.child_docs),
        )
        return result
