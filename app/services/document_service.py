"""文档服务模块，处理完整的文档摄入流程。

本模块实现文档的完整处理流水线：保存 -> 加载 -> 切分 -> 嵌入 -> 存储。

主要特点：
- 使用父子块切分策略：
  - 子块（小而精确）存储在主集合中，用于向量检索
  - 父块（大而丰富）存储在 ``{collection}_parents`` 集合中，
    查询时提供更丰富的上下文
- BM25 索引：子块被索引用于关键词搜索
- SQL 存储：Excel/CSV 文件自动导入 SQLite 用于 Text-to-SQL 查询
"""

import os
import shutil
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.services.bm25_retriever import BM25RetrieverService
from app.services.collection_service import CollectionService
from app.services.document_loader import DocumentLoaderFactory
from app.services.sql_store import SQLStore
from app.services.text_splitter import (
    ParentChildTextSplitter,
    TextSplitterService,
)
from app.models.schemas import UploadResponse
from app.utils.exceptions import FileTooLargeError
from app.utils.logger import get_logger

logger = get_logger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")


class DocumentService:
    """Handle the full pipeline: save -> load -> split -> embed -> store.

    Uses **parent-child chunking** by default:
    - Child chunks (small, precise) are stored in the main collection
      for vector retrieval.
    - Parent chunks (large, context-rich) are stored in a sibling
      ``{collection}_parents`` collection and looked up at query time
      to provide richer context to the LLM.

    Additional pipelines:
    - **BM25 index**: child chunks are indexed for keyword search.
    - **SQL store**: Excel/CSV files are auto-ingested into SQLite for
      Text-to-SQL queries.
    """

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings,
        text_splitter: TextSplitterService,
        parent_child_splitter: ParentChildTextSplitter,
        collection_service: CollectionService,
        bm25_service: BM25RetrieverService | None = None,
        sql_store: SQLStore | None = None,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._flat_splitter = text_splitter
        self._pc_splitter = parent_child_splitter
        self._collection_service = collection_service
        self._bm25 = bm25_service
        self._sql_store = sql_store

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        collection_name: str = "default",
    ) -> UploadResponse:
        # Validate size
        max_bytes = self._settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise FileTooLargeError(self._settings.MAX_UPLOAD_SIZE_MB)

        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Save to temp file with unique name
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        try:
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            # Ensure both collections exist
            self._collection_service.ensure_exists(collection_name)
            parent_collection = f"{collection_name}_parents"
            self._collection_service.ensure_exists(parent_collection)

            # Load documents
            documents = DocumentLoaderFactory.load(file_path)

            # Inject original filename into metadata
            for doc in documents:
                doc.metadata["source"] = filename

            # --- Parent-child chunking ---
            pc_result = self._pc_splitter.split(documents)

            # Store child chunks in main collection (for retrieval)
            child_store = Chroma(
                collection_name=collection_name,
                persist_directory=self._settings.CHROMA_PERSIST_DIR,
                embedding_function=self._embeddings,
            )
            if pc_result.child_docs:
                child_store.add_documents(pc_result.child_docs)

            # Store parent chunks in parent collection (for context lookup)
            parent_store = Chroma(
                collection_name=parent_collection,
                persist_directory=self._settings.CHROMA_PERSIST_DIR,
                embedding_function=self._embeddings,
            )
            if pc_result.parent_docs:
                parent_store.add_documents(pc_result.parent_docs)

            total_chunks = len(pc_result.child_docs)
            parent_chunks = len(pc_result.parent_docs)

            # --- BM25 index: add child chunks for keyword search ---
            if self._bm25 and pc_result.child_docs:
                try:
                    self._bm25.add_documents(collection_name, pc_result.child_docs)
                    logger.info("BM25: indexed %d child chunks for '%s'", total_chunks, filename)
                except Exception as exc:
                    logger.warning("BM25 indexing failed for '%s': %s", filename, exc)

            # --- SQL store: auto-ingest structured data from Excel/CSV ---
            sql_tables_ingested = 0
            ext = Path(filename).suffix.lower()
            if self._sql_store and ext in (".xlsx", ".xls", ".csv", ".tsv"):
                try:
                    if ext in (".xlsx", ".xls"):
                        tables = self._sql_store.ingest_from_excel(
                            collection_name, file_path, source_file=filename,
                        )
                        sql_tables_ingested = len(tables)
                    elif ext in (".csv", ".tsv"):
                        result = self._sql_store.ingest_from_csv(
                            collection_name, file_path, source_file=filename,
                        )
                        sql_tables_ingested = 1 if result else 0
                    if sql_tables_ingested:
                        logger.info(
                            "SQL store: ingested %d table(s) from '%s'",
                            sql_tables_ingested, filename,
                        )
                except Exception as exc:
                    logger.warning("SQL ingestion failed for '%s': %s", filename, exc)

            logger.info(
                "Ingested '%s' -> %d child chunks + %d parent chunks "
                "into collection '%s'",
                filename,
                total_chunks,
                parent_chunks,
                collection_name,
            )

            msg = (
                    f"Successfully ingested {filename} "
                    f"({parent_chunks} parent chunks, "
                    f"{total_chunks} child chunks)"
                )
            if sql_tables_ingested:
                msg += f" + {sql_tables_ingested} SQL table(s)"

            return UploadResponse(
                filename=filename,
                collection_name=collection_name,
                chunks_count=total_chunks,
                message=msg,
            )
        finally:
            # Cleanup temp file
            if os.path.exists(file_path):
                os.remove(file_path)

    def list_documents(self, collection_name: str):
        """Return a list of unique source documents in a collection."""
        self._collection_service.get(collection_name)  # will raise if not found
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=self._settings.CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
        )
        result = vectorstore.get()
        sources: dict = {}
        metadatas = result.get("metadatas") or []
        ids = result.get("ids") or []
        for doc_id, meta in zip(ids, metadatas):
            src = (meta or {}).get("source", "unknown")
            if src not in sources:
                sources[src] = {"source": src, "id": doc_id, "chunks": 0}
            sources[src]["chunks"] += 1
        return list(sources.values())

    def delete_document(self, collection_name: str, doc_source: str) -> int:
        """Delete all chunks belonging to a specific source document.

        Removes from both child (main) and parent collections.
        """
        self._collection_service.get(collection_name)  # will raise if not found

        deleted = 0

        # Delete from child collection
        child_store = Chroma(
            collection_name=collection_name,
            persist_directory=self._settings.CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
        )
        result = child_store.get(where={"source": doc_source})
        ids_to_delete = result.get("ids") or []
        if ids_to_delete:
            child_store.delete(ids=ids_to_delete)
            deleted += len(ids_to_delete)

        # Delete from parent collection
        parent_collection = f"{collection_name}_parents"
        try:
            parent_store = Chroma(
                collection_name=parent_collection,
                persist_directory=self._settings.CHROMA_PERSIST_DIR,
                embedding_function=self._embeddings,
            )
            p_result = parent_store.get(where={"source": doc_source})
            p_ids = p_result.get("ids") or []
            if p_ids:
                parent_store.delete(ids=p_ids)
                deleted += len(p_ids)
        except Exception as exc:
            logger.warning("Could not clean parent collection: %s", exc)

        # Delete from BM25 index
        if self._bm25:
            try:
                self._bm25.delete_source(collection_name, doc_source)
            except Exception as exc:
                logger.warning("BM25 cleanup failed: %s", exc)

        # Delete from SQL store
        if self._sql_store:
            try:
                self._sql_store.delete_source(collection_name, doc_source)
            except Exception as exc:
                logger.warning("SQL store cleanup failed: %s", exc)

        logger.info(
            "Deleted %d chunk(s) for source '%s' from '%s' (incl. parents)",
            deleted,
            doc_source,
            collection_name,
        )
        return deleted
