"""智能检索模块，实现多阶段混合检索流水线。

本模块提供多阶段检索策略：
1. 向量检索 — 通过 ChromaDB 进行语义子块搜索
2. BM25 检索 — 通过 rank_bm25 索引进行关键词搜索
3. 混合融合 — 通过 RRF 算法合并向量和 BM25 结果
4. 父块扩展 — 查找父块以提供更丰富的上下文
5. 重排序 — 交叉编码器 / LLM / 余弦重排序

检索策略（vector / bm25 / hybrid）可通过参数按查询控制。
"""

from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.services.bm25_retriever import BM25RetrieverService
from app.services.reranker import BaseReranker
from app.utils.exceptions import RetrievalError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _rrf_merge(
    doc_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """倒数排名融合算法，合并多个排序列表。
    
    RRF (Reciprocal Rank Fusion) 是一种简单有效的多列表融合方法，
    对每个文档计算其在所有列表中的倒数排名之和作为最终分数。
    
    Args:
        doc_lists: 多个文档列表，每个列表已按相关性排序。
        k: RRF 参数，默认为 60，用于平滑排名影响。
    
    Returns:
        合并后的文档列表，按 RRF 分数降序排列。
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            key = _doc_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
                scores[key] = 0.0
            # RRF 分数累加
            scores[key] += 1.0 / (k + rank)

    # 按分数降序排序
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = scores[key]
        result.append(doc)
    return result


def _doc_key(doc: Document) -> str:
    """生成文档的唯一键，用于去重和合并。
    
    Args:
        doc: 文档对象。
    
    Returns:
        文档的唯一键字符串。
    """
    source = doc.metadata.get("source", "")
    parent_id = doc.metadata.get("parent_id", "")
    if parent_id:
        return f"{source}::{parent_id}"
    return f"{source}::{hash(doc.page_content[:200])}"


class SmartRetriever:
    """智能检索器，实现多阶段混合检索流水线。
    
    检索阶段：
    1. **向量检索** — 通过 ChromaDB 进行语义子块搜索
    2. **BM25 检索** — 通过 rank_bm25 索引进行关键词搜索
    3. **混合融合** — 通过 RRF 算法合并向量和 BM25 结果
    4. **父块扩展** — 查找父块以提供更丰富的上下文
    5. **重排序** — 交叉编码器 / LLM / 余弦重排序
    
    检索策略（vector / bm25 / hybrid）可通过 ``strategy`` 参数按查询控制。
    
    Attributes:
        _settings: 应用配置对象。
        _embeddings: 嵌入模型实例。
        _reranker: 重排序器实例（可选）。
        _bm25: BM25 检索服务（可选）。
    """

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings,
        reranker: Optional[BaseReranker] = None,
        bm25_service: Optional[BM25RetrieverService] = None,
    ) -> None:
        """初始化智能检索器。
        
        Args:
            settings: 应用配置对象。
            embeddings: 嵌入模型实例。
            reranker: 重排序器实例，可选。
            bm25_service: BM25 检索服务，可选。
        """
        self._settings = settings
        self._embeddings = embeddings
        self._reranker = reranker
        self._bm25 = bm25_service

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int | None = None,
        score_threshold: float | None = None,
        strategy: str = "hybrid",
    ) -> List[Document]:
        """使用指定策略检索文档。
        
        Args:
            query: 查询字符串。
            collection_name: 集合名称，默认为 "default"。
            top_k: 返回的最大文档数量，可选。
            score_threshold: 分数阈值，可选。
            strategy: 检索策略，可选值为 "vector"、"bm25" 或 "hybrid"（默认）。
        
        Returns:
            检索到的文档列表。
        
        Raises:
            RetrievalError: 检索失败时抛出。
        """
        top_k = top_k or self._settings.RETRIEVAL_TOP_K
        score_threshold = score_threshold or self._settings.RETRIEVAL_SCORE_THRESHOLD

        try:
            doc_lists: List[List[Document]] = []

            # 向量检索
            if strategy in ("vector", "hybrid"):
                vector_docs = self._retrieve_children(query, collection_name, score_threshold)
                if vector_docs:
                    for d in vector_docs:
                        d.metadata["retrieval_method"] = "vector"
                    doc_lists.append(vector_docs)

            # BM25 检索
            if strategy in ("bm25", "hybrid") and self._bm25 and self._settings.BM25_ENABLED:
                bm25_docs = self._bm25.search(
                    query=query,
                    collection_name=collection_name,
                    top_k=self._settings.BM25_TOP_K,
                )
                if bm25_docs:
                    doc_lists.append(bm25_docs)

            if not doc_lists:
                logger.info("No docs found from any retrieval method.")
                return []

            # 如果有多个来源，使用 RRF 合并
            if len(doc_lists) == 1:
                candidates = doc_lists[0]
            else:
                candidates = _rrf_merge(doc_lists)
                logger.info(
                    "Hybrid fusion: vector(%d) + bm25(%d) -> %d merged",
                    len(doc_lists[0]) if doc_lists else 0,
                    len(doc_lists[1]) if len(doc_lists) > 1 else 0,
                    len(candidates),
                )

            # 父块扩展
            parent_docs = self._expand_to_parents(candidates, collection_name)

            # 重排序
            if self._reranker and self._settings.RERANKER_ENABLED:
                final_docs = self._reranker.rerank(
                    query=query,
                    documents=parent_docs,
                    top_n=self._settings.RERANKER_TOP_N,
                )
            else:
                final_docs = parent_docs[:top_k]

            logger.info(
                "Retrieval pipeline [%s]: %d candidates -> %d parents -> %d final "
                "(collection=%s)",
                strategy,
                len(candidates),
                len(parent_docs),
                len(final_docs),
                collection_name,
            )
            return final_docs

        except RetrievalError:
            raise
        except Exception as exc:
            logger.error("Retrieval error: %s", exc)
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

    # ------------------------------------------------------------------
    # 阶段 1a — 向量子块检索
    # ------------------------------------------------------------------

    def _retrieve_children(
        self, query: str, collection_name: str, score_threshold: float,
    ) -> List[Document]:
        """通过向量相似度检索子块。
        
        Args:
            query: 查询字符串。
            collection_name: 集合名称。
            score_threshold: 分数阈值。
        
        Returns:
            检索到的子块文档列表。
        """
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=self._settings.CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
        )
        initial_k = self._settings.RETRIEVAL_INITIAL_K
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": initial_k, "score_threshold": score_threshold},
        )
        docs = retriever.invoke(query)
        # 过滤出子块文档
        child_docs = [d for d in docs if d.metadata.get("chunk_type") == "child"]
        if not child_docs:
            child_docs = docs
        return child_docs

    # ------------------------------------------------------------------
    # 阶段 2 — 父块扩展
    # ------------------------------------------------------------------

    def _expand_to_parents(
        self, child_docs: List[Document], collection_name: str,
    ) -> List[Document]:
        """将子块扩展为父块，提供更丰富的上下文。
        
        Args:
            child_docs: 子块文档列表。
            collection_name: 集合名称。
        
        Returns:
            父块文档列表，如果无法扩展则返回原始子块列表。
        """
        parent_ids: List[str] = []
        seen: set = set()
        child_by_parent: Dict[str, List[Document]] = {}
        has_parent_meta = False

        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if pid:
                has_parent_meta = True
                if pid not in seen:
                    parent_ids.append(pid)
                    seen.add(pid)
                child_by_parent.setdefault(pid, []).append(child)

        if not has_parent_meta or not parent_ids:
            return child_docs

        parent_collection = f"{collection_name}_parents"
        try:
            parent_store = Chroma(
                collection_name=parent_collection,
                persist_directory=self._settings.CHROMA_PERSIST_DIR,
                embedding_function=self._embeddings,
            )
            result = parent_store.get(
                where={"parent_id": {"$in": parent_ids}},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("Parent fetch failed: %s. Using children.", exc)
            return child_docs

        fetched_parents: Dict[str, Document] = {}
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for doc_id, text, meta in zip(ids, documents, metadatas):
            pid = (meta or {}).get("parent_id")
            if pid and pid not in fetched_parents:
                parent_doc = Document(page_content=text or "", metadata=meta or {})
                parent_doc.metadata["matched_children_count"] = len(child_by_parent.get(pid, []))
                fetched_parents[pid] = parent_doc

        parent_docs: List[Document] = []
        for pid in parent_ids:
            if pid in fetched_parents:
                parent_docs.append(fetched_parents[pid])
            else:
                # 如果父块未找到，使用子块作为降级
                fallback = child_by_parent.get(pid, [None])[0]
                if fallback:
                    parent_docs.append(fallback)
        return parent_docs

    # ------------------------------------------------------------------
    # 旧版扁平检索
    # ------------------------------------------------------------------

    def retrieve_flat(
        self, query: str, collection_name: str = "default",
        top_k: int | None = None, score_threshold: float | None = None,
    ) -> List[Document]:
        """扁平检索，不使用父子块策略。
        
        Args:
            query: 查询字符串。
            collection_name: 集合名称，默认为 "default"。
            top_k: 返回的最大文档数量，可选。
            score_threshold: 分数阈值，可选。
        
        Returns:
            检索到的文档列表。
        """
        top_k = top_k or self._settings.RETRIEVAL_TOP_K
        score_threshold = score_threshold or self._settings.RETRIEVAL_SCORE_THRESHOLD
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=self._settings.CHROMA_PERSIST_DIR,
            embedding_function=self._embeddings,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": top_k, "score_threshold": score_threshold},
        )
        return retriever.invoke(query)
