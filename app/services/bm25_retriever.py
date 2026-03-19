"""BM25 关键词检索服务模块（支持持久化索引）。

本模块实现了基于 Okapi BM25 算法的关键词检索功能，主要特点：
- 使用 rank_bm25 库进行 BM25 评分计算
- 索引基于集合中的所有子块构建，并持久化到磁盘（pickle 格式）
- 进程重启后无需重新索引，直接加载持久化文件
- 中文文本使用 jieba 分词，其他语言使用字符级分词

主要组件：
- BM25Index: 内存中的 BM25 索引实现
- BM25RetrieverService: 管理多集合 BM25 索引的服务类
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """对文本进行分词，用于 BM25 索引构建和查询。
    
    Args:
        text: 待分词的文本字符串。
    
    Returns:
        分词后的 token 列表（已转换为小写，过滤短词）。
    
    Note:
        - 中文文本使用 jieba 搜索模式分词，提高召回率
        - 其他语言使用字符级 + 空格分词作为降级方案
        - 过滤长度小于 2 的 token，减少噪声
    """
    try:
        import jieba  # type: ignore
        # 使用搜索模式分词，提高召回率
        tokens = list(jieba.cut_for_search(text))
    except ImportError:
        # 降级方案：字符级 + 空格分词
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
    # 过滤过短的 token，减少噪声
    return [t.strip().lower() for t in tokens if len(t.strip()) > 1]


class BM25Index:
    """内存中的 BM25 索引，用于对文档集合进行关键词检索。
    
    Attributes:
        _documents: 索引的文档列表。
        _bm25: rank_bm25 的 BM25Okapi 实例。
        _corpus_size: 语料库大小（文档数量）。
    """

    def __init__(self, documents: List[Document]) -> None:
        """初始化 BM25 索引。
        
        Args:
            documents: 要索引的文档列表，每个文档将被分词后加入语料库。
        """
        from rank_bm25 import BM25Okapi  # type: ignore

        self._documents = documents
        # 对所有文档进行分词，构建语料库
        corpus = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._corpus_size = len(corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """使用 BM25 算法检索与查询最相关的文档。
        
        Args:
            query: 查询字符串。
            top_k: 返回的最大文档数量，默认为 10。
        
        Returns:
            (文档, 分数) 元组列表，按 BM25 分数降序排列。
        """
        if not self._bm25 or self._corpus_size == 0:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        # 计算所有文档的 BM25 分数
        scores = self._bm25.get_scores(tokenized_query)
        # 获取 top-k 索引
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[Document, float]] = []
        for idx, score in indexed_scores[:top_k]:
            # 只返回分数大于 0 的结果
            if score > 0:
                doc = self._documents[idx]
                results.append((doc, float(score)))
        return results


class BM25RetrieverService:
    """BM25 索引管理服务，支持多集合索引和磁盘持久化。
    
    为每个集合维护独立的 BM25 索引，索引文件存储在：
    ``{CHROMA_PERSIST_DIR}/../bm25_indices/`` 目录下。
    
    Attributes:
        _settings: 应用配置对象。
        _index_dir: 索引文件存储目录。
        _cache: 内存中的索引缓存（集合名 -> BM25Index）。
    """

    def __init__(self, settings: Settings) -> None:
        """初始化 BM25 检索服务。
        
        Args:
            settings: 应用配置对象，用于获取持久化目录路径。
        """
        self._settings = settings
        # 索引文件存储在 Chroma 持久化目录的同级目录下
        self._index_dir = os.path.join(
            os.path.dirname(settings.CHROMA_PERSIST_DIR), "bm25_indices"
        )
        os.makedirs(self._index_dir, exist_ok=True)
        # 内存缓存：集合名 -> BM25Index
        self._cache: Dict[str, BM25Index] = {}

    def _index_path(self, collection_name: str) -> str:
        """获取指定集合的索引文件路径。
        
        Args:
            collection_name: 集合名称。
        
        Returns:
            索引文件的完整路径。
        """
        return os.path.join(self._index_dir, f"{collection_name}.bm25.pkl")

    def build_index(self, collection_name: str, documents: List[Document]) -> None:
        """为指定文档集合构建并持久化 BM25 索引。
        
        Args:
            collection_name: 集合名称。
            documents: 要索引的文档列表。
        """
        index = BM25Index(documents)
        self._cache[collection_name] = index

        # 持久化到磁盘
        path = self._index_path(collection_name)
        try:
            with open(path, "wb") as f:
                pickle.dump(documents, f)
            logger.info(
                "BM25 index built and saved: %s (%d docs)",
                collection_name,
                len(documents),
            )
        except Exception as exc:
            logger.warning("Failed to persist BM25 index: %s", exc)

    def add_documents(self, collection_name: str, documents: List[Document]) -> None:
        """向现有索引中增量添加文档（通过重建索引实现）。
        
        Args:
            collection_name: 集合名称。
            documents: 要添加的新文档列表。
        """
        existing = self._load_docs(collection_name)
        all_docs = existing + documents
        self.build_index(collection_name, all_docs)

    def _load_docs(self, collection_name: str) -> List[Document]:
        """从磁盘加载持久化的文档列表。
        
        Args:
            collection_name: 集合名称。
        
        Returns:
            文档列表，如果文件不存在或加载失败则返回空列表。
        """
        path = self._index_path(collection_name)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Failed to load BM25 docs: %s", exc)
            return []

    def _get_index(self, collection_name: str) -> Optional[BM25Index]:
        """获取或加载指定集合的 BM25 索引。
        
        优先从内存缓存获取，缓存未命中时从磁盘加载。
        
        Args:
            collection_name: 集合名称。
        
        Returns:
            BM25Index 实例，如果索引不存在则返回 None。
        """
        if collection_name in self._cache:
            return self._cache[collection_name]

        docs = self._load_docs(collection_name)
        if docs:
            index = BM25Index(docs)
            self._cache[collection_name] = index
            return index
        return None

    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
    ) -> List[Document]:
        """使用 BM25 算法进行关键词检索。
        
        Args:
            query: 查询字符串。
            collection_name: 集合名称。
            top_k: 返回的最大文档数量，默认为 10。
        
        Returns:
            匹配的文档列表，每个文档的 metadata 中包含 bm25_score 和 retrieval_method。
        """
        index = self._get_index(collection_name)
        if not index:
            logger.debug("No BM25 index for collection '%s'", collection_name)
            return []

        results = index.search(query, top_k=top_k)
        docs = []
        for doc, score in results:
            # 将 BM25 分数和检索方法添加到元数据
            doc.metadata["bm25_score"] = score
            doc.metadata["retrieval_method"] = "bm25"
            docs.append(doc)

        logger.info(
            "BM25 search: %d results for collection '%s'",
            len(docs),
            collection_name,
        )
        return docs

    def delete_index(self, collection_name: str) -> None:
        """删除指定集合的 BM25 索引。
        
        同时清理内存缓存和磁盘文件。
        
        Args:
            collection_name: 集合名称。
        """
        self._cache.pop(collection_name, None)
        path = self._index_path(collection_name)
        if os.path.exists(path):
            os.remove(path)

    def delete_source(self, collection_name: str, source: str) -> None:
        """删除指定来源的文档并重建索引。
        
        Args:
            collection_name: 集合名称。
            source: 要删除的文档来源标识。
        """
        docs = self._load_docs(collection_name)
        # 过滤掉指定来源的文档
        filtered = [d for d in docs if d.metadata.get("source") != source]
        if len(filtered) != len(docs):
            self.build_index(collection_name, filtered)
