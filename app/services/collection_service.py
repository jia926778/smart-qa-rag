"""ChromaDB 集合管理服务模块。

本模块提供对 ChromaDB 集合的 CRUD 操作，包括：
- 创建集合（带描述和创建时间元数据）
- 列出所有集合
- 获取单个集合信息
- 删除集合
- 获取集合统计信息
- 确保集合存在（不存在则创建）
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

import chromadb

from app.models.schemas import CollectionInfo, CollectionStats
from app.utils.exceptions import CollectionAlreadyExistsError, CollectionNotFoundError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 集合元数据键名常量
_METADATA_KEY_DESC = "description"  # 描述字段键名
_METADATA_KEY_CREATED = "created_at"  # 创建时间字段键名


class CollectionService:
    """ChromaDB 集合管理服务，提供集合的 CRUD 操作。
    
    Attributes:
        _client: ChromaDB 客户端实例。
    """

    def __init__(self, chroma_client: chromadb.ClientAPI) -> None:
        """初始化集合服务。
        
        Args:
            chroma_client: ChromaDB 客户端实例。
        """
        self._client = chroma_client

    def create(self, name: str, description: str = "") -> CollectionInfo:
        """创建新的 ChromaDB 集合。
        
        Args:
            name: 集合名称。
            description: 集合描述，可选。
        
        Returns:
            创建的集合信息对象。
        
        Raises:
            CollectionAlreadyExistsError: 集合已存在时抛出。
        """
        # 检查集合是否已存在
        existing = [c.name for c in self._client.list_collections()]
        if name in existing:
            raise CollectionAlreadyExistsError(name)
        
        # 创建时间戳
        now = datetime.now(timezone.utc).isoformat()
        # 构建元数据
        metadata: Dict[str, str] = {
            _METADATA_KEY_DESC: description,
            _METADATA_KEY_CREATED: now,
        }
        self._client.get_or_create_collection(name=name, metadata=metadata)
        logger.info("Created collection '%s'", name)
        
        return CollectionInfo(
            name=name,
            description=description,
            documents_count=0,
            created_at=now,
        )

    def list_all(self) -> List[CollectionInfo]:
        """列出所有集合及其基本信息。
        
        Returns:
            所有集合的信息列表。
        """
        collections = self._client.list_collections()
        result: List[CollectionInfo] = []
        for col_name in collections:
            # 兼容不同版本的 ChromaDB 返回类型
            name = col_name if isinstance(col_name, str) else col_name.name
            try:
                col = self._client.get_collection(name)
            except Exception:
                continue
            meta = col.metadata or {}
            result.append(
                CollectionInfo(
                    name=name,
                    description=meta.get(_METADATA_KEY_DESC, ""),
                    documents_count=col.count(),
                    created_at=meta.get(_METADATA_KEY_CREATED),
                )
            )
        return result

    def get(self, name: str) -> CollectionInfo:
        """获取指定集合的详细信息。
        
        Args:
            name: 集合名称。
        
        Returns:
            集合信息对象。
        
        Raises:
            CollectionNotFoundError: 集合不存在时抛出。
        """
        try:
            col = self._client.get_collection(name)
        except Exception as exc:
            raise CollectionNotFoundError(name) from exc
        meta = col.metadata or {}
        return CollectionInfo(
            name=name,
            description=meta.get(_METADATA_KEY_DESC, ""),
            documents_count=col.count(),
            created_at=meta.get(_METADATA_KEY_CREATED),
        )

    def delete(self, name: str) -> None:
        """删除指定集合。
        
        Args:
            name: 集合名称。
        
        Raises:
            CollectionNotFoundError: 集合不存在时抛出。
        """
        try:
            self._client.delete_collection(name)
            logger.info("Deleted collection '%s'", name)
        except Exception as exc:
            raise CollectionNotFoundError(name) from exc

    def stats(self, name: str) -> CollectionStats:
        """获取集合的统计信息。
        
        Args:
            name: 集合名称。
        
        Returns:
            集合统计信息对象。
        """
        info = self.get(name)
        return CollectionStats(
            name=info.name,
            documents_count=info.documents_count,
            description=info.description,
        )

    def ensure_exists(self, name: str) -> None:
        """确保集合存在，不存在则创建。
        
        Args:
            name: 集合名称。
        """
        # 获取所有现有集合名称
        existing = [c.name if isinstance(c, str) else c.name if hasattr(c, "name") else c for c in self._client.list_collections()]
        # 兼容字符串和 Collection 对象两种类型
        names = []
        for c in existing:
            if isinstance(c, str):
                names.append(c)
            elif hasattr(c, "name"):
                names.append(c.name)
        # 如果集合不存在，则创建
        if name not in names:
            self.create(name)
