"""嵌入引擎模块，提供文本嵌入向量生成服务。

本模块根据配置的嵌入模型提供商（OpenAI 或本地模型），
创建相应的 Embeddings 实例，用于将文本转换为向量表示。

支持的嵌入模型提供商：
- OpenAI: 使用 OpenAI 的嵌入 API
- Local: 使用本地 HuggingFace 模型
"""

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    """嵌入引擎类，根据配置创建嵌入模型实例。
    
    支持的嵌入模型提供商：
    - "openai": 使用 OpenAI 嵌入 API
    - "local": 使用本地 HuggingFace 模型
    
    Attributes:
        _settings: 应用配置对象。
        _embeddings: 创建的嵌入模型实例。
    """

    def __init__(self, settings: Settings) -> None:
        """初始化嵌入引擎。
        
        Args:
            settings: 应用配置对象。
        """
        self._settings = settings
        self._embeddings: Embeddings = self._build()

    def _build(self) -> Embeddings:
        """根据配置构建嵌入模型实例。
        
        Returns:
            嵌入模型实例。
        
        Raises:
            ValueError: 配置的嵌入模型提供商未知时抛出。
        """
        provider = self._settings.EMBEDDING_PROVIDER.lower()
        if provider == "openai":
            # OpenAI 嵌入模型
            logger.info("Using OpenAI embeddings: %s", self._settings.EMBEDDING_MODEL)
            kwargs: dict = {
                "model": self._settings.EMBEDDING_MODEL,
                "openai_api_key": self._settings.OPENAI_API_KEY,
            }
            # 如果配置了自定义 API 基础 URL
            if self._settings.OPENAI_API_BASE:
                kwargs["openai_api_base"] = self._settings.OPENAI_API_BASE
            return OpenAIEmbeddings(**kwargs)
        elif provider == "local":
            # 本地 HuggingFace 嵌入模型（延迟导入，使依赖可选）
            from langchain_community.embeddings import HuggingFaceEmbeddings

            model_name = self._settings.LOCAL_EMBEDDING_MODEL
            logger.info("Using local HuggingFace embeddings: %s", model_name)
            return HuggingFaceEmbeddings(model_name=model_name)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @property
    def embeddings(self) -> Embeddings:
        """获取嵌入模型实例。
        
        Returns:
            嵌入模型实例。
        """
        return self._embeddings
