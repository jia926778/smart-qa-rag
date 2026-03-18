from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    """Provide an Embeddings instance based on the configured provider."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings: Embeddings = self._build()

    def _build(self) -> Embeddings:
        provider = self._settings.EMBEDDING_PROVIDER.lower()
        if provider == "openai":
            logger.info("Using OpenAI embeddings: %s", self._settings.EMBEDDING_MODEL)
            kwargs: dict = {
                "model": self._settings.EMBEDDING_MODEL,
                "openai_api_key": self._settings.OPENAI_API_KEY,
            }
            if self._settings.OPENAI_API_BASE:
                kwargs["openai_api_base"] = self._settings.OPENAI_API_BASE
            return OpenAIEmbeddings(**kwargs)
        elif provider == "local":
            # Import lazily so HuggingFace deps are optional
            from langchain_community.embeddings import HuggingFaceEmbeddings

            model_name = self._settings.LOCAL_EMBEDDING_MODEL
            logger.info("Using local HuggingFace embeddings: %s", model_name)
            return HuggingFaceEmbeddings(model_name=model_name)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings
