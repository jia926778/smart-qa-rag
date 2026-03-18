from __future__ import annotations

import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI / LLM ---------------------------------------------------------
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: Optional[str] = None
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 1024

    # --- Embedding -------------------------------------------------------------
    EMBEDDING_PROVIDER: str = "openai"  # "openai" or "local"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LOCAL_EMBEDDING_MODEL: str = "shibing624/text2vec-base-chinese"

    # --- ChromaDB --------------------------------------------------------------
    CHROMA_PERSIST_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "chroma_db",
    )

    # --- Retrieval -------------------------------------------------------------
    RETRIEVAL_TOP_K: int = 4
    RETRIEVAL_SCORE_THRESHOLD: float = 0.35

    # --- Text splitting --------------------------------------------------------
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80

    # --- Upload ----------------------------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = 20

    # --- Logging ---------------------------------------------------------------
    LOG_LEVEL: str = "INFO"


settings = Settings()
