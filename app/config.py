"""
应用配置模块

本模块定义了应用的全局配置项，通过环境变量或 .env 文件加载配置。
配置项包括：LLM 模型、嵌入模型、向量数据库、检索参数、文本分割、重排序等。
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    全局应用配置类

    从环境变量或 .env 文件加载配置项，支持类型验证和默认值设置。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI / LLM 配置 ----------------------------------------------------
    OPENAI_API_KEY: str = ""  # OpenAI API 密钥
    OPENAI_API_BASE: Optional[str] = None  # OpenAI API 基础 URL（可选）
    LLM_MODEL: str = "gpt-3.5-turbo"  # 使用的 LLM 模型名称
    LLM_TEMPERATURE: float = 0.3  # LLM 温度参数，控制生成随机性
    LLM_MAX_TOKENS: int = 1024  # LLM 最大生成 token 数

    # --- 嵌入模型配置 ----------------------------------------------------------
    EMBEDDING_PROVIDER: str = "openai"  # 嵌入提供者："openai" 或 "local"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"  # OpenAI 嵌入模型名称
    LOCAL_EMBEDDING_MODEL: str = "shibing624/text2vec-base-chinese"  # 本地嵌入模型名称

    # --- ChromaDB 向量数据库配置 -----------------------------------------------
    CHROMA_PERSIST_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "chroma_db",
    )  # ChromaDB 数据持久化目录

    # --- 检索配置 --------------------------------------------------------------
    RETRIEVAL_TOP_K: int = 4  # 最终返回的文档数量
    RETRIEVAL_SCORE_THRESHOLD: float = 0.35  # 检索分数阈值
    RETRIEVAL_INITIAL_K: int = 20  # 重排序前的初始检索数量

    # --- 文本分割配置（父子分块策略）--------------------------------------------
    CHUNK_SIZE: int = 500  # 传统/后备平面分块大小
    CHUNK_OVERLAP: int = 80  # 分块重叠大小
    PARENT_CHUNK_SIZE: int = 1500  # 父分块大小，携带完整上下文
    PARENT_CHUNK_OVERLAP: int = 200  # 父分块重叠大小
    CHILD_CHUNK_SIZE: int = 300  # 子分块大小，用于精确检索
    CHILD_CHUNK_OVERLAP: int = 50  # 子分块重叠大小

    # --- 重排序器配置 ----------------------------------------------------------
    RERANKER_ENABLED: bool = True  # 是否启用重排序
    RERANKER_PROVIDER: str = "cross-encoder"  # 重排序提供者："cross-encoder" 或 "llm"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"  # 重排序模型名称
    RERANKER_TOP_N: int = 4  # 重排序后返回的文档数量

    # --- BM25 关键词检索配置 ---------------------------------------------------
    BM25_ENABLED: bool = True  # 是否启用 BM25 检索
    BM25_TOP_K: int = 10  # BM25 检索结果数量，用于 RRF 融合

    # --- Text-to-SQL 配置 ------------------------------------------------------
    TEXT_TO_SQL_ENABLED: bool = True  # 是否启用文本转 SQL 功能

    # --- LangGraph 智能体流水线配置 --------------------------------------------
    AGENT_MAX_RETRIES: int = 2  # 质量检查最大重试次数

    # --- OCR 配置（pytesseract）-----------------------------------------------
    OCR_ENABLED: bool = True  # 是否启用 OCR
    OCR_LANGUAGE: str = "chi_sim+eng"  # Tesseract 语言代码

    # --- 音视频转录配置（Whisper）----------------------------------------------
    WHISPER_ENABLED: bool = True  # 是否启用 Whisper 转录
    WHISPER_MODEL: str = "base"  # Whisper 模型大小：tiny, base, small, medium, large

    # --- 文件上传配置 ----------------------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = 50  # 最大上传文件大小（MB），已为音视频文件增加

    # --- 日志配置 --------------------------------------------------------------
    LOG_LEVEL: str = "INFO"  # 日志级别


# 全局配置实例
settings = Settings()
