"""
日志工具模块

本模块提供统一的日志记录功能，为应用程序中的各个组件提供配置好的Logger实例。
所有日志输出到标准输出，格式统一，便于日志收集和分析。

主要功能：
- 提供统一的日志格式
- 支持通过配置文件设置日志级别
- 避免重复添加日志处理器
"""
from __future__ import annotations

import logging
import sys

from app.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    获取配置好的Logger实例

    创建或获取指定名称的Logger实例，并配置统一的日志格式和处理器。
    如果Logger已经配置过处理器，则不会重复添加，避免日志重复输出。

    Args:
        name: Logger名称，通常使用 __name__ 作为模块标识

    Returns:
        logging.Logger: 配置好的Logger实例

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("这是一条信息日志")
        >>> logger.error("这是一条错误日志")
    """
    # 获取或创建指定名称的Logger实例
    logger = logging.getLogger(name)

    # 检查是否已配置处理器，避免重复添加导致日志重复输出
    if not logger.handlers:
        # 创建标准输出处理器，将日志输出到控制台
        handler = logging.StreamHandler(sys.stdout)

        # 创建格式化器，定义统一的日志格式
        # 格式：时间 | 日志级别（8字符宽度左对齐） | Logger名称 | 日志消息
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # 日期格式：年-月-日 时:分:秒
        )
        handler.setFormatter(formatter)

        # 将处理器添加到Logger
        logger.addHandler(handler)

    # 从配置中获取日志级别，如果配置无效则默认为INFO级别
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    return logger
