"""
工具模块包

本包提供应用程序通用的工具类和函数，包括异常定义、日志记录等功能。

模块说明：
- exceptions: 定义应用程序中使用的所有自定义异常类
- logger: 提供统一的日志记录功能

使用示例：
    from app.utils import get_logger, AppException, DocumentLoadError

    # 获取Logger实例
    logger = get_logger(__name__)
    logger.info("应用程序启动")

    # 抛出自定义异常
    raise DocumentLoadError("无法加载文档：文件已损坏")
"""
