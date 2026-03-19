"""
异常定义模块

本模块定义了应用程序中使用的所有自定义异常类。
所有异常都继承自 AppException 基类，提供统一的异常处理接口。

异常层次结构：
- AppException: 应用异常基类
  - DocumentLoadError: 文档加载异常
  - UnsupportedFileTypeError: 不支持的文件类型异常
  - CollectionNotFoundError: 集合不存在异常
  - CollectionAlreadyExistsError: 集合已存在异常
  - FileTooLargeError: 文件过大异常
  - RetrievalError: 检索失败异常
  - LLMError: LLM调用异常
"""
from __future__ import annotations


class AppException(Exception):
    """
    应用程序异常基类

    所有自定义异常的基类，提供统一的异常信息结构，包括错误详情、HTTP状态码和错误代码。

    Args:
        detail: 错误详细信息，描述具体的错误内容
        status_code: HTTP状态码，默认为500（服务器内部错误）
        error_code: 应用内部错误代码，默认为"INTERNAL_ERROR"

    Attributes:
        detail: 错误详细信息
        status_code: HTTP状态码
        error_code: 应用内部错误代码
    """

    def __init__(self, detail: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR"):
        # 保存错误详情，用于API响应
        self.detail = detail
        # 保存HTTP状态码，用于HTTP响应
        self.status_code = status_code
        # 保存应用内部错误代码，用于错误分类和处理
        self.error_code = error_code
        # 调用父类构造函数，设置异常消息
        super().__init__(detail)


class DocumentLoadError(AppException):
    """
    文档加载异常

    当文档无法成功加载时抛出此异常，例如文件损坏、格式错误等情况。

    Args:
        detail: 错误详细信息，默认为"Failed to load document"
    """

    def __init__(self, detail: str = "Failed to load document"):
        # 调用父类构造函数，设置HTTP状态码为400（客户端错误）
        super().__init__(detail=detail, status_code=400, error_code="DOCUMENT_LOAD_ERROR")


class UnsupportedFileTypeError(AppException):
    """
    不支持的文件类型异常

    当用户上传的文件扩展名不在支持列表中时抛出此异常。

    Args:
        ext: 不支持的文件扩展名（如".xyz"）
    """

    def __init__(self, ext: str):
        # 构造包含具体扩展名的错误信息
        super().__init__(
            detail=f"Unsupported file type: {ext}",
            status_code=400,  # 客户端错误
            error_code="UNSUPPORTED_FILE_TYPE",
        )


class CollectionNotFoundError(AppException):
    """
    集合不存在异常

    当尝试访问不存在的ChromaDB集合时抛出此异常。

    Args:
        name: 不存在的集合名称
    """

    def __init__(self, name: str):
        # 构造包含集合名称的错误信息，使用404状态码表示资源不存在
        super().__init__(
            detail=f"Collection '{name}' not found",
            status_code=404,  # 资源不存在
            error_code="COLLECTION_NOT_FOUND",
        )


class CollectionAlreadyExistsError(AppException):
    """
    集合已存在异常

    当尝试创建已存在的ChromaDB集合时抛出此异常。

    Args:
        name: 已存在的集合名称
    """

    def __init__(self, name: str):
        # 构造包含集合名称的错误信息，使用409状态码表示冲突
        super().__init__(
            detail=f"Collection '{name}' already exists",
            status_code=409,  # 资源冲突
            error_code="COLLECTION_ALREADY_EXISTS",
        )


class FileTooLargeError(AppException):
    """
    文件过大异常

    当上传的文件大小超过系统限制时抛出此异常。

    Args:
        max_mb: 系统允许的最大文件大小（单位：MB）
    """

    def __init__(self, max_mb: int):
        # 构造包含大小限制的错误信息，使用413状态码表示请求实体过大
        super().__init__(
            detail=f"File exceeds maximum allowed size of {max_mb} MB",
            status_code=413,  # 请求实体过大
            error_code="FILE_TOO_LARGE",
        )


class RetrievalError(AppException):
    """
    检索失败异常

    当向量检索或文档检索过程中发生错误时抛出此异常。

    Args:
        detail: 错误详细信息，默认为"Retrieval failed"
    """

    def __init__(self, detail: str = "Retrieval failed"):
        # 调用父类构造函数，设置HTTP状态码为500（服务器内部错误）
        super().__init__(detail=detail, status_code=500, error_code="RETRIEVAL_ERROR")


class LLMError(AppException):
    """
    LLM调用异常

    当调用大语言模型（LLM）失败时抛出此异常，例如API调用失败、超时等情况。

    Args:
        detail: 错误详细信息，默认为"LLM call failed"
    """

    def __init__(self, detail: str = "LLM call failed"):
        # 调用父类构造函数，设置HTTP状态码为502（网关错误）
        super().__init__(detail=detail, status_code=502, error_code="LLM_ERROR")
