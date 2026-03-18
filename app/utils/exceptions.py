from __future__ import annotations


class AppException(Exception):
    """Base application exception."""

    def __init__(self, detail: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR"):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(detail)


class DocumentLoadError(AppException):
    """Raised when a document cannot be loaded."""

    def __init__(self, detail: str = "Failed to load document"):
        super().__init__(detail=detail, status_code=400, error_code="DOCUMENT_LOAD_ERROR")


class UnsupportedFileTypeError(AppException):
    """Raised for unsupported file extensions."""

    def __init__(self, ext: str):
        super().__init__(
            detail=f"Unsupported file type: {ext}",
            status_code=400,
            error_code="UNSUPPORTED_FILE_TYPE",
        )


class CollectionNotFoundError(AppException):
    """Raised when a ChromaDB collection does not exist."""

    def __init__(self, name: str):
        super().__init__(
            detail=f"Collection '{name}' not found",
            status_code=404,
            error_code="COLLECTION_NOT_FOUND",
        )


class CollectionAlreadyExistsError(AppException):
    """Raised when trying to create a collection that already exists."""

    def __init__(self, name: str):
        super().__init__(
            detail=f"Collection '{name}' already exists",
            status_code=409,
            error_code="COLLECTION_ALREADY_EXISTS",
        )


class FileTooLargeError(AppException):
    """Raised when an uploaded file exceeds the size limit."""

    def __init__(self, max_mb: int):
        super().__init__(
            detail=f"File exceeds maximum allowed size of {max_mb} MB",
            status_code=413,
            error_code="FILE_TOO_LARGE",
        )


class RetrievalError(AppException):
    """Raised when retrieval fails."""

    def __init__(self, detail: str = "Retrieval failed"):
        super().__init__(detail=detail, status_code=500, error_code="RETRIEVAL_ERROR")


class LLMError(AppException):
    """Raised when the LLM call fails."""

    def __init__(self, detail: str = "LLM call failed"):
        super().__init__(detail=detail, status_code=502, error_code="LLM_ERROR")
