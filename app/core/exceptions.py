"""Custom exceptions"""

from typing import Optional, Dict, Any


class MojiException(Exception):
    """Base exception for MOJI application"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(MojiException):
    """Authentication failed"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, error_code="AUTH_FAILED")


class AuthorizationError(MojiException):
    """Authorization failed"""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, error_code="AUTH_FORBIDDEN")


class ValidationError(MojiException):
    """Validation error"""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(message, error_code="VALIDATION_ERROR", details=details)


class LLMError(MojiException):
    """LLM service error"""

    def __init__(self, message: str, provider: Optional[str] = None):
        details = {"provider": provider} if provider else {}
        super().__init__(message, error_code="LLM_ERROR", details=details)


class RateLimitError(MojiException):
    """Rate limit exceeded"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")


class NotFoundError(MojiException):
    """Resource not found"""

    def __init__(self, resource: str):
        super().__init__(f"{resource} not found", error_code="NOT_FOUND")


class RAGError(MojiException):
    """RAG service error"""

    def __init__(self, message: str, operation: Optional[str] = None):
        details = {"operation": operation} if operation else {}
        super().__init__(message, error_code="RAG_ERROR", details=details)


class VectorStoreError(MojiException):
    """Vector store error"""

    def __init__(self, message: str, store_name: Optional[str] = None):
        details = {"store_name": store_name} if store_name else {}
        super().__init__(message, error_code="VECTOR_STORE_ERROR", details=details)


class AdapterError(MojiException):
    """Platform adapter error"""

    def __init__(self, message: str, platform: Optional[str] = None):
        details = {"platform": platform} if platform else {}
        super().__init__(message, error_code="ADAPTER_ERROR", details=details)


class ConfigurationError(MojiException):
    """Configuration error"""

    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(message, error_code="CONFIG_ERROR", details=details)
