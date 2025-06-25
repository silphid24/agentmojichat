"""Custom exceptions"""

from typing import Optional, Dict, Any


class MojiException(Exception):
    """Base exception for MOJI application"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
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