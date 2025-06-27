"""Error handling utilities and decorators"""

from functools import wraps
from typing import Dict, Type, Tuple, Callable, Any
import traceback

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.core.exceptions import (
    MojiException, 
    AuthenticationError, 
    AuthorizationError,
    ValidationError,
    LLMError,
    RateLimitError,
    NotFoundError,
    RAGError,
    VectorStoreError,
    AdapterError,
    ConfigurationError
)
from app.core.logging import logger


# HTTP status code mapping for exception types
STATUS_CODE_MAP = {
    "AUTH_FAILED": 401,
    "AUTH_FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "VALIDATION_ERROR": 422,
    "RATE_LIMIT_EXCEEDED": 429,
    "LLM_ERROR": 503,
    "RAG_ERROR": 500,
    "VECTOR_STORE_ERROR": 500,
    "ADAPTER_ERROR": 502,
    "CONFIG_ERROR": 500,
}


def handle_errors(
    error_mappings: Dict[Type[Exception], Type[MojiException]] = None
):
    """
    Decorator for standardized error handling
    
    Args:
        error_mappings: Optional mapping of exceptions to MojiException types
        
    Example:
        @handle_errors({
            ValueError: ValidationError,
            KeyError: NotFoundError
        })
        async def my_endpoint():
            ...
    """
    if error_mappings is None:
        error_mappings = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except MojiException:
                # Re-raise custom exceptions
                raise
            except PydanticValidationError as e:
                # Handle Pydantic validation errors
                errors = e.errors()
                if errors:
                    field = errors[0].get("loc", ["unknown"])[0]
                    message = errors[0].get("msg", "Validation error")
                    raise ValidationError(message=message, field=str(field))
                raise ValidationError(message="Validation error")
            except Exception as e:
                # Check error mappings
                for exc_type, moji_exc_type in error_mappings.items():
                    if isinstance(e, exc_type):
                        if moji_exc_type == ValidationError:
                            raise moji_exc_type(message=str(e))
                        elif moji_exc_type == NotFoundError:
                            raise moji_exc_type(resource=str(e))
                        else:
                            raise moji_exc_type(message=str(e))
                
                # Log unexpected errors
                logger.error(
                    f"Unexpected error in {func.__name__}",
                    exc_info=True,
                    extra={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc()
                    }
                )
                
                # Raise generic error
                raise MojiException(
                    message="Internal server error",
                    error_code="INTERNAL_ERROR",
                    details={"original_error": type(e).__name__}
                )
        
        return wrapper
    return decorator


async def moji_exception_handler(request: Request, exc: MojiException) -> JSONResponse:
    """
    Global exception handler for MojiException
    
    This should be registered in main.py:
    app.exception_handler(MojiException)(moji_exception_handler)
    """
    status_code = STATUS_CODE_MAP.get(exc.error_code, 400)
    
    # Log the error
    logger.error(
        f"MojiException: {exc.error_code}",
        extra={
            "error_code": exc.error_code,
            "error_message": exc.message,
            "error_details": exc.details,
            "request_path": request.url.path,
            "request_method": request.method,
            "request_id": getattr(request.state, "request_id", None)
        }
    )
    
    # Build response
    response_content = {
        "error": {
            "message": exc.message,
            "type": exc.error_code,
            "details": exc.details
        }
    }
    
    # Add request ID if available
    if hasattr(request.state, "request_id"):
        response_content["request_id"] = request.state.request_id
    
    # Add special headers for authentication errors
    headers = {}
    if exc.error_code == "AUTH_FAILED":
        headers["WWW-Authenticate"] = "Bearer"
    
    return JSONResponse(
        status_code=status_code,
        content=response_content,
        headers=headers
    )


async def validation_exception_handler(
    request: Request, 
    exc: PydanticValidationError
) -> JSONResponse:
    """
    Global handler for Pydantic validation errors
    
    This should be registered in main.py:
    app.exception_handler(PydanticValidationError)(validation_exception_handler)
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error.get("loc", [])),
            "message": error.get("msg", "Validation error"),
            "type": error.get("type", "validation_error")
        })
    
    logger.warning(
        "Validation error",
        extra={
            "errors": errors,
            "request_path": request.url.path,
            "request_method": request.method
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Validation error",
                "type": "VALIDATION_ERROR",
                "details": {"errors": errors}
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global handler for unexpected exceptions
    
    This should be registered in main.py:
    app.exception_handler(Exception)(general_exception_handler)
    """
    logger.error(
        f"Unexpected error: {type(exc).__name__}",
        exc_info=True,
        extra={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "request_path": request.url.path,
            "request_method": request.method,
            "request_id": getattr(request.state, "request_id", None)
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "INTERNAL_ERROR",
                "details": {}
            }
        }
    )