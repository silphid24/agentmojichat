"""Common schemas"""

from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class HealthCheck(BaseModel):
    """Health check schema"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    services: Dict[str, str] = {}


class ErrorDetail(BaseModel):
    """Error detail schema"""
    message: str
    type: str = "generic_error"
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: ErrorDetail


class SuccessResponse(BaseModel):
    """Success response schema"""
    success: bool = True
    message: str = "Operation completed successfully"


class PaginationParams(BaseModel):
    """Pagination parameters"""
    skip: int = Field(0, ge=0)
    limit: int = Field(20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Paginated response schema"""
    items: List[Any]
    total: int
    skip: int
    limit: int