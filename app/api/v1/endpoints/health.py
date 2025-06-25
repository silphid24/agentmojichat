"""Health check endpoints"""

from fastapi import APIRouter
from app.models.common import HealthResponse
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(version=settings.app_version)