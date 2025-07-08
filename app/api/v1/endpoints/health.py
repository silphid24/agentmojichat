"""Health check endpoints"""

from fastapi import APIRouter
from app.schemas.common import HealthCheck
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    services = {}

    # Check database connection
    try:
        # TODO: Implement actual DB check
        services["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        services["database"] = "unhealthy"

    # Check Redis connection
    try:
        # TODO: Implement actual Redis check
        services["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        services["redis"] = "unhealthy"

    return HealthCheck(version=settings.app_version, services=services)
