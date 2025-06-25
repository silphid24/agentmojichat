"""Main FastAPI application"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import logger
from app.core.middleware import RequestIDMiddleware, LoggingMiddleware
from app.core.exceptions import MojiException
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize LLM router
    from app.llm.router import llm_router
    await llm_router.initialize()
    
    # Initialize agent system
    from app.agents.manager import agent_manager
    await agent_manager.initialize_default_agents()
    
    yield
    
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(MojiException)
async def moji_exception_handler(request: Request, exc: MojiException):
    """Handle custom exceptions"""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_code,
                "details": exc.details
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error"
            }
        }
    )


# Include routers
app.include_router(api_router, prefix="/api/v1")