"""Main FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import ValidationError as PydanticValidationError

from app.core.config import settings
from app.core.logging import logger
from app.core.middleware import RequestIDMiddleware, LoggingMiddleware
from app.core.exceptions import MojiException
from app.core.error_handlers import (
    moji_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # RAG 시스템 사전 초기화 (첫 질문 응답 지연 방지)
    try:
        logger.info("Pre-initializing RAG system...")
        from app.rag.enhanced_rag import rag_pipeline
        
        # 벡터스토어 사전 로드 시도 (안전하게)
        try:
            rag_pipeline._initialize_vectorstore()
            logger.info("RAG vectorstore pre-loaded successfully")
        except Exception as init_error:
            logger.info(f"RAG vectorstore will be initialized on first use: {init_error}")
            
    except Exception as e:
        logger.warning(f"RAG pre-initialization failed (will lazy load): {e}")

    # Note: LLM router and agent system will be initialized on first use (lazy loading)
    logger.info("Core components will be initialized on first use for faster startup")

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
app.exception_handler(MojiException)(moji_exception_handler)
app.exception_handler(PydanticValidationError)(validation_exception_handler)
app.exception_handler(Exception)(general_exception_handler)


# Include routers
app.include_router(api_router, prefix="/api/v1")


# Root route - redirect to webchat
@app.get("/")
async def root():
    """Redirect to webchat interface"""
    return RedirectResponse(url="/webchat")


# WebChat interface route
@app.get("/webchat")
async def webchat():
    """Serve the WebChat interface"""
    return FileResponse("app/static/moji-webchat-v2-modular.html")


# Chat route (alias for webchat)
@app.get("/chat")
async def chat():
    """Alias for webchat interface"""
    return RedirectResponse(url="/webchat")


# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
