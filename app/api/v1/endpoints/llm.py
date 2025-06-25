"""LLM management endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.schemas.auth import UserInDB
from app.api.v1.endpoints.auth import get_current_user
from app.llm.router import llm_router
from app.llm.base import LLMConfig
from app.core.logging import logger

router = APIRouter()


class LLMSwitchRequest(BaseModel):
    """Request to switch LLM provider/model"""
    provider: str = Field(..., description="LLM provider (deepseek, openai, custom)")
    model: Optional[str] = Field(None, description="Model name")
    api_key: Optional[str] = Field(None, description="API key for provider")
    api_base: Optional[str] = Field(None, description="API base URL (for custom provider)")


class LLMTestRequest(BaseModel):
    """Request to test LLM"""
    message: str = Field(default="Hello, can you hear me?", description="Test message")
    provider: Optional[str] = Field(None, description="Provider to test")
    model: Optional[str] = Field(None, description="Model to test")


@router.get("/info")
async def get_llm_info(
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Get current LLM configuration info"""
    return llm_router.get_current_info()


@router.post("/switch")
async def switch_llm(
    request: LLMSwitchRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Switch to a different LLM provider/model"""
    try:
        # Create new config
        config = LLMConfig(
            provider=request.provider,
            model=request.model or llm_router._get_default_model(request.provider),
            api_key=request.api_key or llm_router.config.api_key,
            api_base=request.api_base or llm_router.config.api_base,
            temperature=llm_router.config.temperature if llm_router.config else 0.7,
            max_tokens=llm_router.config.max_tokens if llm_router.config else 1024
        )
        
        # Reinitialize router with new config
        await llm_router.initialize(config)
        
        return {
            "success": True,
            "message": f"Switched to {request.provider} ({config.model})",
            "config": llm_router.get_current_info()
        }
        
    except Exception as e:
        logger.error(f"Failed to switch LLM: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to switch LLM: {str(e)}"
        )


@router.post("/test")
async def test_llm(
    request: LLMTestRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Test LLM with a simple message"""
    try:
        from langchain.schema import HumanMessage
        
        # Test the LLM
        messages = [HumanMessage(content=request.message)]
        response = await llm_router.generate(
            messages,
            provider=request.provider,
            model=request.model,
            max_tokens=100
        )
        
        return {
            "success": True,
            "request": request.message,
            "response": response.content,
            "model": response.model,
            "usage": response.usage
        }
        
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM test failed: {str(e)}"
        )


@router.get("/validate")
async def validate_providers(
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Validate all configured LLM providers"""
    try:
        results = await llm_router.validate_all_providers()
        
        return {
            "current_provider": llm_router.config.provider if llm_router.config else None,
            "validation_results": results
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )


@router.get("/models/{provider}")
async def get_provider_models(
    provider: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Get available models for a provider"""
    # Predefined model lists
    models = {
        "deepseek": [
            {"id": "deepseek-r1", "name": "DeepSeek R1", "description": "Latest reasoning model"},
            {"id": "deepseek-chat", "name": "DeepSeek Chat", "description": "Chat model"},
            {"id": "deepseek-coder", "name": "DeepSeek Coder", "description": "Code generation model"}
        ],
        "openai": [
            {"id": "gpt-4", "name": "GPT-4", "description": "Most capable model"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "Fast and efficient"},
            {"id": "gpt-4-turbo-preview", "name": "GPT-4 Turbo", "description": "Latest GPT-4 with vision"}
        ],
        "custom": [
            {"id": "llama-3", "name": "Llama 3", "description": "Open source model"},
            {"id": "mistral", "name": "Mistral", "description": "Efficient open model"},
            {"id": "custom", "name": "Custom Model", "description": "User-defined model"}
        ]
    }
    
    if provider not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown provider: {provider}"
        )
    
    return {
        "provider": provider,
        "models": models[provider]
    }