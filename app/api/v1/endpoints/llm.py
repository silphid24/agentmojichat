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


@router.get("/info/public")
async def get_llm_info_public():
    """Get current LLM configuration info (public, no auth required)"""
    return llm_router.get_current_info()


@router.post("/switch")
async def switch_llm(
    request: LLMSwitchRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Switch to a different LLM provider/model"""
    try:
        # Determine API key and base URL based on provider
        api_key = request.api_key
        api_base = request.api_base
        
        if not api_key:
            if request.provider == "anthropic":
                from app.core.config import settings
                api_key = settings.anthropic_api_key
            elif request.provider in ["deepseek-local", "exaone-local"]:
                api_key = "not-needed"
                if request.provider == "deepseek-local":
                    from app.core.config import settings
                    api_base = api_base or settings.deepseek_local_url
                else:
                    from app.core.config import settings
                    api_base = api_base or settings.exaone_local_url
            else:
                api_key = llm_router.config.api_key if llm_router.config else ""
        
        if not api_base and llm_router.config:
            api_base = llm_router.config.api_base
        
        # Create new config
        config = LLMConfig(
            provider=request.provider,
            model=request.model or llm_router._get_default_model(request.provider),
            api_key=api_key,
            api_base=api_base,
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
            {"id": "gpt-4", "name": "GPT-4", "description": "OpenAI의 가장 강력한 모델"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "빠르고 효율적인 GPT-4"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "빠르고 경제적"}
        ],
        "anthropic": [
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "description": "Anthropic의 최신 모델, 뛰어난 추론 능력"},
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "description": "가장 강력한 Claude 모델"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "description": "빠르고 효율적"}
        ],
        "deepseek-local": [
            {"id": "deepseek-r1:7b", "name": "DeepSeek R1 7B (Local)", "description": "로컬 워크스테이션에서 실행되는 추론 모델"}
        ],
        "exaone-local": [
            {"id": "exaone:latest", "name": "EXAONE (Local)", "description": "LG AI Research의 한국어 특화 모델"}
        ],
        "custom": [
            {"id": "your-model-name", "name": "Workstation LLM", "description": "워크스테이션 LLM 서버 (192.168.0.7:5000)"},
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


@router.get("/models/{provider}/public")
async def get_provider_models_public(provider: str):
    """Get available models for a provider (public, no auth required)"""
    # Predefined model lists
    models = {
        "deepseek": [
            {"id": "deepseek-r1", "name": "DeepSeek R1", "description": "Latest reasoning model"},
            {"id": "deepseek-chat", "name": "DeepSeek Chat", "description": "Chat model"},
            {"id": "deepseek-coder", "name": "DeepSeek Coder", "description": "Code generation model"}
        ],
        "openai": [
            {"id": "gpt-4", "name": "GPT-4", "description": "OpenAI의 가장 강력한 모델"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "빠르고 효율적인 GPT-4"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "빠르고 경제적"}
        ],
        "anthropic": [
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "description": "Anthropic의 최신 모델, 뛰어난 추론 능력"},
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "description": "가장 강력한 Claude 모델"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "description": "빠르고 효율적"}
        ],
        "deepseek-local": [
            {"id": "deepseek-r1:7b", "name": "DeepSeek R1 7B (Local)", "description": "로컬 워크스테이션에서 실행되는 추론 모델"}
        ],
        "exaone-local": [
            {"id": "exaone:latest", "name": "EXAONE (Local)", "description": "LG AI Research의 한국어 특화 모델"}
        ],
        "custom": [
            {"id": "your-model-name", "name": "Workstation LLM", "description": "워크스테이션 LLM 서버 (192.168.0.7:5000)"},
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