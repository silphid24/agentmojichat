"""Base LLM provider interface"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from pydantic import BaseModel, Field
from langchain.schema import BaseMessage

from app.core.logging import logger


class LLMConfig(BaseModel):
    """LLM configuration"""

    provider: str
    model: str
    api_key: str
    api_base: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1)
    timeout: int = Field(default=30, ge=1)
    retry_count: int = Field(default=3, ge=0)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """LLM response model"""

    content: str
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.name = self.__class__.__name__
        logger.info(f"Initialized LLM provider: {self.name}")

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider"""
        pass

    @abstractmethod
    async def generate(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """Generate a response from messages"""
        pass

    @abstractmethod
    async def stream(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[str]:
        """Stream response tokens"""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate provider connection"""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff"""
        import asyncio

        for attempt in range(self.config.retry_count):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.retry_count - 1:
                    raise

                wait_time = 2**attempt
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
