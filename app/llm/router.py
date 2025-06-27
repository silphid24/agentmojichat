"""LLM Router for dynamic model selection"""

from typing import Dict, Any, List, Optional, Type
from langchain.schema import BaseMessage
from langchain.chat_models.base import BaseChatModel

from app.llm.base import BaseLLMProvider, LLMConfig, LLMResponse
from app.llm.providers.deepseek import DeepSeekProvider
from app.llm.providers.openai import OpenAIProvider
from app.llm.providers.custom import CustomProvider
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import LLMError


class LLMRouter:
    """Routes requests to appropriate LLM providers"""
    
    # Provider registry
    PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
        "deepseek": DeepSeekProvider,
        "openai": OpenAIProvider,
        "custom": CustomProvider
    }
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.current_provider: Optional[BaseLLMProvider] = None
        self.config: Optional[LLMConfig] = None
        logger.info("Initialized LLM Router")
    
    async def initialize(self, config: Optional[LLMConfig] = None) -> None:
        """Initialize router with configuration"""
        if config:
            self.config = config
        else:
            # Load from environment variables
            self.config = LLMConfig(
                provider=settings.llm_provider,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                api_base=settings.llm_api_base,
                temperature=0.7,
                max_tokens=1024,
                timeout=30,
                retry_count=3
            )
        
        # Initialize the provider
        await self._initialize_provider(self.config.provider)
        logger.info(
            f"LLM Router initialized with provider: {self.config.provider}, "
            f"model: {self.config.model}"
        )
    
    async def _initialize_provider(self, provider_name: str) -> None:
        """Initialize a specific provider"""
        if provider_name not in self.PROVIDERS:
            raise LLMError(
                f"Unknown provider: {provider_name}",
                provider=provider_name
            )
        
        # Create provider instance if not exists
        if provider_name not in self.providers:
            provider_class = self.PROVIDERS[provider_name]
            provider = provider_class(self.config)
            await provider.initialize()
            self.providers[provider_name] = provider
        
        self.current_provider = self.providers[provider_name]
    
    async def generate(
        self,
        messages: List[BaseMessage],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using specified or current provider"""
        # Use specified provider or current
        if provider and provider != self.config.provider:
            await self._switch_provider(provider, model)
        
        if not self.current_provider:
            raise LLMError(
                "No LLM provider initialized",
                provider=self.config.provider if self.config else None
            )
        
        # Override model if specified
        if model and model != self.config.model:
            old_model = self.config.model
            self.config.model = model
            try:
                response = await self.current_provider.generate(messages, **kwargs)
            finally:
                self.config.model = old_model
            return response
        
        return await self.current_provider.generate(messages, **kwargs)
    
    async def stream(
        self,
        messages: List[BaseMessage],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """Stream response using specified or current provider"""
        # Use specified provider or current
        if provider and provider != self.config.provider:
            await self._switch_provider(provider, model)
        
        if not self.current_provider:
            raise LLMError(
                "No LLM provider initialized",
                provider=self.config.provider if self.config else None
            )
        
        # Override model if specified
        if model and model != self.config.model:
            old_model = self.config.model
            self.config.model = model
            try:
                async for token in self.current_provider.stream(messages, **kwargs):
                    yield token
            finally:
                self.config.model = old_model
        else:
            async for token in self.current_provider.stream(messages, **kwargs):
                yield token
    
    async def _switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None
    ) -> None:
        """Switch to a different provider"""
        if provider_name not in self.PROVIDERS:
            raise LLMError(
                f"Unknown provider: {provider_name}",
                provider=provider_name
            )
        
        # Update config
        self.config.provider = provider_name
        if model:
            self.config.model = model
        
        # Initialize new provider
        await self._initialize_provider(provider_name)
        logger.info(f"Switched to provider: {provider_name}")
    
    async def validate_all_providers(self) -> Dict[str, bool]:
        """Validate connections for all configured providers"""
        results = {}
        
        for provider_name in self.PROVIDERS:
            try:
                # Skip if no API key configured for this provider
                if provider_name == "openai" and not settings.llm_api_key:
                    results[provider_name] = False
                    continue
                
                # Create temporary config
                temp_config = LLMConfig(
                    provider=provider_name,
                    model=self._get_default_model(provider_name),
                    api_key=settings.llm_api_key,
                    api_base=settings.llm_api_base if provider_name == "custom" else None
                )
                
                # Create and test provider
                provider_class = self.PROVIDERS[provider_name]
                provider = provider_class(temp_config)
                await provider.initialize()
                
                results[provider_name] = await provider.validate_connection()
                
            except Exception as e:
                logger.error(f"Failed to validate {provider_name}: {e}")
                results[provider_name] = False
        
        return results
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider"""
        defaults = {
            "deepseek": "deepseek-r1",
            "openai": "gpt-3.5-turbo",
            "custom": "llama-3"
        }
        return defaults.get(provider, "unknown")
    
    def get_current_info(self) -> Dict[str, Any]:
        """Get information about current provider and model"""
        if not self.current_provider:
            return {"status": "not_initialized"}
        
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "available_providers": list(self.PROVIDERS.keys())
        }
    
    async def get_langchain_model(self) -> BaseChatModel:
        """Get LangChain-compatible chat model"""
        if not self.current_provider:
            await self.initialize()
        
        # Create LangChain wrapper
        from app.llm.langchain_wrapper import LangChainLLMWrapper
        return LangChainLLMWrapper(self)
    
    async def cleanup(self) -> None:
        """Cleanup all providers"""
        for provider in self.providers.values():
            if hasattr(provider, '__aexit__'):
                await provider.__aexit__(None, None, None)
        self.providers.clear()
        self.current_provider = None
        logger.info("LLM Router cleaned up")


# Global router instance
llm_router = LLMRouter()