"""DeepSeek LLM provider implementation"""

from typing import List, Dict, Any, AsyncIterator
import httpx
import json
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.llm.base import BaseLLMProvider, LLMResponse, LLMConfig
from app.core.logging import logger
from app.core.exceptions import LLMError


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize the DeepSeek provider"""
        self.client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers=self.headers
        )
        logger.info(f"DeepSeek provider initialized with model: {self.config.model}")
    
    async def generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> LLMResponse:
        """Generate response using DeepSeek API"""
        if not self.client:
            await self.initialize()
        
        # Convert messages to DeepSeek format
        formatted_messages = self._format_messages(messages)
        
        # Prepare request
        request_data = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False
        }
        
        # Add extra parameters if provided
        request_data.update(self.config.extra_params)
        request_data.update(kwargs)
        
        try:
            # Make API request with retry
            response = await self._retry_with_backoff(
                self._make_request,
                request_data
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise LLMError(f"DeepSeek generation failed: {str(e)}", provider="deepseek")
    
    async def stream(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response from DeepSeek API"""
        if not self.client:
            await self.initialize()
        
        formatted_messages = self._format_messages(messages)
        
        request_data = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True
        }
        
        request_data.update(self.config.extra_params)
        request_data.update(kwargs)
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                json=request_data
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"DeepSeek streaming error: {e}")
            raise LLMError(f"DeepSeek streaming failed: {str(e)}", provider="deepseek")
    
    async def validate_connection(self) -> bool:
        """Validate DeepSeek API connection"""
        try:
            if not self.client:
                await self.initialize()
            
            # Test with a simple request
            test_messages = [HumanMessage(content="Hello")]
            response = await self.generate(test_messages, max_tokens=10)
            
            return bool(response.content)
            
        except Exception as e:
            logger.error(f"DeepSeek connection validation failed: {e}")
            return False
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Format messages for DeepSeek API"""
        formatted = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
        
        return formatted
    
    async def _make_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to DeepSeek"""
        response = await self.client.post(
            f"{self.api_base}/chat/completions",
            json=request_data
        )
        response.raise_for_status()
        return response.json()
    
    def _parse_response(self, response_data: Dict[str, Any]) -> LLMResponse:
        """Parse DeepSeek API response"""
        try:
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage", {})
            
            return LLMResponse(
                content=content,
                model=response_data.get("model", self.config.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                },
                metadata={
                    "id": response_data.get("id"),
                    "created": response_data.get("created")
                }
            )
        except KeyError as e:
            raise LLMError(f"Invalid DeepSeek response format: {e}", provider="deepseek")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()