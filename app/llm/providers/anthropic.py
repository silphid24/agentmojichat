"""Anthropic Claude LLM provider implementation"""

from typing import List, Dict, Any, AsyncIterator
import httpx
import json
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from app.llm.base import BaseLLMProvider, LLMResponse, LLMConfig
from app.core.logging import logger
from app.core.exceptions import LLMError


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        self.client = None

    async def initialize(self) -> None:
        """Initialize the Anthropic provider"""
        self.client = httpx.AsyncClient(
            timeout=self.config.timeout, headers=self.headers
        )
        logger.info(f"Anthropic provider initialized with model: {self.config.model}")

    async def generate(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """Generate response using Anthropic Claude API"""
        if not self.client:
            await self.initialize()

        # Convert messages to Anthropic format
        formatted_messages, system_prompt = self._format_messages(messages)

        # Prepare request
        request_data = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        # Add system prompt if exists
        if system_prompt:
            request_data["system"] = system_prompt

        # Add extra parameters
        request_data.update(self.config.extra_params)

        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}

        try:
            # Make API request with retry
            response = await self._retry_with_backoff(self._make_request, request_data)

            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise LLMError(
                f"Anthropic generation failed: {str(e)}", provider="anthropic"
            )

    async def stream(self, messages: List[BaseMessage], **kwargs) -> AsyncIterator[str]:
        """Stream response from Anthropic API"""
        if not self.client:
            await self.initialize()

        formatted_messages, system_prompt = self._format_messages(messages)

        request_data = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        if system_prompt:
            request_data["system"] = system_prompt

        request_data.update(self.config.extra_params)
        request_data = {k: v for k, v in request_data.items() if v is not None}

        try:
            async with self.client.stream(
                "POST", f"{self.api_base}/messages", json=request_data
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            if chunk.get("type") == "content_block_delta":
                                delta = chunk.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise LLMError(
                f"Anthropic streaming failed: {str(e)}", provider="anthropic"
            )

    async def validate_connection(self) -> bool:
        """Validate Anthropic API connection"""
        try:
            if not self.client:
                await self.initialize()

            # Test with a simple request
            test_messages = [HumanMessage(content="Hi")]
            response = await self.generate(test_messages, max_tokens=10)

            return bool(response.content)

        except Exception as e:
            logger.error(f"Anthropic connection validation failed: {e}")
            return False

    def _format_messages(
        self, messages: List[BaseMessage]
    ) -> tuple[List[Dict[str, str]], str]:
        """Format messages for Anthropic API"""
        formatted = []
        system_prompt = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Anthropic uses a separate system parameter
                if system_prompt:
                    system_prompt += "\n\n" + msg.content
                else:
                    system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})

        # Ensure conversation starts with user message
        if formatted and formatted[0]["role"] != "user":
            formatted.insert(
                0, {"role": "user", "content": "Continue the conversation"}
            )

        # Ensure conversation alternates between user and assistant
        cleaned = []
        last_role = None
        for msg in formatted:
            if msg["role"] != last_role:
                cleaned.append(msg)
                last_role = msg["role"]
            else:
                # Merge consecutive messages from same role
                cleaned[-1]["content"] += "\n\n" + msg["content"]

        return cleaned, system_prompt

    async def _make_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to Anthropic"""
        response = await self.client.post(
            f"{self.api_base}/messages", json=request_data
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response_data: Dict[str, Any]) -> LLMResponse:
        """Parse Anthropic API response"""
        try:
            # Extract content from response
            content = ""
            for content_block in response_data.get("content", []):
                if content_block.get("type") == "text":
                    content += content_block.get("text", "")

            # Extract usage information
            usage = response_data.get("usage", {})

            return LLMResponse(
                content=content,
                model=response_data.get("model", self.config.model),
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                },
                metadata={
                    "id": response_data.get("id"),
                    "type": response_data.get("type"),
                    "role": response_data.get("role"),
                    "stop_reason": response_data.get("stop_reason"),
                },
            )
        except (KeyError, IndexError) as e:
            raise LLMError(
                f"Invalid Anthropic response format: {e}", provider="anthropic"
            )

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry with exponential backoff"""
        import asyncio

        for attempt in range(self.config.retry_count):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = min(2**attempt, 60)
                    logger.warning(f"Rate limited, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                if attempt == self.config.retry_count - 1:
                    raise
                wait_time = min(2**attempt, 30)
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
