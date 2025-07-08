"""LangChain wrapper for LLM router"""

from typing import List, Optional, Any, Dict, Iterator, AsyncIterator
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import (
    ChatResult,
    ChatGeneration,
    ChatGenerationChunk,
)
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)



class LangChainLLMWrapper(BaseChatModel):
    """Wrapper to make LLM router compatible with LangChain"""

    llm_router: Any = None

    def __init__(self, llm_router):
        super().__init__()
        self.llm_router = llm_router

    @property
    def _llm_type(self) -> str:
        """Return type of LLM"""
        if self.llm_router.current_provider:
            return (
                f"moji_{self.llm_router.config.provider}_{self.llm_router.config.model}"
            )
        return "moji_llm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response synchronously"""
        # This is a sync wrapper for async method
        import asyncio

        async def _async_generate():
            response = await self.llm_router.generate(messages, **kwargs)
            return response

        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(_async_generate())
        finally:
            loop.close()

        # Convert to LangChain format
        message = AIMessage(content=response.content)
        generation = ChatGeneration(
            message=message,
            generation_info={
                "model": response.model,
                "usage": response.usage,
                "metadata": response.metadata,
            },
        )

        return ChatResult(
            generations=[generation],
            llm_output={"token_usage": response.usage, "model_name": response.model},
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response asynchronously"""
        response = await self.llm_router.generate(messages, **kwargs)

        # Convert to LangChain format
        message = AIMessage(content=response.content)
        generation = ChatGeneration(
            message=message,
            generation_info={
                "model": response.model,
                "usage": response.usage,
                "metadata": response.metadata,
            },
        )

        return ChatResult(
            generations=[generation],
            llm_output={"token_usage": response.usage, "model_name": response.model},
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response synchronously"""
        # This is a sync wrapper for async method
        import asyncio

        # ChatGenerationChunk는 이미 import됨

        async def _collect_stream():
            chunks = []
            async for token in self.llm_router.stream(messages, **kwargs):
                chunks.append(token)
            return chunks

        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            chunks = loop.run_until_complete(_collect_stream())
        finally:
            loop.close()

        # Yield chunks
        for chunk in chunks:
            yield ChatGenerationChunk(
                message=AIMessage(content=chunk), generation_info={}
            )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream chat response asynchronously"""
        # ChatGenerationChunk는 이미 import됨

        async for token in self.llm_router.stream(messages, **kwargs):
            if run_manager:
                await run_manager.on_llm_new_token(token)

            yield ChatGenerationChunk(
                message=AIMessage(content=token), generation_info={}
            )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters"""
        return self.llm_router.get_current_info()

    def get_num_tokens(self, text: str) -> int:
        """Get number of tokens (approximate)"""
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get number of tokens from messages"""
        total = 0
        for msg in messages:
            total += self.get_num_tokens(msg.content)
        return total
