"""Embeddings management for RAG"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np

from langchain.embeddings.base import Embeddings
from langchain.schema import Document

from app.core.logging import logger
from app.core.config import settings
from app.llm.router import llm_router


class BaseEmbeddings(ABC):
    """Base class for embeddings"""
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass


class LLMEmbeddings(BaseEmbeddings):
    """Embeddings using LLM provider"""
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or settings.llm_provider
        self.model = model or self._get_embedding_model()
        logger.info(f"Initialized LLM embeddings: {self.provider}/{self.model}")
    
    def _get_embedding_model(self) -> str:
        """Get appropriate embedding model for provider"""
        embedding_models = {
            "openai": "text-embedding-3-small",  # OpenAI's small embedding model
            "deepseek": "deepseek-chat",  # DeepSeek uses same model for embeddings
            "custom": "all-MiniLM-L6-v2"  # Default for local models
        }
        return embedding_models.get(self.provider, "text-embedding-3-small")
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = []
        
        for text in texts:
            embedding = await self.embed_query(text)
            embeddings.append(embedding)
        
        return embeddings
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        # For MVP, return mock embeddings
        # TODO: Implement actual embedding API calls
        logger.debug(f"Generating embedding for text of length: {len(text)}")
        
        # Mock 384-dimensional embedding
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(384).tolist()
        
        return embedding


class LocalEmbeddings(BaseEmbeddings):
    """Local embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized local embeddings model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        if not self.model:
            # Fallback to mock embeddings
            return [await self.embed_query(text) for text in texts]
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not self.model:
            # Fallback to mock embeddings
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(384).tolist()
        
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


class LangChainEmbeddingsWrapper(Embeddings):
    """Wrapper to make our embeddings compatible with LangChain"""
    
    def __init__(self, embeddings: BaseEmbeddings):
        self.embeddings = embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Sync wrapper for async embed_documents"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.embeddings.embed_documents(texts))
        finally:
            loop.close()
    
    def embed_query(self, text: str) -> List[float]:
        """Sync wrapper for async embed_query"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.embeddings.embed_query(text))
        finally:
            loop.close()
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents"""
        return await self.embeddings.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query"""
        return await self.embeddings.embed_query(text)


def get_embeddings(
    use_local: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Embeddings:
    """Get appropriate embeddings instance"""
    if use_local:
        base_embeddings = LocalEmbeddings(model or "all-MiniLM-L6-v2")
    else:
        base_embeddings = LLMEmbeddings(provider, model)
    
    return LangChainEmbeddingsWrapper(base_embeddings)