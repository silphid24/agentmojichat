"""Base vector store interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from pydantic import BaseModel, Field


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""

    collection_name: str = "moji_vectors"
    persist_directory: Optional[str] = "data/chroma"
    distance_metric: str = "cosine"  # cosine, euclidean, dot
    index_params: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Search result model"""

    document: Document
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store"""
        pass

    @abstractmethod
    async def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the store"""
        pass

    @abstractmethod
    async def search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[SearchResult]:
        """Search for similar documents"""
        pass

    @abstractmethod
    async def search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search using a vector"""
        pass

    @abstractmethod
    async def delete(
        self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Delete documents"""
        pass

    @abstractmethod
    async def update_document(self, document_id: str, document: Document) -> bool:
        """Update a document"""
        pass

    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all documents"""
        pass

    @abstractmethod
    async def persist(self) -> None:
        """Persist the store to disk"""
        pass

    async def hybrid_search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform hybrid search (vector + keyword)"""
        # Default implementation uses vector search only
        return await self.search(query, k, filter)

    async def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search with relevance score filtering"""
        results = await self.search(query, k * 2, filter)  # Get more results

        # Filter by threshold
        filtered_results = [
            (r.document, r.score) for r in results if r.score >= score_threshold
        ]

        # Return top k
        return filtered_results[:k]
