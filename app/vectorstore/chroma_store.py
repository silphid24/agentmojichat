"""Chroma DB vector store implementation"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from app.vectorstore.base import BaseVectorStore, VectorStoreConfig, SearchResult
from app.rag.embeddings import get_embeddings
from app.core.logging import logger


class ChromaVectorStore(BaseVectorStore):
    """Chroma DB implementation of vector store"""

    def __init__(
        self,
        config: VectorStoreConfig,
        embeddings=None,
        client_settings: Optional[Settings] = None,
    ):
        super().__init__(config)
        self.embeddings = embeddings or get_embeddings(use_local=True)

        # Default Chroma settings with comprehensive telemetry disabling
        self.client_settings = client_settings or Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=config.persist_directory,
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True if config.persist_directory else False,
        )

        self.client = None
        self.collection = None
        self.langchain_chroma = None

        # Ensure persist directory exists
        if config.persist_directory:
            Path(config.persist_directory).mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize Chroma client and collection"""
        try:
            # Initialize Chroma client
            self.client = chromadb.Client(self.client_settings)

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "distance_metric": self.config.distance_metric,
                    "index_params": self.config.index_params,
                },
            )

            # Create LangChain Chroma wrapper
            self.langchain_chroma = Chroma(
                client=self.client,
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.config.persist_directory,
            )

            self.is_initialized = True
            logger.info(f"Initialized Chroma store: {self.config.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Chroma store: {e}")
            raise

    async def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to Chroma"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # If no IDs provided, generate them
            if not ids:
                ids = [
                    f"doc_{i}_{hash(doc.page_content)}"
                    for i, doc in enumerate(documents)
                ]

            # Add documents using LangChain wrapper
            result_ids = self.langchain_chroma.add_documents(
                documents=documents, ids=ids
            )

            # Persist changes
            await self.persist()

            logger.info(f"Added {len(documents)} documents to Chroma")
            return result_ids

        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            raise

    async def search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[SearchResult]:
        """Search for similar documents"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Use LangChain similarity search with score
            results = self.langchain_chroma.similarity_search_with_score(
                query=query, k=k, filter=filter
            )

            # Convert to SearchResult objects
            search_results = []
            for doc, score in results:
                search_results.append(
                    SearchResult(
                        document=doc,
                        score=1 - score,  # Convert distance to similarity
                        metadata=doc.metadata,
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Error searching in Chroma: {e}")
            return []

    async def search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """Search using a vector embedding"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Query collection directly
            results = self.collection.query(
                query_embeddings=[embedding], n_results=k, where=filter
            )

            # Convert results
            search_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    doc = Document(
                        page_content=results["documents"][0][i],
                        metadata=(
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        ),
                    )
                    score = (
                        1 - results["distances"][0][i] if results["distances"] else 0
                    )

                    search_results.append(
                        SearchResult(document=doc, score=score, metadata=doc.metadata)
                    )

            return search_results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    async def delete(
        self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Delete documents from Chroma"""
        if not self.is_initialized:
            await self.initialize()

        try:
            if ids:
                self.collection.delete(ids=ids)
            elif filter:
                self.collection.delete(where=filter)
            else:
                logger.warning("No ids or filter provided for deletion")
                return False

            await self.persist()
            logger.info("Deleted documents from Chroma")
            return True

        except Exception as e:
            logger.error(f"Error deleting from Chroma: {e}")
            return False

    async def update_document(self, document_id: str, document: Document) -> bool:
        """Update a document in Chroma"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Delete old document
            await self.delete(ids=[document_id])

            # Add new document with same ID
            await self.add_documents([document], ids=[document_id])

            return True

        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get Chroma collection statistics"""
        if not self.is_initialized:
            await self.initialize()

        try:
            count = self.collection.count()

            # Get collection metadata
            metadata = self.collection.metadata or {}

            return {
                "store_type": "chroma",
                "collection_name": self.config.collection_name,
                "document_count": count,
                "persist_directory": self.config.persist_directory,
                "distance_metric": metadata.get(
                    "distance_metric", self.config.distance_metric
                ),
                "is_persistent": bool(self.config.persist_directory),
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    async def clear(self) -> bool:
        """Clear all documents from collection"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Delete the collection
            self.client.delete_collection(self.config.collection_name)

            # Recreate it
            await self.initialize()

            logger.info(f"Cleared Chroma collection: {self.config.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    async def persist(self) -> None:
        """Persist the store to disk"""
        if self.langchain_chroma and self.config.persist_directory:
            try:
                self.langchain_chroma.persist()
                logger.debug("Persisted Chroma store to disk")
            except Exception as e:
                logger.error(f"Error persisting Chroma: {e}")

    async def hybrid_search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # Vector search
            vector_results = await self.search(query, k * 2, filter)

            # Keyword search using Chroma's built-in text search
            # For MVP, we'll use simple text matching in metadata
            keyword_results = []

            if filter and "text_match" in filter:
                text_query = filter.pop("text_match")
                all_docs = self.collection.get(where=filter)

                for i, content in enumerate(all_docs["documents"] or []):
                    if text_query.lower() in content.lower():
                        doc = Document(
                            page_content=content,
                            metadata=(
                                all_docs["metadatas"][i]
                                if all_docs["metadatas"]
                                else {}
                            ),
                        )
                        keyword_results.append(
                            SearchResult(
                                document=doc,
                                score=0.5,  # Fixed score for keyword matches
                                metadata=doc.metadata,
                            )
                        )

            # Combine results with alpha weighting
            combined_results = {}

            # Add vector results
            for result in vector_results:
                doc_id = result.document.metadata.get(
                    "chunk_id", hash(result.document.page_content)
                )
                combined_results[doc_id] = {
                    "result": result,
                    "score": result.score * alpha,
                }

            # Add keyword results
            for result in keyword_results:
                doc_id = result.document.metadata.get(
                    "chunk_id", hash(result.document.page_content)
                )
                if doc_id in combined_results:
                    combined_results[doc_id]["score"] += result.score * (1 - alpha)
                else:
                    combined_results[doc_id] = {
                        "result": result,
                        "score": result.score * (1 - alpha),
                    }

            # Sort by combined score
            sorted_results = sorted(
                combined_results.values(), key=lambda x: x["score"], reverse=True
            )

            # Return top k results
            final_results = []
            for item in sorted_results[:k]:
                result = item["result"]
                result.score = item["score"]
                final_results.append(result)

            return final_results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return await self.search(query, k, filter)  # Fallback to vector search
