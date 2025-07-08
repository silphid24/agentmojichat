"""Vector retriever for RAG pipeline"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

from app.core.logging import logger
from app.rag.embeddings import get_embeddings


class VectorRetriever:
    """Handles vector storage and retrieval"""

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        index_path: Optional[str] = None,
        use_local_embeddings: bool = True,
    ):
        self.embeddings = embeddings or get_embeddings(use_local=use_local_embeddings)
        self.index_path = index_path or "data/vector_index"
        self.vector_store: Optional[FAISS] = None

        # Create index directory if it doesn't exist
        Path(self.index_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized VectorRetriever with index path: {self.index_path}")

    async def create_index(self, documents: List[Document]) -> None:
        """Create vector index from documents"""
        if not documents:
            logger.warning("No documents provided for indexing")
            return

        try:
            logger.info(f"Creating index from {len(documents)} documents")

            # Create FAISS index
            self.vector_store = FAISS.from_documents(
                documents=documents, embedding=self.embeddings
            )

            # Save index
            self.save_index()

            logger.info(f"Successfully created index with {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing index"""
        if not self.vector_store:
            # Create new index if none exists
            await self.create_index(documents)
            return

        try:
            logger.info(f"Adding {len(documents)} documents to index")

            # Add to existing index
            self.vector_store.add_documents(documents)

            # Save updated index
            self.save_index()

            logger.info(f"Successfully added {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if not self.vector_store:
            logger.warning("No vector store available for search")
            return []

        try:
            logger.debug(f"Searching for: '{query}' (k={k})")

            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query, k=k, filter=filter_dict, fetch_k=fetch_k
            )

            # Filter by score threshold if provided
            if score_threshold is not None:
                results = [
                    (doc, score) for doc, score in results if score >= score_threshold
                ]

            logger.info(f"Found {len(results)} relevant documents")

            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    async def search_with_metadata(
        self, query: str, k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search and return results with metadata"""
        results = await self.search(query=query, k=k, filter_dict=metadata_filter)

        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                }
            )

        return formatted_results

    def save_index(self, path: Optional[str] = None) -> None:
        """Save vector index to disk"""
        if not self.vector_store:
            logger.warning("No vector store to save")
            return

        save_path = path or self.index_path
        try:
            self.vector_store.save_local(save_path)
            logger.info(f"Saved vector index to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise

    def load_index(self, path: Optional[str] = None) -> bool:
        """Load vector index from disk"""
        load_path = path or self.index_path
        index_file = Path(load_path) / "index.faiss"

        if not index_file.exists():
            logger.warning(f"No index found at: {load_path}")
            return False

        try:
            self.vector_store = FAISS.load_local(
                load_path, self.embeddings, allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded vector index from: {load_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def delete_index(self) -> None:
        """Delete vector index"""
        self.vector_store = None

        # Remove index files
        index_dir = Path(self.index_path)
        if index_dir.exists():
            for file in index_dir.glob("*"):
                file.unlink()
            logger.info(f"Deleted index at: {self.index_path}")

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        if not self.vector_store:
            return {"status": "no_index"}

        try:
            # FAISS specific stats
            index = self.vector_store.index
            return {
                "status": "active",
                "total_vectors": index.ntotal,
                "dimension": index.d,
                "index_type": str(type(index).__name__),
                "index_path": self.index_path,
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"status": "error", "error": str(e)}
