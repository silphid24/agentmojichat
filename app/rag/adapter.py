"""Adapter to make enhanced_rag compatible with existing RAG pipeline interface"""

from typing import Dict, Any, List, Optional
from app.rag.enhanced_rag import EnhancedRAGPipeline
from app.rag.pipeline import RAGQuery, RAGResponse
from app.core.logging import logger


class RAGPipelineAdapter:
    """Adapter to make EnhancedRAGPipeline compatible with RAGPipeline interface"""

    def __init__(self):
        self.enhanced_pipeline = EnhancedRAGPipeline()

    async def query(self, query: RAGQuery) -> RAGResponse:
        """Query the RAG system with backwards compatibility"""
        try:
            # Use enhanced pipeline's answer_with_confidence method
            result = await self.enhanced_pipeline.answer_with_confidence(
                query=query.query,
                k=query.top_k or 5,
                score_threshold=query.score_threshold
                or 1.6,  # Adjusted for ChromaDB distance
            )

            # Extract confidence and sources from result
            confidence = result.get("confidence", "MEDIUM")
            sources = []

            # Parse sources from search metadata
            if (
                "search_metadata" in result
                and "result_details" in result["search_metadata"]
            ):
                for detail in result["search_metadata"]["result_details"]:
                    sources.append(
                        {
                            "content": f"Source: {detail.get('source', 'Unknown')}",
                            "metadata": {
                                "source": detail.get("source"),
                                "score": detail.get("score"),
                                "chunk_id": detail.get("chunk_id"),
                            },
                            "score": detail.get("score", 0.0),
                        }
                    )

            # Return compatible response
            return RAGResponse(
                answer=result.get("answer", "No answer available"),
                sources=sources[: query.top_k] if sources else [],
                query=query.query,
                total_sources=len(sources),
            )

        except Exception as e:
            logger.error(f"Error in RAG query adapter: {e}")
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                query=query.query,
                total_sources=0,
            )

    async def add_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add text to the RAG system"""
        try:
            # Create a temporary document and add to enhanced pipeline
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(text)
                temp_path = f.name

            result = await self.enhanced_pipeline.load_documents([temp_path])

            # Clean up
            os.unlink(temp_path)

            return {
                "success": result.get("total_chunks", 0) > 0,
                "chunks": result.get("total_chunks", 0),
                "error": result.get("error"),
            }

        except Exception as e:
            logger.error(f"Error adding text: {e}")
            return {"success": False, "error": str(e)}

    async def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the RAG system"""
        try:
            result = await self.enhanced_pipeline.load_documents(file_paths)
            return {
                "processed": result.get("processed_files", 0),
                "total_chunks": result.get("total_chunks", 0),
                "success": result.get("processed_files", 0) > 0,
            }
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                "processed": 0,
                "total_chunks": 0,
                "success": False,
                "error": str(e),
            }

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        try:
            # Get collection stats from ChromaDB
            collection = self.enhanced_pipeline.vectorstore._collection
            count = collection.count()

            return {
                "total_documents": count,
                "collection_name": self.enhanced_pipeline.collection_name,
                "chunk_size": self.enhanced_pipeline.chunk_size,
                "chunk_overlap": self.enhanced_pipeline.chunk_overlap,
                "embedding_model": "text-embedding-3-small",
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e), "total_documents": 0}

    def clear_index(self) -> None:
        """Clear the vector index"""
        try:
            # Delete and recreate the collection
            self.enhanced_pipeline.vectorstore.delete_collection()
            self.enhanced_pipeline.vectorstore = (
                self.enhanced_pipeline._create_vectorstore()
            )
            logger.info("RAG index cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise

    async def update_chunk_size(self, chunk_size: int, chunk_overlap: int) -> None:
        """Update chunk size configuration"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.enhanced_pipeline.chunk_size = chunk_size
        self.enhanced_pipeline.chunk_overlap = chunk_overlap
        self.enhanced_pipeline.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        logger.info(
            f"Updated chunk configuration: size={chunk_size}, overlap={chunk_overlap}"
        )

    def _create_vectorstore(self):
        """Helper to create a new vectorstore instance"""
        from langchain_community.vectorstores import Chroma

        return Chroma(
            collection_name=self.enhanced_pipeline.collection_name,
            embedding_function=self.enhanced_pipeline.embeddings,
            persist_directory=str(self.enhanced_pipeline.vectordb_dir),
        )
