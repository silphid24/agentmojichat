"""Enhanced RAG pipeline with multi-store support

DEPRECATED: This module is deprecated in favor of enhanced_rag.py

Multi-store functionality will be migrated to enhanced_rag.py in a future update.

Please use:
- app.rag.enhanced_rag.EnhancedRAGPipeline for direct usage
- app.rag.adapter.RAGPipelineAdapter for API compatibility

See app/rag/MIGRATION_GUIDE.md for migration instructions.
"""

import warnings

warnings.warn(
    "app.rag.enhanced_pipeline is deprecated. Use app.rag.enhanced_rag instead.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import List, Dict, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.rag.document_processor import DocumentProcessor
from app.vectorstore.manager import vector_store_manager, VectorStoreType
from app.vectorstore.base import VectorStoreConfig
from app.llm.router import llm_router


class EnhancedRAGQuery(BaseModel):
    """Enhanced RAG query with vector store selection"""

    query: str
    k: int = Field(default=3, ge=1, le=10)
    score_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    include_sources: bool = True
    store_id: Optional[str] = None  # None uses default store
    use_all_stores: bool = False  # Search across all stores
    use_hybrid: bool = False  # Use hybrid search if available
    hybrid_alpha: float = Field(default=0.5, ge=0, le=1)


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with vector store management"""

    def __init__(
        self,
        document_processor: Optional[DocumentProcessor] = None,
        default_store_type: VectorStoreType = VectorStoreType.CHROMA,
    ):
        self.document_processor = document_processor or DocumentProcessor()
        self.default_store_type = default_store_type

        # Enhanced RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant. Use the following context to answer the user's question.
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        )

        # Store-aware prompt for multi-store queries
        self.multi_store_prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant. Use the following context from multiple sources to answer the user's question.
The context comes from different knowledge stores as indicated.

{multi_context}

Question: {question}

Answer:"""
        )

        logger.info("Initialized enhanced RAG pipeline")

    async def ensure_default_store(self) -> None:
        """Ensure a default vector store exists"""
        if not vector_store_manager.default_store:
            config = VectorStoreConfig(
                collection_name="default_rag", persist_directory="data/default_vectors"
            )
            await vector_store_manager.create_store(
                store_id="default_rag",
                store_type=self.default_store_type,
                config=config,
                set_as_default=True,
            )

    async def add_documents(
        self,
        file_paths: List[str],
        store_id: Optional[str] = None,
        add_to_all: bool = False,
    ) -> Dict[str, Any]:
        """Add documents to vector store(s)"""
        await self.ensure_default_store()

        processed_docs = []
        failed_files = []

        # Process each file
        for file_path in file_paths:
            try:
                docs = await self.document_processor.process_file(file_path)
                if isinstance(docs, list):
                    processed_docs.extend(docs)
                else:
                    processed_docs.append(docs)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed_files.append({"file": file_path, "error": str(e)})

        # Extract all chunks
        all_chunks = []
        for doc in processed_docs:
            if hasattr(doc, "chunks"):
                all_chunks.extend(doc.chunks)
            else:
                all_chunks.append(doc)

        # Add to vector store(s)
        results = {}
        if all_chunks:
            if add_to_all:
                # Add to all stores
                results = await vector_store_manager.add_documents_to_all(all_chunks)
            else:
                # Add to specific store or default
                store = vector_store_manager.get_store(store_id)
                ids = await store.add_documents(all_chunks)
                results[store_id or vector_store_manager.default_store] = ids

        return {
            "processed": len(processed_docs),
            "failed": len(failed_files),
            "total_chunks": len(all_chunks),
            "failed_files": failed_files,
            "store_results": results,
        }

    async def query(self, rag_query: EnhancedRAGQuery) -> Dict[str, Any]:
        """Query the RAG system with enhanced features"""
        await self.ensure_default_store()

        try:
            if rag_query.use_all_stores:
                return await self._query_all_stores(rag_query)
            else:
                return await self._query_single_store(rag_query)

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "query": rag_query.query,
                "total_sources": 0,
                "error": str(e),
            }

    async def _query_single_store(self, rag_query: EnhancedRAGQuery) -> Dict[str, Any]:
        """Query a single vector store"""
        store = vector_store_manager.get_store(rag_query.store_id)

        # Perform search
        if rag_query.use_hybrid and hasattr(store, "hybrid_search"):
            search_results = await store.hybrid_search(
                query=rag_query.query,
                k=rag_query.k,
                alpha=rag_query.hybrid_alpha,
                filter=(
                    {"score_threshold": rag_query.score_threshold}
                    if rag_query.score_threshold
                    else None
                ),
            )
        else:
            search_results = await store.search(query=rag_query.query, k=rag_query.k)

        # Filter by score threshold if needed
        if rag_query.score_threshold:
            search_results = [
                r for r in search_results if r.score >= rag_query.score_threshold
            ]

        if not search_results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "query": rag_query.query,
                "total_sources": 0,
                "store_id": rag_query.store_id or vector_store_manager.default_store,
            }

        # Extract context and sources
        context_parts = []
        sources = []

        for result in search_results:
            context_parts.append(result.document.page_content)

            if rag_query.include_sources:
                sources.append(
                    {
                        "content": result.document.page_content[:200] + "...",
                        "metadata": result.document.metadata,
                        "score": float(result.score),
                    }
                )

        context = "\n\n".join(context_parts)

        # Generate answer
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(
                content=self.rag_prompt.format(
                    context=context, question=rag_query.query
                )
            ),
        ]

        response = await llm_router.generate(messages)

        return {
            "answer": response.content,
            "sources": sources,
            "query": rag_query.query,
            "total_sources": len(search_results),
            "store_id": rag_query.store_id or vector_store_manager.default_store,
        }

    async def _query_all_stores(self, rag_query: EnhancedRAGQuery) -> Dict[str, Any]:
        """Query all vector stores and combine results"""
        all_results = await vector_store_manager.search_all_stores(
            query=rag_query.query, k=rag_query.k
        )

        # Combine and rank results from all stores
        combined_results = []
        sources_by_store = {}

        for store_id, results in all_results.items():
            sources_by_store[store_id] = []

            for result in results:
                # Add store info to metadata
                result.document.metadata["_store_id"] = store_id
                combined_results.append(result)

                if rag_query.include_sources:
                    sources_by_store[store_id].append(
                        {
                            "content": result.document.page_content[:200] + "...",
                            "metadata": result.document.metadata,
                            "score": float(result.score),
                        }
                    )

        # Sort by score
        combined_results.sort(key=lambda x: x.score, reverse=True)

        # Apply score threshold
        if rag_query.score_threshold:
            combined_results = [
                r for r in combined_results if r.score >= rag_query.score_threshold
            ]

        if not combined_results:
            return {
                "answer": "I couldn't find any relevant information across any knowledge stores.",
                "sources_by_store": {},
                "query": rag_query.query,
                "total_sources": 0,
                "stores_searched": list(all_results.keys()),
            }

        # Build multi-context
        context_parts = []
        for result in combined_results[: rag_query.k]:  # Top k across all stores
            store_id = result.document.metadata.get("_store_id", "unknown")
            context_parts.append(
                f"[Source: {store_id}]\n{result.document.page_content}"
            )

        multi_context = "\n\n".join(context_parts)

        # Generate answer
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(
                content=self.multi_store_prompt.format(
                    multi_context=multi_context, question=rag_query.query
                )
            ),
        ]

        response = await llm_router.generate(messages)

        return {
            "answer": response.content,
            "sources_by_store": sources_by_store,
            "query": rag_query.query,
            "total_sources": len(combined_results),
            "stores_searched": list(all_results.keys()),
        }

    async def create_specialized_store(
        self,
        store_id: str,
        store_type: VectorStoreType,
        collection_name: str,
        description: str,
    ) -> Dict[str, Any]:
        """Create a specialized vector store for specific content"""
        config = VectorStoreConfig(
            collection_name=collection_name,
            persist_directory=f"data/vectors/{store_id}",
        )

        store = await vector_store_manager.create_store(
            store_id=store_id, store_type=store_type, config=config
        )

        # Store metadata about the specialized store
        stats = await store.get_collection_stats()
        stats["description"] = description

        return {"store_id": store_id, "store_type": store_type, "stats": stats}

    async def get_rag_stats(self) -> Dict[str, Any]:
        """Get comprehensive RAG system statistics"""
        stats = await vector_store_manager.get_stats()

        # Add processing stats
        stats["document_processor"] = {
            "chunk_size": self.document_processor.chunk_size,
            "chunk_overlap": self.document_processor.chunk_overlap,
            "splitter_type": type(self.document_processor.text_splitter).__name__,
        }

        return stats

    async def optimize_all_stores(self) -> Dict[str, bool]:
        """Optimize all vector stores in the RAG system"""
        return await vector_store_manager.optimize_stores()


# Global enhanced RAG pipeline instance
enhanced_rag_pipeline = EnhancedRAGPipeline()
