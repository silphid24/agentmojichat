"""RAG pipeline implementation

DEPRECATED: This module is deprecated in favor of enhanced_rag.py

Please use:
- app.rag.enhanced_rag.EnhancedRAGPipeline for direct usage
- app.rag.adapter.RAGPipelineAdapter for API compatibility

See app/rag/MIGRATION_GUIDE.md for migration instructions.
"""

import warnings

warnings.warn(
    "app.rag.pipeline is deprecated. Use app.rag.enhanced_rag instead.",
    DeprecationWarning,
    stacklevel=2,
)

from typing import List, Dict, Any, Optional
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.rag.document_processor import DocumentProcessor
from app.rag.retriever import VectorRetriever
from app.llm.router import llm_router


class RAGQuery(BaseModel):
    """RAG query model"""

    query: str
    k: int = Field(default=3, ge=1, le=10)
    score_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    include_sources: bool = True


class RAGResponse(BaseModel):
    """RAG response model"""

    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    query: str
    total_sources: int = 0


class RAGPipeline:
    """Complete RAG pipeline"""

    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        document_processor: Optional[DocumentProcessor] = None,
        use_local_embeddings: bool = True,
    ):
        self.retriever = retriever or VectorRetriever(
            use_local_embeddings=use_local_embeddings
        )
        self.document_processor = document_processor or DocumentProcessor()

        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant. Use the following context to answer the user's question.
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        )

        logger.info("Initialized RAG pipeline")

    async def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the RAG system"""
        processed_docs = []
        failed_files = []

        # Process each file
        for file_path in file_paths:
            try:
                doc = await self.document_processor.process_file(file_path)
                processed_docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed_files.append({"file": file_path, "error": str(e)})

        # Extract all chunks
        all_chunks = []
        for doc in processed_docs:
            all_chunks.extend(doc.chunks)

        # Add to vector store
        if all_chunks:
            await self.retriever.add_documents(all_chunks)

        return {
            "processed": len(processed_docs),
            "failed": len(failed_files),
            "total_chunks": len(all_chunks),
            "failed_files": failed_files,
        }

    async def add_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add raw text to the RAG system"""
        try:
            # Process text
            doc = await self.document_processor.process_text(text, metadata)

            # Add chunks to vector store
            await self.retriever.add_documents(doc.chunks)

            return {"success": True, "chunks": len(doc.chunks), "document_id": doc.id}

        except Exception as e:
            logger.error(f"Failed to add text: {e}")
            return {"success": False, "error": str(e)}

    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """Query the RAG system"""
        try:
            # Search for relevant documents
            search_results = await self.retriever.search(
                query=rag_query.query,
                k=rag_query.k,
                score_threshold=rag_query.score_threshold,
            )

            if not search_results:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    query=rag_query.query,
                    total_sources=0,
                )

            # Extract context from search results
            context_parts = []
            sources = []

            for doc, score in search_results:
                context_parts.append(doc.page_content)

                if rag_query.include_sources:
                    sources.append(
                        {
                            "content": doc.page_content[:200]
                            + "...",  # Truncate for response
                            "metadata": doc.metadata,
                            "score": float(score),
                        }
                    )

            context = "\n\n".join(context_parts)

            # Generate answer using LLM
            messages = [
                SystemMessage(content="You are a helpful AI assistant."),
                HumanMessage(
                    content=self.rag_prompt.format(
                        context=context, question=rag_query.query
                    )
                ),
            ]

            response = await llm_router.generate(messages)

            return RAGResponse(
                answer=response.content,
                sources=sources,
                query=rag_query.query,
                total_sources=len(search_results),
            )

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                query=rag_query.query,
                total_sources=0,
            )

    async def query_with_history(
        self, query: str, history: List[BaseMessage], k: int = 3
    ) -> RAGResponse:
        """Query with conversation history"""
        # Build context-aware query
        history_context = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in history[-3:]]  # Last 3 messages
        )

        enhanced_query = f"""Based on this conversation:
{history_context}

Current question: {query}"""

        # Create RAG query
        rag_query = RAGQuery(query=enhanced_query, k=k)

        return await self.query(rag_query)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return self.retriever.get_index_stats()

    def clear_index(self) -> None:
        """Clear the vector index"""
        self.retriever.delete_index()
        logger.info("Cleared RAG index")

    async def update_chunk_size(self, chunk_size: int, chunk_overlap: int) -> None:
        """Update document processing parameters"""
        self.document_processor.update_chunk_size(chunk_size, chunk_overlap)
        logger.info(f"Updated chunk size: {chunk_size}, overlap: {chunk_overlap}")
