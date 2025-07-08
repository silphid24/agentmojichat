"""RAG-enabled agent"""

from typing import List, Optional, Dict, Any
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from app.agents.base import BaseAgent
from app.rag.pipeline import RAGPipeline, RAGQuery
from app.core.logging import logger


class RAGAgent(BaseAgent):
    """Agent with RAG capabilities"""

    def __init__(
        self,
        agent_id: str = "rag_agent",
        name: str = "MOJI RAG Agent",
        description: str = "Agent with document retrieval capabilities",
        rag_pipeline: Optional[RAGPipeline] = None,
        memory_window: int = 5,
        rag_k: int = 3,
    ):
        super().__init__(agent_id, name, description, memory_window)
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.rag_k = rag_k

        self.system_prompt = """You are MOJI RAG Agent, an AI assistant with access to a knowledge base.
When answering questions:
1. Search the knowledge base for relevant information
2. Base your answers on the retrieved context
3. If the information isn't in the knowledge base, say so
4. Always cite your sources when using retrieved information
"""

    async def initialize(self) -> None:
        """Initialize the RAG agent"""
        logger.info(f"Initialized RAG agent: {self.name}")

    async def process(self, messages: List[BaseMessage]) -> BaseMessage:
        """Process messages with RAG"""
        try:
            # Get the last user message
            last_message = messages[-1] if messages else None
            if not last_message or not isinstance(last_message, HumanMessage):
                return AIMessage(content="I didn't receive a valid message.")

            # Query RAG system
            rag_query = RAGQuery(
                query=last_message.content, k=self.rag_k, include_sources=True
            )

            rag_response = await self.rag_pipeline.query(rag_query)

            # Format response with sources
            if rag_response.sources:
                source_text = "\n\nSources:\n"
                for i, source in enumerate(rag_response.sources, 1):
                    source_text += (
                        f"{i}. {source['metadata'].get('filename', 'Unknown')} "
                    )
                    source_text += f"(relevance: {source['score']:.2f})\n"

                full_response = rag_response.answer + source_text
            else:
                full_response = rag_response.answer

            # Update memory
            self.memory.chat_memory.add_user_message(last_message.content)
            self.memory.chat_memory.add_ai_message(full_response)

            return AIMessage(content=full_response)

        except Exception as e:
            logger.error(f"Error in RAG agent processing: {e}")
            return AIMessage(
                content="I encountered an error while searching the knowledge base."
            )

    async def add_knowledge(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add knowledge to the RAG system"""
        try:
            result = await self.rag_pipeline.add_text(text, metadata)
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics"""
        base_stats = self.get_state()
        rag_stats = self.rag_pipeline.get_index_stats()

        return {**base_stats, "rag_stats": rag_stats}
