"""Base agent class"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage
from langchain.memory import ConversationBufferWindowMemory

from app.core.logging import logger


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        memory_window: int = 5
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True
        )
        logger.info(f"Initialized agent: {name} ({agent_id})")
    
    @abstractmethod
    async def process(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        """Process messages and return response"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent resources"""
        pass
    
    async def reset(self) -> None:
        """Reset agent state"""
        self.memory.clear()
        logger.info(f"Reset agent: {self.name}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "memory": self.memory.buffer
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set agent state"""
        if "memory" in state:
            self.memory.buffer = state["memory"]