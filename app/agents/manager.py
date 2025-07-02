"""Agent manager for handling multiple agents"""

from typing import Dict, Optional, List, Any
from langchain.schema import BaseMessage

from app.agents.base import BaseAgent
from app.agents.chat_agent import ChatAgent
from app.core.logging import logger
from app.core.exceptions import NotFoundError


class AgentManager:
    """Manages multiple agents and routes requests"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.default_agent_id: Optional[str] = None
        logger.info("Initialized AgentManager")
    
    async def initialize_default_agents(self) -> None:
        """Initialize default agents"""
        # Create default chat agent
        chat_agent = ChatAgent()
        await self.register_agent(chat_agent)
        self.set_default_agent(chat_agent.agent_id)
        
        logger.info("Default agents initialized")
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent"""
        await agent.initialize()
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
            
            # Reset default if needed
            if self.default_agent_id == agent_id:
                self.default_agent_id = None
    
    def get_agent(self, agent_id: str) -> BaseAgent:
        """Get agent by ID"""
        if agent_id not in self.agents:
            raise NotFoundError(f"Agent {agent_id}")
        return self.agents[agent_id]
    
    def set_default_agent(self, agent_id: str) -> None:
        """Set default agent"""
        if agent_id not in self.agents:
            raise NotFoundError(f"Agent {agent_id}")
        self.default_agent_id = agent_id
        logger.info(f"Set default agent: {agent_id}")
    
    def get_default_agent(self) -> BaseAgent:
        """Get default agent"""
        if not self.default_agent_id:
            raise ValueError("No default agent set")
        return self.get_agent(self.default_agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "is_default": agent.agent_id == self.default_agent_id
            }
            for agent in self.agents.values()
        ]
    
    async def process_with_agent(
        self,
        messages: List[BaseMessage],
        agent_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseMessage:
        """Process messages with specified or default agent"""
        if agent_id:
            agent = self.get_agent(agent_id)
        else:
            agent = self.get_default_agent()
        
        logger.info(f"Processing with agent: {agent.name}")
        
        # Pass provider and model info to agent if supported
        process_kwargs = {}
        if provider:
            process_kwargs['provider'] = provider
        if model:
            process_kwargs['model'] = model
        process_kwargs.update(kwargs)
        
        return await agent.process(messages, **process_kwargs)
    
    async def reset_agent(self, agent_id: str) -> None:
        """Reset specific agent"""
        agent = self.get_agent(agent_id)
        await agent.reset()
    
    async def reset_all_agents(self) -> None:
        """Reset all agents"""
        for agent in self.agents.values():
            await agent.reset()
        logger.info("Reset all agents")


# Global agent manager instance
agent_manager = AgentManager()