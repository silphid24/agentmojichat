"""Chat agent implementation"""

from typing import List, Optional, Dict, Any
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.callbacks import AsyncCallbackHandler

from app.agents.base import BaseAgent
from app.core.logging import logger
from app.core.config import settings


class ChatAgent(BaseAgent):
    """Basic chat agent for conversations"""
    
    def __init__(
        self,
        agent_id: str = "chat_agent",
        name: str = "MOJI Chat Agent",
        description: str = "General purpose chat agent",
        system_prompt: Optional[str] = None,
        memory_window: int = 5
    ):
        super().__init__(agent_id, name, description, memory_window)
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.llm = None
        self.chain = None
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are MOJI, a helpful AI assistant. You are:
- Friendly and professional
- Concise but thorough in your responses
- Honest about what you know and don't know
- Respectful of user privacy and preferences

Current context:
- Application: {app_name}
- Version: {version}
"""
    
    async def initialize(self) -> None:
        """Initialize the chat agent"""
        # Get LLM from router
        from app.llm.router import llm_router
        
        # Initialize router if needed
        if not llm_router.current_provider:
            await llm_router.initialize()
        
        # Get LangChain model
        self.llm = await llm_router.get_langchain_model()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt.format(
                app_name=settings.app_name,
                version=settings.app_version
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory
        )
        
        logger.info(f"Chat agent initialized with LLM: {llm_router.config.provider}")
    
    async def process(self, messages: List[BaseMessage]) -> BaseMessage:
        """Process messages and return response"""
        try:
            # Get the last user message
            last_message = messages[-1] if messages else None
            if not last_message or not isinstance(last_message, HumanMessage):
                return AIMessage(content="I didn't receive a valid message.")
            
            # Add messages to memory
            for msg in messages[:-1]:  # Add all but the last message
                if isinstance(msg, HumanMessage):
                    self.memory.chat_memory.add_user_message(msg.content)
                elif isinstance(msg, AIMessage):
                    self.memory.chat_memory.add_ai_message(msg.content)
            
            # For MVP, return a simple response
            # TODO: Integrate with actual LLM when available
            response_content = await self._generate_response(last_message.content)
            
            # Add the interaction to memory
            self.memory.chat_memory.add_user_message(last_message.content)
            self.memory.chat_memory.add_ai_message(response_content)
            
            return AIMessage(content=response_content)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return AIMessage(content="I'm sorry, I encountered an error processing your message.")
    
    async def _generate_response(self, input_text: str) -> str:
        """Generate response using LLM"""
        if not self.chain:
            # Fallback if chain not initialized
            return f"I received your message: '{input_text}'. The LLM is not yet initialized."
        
        try:
            # Use the chain to generate response
            response = await self.chain.arun(input=input_text)
            return response
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I apologize, but I encountered an error generating a response."
    
    def add_tool(self, tool: Any) -> None:
        """Add a tool to the agent"""
        # TODO: Implement tool integration
        logger.info(f"Tool integration not yet implemented for {self.name}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts"""
        history = []
        messages = self.memory.chat_memory.messages
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                history.append({"role": "system", "content": msg.content})
        
        return history