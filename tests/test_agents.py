"""Agent system tests"""

import pytest
from langchain.schema import HumanMessage, AIMessage

from app.agents.chat_agent import ChatAgent
from app.agents.manager import AgentManager
from app.agents.state import StateManager
from app.agents.tools import tool_registry


@pytest.mark.asyncio
async def test_chat_agent_creation():
    """Test chat agent creation"""
    agent = ChatAgent(
        agent_id="test_agent",
        name="Test Agent",
        description="Test description"
    )
    assert agent.agent_id == "test_agent"
    assert agent.name == "Test Agent"
    assert agent.description == "Test description"


@pytest.mark.asyncio
async def test_chat_agent_initialization():
    """Test chat agent initialization"""
    agent = ChatAgent()
    await agent.initialize()
    assert agent.system_prompt is not None
    assert agent.memory is not None


@pytest.mark.asyncio
async def test_chat_agent_process():
    """Test chat agent message processing"""
    agent = ChatAgent()
    await agent.initialize()
    
    messages = [HumanMessage(content="Hello, MOJI!")]
    response = await agent.process(messages)
    
    assert isinstance(response, AIMessage)
    assert "Hello, MOJI!" in response.content


@pytest.mark.asyncio
async def test_agent_manager():
    """Test agent manager"""
    manager = AgentManager()
    
    # Create and register agent
    agent = ChatAgent(agent_id="test_manager_agent")
    await manager.register_agent(agent)
    
    # Test agent retrieval
    retrieved_agent = manager.get_agent("test_manager_agent")
    assert retrieved_agent.agent_id == "test_manager_agent"
    
    # Test listing agents
    agents = manager.list_agents()
    assert any(a["agent_id"] == "test_manager_agent" for a in agents)
    
    # Test processing with agent
    messages = [HumanMessage(content="Test message")]
    response = await manager.process_with_agent(messages, "test_manager_agent")
    assert isinstance(response, AIMessage)


def test_state_manager():
    """Test state manager"""
    manager = StateManager()
    
    # Create state
    state = manager.create_state(
        session_id="test_session",
        agent_id="test_agent",
        user_id="test_user"
    )
    assert state.session_id == "test_session"
    assert state.agent_id == "test_agent"
    
    # Add message
    manager.add_message(
        "test_session",
        "user",
        "Test message"
    )
    
    # Get history
    history = manager.get_conversation_history("test_session")
    assert len(history) == 1
    assert history[0]["content"] == "Test message"
    
    # Update context
    manager.update_context("test_session", "test_key", "test_value")
    context = manager.get_context("test_session", "test_key")
    assert context == "test_value"
    
    # Delete state
    assert manager.delete_state("test_session") is True


def test_tool_registry():
    """Test tool registry"""
    # Test default tools
    tools = tool_registry.list_tools()
    assert "calculator" in tools
    assert "datetime" in tools
    assert "search" in tools
    
    # Test calculator tool
    calc_tool = tool_registry.get_tool("calculator")
    result = calc_tool.func("2 + 2")
    assert "4" in result
    
    # Test datetime tool
    dt_tool = tool_registry.get_tool("datetime")
    result = dt_tool.func()
    assert len(result) > 0
    
    # Test tools for agent type
    general_tools = tool_registry.get_tools_for_agent("general")
    assert len(general_tools) > 0