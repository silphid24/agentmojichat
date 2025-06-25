# MOJI Agent System Documentation

## Overview

The MOJI Agent System is built on LangChain and provides a flexible, extensible framework for creating AI agents with different capabilities and specializations.

## Architecture

### Core Components

1. **BaseAgent**: Abstract base class for all agents
2. **ChatAgent**: Basic conversational agent implementation
3. **AgentManager**: Manages multiple agents and routes requests
4. **StateManager**: Handles conversation state and history
5. **ToolRegistry**: Manages available tools for agents

## Agent Types

### Chat Agent (Default)
- General-purpose conversational agent
- Maintains conversation history
- Supports context window management
- Can be extended with tools

### Future Agent Types (Planned)
- **Task Agent**: Specialized in task management and execution
- **Knowledge Agent**: Focused on information retrieval and analysis
- **Technical Agent**: Programming and technical assistance
- **Creative Agent**: Creative content generation

## Usage

### Basic Chat Interaction

```python
from app.agents.manager import agent_manager
from langchain.schema import HumanMessage

# Process a message with default agent
messages = [HumanMessage(content="Hello, MOJI!")]
response = await agent_manager.process_with_agent(messages)
print(response.content)
```

### Creating Custom Agents

```python
from app.agents.chat_agent import ChatAgent

# Create custom agent with specific prompt
agent = ChatAgent(
    agent_id="custom_agent",
    name="Custom MOJI Agent",
    description="Specialized agent for specific tasks",
    system_prompt="You are a specialized assistant focused on..."
)

# Register with manager
await agent_manager.register_agent(agent)
```

### Managing Conversation State

```python
from app.agents.state import state_manager

# Create session
state = state_manager.create_state(
    session_id="unique_session_id",
    agent_id="chat_agent",
    user_id="user123"
)

# Add messages
state_manager.add_message(session_id, "user", "Hello!")
state_manager.add_message(session_id, "assistant", "Hi there!")

# Get history
history = state_manager.get_conversation_history(session_id)
```

## Available Tools

### Calculator
- Mathematical calculations
- Basic arithmetic operations

### DateTime
- Current date and time
- Custom formatting support

### Search (Mock)
- Placeholder for search functionality
- Returns mock results for testing

## API Endpoints

### Agent Management
- `GET /api/v1/agents/` - List all agents
- `GET /api/v1/agents/{agent_id}` - Get agent info
- `POST /api/v1/agents/{agent_id}/reset` - Reset agent state

### Session Management
- `GET /api/v1/agents/sessions/list` - List user sessions
- `GET /api/v1/agents/sessions/{session_id}` - Get session state
- `DELETE /api/v1/agents/sessions/{session_id}` - Delete session

### Tools
- `GET /api/v1/agents/tools/list` - List available tools
- `GET /api/v1/agents/tools/{agent_type}` - Get tools for agent type

## Configuration

Agent behavior can be configured through:
- System prompts
- Memory window size
- Tool availability
- Context parameters

## Best Practices

1. **Session Management**: Always use session IDs for multi-turn conversations
2. **Error Handling**: Agents include error handling for graceful failures
3. **State Persistence**: Current implementation uses in-memory storage (production will use database)
4. **Tool Usage**: Tools are automatically selected based on agent type

## Future Enhancements

1. **LangGraph Integration**: Migration to LangGraph for advanced workflows
2. **Multi-Agent Orchestration**: Coordination between multiple specialized agents
3. **Advanced Tools**: Integration with external APIs and services
4. **Persistent Storage**: Database-backed state management
5. **Streaming Responses**: Real-time streaming for long responses