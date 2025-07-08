"""Agent management endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, List, Dict, Any

from app.schemas.auth import UserInDB
from app.api.v1.endpoints.auth import get_current_user
from app.agents.manager import agent_manager
from app.agents.state import state_manager
from app.agents.tools import tool_registry

router = APIRouter()


@router.get("/", response_model=List[Dict[str, Any]])
async def list_agents(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    """List all available agents"""
    return agent_manager.list_agents()


@router.get("/{agent_id}")
async def get_agent_info(
    agent_id: str, current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Get information about a specific agent"""
    try:
        agent = agent_manager.get_agent(agent_id)
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "state": agent.get_state(),
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent not found: {agent_id}"
        )


@router.post("/{agent_id}/reset")
async def reset_agent(
    agent_id: str, current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Reset an agent's state"""
    try:
        await agent_manager.reset_agent(agent_id)
        return {"message": f"Agent {agent_id} reset successfully"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent not found: {agent_id}"
        )


@router.get("/sessions/list")
async def list_sessions(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    """List all sessions for the current user"""
    return state_manager.list_sessions(user_id=str(current_user.id))


@router.get("/sessions/{session_id}")
async def get_session_state(
    session_id: str, current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Get session state"""
    state = state_manager.get_state(session_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    # Verify user access
    if state.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    return {
        "session_id": state.session_id,
        "agent_id": state.agent_id,
        "message_count": len(state.messages),
        "context": state.context,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Delete a session"""
    state = state_manager.get_state(session_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    # Verify user access
    if state.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    state_manager.delete_state(session_id)
    return {"message": "Session deleted successfully"}


@router.get("/tools/list")
async def list_tools(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    """List available tools"""
    return tool_registry.list_tools()


@router.get("/tools/{agent_type}")
async def get_agent_tools(
    agent_type: str, current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Get tools available for a specific agent type"""
    tools = tool_registry.get_tools_for_agent(agent_type)
    return {
        "agent_type": agent_type,
        "tools": [
            {"name": tool.name, "description": tool.description} for tool in tools
        ],
    }
