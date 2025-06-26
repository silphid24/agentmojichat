"""Chat endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, List
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.schemas.chat import (
    ChatRequest, ChatResponse, ChatMessage, ChatChoice, ChatUsage,
    ChatSessionCreate, ChatSessionResponse
)
from app.schemas.auth import UserInDB
from app.api.v1.endpoints.auth import get_current_user
from app.core.config import settings
from app.core.logging import logger
from app.agents.manager import agent_manager
from app.agents.state import state_manager

router = APIRouter()

# MVP: In-memory session storage
SESSIONS = {}


@router.post("/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Create a chat completion"""
    logger.info(f"Chat request from user: {current_user.username}")
    
    # Convert messages to LangChain format
    lc_messages = []
    for msg in request.messages:
        if msg.role == "system":
            lc_messages.append(SystemMessage(content=msg.content))
        elif msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))
    
    # Create or get session state
    session_id = request.session_id or str(uuid.uuid4())
    state = state_manager.get_state(session_id)
    if not state:
        state = state_manager.create_state(
            session_id=session_id,
            agent_id="chat_agent",
            user_id=str(current_user.id)
        )
    
    # Process with agent
    try:
        ai_message = await agent_manager.process_with_agent(lc_messages)
        response_content = ai_message.content
    except Exception as e:
        logger.error(f"Agent processing error: {e}")
        response_content = "I apologize, but I encountered an error processing your request."
    
    # Update state
    for msg in request.messages:
        state_manager.add_message(session_id, msg.role, msg.content)
    state_manager.add_message(session_id, "assistant", response_content)
    
    # Create response
    response_message = ChatMessage(
        role="assistant",
        content=response_content
    )
    
    return ChatResponse(
        id=str(uuid.uuid4()),
        model=request.model or settings.llm_model,
        choices=[ChatChoice(message=response_message)],
        usage=ChatUsage(
            prompt_tokens=sum(len(msg.content.split()) for msg in request.messages),
            completion_tokens=len(response_message.content.split()),
            total_tokens=sum(len(msg.content.split()) for msg in request.messages) + len(response_message.content.split())
        )
    )


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    session_create: ChatSessionCreate,
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    
    SESSIONS[session_id] = {
        "user_id": current_user.id,
        "messages": [],
        "created_at": datetime.utcnow()
    }
    
    # Add initial message if provided
    if session_create.initial_message:
        SESSIONS[session_id]["messages"].append(
            ChatMessage(role="user", content=session_create.initial_message)
        )
    
    return ChatSessionResponse(
        id=session_id,
        created_at=SESSIONS[session_id]["created_at"],
        message_count=len(SESSIONS[session_id]["messages"])
    )


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_chat_history(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Get chat history for a session"""
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    session = SESSIONS[session_id]
    if session["user_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return session["messages"]


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)]
):
    """Delete a chat session"""
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    session = SESSIONS[session_id]
    if session["user_id"] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    del SESSIONS[session_id]
    return {"message": "Session deleted successfully"}