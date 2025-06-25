"""Chat endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated, List
import uuid
from datetime import datetime

from app.schemas.chat import (
    ChatRequest, ChatResponse, ChatMessage, ChatChoice, ChatUsage,
    ChatSessionCreate, ChatSessionResponse
)
from app.schemas.auth import UserInDB
from app.api.v1.endpoints.auth import get_current_user
from app.core.config import settings
from app.core.logging import logger

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
    
    # MVP: Echo response for testing
    # TODO: Integrate with LLM router (task_05)
    response_message = ChatMessage(
        role="assistant",
        content=f"Echo: {request.messages[-1].content if request.messages else 'No message'}"
    )
    
    # Store in session if session_id provided
    if request.session_id:
        if request.session_id not in SESSIONS:
            SESSIONS[request.session_id] = {
                "user_id": current_user.id,
                "messages": []
            }
        SESSIONS[request.session_id]["messages"].extend(request.messages)
        SESSIONS[request.session_id]["messages"].append(response_message)
    
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