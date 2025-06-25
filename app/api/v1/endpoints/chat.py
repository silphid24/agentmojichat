"""Chat endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
import uuid

from app.models.chat import ChatRequest, ChatResponse, Message, Choice, Usage
from app.models.auth import User
from app.api.v1.endpoints.auth import get_current_user
from app.core.logging import logger

router = APIRouter()


@router.post("/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Create a chat completion"""
    logger.info(f"Chat request from user: {current_user.username}")
    
    # MVP: Echo response for testing
    # TODO: Integrate with LLM router (task_05)
    response_message = Message(
        role="assistant",
        content=f"Echo: {request.messages[-1].content if request.messages else 'No message'}"
    )
    
    return ChatResponse(
        id=str(uuid.uuid4()),
        choices=[Choice(message=response_message)],
        usage=Usage(
            prompt_tokens=len(request.messages) * 10,  # Mock calculation
            completion_tokens=len(response_message.content),
            total_tokens=len(request.messages) * 10 + len(response_message.content)
        )
    )


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Get chat history for a session"""
    # MVP: Return empty history
    # TODO: Implement with database integration
    return {
        "session_id": session_id,
        "messages": []
    }