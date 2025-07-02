"""Chat schemas"""

from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message schema"""
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Chat request schema"""
    messages: List[ChatMessage]
    model: Optional[str] = Field(None, description="Override default model")
    provider: Optional[str] = Field(None, description="LLM provider (openai, custom, deepseek, anthropic)")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1024, ge=1, le=4096)
    stream: bool = False
    session_id: Optional[str] = None


class ChatChoice(BaseModel):
    """Chat completion choice"""
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "error"] = "stop"


class ChatUsage(BaseModel):
    """Token usage information"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Chat response schema"""
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage


class ChatSessionCreate(BaseModel):
    """Create chat session schema"""
    initial_message: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Chat session response schema"""
    id: str
    created_at: datetime
    message_count: int = 0