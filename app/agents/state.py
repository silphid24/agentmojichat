"""Agent state management"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import json

from app.core.logging import logger


class ConversationState(BaseModel):
    """Conversation state model"""

    session_id: str
    agent_id: str
    user_id: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateManager:
    """Manages conversation states"""

    def __init__(self):
        # MVP: In-memory storage
        self.states: Dict[str, ConversationState] = {}
        logger.info("Initialized StateManager")

    def create_state(
        self,
        session_id: str,
        agent_id: str,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> ConversationState:
        """Create new conversation state"""
        state = ConversationState(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            context=initial_context or {},
        )
        self.states[session_id] = state
        logger.info(f"Created state for session: {session_id}")
        return state

    def get_state(self, session_id: str) -> Optional[ConversationState]:
        """Get conversation state"""
        return self.states.get(session_id)

    def update_state(
        self,
        session_id: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationState]:
        """Update conversation state"""
        state = self.get_state(session_id)
        if not state:
            return None

        if messages:
            state.messages.extend(messages)

        if context:
            state.context.update(context)

        if metadata:
            state.metadata.update(metadata)

        state.updated_at = datetime.utcnow()
        logger.debug(f"Updated state for session: {session_id}")
        return state

    def delete_state(self, session_id: str) -> bool:
        """Delete conversation state"""
        if session_id in self.states:
            del self.states[session_id]
            logger.info(f"Deleted state for session: {session_id}")
            return True
        return False

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add message to conversation state"""
        state = self.get_state(session_id)
        if not state:
            return False

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        state.messages.append(message)
        state.updated_at = datetime.utcnow()
        return True

    def get_conversation_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history"""
        state = self.get_state(session_id)
        if not state:
            return []

        messages = state.messages
        if limit:
            messages = messages[-limit:]

        return messages

    def update_context(self, session_id: str, key: str, value: Any) -> bool:
        """Update specific context value"""
        state = self.get_state(session_id)
        if not state:
            return False

        state.context[key] = value
        state.updated_at = datetime.utcnow()
        return True

    def get_context(self, session_id: str, key: Optional[str] = None) -> Any:
        """Get context value or entire context"""
        state = self.get_state(session_id)
        if not state:
            return None

        if key:
            return state.context.get(key)
        return state.context

    def export_state(self, session_id: str) -> Optional[str]:
        """Export state as JSON"""
        state = self.get_state(session_id)
        if not state:
            return None

        return state.model_dump_json(indent=2)

    def import_state(self, session_id: str, state_json: str) -> bool:
        """Import state from JSON"""
        try:
            state_dict = json.loads(state_json)
            state = ConversationState(**state_dict)
            self.states[session_id] = state
            logger.info(f"Imported state for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to import state: {e}")
            return False

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all sessions, optionally filtered by user"""
        sessions = []
        for session_id, state in self.states.items():
            if user_id and state.user_id != user_id:
                continue

            sessions.append(
                {
                    "session_id": session_id,
                    "agent_id": state.agent_id,
                    "user_id": state.user_id,
                    "message_count": len(state.messages),
                    "created_at": state.created_at.isoformat(),
                    "updated_at": state.updated_at.isoformat(),
                }
            )

        return sessions


# Global state manager
state_manager = StateManager()
