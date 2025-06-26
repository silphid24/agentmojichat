"""Base adapter interface for platform integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


class MessageType(str, Enum):
    """Message types supported across platforms."""
    
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    LOCATION = "location"
    CARD = "card"
    CAROUSEL = "carousel"
    BUTTONS = "buttons"
    SYSTEM = "system"


class AttachmentType(str, Enum):
    """Attachment types for messages."""
    
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class Attachment:
    """Message attachment."""
    
    type: AttachmentType
    url: Optional[str] = None
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    thumbnail_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Button:
    """Interactive button for messages."""
    
    text: str
    value: str
    action: str = "postback"
    url: Optional[str] = None
    style: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Card:
    """Rich card for structured content."""
    
    title: str
    subtitle: Optional[str] = None
    text: Optional[str] = None
    image_url: Optional[str] = None
    buttons: List[Button] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Location:
    """Location data."""
    
    latitude: float
    longitude: float
    title: Optional[str] = None
    address: Optional[str] = None


@dataclass
class User:
    """Platform user information."""
    
    id: str
    name: str
    platform: str
    avatar_url: Optional[str] = None
    email: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Conversation context."""
    
    id: str
    platform: str
    type: str = "direct"  # direct, group, channel
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlatformMessage:
    """Unified message format across platforms."""
    
    id: Union[str, UUID] = field(default_factory=uuid4)
    type: MessageType = MessageType.TEXT
    text: Optional[str] = None
    user: Optional[User] = None
    conversation: Optional[Conversation] = None
    attachments: List[Attachment] = field(default_factory=list)
    buttons: List[Button] = field(default_factory=list)
    cards: List[Card] = field(default_factory=list)
    location: Optional[Location] = None
    reply_to: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    platform_specific: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = {
            "id": str(self.id),
            "type": self.type.value,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        
        if self.user:
            data["user"] = {
                "id": self.user.id,
                "name": self.user.name,
                "platform": self.user.platform,
            }
        
        if self.conversation:
            data["conversation"] = {
                "id": self.conversation.id,
                "platform": self.conversation.platform,
                "type": self.conversation.type,
            }
        
        if self.attachments:
            data["attachments"] = [
                {
                    "type": att.type.value,
                    "url": att.url,
                    "file_name": att.file_name,
                }
                for att in self.attachments
            ]
        
        if self.buttons:
            data["buttons"] = [
                {"text": btn.text, "value": btn.value, "action": btn.action}
                for btn in self.buttons
            ]
        
        if self.cards:
            data["cards"] = [
                {
                    "title": card.title,
                    "subtitle": card.subtitle,
                    "text": card.text,
                    "buttons": [
                        {"text": btn.text, "value": btn.value}
                        for btn in card.buttons
                    ],
                }
                for card in self.cards
            ]
        
        return data


class BaseAdapter(ABC):
    """Base adapter interface for platform integrations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adapter with configuration."""
        self.config = config
        self.platform_name = self.__class__.__name__.replace("Adapter", "").lower()
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the platform."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the platform."""
        pass
    
    @abstractmethod
    async def send_message(self, message: PlatformMessage) -> Dict[str, Any]:
        """Send a message to the platform."""
        pass
    
    @abstractmethod
    async def receive_message(self, raw_message: Dict[str, Any]) -> PlatformMessage:
        """Convert platform-specific message to unified format."""
        pass
    
    @abstractmethod
    async def get_user_info(self, user_id: str) -> User:
        """Get user information from the platform."""
        pass
    
    @abstractmethod
    async def get_conversation_info(self, conversation_id: str) -> Conversation:
        """Get conversation information from the platform."""
        pass
    
    async def handle_error(self, error: Exception) -> None:
        """Handle platform-specific errors."""
        print(f"[{self.platform_name}] Error: {error}")
    
    async def validate_message(self, message: PlatformMessage) -> bool:
        """Validate message before sending."""
        if message.type == MessageType.TEXT and not message.text:
            return False
        
        if message.type == MessageType.IMAGE and not message.attachments:
            return False
        
        return True
    
    def format_text(self, text: str, platform_specific: bool = True) -> str:
        """Format text for the platform."""
        return text
    
    def supports_feature(self, feature: str) -> bool:
        """Check if platform supports a specific feature."""
        features = {
            "buttons": True,
            "cards": True,
            "files": True,
            "images": True,
            "audio": True,
            "video": True,
            "location": False,
            "typing_indicator": True,
            "read_receipts": False,
            "reactions": False,
        }
        return features.get(feature, False)