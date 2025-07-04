"""Platform adapters for MOJI AI Agent."""

from .base import BaseAdapter, PlatformMessage, MessageType, AttachmentType, Button, Card
# from .teams import TeamsAdapter  # TODO: Install botbuilder-core to enable Teams support
from .kakaotalk import KakaoTalkAdapter
from .webchat import WebChatAdapter

__all__ = [
    "BaseAdapter",
    "PlatformMessage",
    "MessageType",
    "AttachmentType",
    "Button",
    "Card",
    # "TeamsAdapter",  # TODO: Install botbuilder-core to enable Teams support
    "KakaoTalkAdapter",
    "WebChatAdapter",
]