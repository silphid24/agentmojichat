"""KakaoTalk adapter implementation."""

import aiohttp
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import (
    BaseAdapter,
    PlatformMessage,
    MessageType,
    User,
    Conversation,
    Attachment,
    AttachmentType,
)


class KakaoTalkAdapter(BaseAdapter):
    """KakaoTalk adapter for MOJI."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize KakaoTalk adapter."""
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.channel_id = config.get("channel_id")
        self.base_url = "https://kapi.kakao.com"
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> None:
        """Connect to KakaoTalk API."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"KakaoAK {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    async def disconnect(self) -> None:
        """Disconnect from KakaoTalk."""
        if self.session:
            await self.session.close()
            self.session = None

    async def send_message(self, message: PlatformMessage) -> Dict[str, Any]:
        """Send message to KakaoTalk."""
        if not await self.validate_message(message):
            raise ValueError("Invalid message format")

        template = await self._create_message_template(message)

        if not self.session:
            await self.connect()

        # Send through KakaoTalk API
        async with self.session.post(
            f"{self.base_url}/v2/api/talk/memo/default/send",
            json={"template_object": template},
        ) as response:
            result = await response.json()

        return {
            "id": str(message.id),
            "status": "sent",
            "platform": "kakaotalk",
            "timestamp": message.timestamp.isoformat(),
            "result": result,
        }

    async def receive_message(self, raw_message: Dict[str, Any]) -> PlatformMessage:
        """Convert KakaoTalk webhook data to platform message."""
        msg = PlatformMessage(
            id=raw_message.get("message_id", ""),
            type=self._get_message_type(raw_message),
            text=raw_message.get("content", {}).get("text"),
            timestamp=datetime.fromtimestamp(raw_message.get("timestamp", 0) / 1000),
            platform_specific={"kakao_data": raw_message},
        )

        # Extract user information
        user_data = raw_message.get("user", {})
        if user_data:
            msg.user = User(
                id=str(user_data.get("id")),
                name=user_data.get("properties", {}).get("nickname", ""),
                platform="kakaotalk",
                avatar_url=user_data.get("properties", {}).get("profile_image"),
                metadata={"kakao_user": user_data},
            )

        # Extract conversation information
        msg.conversation = Conversation(
            id=raw_message.get("chat_id", ""),
            platform="kakaotalk",
            type="direct",  # KakaoTalk channel messages are typically direct
            metadata={"channel_id": self.channel_id},
        )

        # Extract attachments
        if raw_message.get("attachment"):
            msg.attachments = await self._convert_attachments(raw_message["attachment"])

        return msg

    async def get_user_info(self, user_id: str) -> User:
        """Get KakaoTalk user information."""
        if not self.session:
            await self.connect()

        async with self.session.get(
            f"{self.base_url}/v2/user/me",
            params={"target_id_type": "user_id", "target_id": user_id},
        ) as response:
            data = await response.json()

        return User(
            id=str(data.get("id")),
            name=data.get("properties", {}).get("nickname", ""),
            platform="kakaotalk",
            avatar_url=data.get("properties", {}).get("profile_image"),
            metadata={"kakao_account": data.get("kakao_account", {})},
        )

    async def get_conversation_info(self, conversation_id: str) -> Conversation:
        """Get KakaoTalk conversation information."""
        return Conversation(
            id=conversation_id,
            platform="kakaotalk",
            type="direct",
            metadata={"channel_id": self.channel_id},
        )

    async def _create_message_template(
        self, message: PlatformMessage
    ) -> Dict[str, Any]:
        """Create KakaoTalk message template."""
        if message.cards:
            # Use list template for cards
            return await self._create_list_template(message)
        elif message.buttons:
            # Use text template with buttons
            return await self._create_text_template_with_buttons(message)
        else:
            # Simple text template
            return {
                "object_type": "text",
                "text": message.text or "",
                "link": {
                    "web_url": "https://moji.ai",
                    "mobile_web_url": "https://moji.ai",
                },
            }

    async def _create_list_template(self, message: PlatformMessage) -> Dict[str, Any]:
        """Create list template for cards."""
        contents = []

        for card in message.cards[:3]:  # KakaoTalk limits to 3 items
            content = {
                "title": card.title[:200],  # 200 char limit
                "description": card.subtitle or card.text or "",
                "image_url": card.image_url,
                "image_width": 640,
                "image_height": 480,
                "link": {
                    "web_url": "https://moji.ai",
                    "mobile_web_url": "https://moji.ai",
                },
            }
            contents.append(content)

        template = {
            "object_type": "list",
            "header_title": "MOJI AI Assistant",
            "header_link": {
                "web_url": "https://moji.ai",
                "mobile_web_url": "https://moji.ai",
            },
            "contents": contents,
        }

        # Add buttons if any
        if message.cards[0].buttons:
            template["buttons"] = [
                {
                    "title": btn.text[:14],  # 14 char limit
                    "link": {
                        "web_url": btn.url or "https://moji.ai",
                        "mobile_web_url": btn.url or "https://moji.ai",
                    },
                }
                for btn in message.cards[0].buttons[:2]  # Max 2 buttons
            ]

        return template

    async def _create_text_template_with_buttons(
        self, message: PlatformMessage
    ) -> Dict[str, Any]:
        """Create text template with buttons."""
        template = {
            "object_type": "text",
            "text": message.text or "",
            "link": {
                "web_url": "https://moji.ai",
                "mobile_web_url": "https://moji.ai",
            },
            "button_title": (
                message.buttons[0].text[:14] if message.buttons else "더보기"
            ),
        }

        return template

    async def _convert_attachments(
        self, kakao_attachments: Dict[str, Any]
    ) -> List[Attachment]:
        """Convert KakaoTalk attachments to platform attachments."""
        attachments = []

        if kakao_attachments.get("image"):
            attachments.append(
                Attachment(
                    type=AttachmentType.IMAGE,
                    url=kakao_attachments["image"],
                    metadata={"kakao_type": "image"},
                )
            )

        if kakao_attachments.get("file"):
            attachments.append(
                Attachment(
                    type=AttachmentType.DOCUMENT,
                    url=kakao_attachments["file"]["url"],
                    file_name=kakao_attachments["file"].get("name"),
                    file_size=kakao_attachments["file"].get("size"),
                    metadata={"kakao_type": "file"},
                )
            )

        return attachments

    def _get_message_type(self, raw_message: Dict[str, Any]) -> MessageType:
        """Determine message type from KakaoTalk data."""
        content = raw_message.get("content", {})

        if content.get("type") == "photo":
            return MessageType.IMAGE
        elif content.get("type") == "video":
            return MessageType.VIDEO
        elif content.get("type") == "file":
            return MessageType.FILE

        return MessageType.TEXT

    def supports_feature(self, feature: str) -> bool:
        """Check KakaoTalk feature support."""
        features = {
            "buttons": True,  # Limited to button_title
            "cards": True,  # Through list template
            "files": True,
            "images": True,
            "audio": False,
            "video": True,
            "location": True,
            "typing_indicator": False,
            "read_receipts": False,
            "reactions": False,
            "templates": True,  # KakaoTalk specific
            "quick_replies": True,
        }
        return features.get(feature, False)
