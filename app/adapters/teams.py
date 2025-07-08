"""Microsoft Teams adapter implementation."""

from typing import Any, Dict, List
from datetime import datetime

from botbuilder.core import CardFactory, MessageFactory
from botbuilder.schema import (
    Activity,
    Attachment as BotAttachment,
    HeroCard,
    ActionTypes,
    CardImage,
    SuggestedActions,
    CardAction,
)

from .base import (
    BaseAdapter,
    PlatformMessage,
    MessageType,
    User,
    Conversation,
    Attachment,
    AttachmentType,
)


class TeamsAdapter(BaseAdapter):
    """Microsoft Teams adapter for MOJI."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Teams adapter."""
        super().__init__(config)
        self.app_id = config.get("app_id")
        self.app_password = config.get("app_password")
        self.tenant_id = config.get("tenant_id")
        self._client = None

    async def connect(self) -> None:
        """Connect to Teams Bot Framework."""
        # In production, initialize BotFrameworkAdapter here
        pass

    async def disconnect(self) -> None:
        """Disconnect from Teams."""
        if self._client:
            # Clean up resources
            self._client = None

    async def send_message(self, message: PlatformMessage) -> Dict[str, Any]:
        """Send message to Teams."""
        if not await self.validate_message(message):
            raise ValueError("Invalid message format")

        # activity = await self._convert_to_teams_activity(message)

        # In production, send activity through Bot Framework
        # response = await self._client.send_activity(activity)

        return {
            "id": str(message.id),
            "status": "sent",
            "platform": "teams",
            "timestamp": message.timestamp.isoformat(),
        }

    async def receive_message(self, raw_message: Dict[str, Any]) -> PlatformMessage:
        """Convert Teams activity to platform message."""
        activity = Activity().deserialize(raw_message)

        msg = PlatformMessage(
            id=activity.id,
            type=self._get_message_type(activity),
            text=activity.text,
            timestamp=activity.timestamp or datetime.utcnow(),
            platform_specific={"activity": raw_message},
        )

        # Extract user information
        if activity.from_property:
            msg.user = User(
                id=activity.from_property.id,
                name=activity.from_property.name,
                platform="teams",
                metadata={
                    "aad_object_id": activity.from_property.aad_object_id,
                },
            )

        # Extract conversation information
        if activity.conversation:
            msg.conversation = Conversation(
                id=activity.conversation.id,
                platform="teams",
                type=self._get_conversation_type(activity),
                name=activity.conversation.name,
            )

        # Extract attachments
        if activity.attachments:
            msg.attachments = await self._convert_attachments(activity.attachments)

        return msg

    async def get_user_info(self, user_id: str) -> User:
        """Get Teams user information."""
        # In production, query Graph API for user details
        return User(
            id=user_id,
            name="Teams User",
            platform="teams",
            metadata={"teams_id": user_id},
        )

    async def get_conversation_info(self, conversation_id: str) -> Conversation:
        """Get Teams conversation information."""
        # In production, query Bot Framework for conversation details
        return Conversation(
            id=conversation_id,
            platform="teams",
            type="channel",
            name="Teams Channel",
        )

    async def _convert_to_teams_activity(self, message: PlatformMessage) -> Activity:
        """Convert platform message to Teams activity."""
        activity = MessageFactory.text(message.text or "")

        # Add cards
        if message.cards:
            activity.attachments = []
            for card in message.cards:
                hero_card = HeroCard(
                    title=card.title,
                    subtitle=card.subtitle,
                    text=card.text,
                    images=[CardImage(url=card.image_url)] if card.image_url else [],
                    buttons=[
                        CardAction(
                            type=(
                                ActionTypes.open_url if btn.url else ActionTypes.im_back
                            ),
                            title=btn.text,
                            value=btn.url or btn.value,
                        )
                        for btn in card.buttons
                    ],
                )
                activity.attachments.append(CardFactory.hero_card(hero_card))

        # Add buttons as suggested actions
        elif message.buttons:
            activity.suggested_actions = SuggestedActions(
                actions=[
                    CardAction(
                        type=ActionTypes.im_back,
                        title=btn.text,
                        value=btn.value,
                    )
                    for btn in message.buttons
                ]
            )

        # Add attachments
        if message.attachments:
            if not activity.attachments:
                activity.attachments = []

            for att in message.attachments:
                if att.type == AttachmentType.IMAGE:
                    activity.attachments.append(
                        CardFactory.adaptive_card(
                            {
                                "type": "AdaptiveCard",
                                "body": [
                                    {
                                        "type": "Image",
                                        "url": att.url,
                                        "size": "auto",
                                    }
                                ],
                            }
                        )
                    )

        return activity

    async def _convert_attachments(
        self, teams_attachments: List[BotAttachment]
    ) -> List[Attachment]:
        """Convert Teams attachments to platform attachments."""
        attachments = []

        for att in teams_attachments:
            attachment = Attachment(
                type=self._get_attachment_type(att.content_type),
                file_name=att.name,
                mime_type=att.content_type,
                metadata={"content": att.content},
            )

            if att.content_url:
                attachment.url = att.content_url

            attachments.append(attachment)

        return attachments

    def _get_message_type(self, activity: Activity) -> MessageType:
        """Determine message type from Teams activity."""
        if activity.attachments:
            # Check for adaptive cards or hero cards
            for att in activity.attachments:
                if att.content_type in [
                    "application/vnd.microsoft.card.adaptive",
                    "application/vnd.microsoft.card.hero",
                ]:
                    return MessageType.CARD

        if activity.value:
            return MessageType.BUTTONS

        return MessageType.TEXT

    def _get_conversation_type(self, activity: Activity) -> str:
        """Determine conversation type."""
        if activity.conversation.is_group:
            return "group"
        elif hasattr(activity, "channel_data") and activity.channel_data:
            if activity.channel_data.get("channel"):
                return "channel"

        return "direct"

    def _get_attachment_type(self, content_type: str) -> AttachmentType:
        """Map content type to attachment type."""
        if content_type.startswith("image/"):
            return AttachmentType.IMAGE
        elif content_type.startswith("audio/"):
            return AttachmentType.AUDIO
        elif content_type.startswith("video/"):
            return AttachmentType.VIDEO
        else:
            return AttachmentType.DOCUMENT

    def supports_feature(self, feature: str) -> bool:
        """Check Teams feature support."""
        features = {
            "buttons": True,
            "cards": True,
            "adaptive_cards": True,
            "files": True,
            "images": True,
            "audio": True,
            "video": True,
            "location": False,
            "typing_indicator": True,
            "read_receipts": False,
            "reactions": True,
            "mentions": True,
            "threads": True,
        }
        return features.get(feature, False)
