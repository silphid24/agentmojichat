"""Tests for platform adapters."""

import pytest
from datetime import datetime

from app.adapters import (
    BaseAdapter,
    PlatformMessage,
    MessageType,
    AttachmentType,
    User,
    Conversation,
    Attachment,
    Button,
    Card,
    TeamsAdapter,
    KakaoTalkAdapter,
    WebChatAdapter,
)


class TestPlatformMessage:
    """Test PlatformMessage functionality."""

    def test_create_text_message(self):
        """Test creating a simple text message."""
        msg = PlatformMessage(
            type=MessageType.TEXT,
            text="Hello, world!",
        )

        assert msg.type == MessageType.TEXT
        assert msg.text == "Hello, world!"
        assert msg.id is not None
        assert isinstance(msg.timestamp, datetime)

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        user = User(id="user123", name="Test User", platform="test")
        conversation = Conversation(id="conv123", platform="test")

        msg = PlatformMessage(
            type=MessageType.TEXT,
            text="Test message",
            user=user,
            conversation=conversation,
        )

        data = msg.to_dict()

        assert data["type"] == "text"
        assert data["text"] == "Test message"
        assert data["user"]["id"] == "user123"
        assert data["conversation"]["id"] == "conv123"

    def test_message_with_attachments(self):
        """Test message with attachments."""
        attachment = Attachment(
            type=AttachmentType.IMAGE,
            url="https://example.com/image.jpg",
            file_name="image.jpg",
        )

        msg = PlatformMessage(
            type=MessageType.IMAGE,
            attachments=[attachment],
        )

        assert len(msg.attachments) == 1
        assert msg.attachments[0].type == AttachmentType.IMAGE

    def test_message_with_buttons(self):
        """Test message with buttons."""
        button = Button(text="Click me", value="click_action")

        msg = PlatformMessage(
            type=MessageType.BUTTONS,
            text="Choose an option:",
            buttons=[button],
        )

        assert len(msg.buttons) == 1
        assert msg.buttons[0].text == "Click me"

    def test_message_with_cards(self):
        """Test message with cards."""
        card = Card(
            title="Card Title",
            subtitle="Card Subtitle",
            text="Card content",
            buttons=[Button(text="Action", value="action1")],
        )

        msg = PlatformMessage(
            type=MessageType.CARD,
            cards=[card],
        )

        assert len(msg.cards) == 1
        assert msg.cards[0].title == "Card Title"
        assert len(msg.cards[0].buttons) == 1


class TestBaseAdapter:
    """Test BaseAdapter functionality."""

    class TestAdapter(BaseAdapter):
        """Test implementation of BaseAdapter."""

        async def connect(self):
            pass

        async def disconnect(self):
            pass

        async def send_message(self, message):
            return {"status": "sent"}

        async def receive_message(self, raw_message):
            return PlatformMessage(type=MessageType.TEXT, text="Test")

        async def get_user_info(self, user_id):
            return User(id=user_id, name="Test User", platform="test")

        async def get_conversation_info(self, conversation_id):
            return Conversation(id=conversation_id, platform="test")

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        config = {"api_key": "test123"}
        adapter = self.TestAdapter(config)

        assert adapter.config == config
        assert adapter.platform_name == "test"

    def test_supports_feature(self):
        """Test feature support checking."""
        adapter = self.TestAdapter({})

        assert adapter.supports_feature("buttons") is True
        assert adapter.supports_feature("cards") is True
        assert adapter.supports_feature("location") is False

    @pytest.mark.asyncio
    async def test_validate_message(self):
        """Test message validation."""
        adapter = self.TestAdapter({})

        # Valid text message
        msg = PlatformMessage(type=MessageType.TEXT, text="Hello")
        assert await adapter.validate_message(msg) is True

        # Invalid text message (no text)
        msg = PlatformMessage(type=MessageType.TEXT)
        assert await adapter.validate_message(msg) is False

        # Invalid image message (no attachments)
        msg = PlatformMessage(type=MessageType.IMAGE)
        assert await adapter.validate_message(msg) is False


class TestTeamsAdapter:
    """Test Microsoft Teams adapter."""

    @pytest.fixture
    def teams_config(self):
        """Teams adapter configuration."""
        return {
            "app_id": "test-app-id",
            "app_password": "test-password",
            "tenant_id": "test-tenant",
        }

    @pytest.fixture
    def teams_adapter(self, teams_config):
        """Create Teams adapter instance."""
        return TeamsAdapter(teams_config)

    def test_teams_initialization(self, teams_adapter, teams_config):
        """Test Teams adapter initialization."""
        assert teams_adapter.app_id == teams_config["app_id"]
        assert teams_adapter.app_password == teams_config["app_password"]
        assert teams_adapter.tenant_id == teams_config["tenant_id"]

    def test_teams_features(self, teams_adapter):
        """Test Teams feature support."""
        assert teams_adapter.supports_feature("adaptive_cards") is True
        assert teams_adapter.supports_feature("mentions") is True
        assert teams_adapter.supports_feature("threads") is True
        assert teams_adapter.supports_feature("location") is False

    @pytest.mark.asyncio
    async def test_teams_send_message(self, teams_adapter):
        """Test sending message through Teams."""
        msg = PlatformMessage(
            type=MessageType.TEXT,
            text="Hello Teams!",
        )

        result = await teams_adapter.send_message(msg)

        assert result["status"] == "sent"
        assert result["platform"] == "teams"

    @pytest.mark.asyncio
    async def test_teams_receive_message(self, teams_adapter):
        """Test receiving Teams message."""
        raw_activity = {
            "id": "msg123",
            "type": "message",
            "text": "Hello from Teams",
            "from": {
                "id": "user123",
                "name": "Teams User",
                "aadObjectId": "aad123",
            },
            "conversation": {
                "id": "conv123",
                "isGroup": False,
            },
            "timestamp": "2024-01-01T00:00:00Z",
        }

        msg = await teams_adapter.receive_message(raw_activity)

        assert msg.text == "Hello from Teams"
        assert msg.user.id == "user123"
        assert msg.user.name == "Teams User"
        assert msg.conversation.id == "conv123"


class TestKakaoTalkAdapter:
    """Test KakaoTalk adapter."""

    @pytest.fixture
    def kakao_config(self):
        """KakaoTalk adapter configuration."""
        return {
            "api_key": "test-api-key",
            "channel_id": "test-channel",
        }

    @pytest.fixture
    def kakao_adapter(self, kakao_config):
        """Create KakaoTalk adapter instance."""
        return KakaoTalkAdapter(kakao_config)

    def test_kakao_initialization(self, kakao_adapter, kakao_config):
        """Test KakaoTalk adapter initialization."""
        assert kakao_adapter.api_key == kakao_config["api_key"]
        assert kakao_adapter.channel_id == kakao_config["channel_id"]
        assert kakao_adapter.base_url == "https://kapi.kakao.com"

    def test_kakao_features(self, kakao_adapter):
        """Test KakaoTalk feature support."""
        assert kakao_adapter.supports_feature("templates") is True
        assert kakao_adapter.supports_feature("quick_replies") is True
        assert kakao_adapter.supports_feature("audio") is False
        assert kakao_adapter.supports_feature("typing_indicator") is False

    @pytest.mark.asyncio
    async def test_kakao_connect_disconnect(self, kakao_adapter):
        """Test KakaoTalk connection lifecycle."""
        await kakao_adapter.connect()
        assert kakao_adapter.session is not None

        await kakao_adapter.disconnect()
        assert kakao_adapter.session is None

    @pytest.mark.asyncio
    async def test_kakao_receive_message(self, kakao_adapter):
        """Test receiving KakaoTalk message."""
        raw_message = {
            "message_id": "msg123",
            "content": {"text": "안녕하세요"},
            "timestamp": 1704067200000,  # milliseconds
            "user": {
                "id": "12345",
                "properties": {
                    "nickname": "카카오 사용자",
                    "profile_image": "https://example.com/profile.jpg",
                },
            },
            "chat_id": "chat123",
        }

        msg = await kakao_adapter.receive_message(raw_message)

        assert msg.text == "안녕하세요"
        assert msg.user.id == "12345"
        assert msg.user.name == "카카오 사용자"
        assert msg.conversation.id == "chat123"


class TestWebChatAdapter:
    """Test WebChat adapter."""

    @pytest.fixture
    def webchat_config(self):
        """WebChat adapter configuration."""
        return {
            "widget": {
                "script_url": "/static/moji-webchat.js",
                "api_url": "/api/v1/webchat",
                "theme": "light",
                "position": "bottom-right",
            }
        }

    @pytest.fixture
    def webchat_adapter(self, webchat_config):
        """Create WebChat adapter instance."""
        return WebChatAdapter(webchat_config)

    def test_webchat_initialization(self, webchat_adapter):
        """Test WebChat adapter initialization."""
        assert webchat_adapter.connections == {}
        assert webchat_adapter.conversations == {}
        assert webchat_adapter.widget_config is not None

    def test_webchat_features(self, webchat_adapter):
        """Test WebChat feature support."""
        assert webchat_adapter.supports_feature("file_upload") is True
        assert webchat_adapter.supports_feature("persistent_history") is True
        assert webchat_adapter.supports_feature("typing_indicator") is True
        assert webchat_adapter.supports_feature("reactions") is False

    @pytest.mark.asyncio
    async def test_webchat_connect_disconnect(self, webchat_adapter):
        """Test WebChat connection lifecycle."""
        await webchat_adapter.connect()
        assert webchat_adapter._cleanup_task is not None

        await webchat_adapter.disconnect()
        assert len(webchat_adapter.connections) == 0

    def test_webchat_widget_html(self, webchat_adapter):
        """Test widget HTML generation."""
        html = webchat_adapter.get_widget_html()

        assert "moji-webchat-widget" in html
        assert "/static/moji-webchat.js" in html
        assert "MojiWebChat.init" in html

    @pytest.mark.asyncio
    async def test_webchat_message_format(self, webchat_adapter):
        """Test WebChat message formatting."""
        msg = PlatformMessage(
            type=MessageType.TEXT,
            text="Hello WebChat!",
            buttons=[Button(text="Click", value="action1")],
        )

        formatted = await webchat_adapter._convert_to_webchat_format(msg)

        assert formatted["type"] == "text"
        assert formatted["text"] == "Hello WebChat!"
        assert len(formatted["buttons"]) == 1
        assert formatted["buttons"][0]["text"] == "Click"


@pytest.mark.asyncio
class TestAdapterIntegration:
    """Integration tests for adapters."""

    async def test_adapter_message_flow(self):
        """Test message flow through adapters."""
        # Create adapters
        teams = TeamsAdapter({"app_id": "test", "app_password": "test"})
        kakao = KakaoTalkAdapter({"api_key": "test", "channel_id": "test"})
        webchat = WebChatAdapter({})

        # Create a message
        original_msg = PlatformMessage(
            type=MessageType.TEXT,
            text="Cross-platform message",
            user=User(id="user123", name="Test User", platform="test"),
        )

        # Each adapter should be able to process the message
        for adapter in [teams, kakao, webchat]:
            assert await adapter.validate_message(original_msg) is True

    async def test_adapter_feature_compatibility(self):
        """Test feature compatibility across adapters."""
        adapters = [
            TeamsAdapter({"app_id": "test", "app_password": "test"}),
            KakaoTalkAdapter({"api_key": "test", "channel_id": "test"}),
            WebChatAdapter({}),
        ]

        # Common features that should be supported
        common_features = ["buttons", "cards", "files", "images"]

        for feature in common_features:
            for adapter in adapters:
                assert adapter.supports_feature(feature) is True
