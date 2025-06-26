# Platform Adapters Guide

## Overview

MOJI supports multiple messaging platforms through a unified adapter interface. This guide explains how to integrate and use platform adapters.

## Supported Platforms

### 1. Microsoft Teams
- Full Bot Framework integration
- Adaptive Cards support
- SSO authentication
- Channel and group conversations
- Meeting integration

### 2. KakaoTalk
- Channel API integration
- Message templates
- Button and carousel UI
- User profile integration

### 3. Web Chat
- Embeddable widget
- WebSocket real-time communication
- Customizable UI
- File attachments
- Persistent chat history

## Architecture

### Base Components

#### PlatformMessage
Unified message format across all platforms:
```python
message = PlatformMessage(
    type=MessageType.TEXT,
    text="Hello, world!",
    user=User(id="user123", name="John Doe", platform="teams"),
    conversation=Conversation(id="conv123", platform="teams"),
)
```

#### Message Types
- `TEXT`: Plain text messages
- `IMAGE`: Image attachments
- `FILE`: Document attachments
- `AUDIO`: Audio messages
- `VIDEO`: Video content
- `CARD`: Rich cards with structured content
- `CAROUSEL`: Multiple cards in a carousel
- `BUTTONS`: Interactive buttons
- `SYSTEM`: System notifications

#### BaseAdapter
All platform adapters inherit from `BaseAdapter`:
```python
class CustomAdapter(BaseAdapter):
    async def connect(self) -> None:
        # Initialize platform connection
        
    async def disconnect(self) -> None:
        # Clean up resources
        
    async def send_message(self, message: PlatformMessage) -> Dict[str, Any]:
        # Send message to platform
        
    async def receive_message(self, raw_message: Dict[str, Any]) -> PlatformMessage:
        # Convert platform message to unified format
```

## API Endpoints

### Send Message
```http
POST /api/v1/adapters/send
Content-Type: application/json

{
  "platform": "teams",
  "conversation_id": "conv123",
  "text": "Hello from MOJI!",
  "type": "text"
}
```

### Platform Webhooks
```http
POST /api/v1/adapters/webhook/{platform}
```

### WebChat WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/adapters/webchat/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    user_id: 'user123',
    user_name: 'John Doe'
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};
```

### Get WebChat Widget
```http
GET /api/v1/adapters/webchat/widget
```

Response:
```html
<!-- MOJI Web Chat Widget -->
<div id="moji-webchat-widget"></div>
<script src="/static/moji-webchat.js"></script>
<script>
  MojiWebChat.init({
    apiUrl: '/api/v1/webchat',
    theme: 'light',
    position: 'bottom-right',
    title: 'MOJI Assistant',
    placeholder: 'Type your message...',
  });
</script>
```

## Platform-Specific Features

### Microsoft Teams

#### Adaptive Cards
```python
card = Card(
    title="Task Update",
    subtitle="Project Status",
    text="The project is on track",
    buttons=[
        Button(text="View Details", url="https://example.com/project"),
        Button(text="Mark Complete", value="complete_task")
    ]
)
```

#### Mentions
Teams adapter supports @mentions for users and channels.

### KakaoTalk

#### Message Templates
KakaoTalk uses predefined templates:
- Text template
- List template (for cards)
- Commerce template
- Location template

#### Limitations
- Button text: 14 characters max
- List items: 3 max
- Buttons per template: 2 max

### Web Chat

#### Custom Styling
```javascript
MojiWebChat.init({
  theme: {
    primaryColor: '#007bff',
    backgroundColor: '#ffffff',
    fontFamily: 'Arial, sans-serif'
  }
});
```

#### File Upload
```javascript
const fileInput = document.getElementById('file-input');
fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  MojiWebChat.sendFile(file);
});
```

## Error Handling

All adapters implement consistent error handling:

```python
try:
    await adapter.send_message(message)
except ConnectionError as e:
    # Handle connection issues
except ValidationError as e:
    # Handle message validation errors
except PlatformError as e:
    # Handle platform-specific errors
```

## Testing

### Unit Tests
```python
async def test_adapter_send_message():
    adapter = TeamsAdapter(config)
    message = PlatformMessage(type=MessageType.TEXT, text="Test")
    result = await adapter.send_message(message)
    assert result["status"] == "sent"
```

### Integration Tests
Test with actual platform APIs in staging environment.

### Load Testing
- Concurrent connections: 1000+
- Message throughput: 100 msg/sec
- Response time: < 500ms

## Configuration

### Environment Variables
```env
# Microsoft Teams
TEAMS_APP_ID=your-app-id
TEAMS_APP_PASSWORD=your-password
TEAMS_TENANT_ID=your-tenant-id

# KakaoTalk
KAKAO_API_KEY=your-api-key
KAKAO_CHANNEL_ID=your-channel-id

# WebChat
WEBCHAT_SESSION_TIMEOUT=3600
WEBCHAT_MAX_CONNECTIONS=1000
```

## Best Practices

1. **Message Validation**: Always validate messages before sending
2. **Feature Detection**: Check platform capabilities before using features
3. **Error Recovery**: Implement retry logic for transient failures
4. **Rate Limiting**: Respect platform rate limits
5. **Security**: Validate webhooks and authenticate users
6. **Monitoring**: Track message delivery and error rates

## Extending Adapters

To add a new platform:

1. Create adapter class inheriting from `BaseAdapter`
2. Implement required methods
3. Add platform-specific features
4. Register in adapter registry
5. Add API endpoints if needed
6. Write comprehensive tests
7. Document platform limitations

Example:
```python
class SlackAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.token = config.get("token")
        # Initialize Slack client
        
    async def connect(self) -> None:
        # Connect to Slack RTM or Events API
        
    # Implement other required methods
```