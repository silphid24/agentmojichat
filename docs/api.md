# MOJI AI Agent API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication

All API endpoints (except health check and authentication) require JWT authentication.

### Get Access Token
```http
POST /auth/token
Content-Type: application/json

{
  "username": "admin",
  "password": "secret"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Using the Token
Include the token in the Authorization header:
```
Authorization: Bearer <access_token>
```

## Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy"
  }
}
```

### Authentication

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "username": "newuser",
  "password": "password123",
  "email": "user@example.com"
}
```

#### Get Current User
```http
GET /auth/me
Authorization: Bearer <token>
```

### Chat

#### Create Chat Completion
```http
POST /chat/completions
Authorization: Bearer <token>
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "model": "deepseek-r1",  // optional, defaults to configured model
  "temperature": 0.7,
  "max_tokens": 1024,
  "session_id": "uuid"  // optional
}
```

Response:
```json
{
  "id": "chat-completion-id",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "deepseek-r1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

#### Create Chat Session
```http
POST /chat/sessions
Authorization: Bearer <token>
Content-Type: application/json

{
  "initial_message": "Hello, I need help with..."  // optional
}
```

#### Get Chat History
```http
GET /chat/sessions/{session_id}/messages
Authorization: Bearer <token>
```

#### Delete Chat Session
```http
DELETE /chat/sessions/{session_id}
Authorization: Bearer <token>
```

## Error Responses

All errors follow this format:
```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "code": "ERROR_CODE",
    "details": {}
  }
}
```

Common error types:
- `AUTH_FAILED`: Authentication failed
- `AUTH_FORBIDDEN`: Insufficient permissions
- `VALIDATION_ERROR`: Request validation failed
- `NOT_FOUND`: Resource not found
- `RATE_LIMIT_EXCEEDED`: Too many requests

## Rate Limiting

- Default: 100 requests per hour per user
- Rate limit headers are included in responses:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

## Request IDs

All responses include a `X-Request-ID` header for tracking purposes.