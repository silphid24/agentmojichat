# LLM Router Documentation

## Overview

The LLM Router provides a unified interface for managing multiple Language Model providers. It supports dynamic provider switching, model selection, and seamless integration with the agent system.

## Default Configuration

By default, MOJI uses **DeepSeek R1** as the primary LLM:
- Provider: `deepseek`
- Model: `deepseek-r1`
- Configurable via environment variables

## Supported Providers

### 1. DeepSeek (Default)
- Models: `deepseek-r1`, `deepseek-chat`, `deepseek-coder`
- API Base: `https://api.deepseek.com/v1`
- Environment: `LLM_PROVIDER=deepseek`

### 2. OpenAI
- Models: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo-preview`
- API Base: `https://api.openai.com/v1`
- Environment: `LLM_PROVIDER=openai`

### 3. Custom (OpenAI-compatible)
- Models: Any model supported by your endpoint
- API Base: User-defined (e.g., `http://localhost:11434` for Ollama)
- Environment: `LLM_PROVIDER=custom`

## Configuration

### Environment Variables

```bash
# Provider selection
LLM_PROVIDER=deepseek  # deepseek, openai, custom

# Model selection
LLM_MODEL=deepseek-r1  # Model name

# Authentication
LLM_API_KEY=your-api-key-here

# Custom endpoint (for custom provider)
LLM_API_BASE=http://localhost:11434/v1
```

### Example Configurations

#### DeepSeek (Default)
```bash
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-r1
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
```

#### OpenAI
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
```

#### Local Model (Ollama)
```bash
LLM_PROVIDER=custom
LLM_MODEL=llama3
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=  # Leave empty for local models
```

## API Endpoints

### Get LLM Info
```http
GET /api/v1/llm/info
Authorization: Bearer <token>
```

Response:
```json
{
  "provider": "deepseek",
  "model": "deepseek-r1",
  "temperature": 0.7,
  "max_tokens": 1024,
  "available_providers": ["deepseek", "openai", "custom"]
}
```

### Switch Provider/Model
```http
POST /api/v1/llm/switch
Authorization: Bearer <token>
Content-Type: application/json

{
  "provider": "openai",
  "model": "gpt-4",
  "api_key": "sk-xxxxxxxx"  // Optional
}
```

### Test LLM
```http
POST /api/v1/llm/test
Authorization: Bearer <token>
Content-Type: application/json

{
  "message": "Hello, can you hear me?",
  "provider": "deepseek",  // Optional
  "model": "deepseek-r1"    // Optional
}
```

### Validate Providers
```http
GET /api/v1/llm/validate
Authorization: Bearer <token>
```

### Get Available Models
```http
GET /api/v1/llm/models/{provider}
Authorization: Bearer <token>
```

## Usage in Code

### Basic Usage
```python
from app.llm.router import llm_router
from langchain.schema import HumanMessage

# Initialize router (happens automatically on app startup)
await llm_router.initialize()

# Generate response
messages = [HumanMessage(content="Hello!")]
response = await llm_router.generate(messages)
print(response.content)
```

### Switching Providers
```python
# Switch to OpenAI
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="sk-xxxxxxxx"
)
await llm_router.initialize(config)
```

### Streaming Responses
```python
# Stream tokens
async for token in llm_router.stream(messages):
    print(token, end="")
```

## Integration with Agents

The LLM router automatically integrates with the agent system:

```python
# Agents use the router internally
agent = ChatAgent()
await agent.initialize()  # Uses llm_router

# Process messages
response = await agent.process([HumanMessage(content="Hello")])
```

## Error Handling

The router includes:
- Automatic retry with exponential backoff (3 attempts)
- Provider validation
- Graceful fallback for connection issues
- Detailed error logging

## Performance Considerations

1. **Provider Selection**: Choose based on your needs
   - DeepSeek R1: Best for reasoning tasks
   - GPT-3.5 Turbo: Fast and cost-effective
   - GPT-4: Most capable but slower
   - Local models: Privacy-focused, no API costs

2. **Token Limits**: Configure `max_tokens` appropriately
3. **Temperature**: Adjust for creativity vs consistency
4. **Caching**: Responses are not cached by default

## Troubleshooting

### Common Issues

1. **Invalid API Key**
   - Check environment variable: `echo $LLM_API_KEY`
   - Ensure key has proper permissions

2. **Connection Failed**
   - Verify API endpoint is accessible
   - Check network/firewall settings
   - For custom providers, ensure endpoint is running

3. **Model Not Found**
   - Verify model name is correct
   - Check provider documentation for available models

### Debug Mode

Enable debug logging:
```bash
DEBUG=true
```

This will show detailed LLM request/response information.