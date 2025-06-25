"""LLM router and provider tests"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain.schema import HumanMessage, AIMessage

from app.llm.base import LLMConfig, LLMResponse
from app.llm.router import LLMRouter
from app.llm.providers.deepseek import DeepSeekProvider
from app.llm.providers.openai import OpenAIProvider
from app.llm.providers.custom import CustomProvider


@pytest.fixture
def llm_config():
    """Test LLM configuration"""
    return LLMConfig(
        provider="deepseek",
        model="deepseek-r1",
        api_key="test-key",
        api_base=None,
        temperature=0.7,
        max_tokens=100
    )


@pytest.fixture
def mock_response():
    """Mock LLM response"""
    return LLMResponse(
        content="Test response",
        model="deepseek-r1",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    )


@pytest.mark.asyncio
async def test_llm_router_initialization(llm_config):
    """Test LLM router initialization"""
    router = LLMRouter()
    
    with patch.object(DeepSeekProvider, 'initialize', new_callable=AsyncMock):
        await router.initialize(llm_config)
    
    assert router.config == llm_config
    assert router.current_provider is not None
    assert isinstance(router.current_provider, DeepSeekProvider)


@pytest.mark.asyncio
async def test_llm_router_generate(llm_config, mock_response):
    """Test LLM router generation"""
    router = LLMRouter()
    
    # Mock provider
    mock_provider = Mock(spec=DeepSeekProvider)
    mock_provider.generate = AsyncMock(return_value=mock_response)
    mock_provider.initialize = AsyncMock()
    
    # Patch provider creation
    with patch('app.llm.router.DeepSeekProvider', return_value=mock_provider):
        await router.initialize(llm_config)
        
        messages = [HumanMessage(content="Test message")]
        response = await router.generate(messages)
    
    assert response == mock_response
    mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_llm_router_switch_provider(llm_config):
    """Test switching providers"""
    router = LLMRouter()
    
    # Initialize with DeepSeek
    with patch.object(DeepSeekProvider, 'initialize', new_callable=AsyncMock):
        await router.initialize(llm_config)
    
    assert router.config.provider == "deepseek"
    
    # Switch to OpenAI
    with patch.object(OpenAIProvider, 'initialize', new_callable=AsyncMock):
        await router._switch_provider("openai", "gpt-3.5-turbo")
    
    assert router.config.provider == "openai"
    assert router.config.model == "gpt-3.5-turbo"
    assert isinstance(router.current_provider, OpenAIProvider)


@pytest.mark.asyncio
async def test_deepseek_provider_format_messages():
    """Test DeepSeek message formatting"""
    config = LLMConfig(
        provider="deepseek",
        model="deepseek-r1",
        api_key="test-key"
    )
    
    provider = DeepSeekProvider(config)
    
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!")
    ]
    
    formatted = provider._format_messages(messages)
    
    assert len(formatted) == 2
    assert formatted[0] == {"role": "user", "content": "Hello"}
    assert formatted[1] == {"role": "assistant", "content": "Hi there!"}


@pytest.mark.asyncio
async def test_custom_provider_initialization():
    """Test custom provider initialization"""
    config = LLMConfig(
        provider="custom",
        model="llama-3",
        api_key="",
        api_base="http://localhost:11434"
    )
    
    provider = CustomProvider(config)
    assert provider.api_base == "http://localhost:11434"
    assert "Authorization" not in provider.headers  # No auth when key is empty


def test_llm_response_model():
    """Test LLM response model"""
    response = LLMResponse(
        content="Test content",
        model="test-model",
        usage={"total_tokens": 100},
        metadata={"test": "data"}
    )
    
    assert response.content == "Test content"
    assert response.model == "test-model"
    assert response.usage["total_tokens"] == 100
    assert response.metadata["test"] == "data"


@pytest.mark.asyncio
async def test_provider_retry_logic(llm_config):
    """Test provider retry with backoff"""
    provider = DeepSeekProvider(llm_config)
    
    # Mock function that fails twice then succeeds
    call_count = 0
    async def mock_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Test error")
        return "Success"
    
    # Test retry
    with patch('asyncio.sleep', new_callable=AsyncMock):
        result = await provider._retry_with_backoff(mock_func)
    
    assert result == "Success"
    assert call_count == 3