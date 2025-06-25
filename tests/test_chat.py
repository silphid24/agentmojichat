"""Chat endpoint tests"""

import pytest
from fastapi import status


def test_chat_completion_unauthorized(client):
    """Test chat completion without authentication"""
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_chat_completion_success(client):
    """Test successful chat completion"""
    # First login
    login_response = client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "secret"}
    )
    token = login_response.json()["access_token"]
    
    # Create chat completion
    response = client.post(
        "/api/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}]
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "id" in data
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_create_chat_session(client):
    """Test creating a chat session"""
    # First login
    login_response = client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "secret"}
    )
    token = login_response.json()["access_token"]
    
    # Create session
    response = client.post(
        "/api/v1/chat/sessions",
        json={"initial_message": "Hello"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "id" in data
    assert "created_at" in data
    assert data["message_count"] == 1