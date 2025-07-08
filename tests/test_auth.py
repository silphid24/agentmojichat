"""Authentication tests"""

from fastapi import status


def test_login_success(client):
    """Test successful login"""
    response = client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "secret"}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_credentials(client):
    """Test login with invalid credentials"""
    response = client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "wrong"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_register_new_user(client):
    """Test user registration"""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "newuser",
            "password": "password123",
            "email": "newuser@example.com",
        },
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["username"] == "newuser"
    assert data["email"] == "newuser@example.com"
    assert data["is_active"] is True


def test_get_current_user(client):
    """Test get current user"""
    # First login
    login_response = client.post(
        "/api/v1/auth/token", json={"username": "admin", "password": "secret"}
    )
    token = login_response.json()["access_token"]

    # Get current user
    response = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["username"] == "admin"
