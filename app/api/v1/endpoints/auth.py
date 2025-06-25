"""Authentication endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated

from app.models.auth import TokenRequest, TokenResponse, User
from app.core.security import verify_password, create_access_token, decode_access_token
from app.core.exceptions import AuthenticationError

router = APIRouter()
security = HTTPBearer()


# MVP: Simple in-memory user store
USERS_DB = {
    "admin": {
        "username": "admin",
        "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "email": "admin@example.com"
    }
}


@router.post("/token", response_model=TokenResponse)
async def login(request: TokenRequest):
    """Authenticate user and return JWT token"""
    user = USERS_DB.get(request.username)
    if not user or not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": request.username})
    return TokenResponse(access_token=access_token)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> User:
    """Get current authenticated user"""
    try:
        payload = decode_access_token(credentials.credentials)
        username = payload.get("sub")
        if username is None:
            raise AuthenticationError("Invalid token")
        
        user_data = USERS_DB.get(username)
        if user_data is None:
            raise AuthenticationError("User not found")
        
        return User(username=username, email=user_data.get("email"))
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.get("/me", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    """Get current user information"""
    return current_user