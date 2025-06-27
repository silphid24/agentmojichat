"""Authentication endpoints"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated

from app.schemas.auth import Token, UserInDB, UserCreate, TokenRequest
from app.core.security import verify_password, create_access_token, decode_access_token, get_password_hash
from app.core.exceptions import AuthenticationError

router = APIRouter()
security = HTTPBearer()


# MVP: Simple in-memory user store
USERS_DB = {
    "admin": {
        "id": 1,
        "username": "admin",
        "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "email": "admin@example.com",
        "is_active": True
    }
}


@router.post("/token", response_model=Token)
async def login(request: TokenRequest):
    """Authenticate user and return JWT token"""
    user = USERS_DB.get(request.username)
    if not user or not verify_password(request.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": request.username})
    return Token(access_token=access_token)


@router.post("/register", response_model=UserInDB, status_code=status.HTTP_201_CREATED)
async def register(user_create: UserCreate):
    """Register a new user"""
    # Check if user exists
    if user_create.username in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    user_id = len(USERS_DB) + 1
    hashed_password = get_password_hash(user_create.password)
    
    new_user = {
        "id": user_id,
        "username": user_create.username,
        "password": hashed_password,
        "email": user_create.email,
        "is_active": True
    }
    
    USERS_DB[user_create.username] = new_user
    
    return UserInDB(
        id=user_id,
        username=user_create.username,
        email=user_create.email,
        is_active=True
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> UserInDB:
    """Get current authenticated user"""
    try:
        payload = decode_access_token(credentials.credentials)
        username = payload.get("sub")
        if username is None:
            raise AuthenticationError("Invalid token")
        
        user_data = USERS_DB.get(username)
        if user_data is None:
            raise AuthenticationError("User not found")
        
        return UserInDB(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data.get("email"),
            is_active=user_data.get("is_active", True)
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserInDB)
async def read_users_me(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    """Get current user information"""
    return current_user