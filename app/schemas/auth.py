"""Authentication schemas"""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema"""

    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[EmailStr] = None


class UserCreate(UserBase):
    """User creation schema"""

    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    """User update schema"""

    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6)


class UserInDB(UserBase):
    """User in database schema"""

    id: int
    is_active: bool

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token schema"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400


class TokenData(BaseModel):
    """Token data schema"""

    username: Optional[str] = None


class TokenRequest(BaseModel):
    """Token request schema"""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
