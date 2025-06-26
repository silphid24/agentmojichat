"""Configuration management"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "MOJI AI Agent"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database
    database_url: str = Field(
        default="postgresql://user:pass@localhost/moji",
        env="DATABASE_URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL"
    )
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    llm_api_base: Optional[str] = Field(default=None, env="LLM_API_BASE")
    llm_api_key: str = Field(default="", env="LLM_API_KEY")
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        env="SECRET_KEY"
    )
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # CORS
    cors_origins: list[str] = ["*"]
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings()