"""
Configuration and settings management for snake-pipe
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Global settings and configuration"""
    
    # Database settings
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "snake_pipe")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    # API settings
    API_BASE_URL: str = os.getenv("API_BASE_URL", "")
    API_KEY: Optional[str] = os.getenv("API_KEY")
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))
    
    # File paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "./logs"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Pipeline settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1000"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    def __init__(self):
        # Ensure directories exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()