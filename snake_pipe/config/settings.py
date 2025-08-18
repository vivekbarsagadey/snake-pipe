"""Basic settings for snake_pipe configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Basic settings class for snake_pipe."""
    
    # Logging settings
    log_level: str = "INFO"
    
    # Performance settings
    max_workers: int = 4
    
    # Database settings (for future use)
    database_url: Optional[str] = None


# Global settings instance
settings = Settings()