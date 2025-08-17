"""
Logging utilities for snake-pipe
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from ..config.settings import settings


def setup_logger(name: str, level: Optional[str] = None, 
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers to the same logger
    if logger.handlers:
        return logger
    
    # Set log level
    log_level = level or settings.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified or settings has LOG_DIR)
    if log_file:
        file_path = Path(log_file)
    else:
        log_dir = settings.LOG_DIR
        log_dir.mkdir(exist_ok=True, parents=True)
        file_path = log_dir / f"{name.replace('.', '_')}.log"
    
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Optional log level override
        
    Returns:
        Logger instance
    """
    return setup_logger(name, level)


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


# Create a default logger for the package
default_logger = get_logger('snake_pipe')