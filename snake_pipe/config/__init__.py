"""
Config package initialization
"""

# Import extract config for TASK-001
try:
    from .extract_config import ExtractConfig, create_default_config
    __all__ = ["ExtractConfig", "create_default_config"]
except ImportError:
    __all__ = []
