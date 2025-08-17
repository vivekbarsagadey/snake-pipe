"""
Snake Pipe - Python-based ETL pipeline framework
"""

__version__ = "0.1.0"
__author__ = "Vivek Barsagadey"

# Import main components when dependencies are available
try:
    from .pipeline import Pipeline, run_pipeline
    __all__ = ["Pipeline", "run_pipeline"]
except ImportError:
    # If dependencies aren't installed, provide a helpful message
    def Pipeline(*args, **kwargs):
        raise ImportError(
            "Dependencies not installed. Please run: uv sync or pip install -e ."
        )
    
    def run_pipeline():
        raise ImportError(
            "Dependencies not installed. Please run: uv sync or pip install -e ."
        )
    
    __all__ = ["Pipeline", "run_pipeline"]