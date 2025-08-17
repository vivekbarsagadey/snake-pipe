"""
Snake Pipe - Python-based ETL pipeline framework
"""

from typing import Any, Callable

__version__ = "0.1.0"
__author__ = "Vivek Barsagadey"

# Import main components when dependencies are available
try:
    from .pipeline import Pipeline, run_pipeline

    __all__ = ["Pipeline", "run_pipeline"]
except ImportError:
    # If dependencies aren't installed, provide a helpful message
    def _pipeline_fallback(*args: Any, **kwargs: Any) -> None:
        raise ImportError("Dependencies not installed. Please run: uv sync or pip install -e .")

    def _run_pipeline_fallback() -> None:
        raise ImportError("Dependencies not installed. Please run: uv sync or pip install -e .")

    # Type ignore here because we're dynamically replacing the imports
    Pipeline: Callable[..., None] = _pipeline_fallback  # type: ignore
    run_pipeline: Callable[[], None] = _run_pipeline_fallback  # type: ignore
    __all__ = ["Pipeline", "run_pipeline"]
