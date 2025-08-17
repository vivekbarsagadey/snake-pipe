"""
Helper utilities for snake-pipe
"""

import hashlib
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import pandas as pd
except ImportError:
    pd = None


def sanitize_column_name(column_name: str) -> str:
    """
    Sanitize column name for database compatibility

    Args:
        column_name: Original column name

    Returns:
        Sanitized column name
    """
    # Convert to lowercase
    sanitized = column_name.lower()

    # Replace spaces and special characters with underscores
    sanitized = re.sub(r"[^a-z0-9_]", "_", sanitized)

    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = f"col_{sanitized}"

    # Handle empty string
    if not sanitized:
        sanitized = "unnamed_column"

    return sanitized


def generate_hash(data: str, algorithm: str = "sha256") -> str:
    """
    Generate hash for a given string

    Args:
        data: String to hash
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

    Returns:
        Hash string
    """
    if algorithm == "md5":
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(data.encode(), usedforsecurity=False).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size

    Args:
        data: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i: i + chunk_size])
    return chunks


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely divide two numbers, returning default if division by zero

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Division result or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """
    Flatten a nested dictionary

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items: List[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Get memory usage information for a DataFrame

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with memory usage info
    """
    memory_usage = df.memory_usage(deep=True)

    return {
        "total_bytes": memory_usage.sum(),
        "total_mb": memory_usage.sum() / (1024 * 1024),
        "total_gb": memory_usage.sum() / (1024 * 1024 * 1024),
        "by_column": memory_usage.to_dict(),
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
    }


def parse_date_string(date_str: str, formats: Optional[List[str]] = None) -> Optional[datetime]:
    """
    Parse date string using multiple format attempts

    Args:
        date_str: Date string to parse
        formats: List of date formats to try

    Returns:
        Parsed datetime object or None
    """
    if formats is None:
        formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%SZ"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def validate_email(email: str) -> bool:
    """
    Validate email address format

    Args:
        email: Email address to validate

    Returns:
        True if valid email format
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def retry_operation(func: Callable, max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0, exceptions: tuple = (Exception,)) -> Callable:
    """
    Retry decorator for operations that might fail

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorator function
    """

    def decorator(*args: Any, **kwargs: Any) -> Any:
        last_exception = None
        current_delay = delay

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt == max_retries:
                    break

                import time

                time.sleep(current_delay)
                current_delay *= backoff_factor

        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("Function failed without raising an exception")

    return decorator


def convert_size(size_bytes: int) -> str:
    """
    Convert bytes to human readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Human readable size string
    """
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math

    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"


def get_dataframe_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a comprehensive profile of a DataFrame

    Args:
        df: DataFrame to profile

    Returns:
        Dictionary with DataFrame profile information
    """
    profile = {
        "shape": df.shape,
        "memory_usage": get_memory_usage(df),
        "column_info": {},
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
        "dtypes": df.dtypes.to_dict(),
        "duplicate_rows": df.duplicated().sum(),
    }

    # Column-specific information
    for col in df.columns:
        col_info = {"dtype": str(df[col].dtype), "null_count": df[col].isnull().sum(), "unique_count": df[col].nunique(), "memory_usage": df[col].memory_usage(deep=True)}

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({"min": df[col].min(), "max": df[col].max(), "mean": df[col].mean(), "std": df[col].std(), "median": df[col].median()})
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
            col_info.update({"avg_length": df[col].astype(str).str.len().mean(), "max_length": df[col].astype(str).str.len().max(), "min_length": df[col].astype(str).str.len().min()})

        profile["column_info"][col] = col_info

    return profile
