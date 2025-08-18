"""File system utilities for AST processing pipeline.

This module provides common file system operations, path utilities,
and file handling functions used across the AST processing pipeline.
"""

import asyncio
import json
import mmap
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple, Any
import logging

from snake_pipe.extract.models import ASTMetadata, LanguageType


logger = logging.getLogger(__name__)


class FileSystemError(Exception):
    """Base exception for file system operations."""
    pass


class FilePermissionError(FileSystemError):
    """Raised when file permissions prevent access."""
    pass


class FileSizeError(FileSystemError):
    """Raised when file size exceeds limits."""
    pass


async def get_file_stats(file_path: Path) -> Tuple[int, datetime, Optional[datetime]]:
    """Get file statistics asynchronously.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (file_size, modified_time, created_time)
        
    Raises:
        FileSystemError: If file cannot be accessed
    """
    try:
        # Use asyncio to run the blocking stat call
        loop = asyncio.get_event_loop()
        stat_result = await loop.run_in_executor(None, file_path.stat)
        
        file_size = stat_result.st_size
        modified_time = datetime.fromtimestamp(stat_result.st_mtime)
        
        # Created time is platform-dependent
        created_time = None
        if hasattr(stat_result, 'st_birthtime'):  # macOS
            created_time = datetime.fromtimestamp(stat_result.st_birthtime)
        elif hasattr(stat_result, 'st_ctime') and os.name == 'nt':  # Windows
            created_time = datetime.fromtimestamp(stat_result.st_ctime)
        
        return file_size, modified_time, created_time
        
    except (OSError, PermissionError) as e:
        raise FileSystemError(f"Cannot access file statistics for {file_path}: {e}")


async def check_file_permissions(file_path: Path) -> bool:
    """Check if file is readable.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file is readable, False otherwise
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, os.access, file_path, os.R_OK)
    except Exception:
        return False


async def is_file_json(file_path: Path) -> bool:
    """Check if file appears to be valid JSON.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file appears to be JSON, False otherwise
    """
    try:
        # Check file extension first
        if not file_path.suffix.lower() == '.json':
            return False
        
        # Quick check - read first few bytes to look for JSON structure
        loop = asyncio.get_event_loop()
        
        def _check_json_header():
            try:
                with open(file_path, 'rb') as f:
                    # Read first 1KB to check for JSON structure
                    header = f.read(1024).decode('utf-8', errors='ignore').strip()
                    return header.startswith('{') or header.startswith('[')
            except Exception:
                return False
        
        return await loop.run_in_executor(None, _check_json_header)
        
    except Exception:
        return False


async def read_json_metadata(file_path: Path, max_size: int = 100 * 1024) -> Dict[str, Any]:
    """Read JSON file metadata without loading full content.
    
    Args:
        file_path: Path to JSON file
        max_size: Maximum size to read for metadata extraction
        
    Returns:
        Dictionary containing metadata from JSON file
        
    Raises:
        FileSystemError: If file cannot be read
        json.JSONDecodeError: If JSON is malformed
    """
    try:
        loop = asyncio.get_event_loop()
        
        def _read_json_partial():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # For small files, read completely
                    if file_path.stat().st_size <= max_size:
                        return json.load(f)
                    
                    # For large files, read partial content
                    content = f.read(max_size)
                    
                    # Try to find complete JSON objects at the beginning
                    brace_count = 0
                    end_pos = 0
                    
                    for i, char in enumerate(content):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    if end_pos > 0:
                        partial_content = content[:end_pos]
                        return json.loads(partial_content)
                    else:
                        # Fallback to trying the truncated content
                        # Add closing brace if needed
                        if content.count('{') > content.count('}'):
                            content += '}'
                        return json.loads(content)
                        
            except json.JSONDecodeError as e:
                # Try to extract at least some basic information
                logger.warning(f"JSON parsing failed for {file_path}: {e}")
                return {"error": "partial_json_parse_failed", "file_path": str(file_path)}
            except Exception as e:
                raise FileSystemError(f"Cannot read JSON file {file_path}: {e}")
        
        return await loop.run_in_executor(None, _read_json_partial)
        
    except Exception as e:
        raise FileSystemError(f"Error reading JSON metadata from {file_path}: {e}")


async def extract_ast_metadata(file_path: Path) -> ASTMetadata:
    """Extract comprehensive metadata from AST file.
    
    Args:
        file_path: Path to AST JSON file
        
    Returns:
        ASTMetadata object with extracted information
        
    Raises:
        FileSystemError: If file cannot be processed
    """
    try:
        # Get basic file statistics
        file_size, modified_time, created_time = await get_file_stats(file_path)
        
        # Initialize metadata object
        metadata = ASTMetadata(
            file_size=file_size,
            modified_time=modified_time,
            created_time=created_time
        )
        
        # Read JSON content for additional metadata
        json_data = await read_json_metadata(file_path)
        
        # Extract AST-specific information
        if isinstance(json_data, dict):
            # Look for common AST fields
            if 'uniqueId' in json_data:
                metadata.unique_id = json_data['uniqueId']
            
            # Count nodes for complexity estimation
            if 'classes' in json_data:
                classes = json_data.get('classes', [])
                if isinstance(classes, list):
                    metadata.node_count = len(classes)
                    
                    # Calculate complexity based on classes and methods
                    total_methods = 0
                    for cls in classes:
                        if isinstance(cls, dict) and 'methods' in cls:
                            methods = cls.get('methods', [])
                            if isinstance(methods, list):
                                total_methods += len(methods)
                    
                    # Simple complexity score based on classes and methods
                    metadata.complexity_score = (len(classes) * 2) + total_methods
            
            # Store parser information
            parser_info = {}
            for key in ['parser_version', 'language', 'timestamp', 'tool']:
                if key in json_data:
                    parser_info[key] = json_data[key]
            metadata.parser_info = parser_info
            
            # Store structure information
            structure_info = {}
            for key in ['imports', 'classes', 'functions', 'modules', 'packages']:
                if key in json_data:
                    value = json_data[key]
                    if isinstance(value, list):
                        structure_info[f"{key}_count"] = len(value)
                    elif isinstance(value, dict):
                        structure_info[f"{key}_count"] = len(value)
            metadata.structure_info = structure_info
        
        return metadata
        
    except Exception as e:
        # Return basic metadata even if JSON parsing fails
        logger.warning(f"Failed to extract full metadata from {file_path}: {e}")
        try:
            file_size, modified_time, created_time = await get_file_stats(file_path)
            return ASTMetadata(
                file_size=file_size,
                modified_time=modified_time,
                created_time=created_time
            )
        except Exception:
            raise FileSystemError(f"Cannot extract any metadata from {file_path}: {e}")


async def count_lines_in_file(file_path: Path) -> int:
    """Count lines in a text file efficiently.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Number of lines in the file
        
    Raises:
        FileSystemError: If file cannot be read
    """
    try:
        loop = asyncio.get_event_loop()
        
        def _count_lines():
            try:
                # Use memory mapping for large files
                with open(file_path, 'rb') as f:
                    if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            return mm.count(b'\n')
                    else:
                        return sum(1 for _ in f)
            except Exception as e:
                raise FileSystemError(f"Cannot count lines in {file_path}: {e}")
        
        return await loop.run_in_executor(None, _count_lines)
        
    except Exception as e:
        raise FileSystemError(f"Error counting lines in {file_path}: {e}")


async def scan_directory_async(directory: Path, 
                             max_depth: int = 20,
                             follow_symlinks: bool = False) -> AsyncIterator[Path]:
    """Asynchronously scan directory for files.
    
    Args:
        directory: Directory to scan
        max_depth: Maximum depth to traverse
        follow_symlinks: Whether to follow symbolic links
        
    Yields:
        Path objects for each file found
        
    Raises:
        FileSystemError: If directory cannot be accessed
    """
    if not directory.exists():
        raise FileSystemError(f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise FileSystemError(f"Path is not a directory: {directory}")
    
    try:
        loop = asyncio.get_event_loop()
        
        def _scan_recursive(path: Path, current_depth: int = 0):
            """Recursive directory scanning generator."""
            if current_depth > max_depth:
                return
            
            try:
                for entry in path.iterdir():
                    try:
                        # Check if following symlinks based on parameter
                        if not follow_symlinks and entry.is_symlink():
                            continue
                            
                        if entry.is_file():
                            yield entry
                        elif entry.is_dir():
                            yield from _scan_recursive(entry, current_depth + 1)
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Cannot access {entry}: {e}")
                        continue
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot scan directory {path}: {e}")
        
        # Run the recursive scan in an executor to avoid blocking
        for file_path in await loop.run_in_executor(None, list, _scan_recursive(directory)):
            yield file_path
            # Yield control to allow other tasks to run
            await asyncio.sleep(0)
            
    except Exception as e:
        raise FileSystemError(f"Error scanning directory {directory}: {e}")


async def ensure_directory_exists(directory: Path) -> None:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
        
    Raises:
        FileSystemError: If directory cannot be created
    """
    try:
        if not directory.exists():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, directory.mkdir, True, True)  # parents=True, exist_ok=True
        elif not directory.is_dir():
            raise FileSystemError(f"Path exists but is not a directory: {directory}")
    except Exception as e:
        raise FileSystemError(f"Cannot ensure directory exists {directory}: {e}")


def get_file_extension_hints(file_path: Path) -> List[str]:
    """Get language hints from file extensions.
    
    Args:
        file_path: Path to analyze
        
    Returns:
        List of potential language hints based on filename
    """
    filename = file_path.name.lower()
    hints = []
    
    # Check for language indicators in filename
    language_indicators = {
        'java': ['java', 'javax'],
        'python': ['py', 'python'],
        'javascript': ['js', 'javascript', 'node'],
        'typescript': ['ts', 'typescript'],
        'c': ['.c'],
        'cpp': ['cpp', 'cxx', 'c++'],
        'csharp': ['cs', 'csharp', 'dotnet'],
        'go': ['go', 'golang'],
        'rust': ['rs', 'rust'],
        'kotlin': ['kt', 'kotlin'],
        'scala': ['scala']
    }
    
    for language, indicators in language_indicators.items():
        for indicator in indicators:
            if indicator in filename:
                hints.append(language)
                break
    
    return hints


def normalize_path(path: Path, base_path: Optional[Path] = None) -> Path:
    """Normalize path for consistent handling across platforms.
    
    Args:
        path: Path to normalize
        base_path: Optional base path for relative conversion
        
    Returns:
        Normalized path
    """
    # Resolve path to handle symlinks and relative components
    normalized = path.resolve()
    
    # Convert to relative if base_path provided
    if base_path:
        try:
            normalized = normalized.relative_to(base_path.resolve())
        except ValueError:
            # If path is not relative to base_path, keep absolute
            pass
    
    return normalized


async def batch_file_stats(file_paths: List[Path], 
                          max_concurrent: int = 50) -> List[Tuple[Path, Optional[ASTMetadata]]]:
    """Get file statistics for multiple files concurrently.
    
    Args:
        file_paths: List of file paths to process
        max_concurrent: Maximum number of concurrent operations
        
    Returns:
        List of tuples (path, metadata) where metadata may be None if failed
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _get_metadata(path: Path) -> Tuple[Path, Optional[ASTMetadata]]:
        async with semaphore:
            try:
                metadata = await extract_ast_metadata(path)
                return (path, metadata)
            except Exception as e:
                logger.warning(f"Failed to get metadata for {path}: {e}")
                return (path, None)
    
    tasks = [_get_metadata(path) for path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and return valid results
    valid_results = []
    for result in results:
        if isinstance(result, tuple):
            valid_results.append(result)
        else:
            logger.error(f"Unexpected error in batch processing: {result}")
    
    return valid_results
