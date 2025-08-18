"""AST file discovery and extraction service.

This module provides the main file discovery functionality for the AST extraction phase,
including language detection, file filtering, and metadata extraction.
"""

import asyncio
import fnmatch
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, AsyncIterator, Callable, Any
import logging

from snake_pipe.extract.models import (
    ASTFile, ASTMetadata, LanguageInfo, LanguageType, FilterConfig,
    ValidationResult, DiscoveryResult, DiscoveryProgress, FileStatus,
    DirectoryIndex
)
from snake_pipe.config.extract_config import ExtractConfig, LanguageDetectionConfig, FILENAME_PATTERNS
from snake_pipe.utils.file_utils import (
    scan_directory_async, get_file_stats, is_file_json, read_json_metadata,
    extract_ast_metadata, get_file_extension_hints, normalize_path,
    FileSystemError
)


logger = logging.getLogger(__name__)


class DiscoveryError(Exception):
    """Base exception for file discovery operations."""
    pass


class LanguageDetectionError(DiscoveryError):
    """Raised when language detection fails."""
    pass


class FilterValidationError(DiscoveryError):
    """Raised when filter configuration is invalid."""
    pass


class LanguageDetector:
    """Service for detecting programming languages from AST files."""
    
    def __init__(self, config: LanguageDetectionConfig):
        self.config = config
        self._content_patterns_compiled = self._compile_content_patterns()
    
    def _compile_content_patterns(self) -> Dict[LanguageType, List[re.Pattern]]:
        """Compile content detection patterns for performance."""
        compiled = {}
        for language, patterns in self.config.content_patterns.items():
            compiled[language] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled
    
    async def detect_language(self, file_path: Path) -> LanguageInfo:
        """Detect programming language from AST file.
        
        Args:
            file_path: Path to AST JSON file
            
        Returns:
            LanguageInfo with detected language and confidence
            
        Raises:
            LanguageDetectionError: If detection fails
        """
        try:
            # Strategy 1: Filename pattern detection
            filename_result = await self._detect_by_filename(file_path)
            if filename_result.confidence >= self.config.min_confidence_filename:
                return filename_result
            
            # Strategy 2: Content-based detection
            if self.config.enable_content_detection:
                content_result = await self._detect_by_content(file_path)
                if content_result.confidence >= self.config.min_confidence_content:
                    return content_result
                
                # Combine filename and content scores
                if filename_result.language == content_result.language:
                    combined_confidence = min(1.0, filename_result.confidence + content_result.confidence)
                    return LanguageInfo(
                        language=filename_result.language,
                        confidence=combined_confidence,
                        detection_method="filename_content_combined"
                    )
            
            # Strategy 3: Directory structure hints
            if self.config.enable_directory_detection:
                directory_result = await self._detect_by_directory(file_path)
                if directory_result.confidence >= self.config.min_confidence_directory:
                    return directory_result
            
            # Return best guess or unknown
            best_result = max([filename_result, content_result, directory_result], 
                            key=lambda x: x.confidence)
            
            if best_result.confidence > 0.3:  # Minimum threshold for any detection
                return best_result
            
            return LanguageInfo.unknown()
            
        except Exception as e:
            logger.warning(f"Language detection failed for {file_path}: {e}")
            return LanguageInfo.unknown()
    
    async def _detect_by_filename(self, file_path: Path) -> LanguageInfo:
        """Detect language based on filename patterns."""
        filename = file_path.name.lower()
        
        for language, patterns in FILENAME_PATTERNS.items():
            for pattern in patterns:
                if pattern.match(filename):
                    return LanguageInfo(
                        language=language,
                        confidence=0.9,
                        detection_method="filename_pattern"
                    )
        
        # Check file extension hints
        hints = get_file_extension_hints(file_path)
        if hints:
            try:
                detected_language = LanguageType(hints[0])
                return LanguageInfo(
                    language=detected_language,
                    confidence=0.7,
                    detection_method="filename_extension"
                )
            except ValueError:
                pass
        
        return LanguageInfo(
            language=LanguageType.UNKNOWN,
            confidence=0.0,
            detection_method="filename_none"
        )
    
    async def _detect_by_content(self, file_path: Path) -> LanguageInfo:
        """Detect language based on JSON content analysis."""
        try:
            # Read JSON metadata for content analysis
            json_data = await read_json_metadata(file_path, max_size=50 * 1024)  # 50KB
            
            if not isinstance(json_data, dict):
                return LanguageInfo.unknown()
            
            # Convert to string for pattern matching
            content_str = str(json_data).lower()
            
            # Score each language based on content patterns
            language_scores = {}
            for language, patterns in self._content_patterns_compiled.items():
                score = 0
                for pattern in patterns:
                    matches = len(pattern.findall(content_str))
                    score += matches * 0.1  # Each match adds 0.1 to confidence
                
                if score > 0:
                    language_scores[language] = min(1.0, score)
            
            if language_scores:
                best_language = max(language_scores.items(), key=lambda x: x[1])
                return LanguageInfo(
                    language=best_language[0],
                    confidence=best_language[1],
                    detection_method="content_analysis"
                )
            
            return LanguageInfo.unknown()
            
        except Exception as e:
            logger.debug(f"Content-based detection failed for {file_path}: {e}")
            return LanguageInfo.unknown()
    
    async def _detect_by_directory(self, file_path: Path) -> LanguageInfo:
        """Detect language based on directory structure."""
        path_str = str(file_path).lower()
        
        language_scores = {}
        for language, hints in self.config.directory_hints.items():
            score = 0
            for hint in hints:
                if hint in path_str:
                    score += 0.2  # Each directory hint adds 0.2 to confidence
            
            if score > 0:
                language_scores[language] = min(1.0, score)
        
        if language_scores:
            best_language = max(language_scores.items(), key=lambda x: x[1])
            return LanguageInfo(
                language=best_language[0],
                confidence=best_language[1] * 0.8,  # Directory hints are less reliable
                detection_method="directory_structure"
            )
        
        return LanguageInfo.unknown()


class FileFilter:
    """Service for filtering files based on configuration."""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self._include_patterns = [self._glob_to_regex(p) for p in config.include_patterns]
        self._exclude_patterns = [self._glob_to_regex(p) for p in config.exclude_patterns]
    
    def _glob_to_regex(self, pattern: str) -> re.Pattern:
        """Convert glob pattern to compiled regex."""
        regex_pattern = fnmatch.translate(pattern)
        return re.compile(regex_pattern, re.IGNORECASE)
    
    async def should_include_file(self, file_path: Path, depth: int = 0) -> bool:
        """Check if file should be included based on filters.
        
        Args:
            file_path: Path to check
            depth: Current directory depth
            
        Returns:
            True if file should be included, False otherwise
        """
        try:
            # Check depth limit
            if depth > self.config.max_depth:
                return False
            
            # Check if it's a JSON file
            if not await is_file_json(file_path):
                return False
            
            # Get file stats
            file_size, _, _ = await get_file_stats(file_path)
            
            # Check file size limits
            if file_size < self.config.min_file_size or file_size > self.config.max_file_size:
                return False
            
            filename = file_path.name
            
            # Check include patterns
            if self._include_patterns:
                if not any(pattern.match(filename) for pattern in self._include_patterns):
                    return False
            
            # Check exclude patterns
            if any(pattern.match(filename) for pattern in self._exclude_patterns):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking file filter for {file_path}: {e}")
            return False


class DirectoryIndexer:
    """Service for creating and managing directory indexes."""
    
    def __init__(self):
        self._cache: Dict[Path, DirectoryIndex] = {}
    
    async def build_index(self, directory: Path, cache_duration: int = 3600) -> DirectoryIndex:
        """Build directory index with file statistics.
        
        Args:
            directory: Directory to index
            cache_duration: Cache duration in seconds
            
        Returns:
            DirectoryIndex with directory statistics
        """
        # Check cache first
        if directory in self._cache:
            cached_index = self._cache[directory]
            if not cached_index.is_expired:
                return cached_index
        
        # Build new index
        file_count = 0
        directory_count = 0
        total_size = 0
        languages_detected: Set[LanguageType] = set()
        
        try:
            async for file_path in scan_directory_async(directory):
                if file_path.is_file():
                    file_count += 1
                    try:
                        file_size, _, _ = await get_file_stats(file_path)
                        total_size += file_size
                        
                        # Quick language detection for indexing
                        hints = get_file_extension_hints(file_path)
                        for hint in hints:
                            try:
                                lang = LanguageType(hint)
                                languages_detected.add(lang)
                            except ValueError:
                                continue
                                
                    except Exception:
                        continue
                elif file_path.is_dir():
                    directory_count += 1
        
        except Exception as e:
            logger.warning(f"Error building directory index for {directory}: {e}")
        
        # Create index
        index = DirectoryIndex(
            base_path=directory,
            file_count=file_count,
            directory_count=directory_count,
            total_size=total_size,
            languages_detected=languages_detected,
            cache_duration=cache_duration
        )
        
        # Cache the index
        self._cache[directory] = index
        
        return index
    
    def clear_cache(self):
        """Clear the directory index cache."""
        self._cache.clear()


class ASTFileDiscovery:
    """Main service for discovering AST files in directory structures."""
    
    def __init__(self, config: ExtractConfig):
        self.config = config
        self.language_detector = LanguageDetector(config.language_detection)
        self.file_filter = FileFilter(config.filter_config)
        self.directory_indexer = DirectoryIndexer()
        self.progress_callbacks: List[Callable[[DiscoveryProgress], None]] = []
    
    def add_progress_callback(self, callback: Callable[[DiscoveryProgress], None]):
        """Add a progress tracking callback."""
        self.progress_callbacks.append(callback)
    
    async def discover_ast_files(self, source_path: Path) -> DiscoveryResult:
        """Discover all AST files in the given directory structure.
        
        Args:
            source_path: Root directory to search
            
        Returns:
            DiscoveryResult with discovered files and statistics
            
        Raises:
            DiscoveryError: If discovery fails
        """
        start_time = time.time()
        discovered_files = []
        total_files_found = 0
        total_errors = 0
        
        # Initialize progress tracking
        progress = DiscoveryProgress()
        
        try:
            # Build directory index if caching is enabled
            directory_index = None
            if self.config.performance.enable_directory_cache:
                directory_index = await self.directory_indexer.build_index(
                    source_path, 
                    self.config.performance.cache_duration_seconds
                )
                progress.total_files = directory_index.file_count
            
            # Track progress
            if self.config.enable_progress_tracking:
                for callback in self.progress_callbacks:
                    callback(progress)
            
            # Discover files with concurrency control
            semaphore = asyncio.Semaphore(self.config.performance.max_concurrent_files)
            
            async def process_file(file_path: Path, depth: int) -> Optional[ASTFile]:
                async with semaphore:
                    nonlocal total_files_found, total_errors
                    total_files_found += 1
                    progress.processed_files += 1
                    
                    try:
                        # Check if file should be included
                        if not await self.file_filter.should_include_file(file_path, depth):
                            return None
                        
                        # Create AST file object
                        ast_file = await self._create_ast_file(file_path, source_path)
                        
                        # Update progress
                        if self.config.enable_progress_tracking:
                            for callback in self.progress_callbacks:
                                callback(progress)
                        
                        return ast_file
                        
                    except Exception as e:
                        total_errors += 1
                        logger.warning(f"Error processing file {file_path}: {e}")
                        
                        if not self.config.continue_on_error:
                            raise DiscoveryError(f"File processing failed: {e}")
                        
                        return None
            
            # Process files in batches for memory efficiency
            batch_size = self.config.performance.batch_size
            current_batch = []
            
            async for file_path in scan_directory_async(
                source_path, 
                self.config.filter_config.max_depth,
                self.config.filter_config.follow_symlinks
            ):
                if file_path.is_file():
                    # Calculate depth
                    try:
                        relative_path = file_path.relative_to(source_path)
                        depth = len(relative_path.parts) - 1
                    except ValueError:
                        depth = 0
                    
                    current_batch.append((file_path, depth))
                    
                    # Process batch when full
                    if len(current_batch) >= batch_size:
                        batch_results = await asyncio.gather(
                            *[process_file(fp, d) for fp, d in current_batch],
                            return_exceptions=True
                        )
                        
                        for result in batch_results:
                            if isinstance(result, ASTFile):
                                discovered_files.append(result)
                            elif isinstance(result, Exception):
                                total_errors += 1
                                if not self.config.continue_on_error:
                                    raise DiscoveryError(f"Batch processing failed: {result}")
                        
                        current_batch.clear()
            
            # Process remaining files in batch
            if current_batch:
                batch_results = await asyncio.gather(
                    *[process_file(fp, d) for fp, d in current_batch],
                    return_exceptions=True
                )
                
                for result in batch_results:
                    if isinstance(result, ASTFile):
                        discovered_files.append(result)
                    elif isinstance(result, Exception):
                        total_errors += 1
            
            processing_time = time.time() - start_time
            
            result = DiscoveryResult(
                files=discovered_files,
                total_files_found=total_files_found,
                total_files_processed=len(discovered_files),
                total_errors=total_errors,
                processing_time=processing_time,
                directory_index=directory_index
            )
            
            logger.info(f"Discovery completed: {len(discovered_files)} files discovered in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Discovery failed after {processing_time:.2f}s: {e}")
            raise DiscoveryError(f"File discovery failed: {e}")
    
    async def _create_ast_file(self, file_path: Path, base_path: Path) -> ASTFile:
        """Create ASTFile object with full metadata.
        
        Args:
            file_path: Path to the AST file
            base_path: Base directory for relative path calculation
            
        Returns:
            ASTFile object with complete metadata
            
        Raises:
            DiscoveryError: If file processing fails
        """
        try:
            # Detect language
            language_info = await self.language_detector.detect_language(file_path)
            
            # Extract metadata
            metadata = await extract_ast_metadata(file_path)
            
            # Normalize path relative to base
            relative_path = normalize_path(file_path, base_path)
            
            # Create AST file object
            ast_file = ASTFile(
                path=relative_path,
                language_info=language_info,
                metadata=metadata,
                status=FileStatus.DISCOVERED
            )
            
            # Apply language filtering if configured
            if self.config.filter_config.languages:
                if language_info.language not in self.config.filter_config.languages:
                    ast_file.add_error(f"Language {language_info.language} not in target languages")
            
            return ast_file
            
        except Exception as e:
            raise DiscoveryError(f"Failed to create AST file for {file_path}: {e}")


def validate_discovery_config(config: FilterConfig) -> ValidationResult:
    """Validate file discovery configuration.
    
    Args:
        config: FilterConfig to validate
        
    Returns:
        ValidationResult with validation status and errors
    """
    result = ValidationResult(is_valid=True)
    
    try:
        # Validate patterns
        for pattern in config.include_patterns:
            try:
                re.compile(fnmatch.translate(pattern))
            except re.error as e:
                result.add_error(f"Invalid include pattern '{pattern}': {e}")
        
        for pattern in config.exclude_patterns:
            try:
                re.compile(fnmatch.translate(pattern))
            except re.error as e:
                result.add_error(f"Invalid exclude pattern '{pattern}': {e}")
        
        # Validate numeric limits
        if config.max_depth < 1:
            result.add_error("max_depth must be at least 1")
        
        if config.min_file_size < 0:
            result.add_error("min_file_size cannot be negative")
        
        if config.max_file_size < config.min_file_size:
            result.add_error("max_file_size must be greater than min_file_size")
        
        # Validate languages
        for language in config.languages:
            if not isinstance(language, LanguageType):
                result.add_error(f"Invalid language type: {language}")
        
        # Create normalized config if valid
        if result.is_valid:
            result.normalized_config = config
        
    except Exception as e:
        result.add_error(f"Configuration validation failed: {e}")
    
    return result


# Factory function for creating discovery service
def create_discovery_service(config: ExtractConfig) -> ASTFileDiscovery:
    """Factory function to create configured ASTFileDiscovery service.
    
    Args:
        config: ExtractConfig with discovery settings
        
    Returns:
        Configured ASTFileDiscovery instance
        
    Raises:
        FilterValidationError: If configuration is invalid
    """
    # Validate configuration
    validation_result = validate_discovery_config(config.filter_config)
    if not validation_result.is_valid:
        raise FilterValidationError(f"Invalid configuration: {validation_result.errors}")
    
    return ASTFileDiscovery(config)
