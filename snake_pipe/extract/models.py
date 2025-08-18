"""Domain models for AST file extraction and discovery.

This module contains the core data models used in the AST extraction phase,
including file metadata, language detection, and discovery configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import json


class LanguageType(Enum):
    """Supported programming languages for AST processing."""
    JAVA = "java"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    KOTLIN = "kotlin"
    SCALA = "scala"
    UNKNOWN = "unknown"


class FileStatus(Enum):
    """Status of AST file processing."""
    DISCOVERED = auto()
    VALIDATED = auto()
    PROCESSED = auto()
    ERROR = auto()
    QUARANTINED = auto()


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    LENIENT = "lenient"
    SKIP = "skip"


@dataclass
class LanguageInfo:
    """Information about detected programming language."""
    language: LanguageType
    confidence: float  # 0.0 to 1.0
    detection_method: str  # e.g., "filename_pattern", "ast_content", "directory_structure"
    parser_version: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unknown(cls) -> "LanguageInfo":
        """Create an unknown language info instance."""
        return cls(
            language=LanguageType.UNKNOWN,
            confidence=0.0,
            detection_method="none"
        )


@dataclass
class ASTMetadata:
    """Comprehensive metadata extracted from AST files."""
    file_size: int  # bytes
    modified_time: datetime
    created_time: Optional[datetime] = None
    line_count: Optional[int] = None
    node_count: Optional[int] = None
    complexity_score: Optional[float] = None
    unique_id: Optional[str] = None
    parser_info: Dict[str, Any] = field(default_factory=dict)
    structure_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_large_file(self) -> bool:
        """Check if file is considered large (>10MB)."""
        return self.file_size > 10 * 1024 * 1024


@dataclass
class ASTFile:
    """Core domain entity representing a discovered AST file."""
    path: Path
    language_info: LanguageInfo
    metadata: ASTMetadata
    status: FileStatus = FileStatus.DISCOVERED
    errors: List[str] = field(default_factory=list)
    discovery_time: datetime = field(default_factory=datetime.now)
    
    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return self.path.name
    
    @property
    def relative_path(self) -> Path:
        """Get relative path from a base directory."""
        # This will be set during discovery relative to the source directory
        return self.path
    
    @property
    def is_valid(self) -> bool:
        """Check if file is in a valid state for processing."""
        return self.status != FileStatus.ERROR and not self.errors
    
    def add_error(self, error: str) -> None:
        """Add an error message and update status."""
        self.errors.append(error)
        self.status = FileStatus.ERROR
    
    def mark_validated(self) -> None:
        """Mark file as successfully validated."""
        if self.status == FileStatus.DISCOVERED:
            self.status = FileStatus.VALIDATED


@dataclass
class FilterConfig:
    """Configuration for AST file discovery filtering."""
    include_patterns: List[str] = field(default_factory=lambda: ["*.json"])
    exclude_patterns: List[str] = field(default_factory=list)
    max_depth: int = 20
    min_file_size: int = 1  # bytes
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    languages: Set[LanguageType] = field(default_factory=set)
    follow_symlinks: bool = False
    validation_level: ValidationLevel = ValidationLevel.LENIENT
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.min_file_size < 0:
            raise ValueError("min_file_size cannot be negative")
        if self.max_file_size < self.min_file_size:
            raise ValueError("max_file_size must be greater than min_file_size")


@dataclass
class DirectoryIndex:
    """Optimized index of directory structure for fast lookups."""
    base_path: Path
    file_count: int
    directory_count: int
    total_size: int
    languages_detected: Set[LanguageType]
    created_time: datetime = field(default_factory=datetime.now)
    cache_duration: int = 3600  # seconds
    
    @property
    def is_expired(self) -> bool:
        """Check if the directory index has expired."""
        age = (datetime.now() - self.created_time).total_seconds()
        return age > self.cache_duration


@dataclass
class ValidationResult:
    """Result of configuration or file validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalized_config: Optional[FilterConfig] = None
    
    def add_error(self, error: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(warning)


@dataclass
class DiscoveryResult:
    """Result of file discovery operation."""
    files: List[ASTFile]
    total_files_found: int
    total_files_processed: int
    total_errors: int
    processing_time: float  # seconds
    directory_index: Optional[DirectoryIndex] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of file discovery."""
        if self.total_files_found == 0:
            return 1.0
        return (self.total_files_processed - self.total_errors) / self.total_files_found
    
    @property
    def languages_found(self) -> Set[LanguageType]:
        """Get all languages found during discovery."""
        return {file.language_info.language for file in self.files}


@dataclass
class DiscoveryProgress:
    """Progress tracking for file discovery operations."""
    total_directories: int = 0
    processed_directories: int = 0
    total_files: int = 0
    processed_files: int = 0
    current_directory: Optional[Path] = None
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def files_per_second(self) -> float:
        """Calculate processing rate."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.processed_files / elapsed


# Type aliases for better code readability
ASTFileList = List[ASTFile]
LanguageDetectionStrategy = str
FilePathPattern = str