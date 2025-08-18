"""Extract phase configuration management.

This module provides configuration classes and utilities for the AST extraction phase,
including file discovery, language detection, and processing settings.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Set

from snake_pipe.extract.models import FilterConfig, LanguageType, ValidationLevel


@dataclass
class LanguageDetectionConfig:
    """Configuration for language detection strategies."""
    
    # Filename patterns for language detection
    filename_patterns: Dict[LanguageType, List[str]] = field(default_factory=lambda: {
        LanguageType.JAVA: ["*.java.json", "*_java.json", "*.ast.json"],
        LanguageType.PYTHON: ["*.py.json", "*_python.json", "*.python.ast.json"],
        LanguageType.JAVASCRIPT: ["*.js.json", "*_javascript.json", "*.javascript.ast.json"],
        LanguageType.TYPESCRIPT: ["*.ts.json", "*_typescript.json", "*.typescript.ast.json"],
        LanguageType.C: ["*.c.json", "*_c.json", "*.c.ast.json"],
        LanguageType.CPP: ["*.cpp.json", "*.cxx.json", "*_cpp.json", "*.cpp.ast.json"],
        LanguageType.CSHARP: ["*.cs.json", "*_csharp.json", "*.csharp.ast.json"],
        LanguageType.GO: ["*.go.json", "*_go.json", "*.go.ast.json"],
        LanguageType.RUST: ["*.rs.json", "*_rust.json", "*.rust.ast.json"],
        LanguageType.KOTLIN: ["*.kt.json", "*_kotlin.json", "*.kotlin.ast.json"],
        LanguageType.SCALA: ["*.scala.json", "*_scala.json", "*.scala.ast.json"]
    })
    
    # Content-based detection patterns (JSON field patterns)
    content_patterns: Dict[LanguageType, List[str]] = field(default_factory=lambda: {
        LanguageType.JAVA: ["classes", "imports", "qualifiedName", "java"],
        LanguageType.PYTHON: ["modules", "functions", "classes", "python"],
        LanguageType.JAVASCRIPT: ["functions", "exports", "requires", "javascript"],
        LanguageType.TYPESCRIPT: ["interfaces", "types", "typescript"],
        LanguageType.C: ["functions", "includes", "structs"],
        LanguageType.CPP: ["classes", "namespaces", "includes", "cpp"],
        LanguageType.CSHARP: ["classes", "namespaces", "using", "csharp"],
        LanguageType.GO: ["packages", "imports", "functions", "golang"],
        LanguageType.RUST: ["crates", "modules", "traits", "rust"],
        LanguageType.KOTLIN: ["classes", "packages", "kotlin"],
        LanguageType.SCALA: ["classes", "objects", "packages", "scala"]
    })
    
    # Directory structure hints
    directory_hints: Dict[LanguageType, List[str]] = field(default_factory=lambda: {
        LanguageType.JAVA: ["src/main/java", "src/test/java", "java", "javax"],
        LanguageType.PYTHON: ["src/python", "python", "py"],
        LanguageType.JAVASCRIPT: ["src/js", "javascript", "js", "node_modules"],
        LanguageType.TYPESCRIPT: ["src/ts", "typescript", "ts"],
        LanguageType.C: ["src/c", "include"],
        LanguageType.CPP: ["src/cpp", "src/cxx", "include"],
        LanguageType.CSHARP: ["src/cs", "csharp"],
        LanguageType.GO: ["src/go", "golang"],
        LanguageType.RUST: ["src/rust", "rs"],
        LanguageType.KOTLIN: ["src/kotlin", "kt"],
        LanguageType.SCALA: ["src/scala"]
    })
    
    # Minimum confidence thresholds
    min_confidence_filename: float = 0.8
    min_confidence_content: float = 0.6
    min_confidence_directory: float = 0.4
    
    # Enable/disable detection strategies
    enable_filename_detection: bool = True
    enable_content_detection: bool = True
    enable_directory_detection: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Concurrency settings
    max_concurrent_files: int = 50
    max_concurrent_directories: int = 10
    
    # Batch processing
    batch_size: int = 100
    
    # Memory management
    max_memory_usage_mb: int = 500
    stream_large_files: bool = True
    large_file_threshold_mb: int = 10
    
    # Caching
    enable_directory_cache: bool = True
    cache_duration_seconds: int = 3600
    max_cache_entries: int = 1000
    
    # Timeouts
    file_read_timeout_seconds: float = 30.0
    directory_scan_timeout_seconds: float = 300.0


@dataclass
class ExtractConfig:
    """Main configuration for AST extraction phase."""
    
    # Core settings
    source_paths: List[Path] = field(default_factory=list)
    output_path: Optional[Path] = None
    
    # Filtering
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    
    # Language detection
    language_detection: LanguageDetectionConfig = field(default_factory=LanguageDetectionConfig)
    
    # Performance
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Validation
    validation_level: ValidationLevel = ValidationLevel.LENIENT
    strict_json_validation: bool = False
    
    # Error handling
    continue_on_error: bool = True
    max_errors_per_directory: int = 10
    quarantine_invalid_files: bool = True
    
    # Logging and monitoring
    enable_progress_tracking: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.source_paths:
            raise ValueError("At least one source path must be specified")
        
        # Validate source paths exist
        for path in self.source_paths:
            if not isinstance(path, Path):
                raise TypeError(f"source_paths must contain Path objects, got {type(path)}")
            if not path.exists():
                raise ValueError(f"Source path does not exist: {path}")
            if not path.is_dir():
                raise ValueError(f"Source path is not a directory: {path}")


class ExtractConfigBuilder:
    """Builder pattern for creating ExtractConfig instances."""
    
    def __init__(self):
        self._config = ExtractConfig(source_paths=[Path("/tmp")])  # Temporary path, will be overridden
    
    def with_source_paths(self, *paths: Path) -> "ExtractConfigBuilder":
        """Set source paths for file discovery."""
        self._config.source_paths = list(paths)
        return self
    
    def with_output_path(self, path: Path) -> "ExtractConfigBuilder":
        """Set output path for processed files."""
        self._config.output_path = path
        return self
    
    def with_filter_patterns(self, include: List[str], exclude: List[str] = None) -> "ExtractConfigBuilder":
        """Set file filtering patterns."""
        self._config.filter_config.include_patterns = include
        if exclude:
            self._config.filter_config.exclude_patterns = exclude
        return self
    
    def with_max_depth(self, depth: int) -> "ExtractConfigBuilder":
        """Set maximum directory traversal depth."""
        self._config.filter_config.max_depth = depth
        return self
    
    def with_file_size_limits(self, min_size: int, max_size: int) -> "ExtractConfigBuilder":
        """Set file size filtering limits."""
        self._config.filter_config.min_file_size = min_size
        self._config.filter_config.max_file_size = max_size
        return self
    
    def with_languages(self, *languages: LanguageType) -> "ExtractConfigBuilder":
        """Set target languages for discovery."""
        self._config.filter_config.languages = set(languages)
        return self
    
    def with_validation_level(self, level: ValidationLevel) -> "ExtractConfigBuilder":
        """Set validation strictness level."""
        self._config.validation_level = level
        self._config.filter_config.validation_level = level
        return self
    
    def with_performance_settings(self, 
                                 max_concurrent_files: int = None,
                                 batch_size: int = None,
                                 max_memory_mb: int = None) -> "ExtractConfigBuilder":
        """Set performance optimization settings."""
        if max_concurrent_files is not None:
            self._config.performance.max_concurrent_files = max_concurrent_files
        if batch_size is not None:
            self._config.performance.batch_size = batch_size
        if max_memory_mb is not None:
            self._config.performance.max_memory_usage_mb = max_memory_mb
        return self
    
    def enable_caching(self, duration_seconds: int = 3600) -> "ExtractConfigBuilder":
        """Enable directory caching with specified duration."""
        self._config.performance.enable_directory_cache = True
        self._config.performance.cache_duration_seconds = duration_seconds
        return self
    
    def disable_caching(self) -> "ExtractConfigBuilder":
        """Disable directory caching."""
        self._config.performance.enable_directory_cache = False
        return self
    
    def strict_validation(self) -> "ExtractConfigBuilder":
        """Enable strict validation mode."""
        self._config.validation_level = ValidationLevel.STRICT
        self._config.filter_config.validation_level = ValidationLevel.STRICT
        self._config.strict_json_validation = True
        return self
    
    def lenient_validation(self) -> "ExtractConfigBuilder":
        """Enable lenient validation mode (default)."""
        self._config.validation_level = ValidationLevel.LENIENT
        self._config.filter_config.validation_level = ValidationLevel.LENIENT
        self._config.strict_json_validation = False
        return self
    
    def build(self) -> ExtractConfig:
        """Build and return the configuration."""
        # Create a copy to prevent modification after building
        config = ExtractConfig(
            source_paths=self._config.source_paths.copy(),
            output_path=self._config.output_path,
            filter_config=self._config.filter_config,
            language_detection=self._config.language_detection,
            performance=self._config.performance,
            validation_level=self._config.validation_level,
            strict_json_validation=self._config.strict_json_validation,
            continue_on_error=self._config.continue_on_error,
            max_errors_per_directory=self._config.max_errors_per_directory,
            quarantine_invalid_files=self._config.quarantine_invalid_files,
            enable_progress_tracking=self._config.enable_progress_tracking,
            log_level=self._config.log_level,
            enable_metrics=self._config.enable_metrics
        )
        
        # Validate the built configuration
        config.__post_init__()
        return config


def create_default_config(source_path: Path) -> ExtractConfig:
    """Create a default configuration for common use cases."""
    return (ExtractConfigBuilder()
            .with_source_paths(source_path)
            .with_filter_patterns(["*.json", "*.ast.json"])
            .with_max_depth(20)
            .with_file_size_limits(1, 100 * 1024 * 1024)  # 1 byte to 100MB
            .lenient_validation()
            .enable_caching()
            .build())


def create_high_performance_config(source_path: Path) -> ExtractConfig:
    """Create a configuration optimized for high-throughput processing."""
    return (ExtractConfigBuilder()
            .with_source_paths(source_path)
            .with_filter_patterns(["*.json", "*.ast.json"])
            .with_performance_settings(
                max_concurrent_files=100,
                batch_size=200,
                max_memory_mb=1000
            )
            .lenient_validation()
            .enable_caching(duration_seconds=7200)  # 2 hours
            .build())


def create_strict_config(source_path: Path) -> ExtractConfig:
    """Create a configuration with strict validation for production use."""
    return (ExtractConfigBuilder()
            .with_source_paths(source_path)
            .with_filter_patterns(["*.json", "*.ast.json"])
            .strict_validation()
            .with_performance_settings(max_concurrent_files=25)  # Conservative for stability
            .enable_caching()
            .build())


# Compiled regex patterns for performance
FILENAME_PATTERNS: Dict[LanguageType, List[Pattern]] = {}

def _compile_filename_patterns():
    """Compile filename patterns for performance."""
    global FILENAME_PATTERNS
    config = LanguageDetectionConfig()
    
    for language, patterns in config.filename_patterns.items():
        compiled_patterns = []
        for pattern in patterns:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            compiled_patterns.append(re.compile(regex_pattern, re.IGNORECASE))
        FILENAME_PATTERNS[language] = compiled_patterns

# Initialize compiled patterns on module import
_compile_filename_patterns()
