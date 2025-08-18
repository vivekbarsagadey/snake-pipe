"""Configuration management for real-time file watcher.

This module provides comprehensive configuration management for the file watcher
service with environment-specific presets and validation.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum, auto

from snake_pipe.extract.models import LanguageType
from snake_pipe.config.extract_config import ExtractConfig

logger = logging.getLogger(__name__)


class WatcherMode(Enum):
    """Operating modes for file watcher."""
    DEVELOPMENT = "development"    # Optimized for development workflows
    PRODUCTION = "production"      # Optimized for production stability
    TESTING = "testing"           # Optimized for testing scenarios
    HIGH_PERFORMANCE = "high_performance"  # Maximum throughput


class EventFilterLevel(Enum):
    """Event filtering strictness levels."""
    STRICT = auto()     # Only AST files, maximum filtering
    NORMAL = auto()     # Balanced filtering
    PERMISSIVE = auto() # Minimal filtering, more events


@dataclass
class PathConfig:
    """Configuration for a single watch path."""
    path: Path
    recursive: bool = True
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    priority: int = 1  # 1=high, 2=medium, 3=low
    
    def __post_init__(self) -> None:
        """Validate path configuration."""
        if not self.path.exists():
            logger.warning(f"Watch path does not exist: {self.path}")
        
        if not self.include_patterns:
            self.include_patterns = ['*.json', '**/*.ast.json']


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    max_concurrent_watches: int = 10000
    max_memory_mb: int = 500
    max_cpu_percent: float = 10.0
    queue_size: int = 50000
    batch_size: int = 100
    batch_timeout: float = 2.0
    worker_threads: int = 4
    
    def __post_init__(self) -> None:
        """Validate performance configuration."""
        if self.max_concurrent_watches < 1:
            self.max_concurrent_watches = 1000
            
        if self.max_memory_mb < 10:
            self.max_memory_mb = 100
            
        if self.max_cpu_percent < 0.1:
            self.max_cpu_percent = 1.0


@dataclass
class FilterConfig:
    """Event filtering configuration."""
    level: EventFilterLevel = EventFilterLevel.NORMAL
    ast_file_patterns: List[str] = field(default_factory=lambda: [
        '*.json', '**/*.ast.json', '**/ast/*.json', '**/*.tree.json'
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '**/.git/**', '**/node_modules/**', '**/build/**', '**/dist/**',
        '**/__pycache__/**', '**/*.tmp', '**/*.swp', '**/*.log',
        '**/coverage/**', '**/.vscode/**', '**/.idea/**'
    ])
    min_file_size: int = 10  # Minimum file size in bytes
    max_file_size: int = 100 * 1024 * 1024  # Maximum file size (100MB)
    allowed_languages: List[LanguageType] = field(default_factory=lambda: [
        LanguageType.JAVA, LanguageType.PYTHON, LanguageType.JAVASCRIPT,
        LanguageType.TYPESCRIPT, LanguageType.C, LanguageType.CPP
    ])


@dataclass
class DebounceConfig:
    """Event debouncing configuration."""
    enable_debouncing: bool = True
    base_period: float = 0.5
    max_period: float = 5.0
    min_period: float = 0.1
    adaptive_factor: float = 0.2
    bulk_operation_threshold: int = 10  # Number of files changed to trigger bulk mode
    bulk_operation_period: float = 2.0  # Extended debounce for bulk operations


@dataclass
class MonitoringConfig:
    """Monitoring and health check configuration."""
    enable_health_checks: bool = True
    health_check_interval: float = 30.0
    enable_performance_monitoring: bool = True
    performance_sample_interval: float = 10.0
    log_statistics_interval: float = 60.0
    enable_metrics_export: bool = False
    metrics_port: int = 9090


@dataclass
class WatcherSettings:
    """Complete watcher configuration settings."""
    mode: WatcherMode = WatcherMode.DEVELOPMENT
    watch_paths: List[PathConfig] = field(default_factory=list)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    debouncing: DebounceConfig = field(default_factory=DebounceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Apply mode-specific optimizations."""
        self._apply_mode_optimizations()
        self._apply_environment_overrides()
    
    def _apply_mode_optimizations(self) -> None:
        """Apply optimizations based on watcher mode."""
        if self.mode == WatcherMode.DEVELOPMENT:
            # Optimize for quick feedback
            self.debouncing.base_period = 0.3
            self.performance.batch_size = 50
            self.performance.batch_timeout = 1.0
            self.filtering.level = EventFilterLevel.NORMAL
            
        elif self.mode == WatcherMode.PRODUCTION:
            # Optimize for stability and resource usage
            self.debouncing.base_period = 1.0
            self.performance.batch_size = 200
            self.performance.batch_timeout = 3.0
            self.filtering.level = EventFilterLevel.STRICT
            self.monitoring.enable_health_checks = True
            
        elif self.mode == WatcherMode.TESTING:
            # Optimize for test scenarios
            self.debouncing.base_period = 0.1
            self.performance.batch_size = 10
            self.performance.batch_timeout = 0.5
            self.filtering.level = EventFilterLevel.PERMISSIVE
            self.monitoring.enable_health_checks = False
            
        elif self.mode == WatcherMode.HIGH_PERFORMANCE:
            # Optimize for maximum throughput
            self.debouncing.base_period = 0.1
            self.performance.batch_size = 500
            self.performance.batch_timeout = 0.5
            self.performance.worker_threads = 8
            self.filtering.level = EventFilterLevel.NORMAL
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides."""
        env_name = os.getenv('SNAKE_PIPE_ENV', 'development').lower()
        
        if env_name in self.environment_overrides:
            overrides = self.environment_overrides[env_name]
            for key, value in overrides.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def add_watch_path(
        self, 
        path: Union[str, Path], 
        recursive: bool = True,
        priority: int = 1
    ) -> None:
        """Add a path to be watched."""
        path_obj = Path(path) if isinstance(path, str) else path
        
        # Check if path already exists
        for existing in self.watch_paths:
            if existing.path == path_obj:
                logger.warning(f"Path already being watched: {path_obj}")
                return
        
        path_config = PathConfig(
            path=path_obj,
            recursive=recursive,
            priority=priority
        )
        self.watch_paths.append(path_config)
        logger.info(f"Added watch path: {path_obj} (recursive={recursive})")
    
    def remove_watch_path(self, path: Union[str, Path]) -> bool:
        """Remove a path from being watched."""
        path_obj = Path(path) if isinstance(path, str) else path
        
        for i, existing in enumerate(self.watch_paths):
            if existing.path == path_obj:
                del self.watch_paths[i]
                logger.info(f"Removed watch path: {path_obj}")
                return True
        
        logger.warning(f"Path not found in watch list: {path_obj}")
        return False
    
    def get_all_patterns(self) -> Dict[str, List[str]]:
        """Get all file patterns from watch paths and filtering config."""
        include_patterns = set(self.filtering.ast_file_patterns)
        exclude_patterns = set(self.filtering.exclude_patterns)
        
        # Collect patterns from individual path configs
        for path_config in self.watch_paths:
            include_patterns.update(path_config.include_patterns)
            exclude_patterns.update(path_config.exclude_patterns)
        
        return {
            'include': list(include_patterns),
            'exclude': list(exclude_patterns)
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate watch paths
        if not self.watch_paths:
            issues.append("No watch paths configured")
        
        for path_config in self.watch_paths:
            if not path_config.path.exists():
                issues.append(f"Watch path does not exist: {path_config.path}")
            elif not path_config.path.is_dir():
                issues.append(f"Watch path is not a directory: {path_config.path}")
        
        # Validate performance settings
        if self.performance.max_concurrent_watches < 1:
            issues.append("max_concurrent_watches must be positive")
        
        if self.performance.batch_size < 1:
            issues.append("batch_size must be positive")
        
        # Validate debouncing settings
        if self.debouncing.base_period < 0:
            issues.append("debounce base_period must be non-negative")
        
        if self.debouncing.max_period < self.debouncing.base_period:
            issues.append("debounce max_period must be >= base_period")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'mode': self.mode.value,
            'watch_paths': [
                {
                    'path': str(pc.path),
                    'recursive': pc.recursive,
                    'include_patterns': pc.include_patterns,
                    'exclude_patterns': pc.exclude_patterns,
                    'priority': pc.priority
                }
                for pc in self.watch_paths
            ],
            'performance': {
                'max_concurrent_watches': self.performance.max_concurrent_watches,
                'max_memory_mb': self.performance.max_memory_mb,
                'max_cpu_percent': self.performance.max_cpu_percent,
                'queue_size': self.performance.queue_size,
                'batch_size': self.performance.batch_size,
                'batch_timeout': self.performance.batch_timeout,
                'worker_threads': self.performance.worker_threads
            },
            'filtering': {
                'level': self.filtering.level.name,
                'ast_file_patterns': self.filtering.ast_file_patterns,
                'exclude_patterns': self.filtering.exclude_patterns,
                'min_file_size': self.filtering.min_file_size,
                'max_file_size': self.filtering.max_file_size,
                'allowed_languages': [lang.value for lang in self.filtering.allowed_languages]
            },
            'debouncing': {
                'enable_debouncing': self.debouncing.enable_debouncing,
                'base_period': self.debouncing.base_period,
                'max_period': self.debouncing.max_period,
                'min_period': self.debouncing.min_period,
                'adaptive_factor': self.debouncing.adaptive_factor,
                'bulk_operation_threshold': self.debouncing.bulk_operation_threshold,
                'bulk_operation_period': self.debouncing.bulk_operation_period
            },
            'monitoring': {
                'enable_health_checks': self.monitoring.enable_health_checks,
                'health_check_interval': self.monitoring.health_check_interval,
                'enable_performance_monitoring': self.monitoring.enable_performance_monitoring,
                'performance_sample_interval': self.monitoring.performance_sample_interval,
                'log_statistics_interval': self.monitoring.log_statistics_interval,
                'enable_metrics_export': self.monitoring.enable_metrics_export,
                'metrics_port': self.monitoring.metrics_port
            }
        }


class WatcherConfigBuilder:
    """Builder pattern for creating watcher configurations."""
    
    def __init__(self, mode: WatcherMode = WatcherMode.DEVELOPMENT):
        self.settings = WatcherSettings(mode=mode)
    
    def add_path(
        self, 
        path: Union[str, Path], 
        recursive: bool = True,
        priority: int = 1
    ) -> 'WatcherConfigBuilder':
        """Add a watch path."""
        self.settings.add_watch_path(path, recursive, priority)
        return self
    
    def with_performance(
        self,
        batch_size: Optional[int] = None,
        batch_timeout: Optional[float] = None,
        max_memory_mb: Optional[int] = None
    ) -> 'WatcherConfigBuilder':
        """Configure performance settings."""
        if batch_size is not None:
            self.settings.performance.batch_size = batch_size
        if batch_timeout is not None:
            self.settings.performance.batch_timeout = batch_timeout
        if max_memory_mb is not None:
            self.settings.performance.max_memory_mb = max_memory_mb
        return self
    
    def with_debouncing(
        self,
        base_period: Optional[float] = None,
        enable: Optional[bool] = None
    ) -> 'WatcherConfigBuilder':
        """Configure debouncing settings."""
        if base_period is not None:
            self.settings.debouncing.base_period = base_period
        if enable is not None:
            self.settings.debouncing.enable_debouncing = enable
        return self
    
    def with_filtering(
        self,
        level: Optional[EventFilterLevel] = None,
        patterns: Optional[List[str]] = None
    ) -> 'WatcherConfigBuilder':
        """Configure filtering settings."""
        if level is not None:
            self.settings.filtering.level = level
        if patterns is not None:
            self.settings.filtering.ast_file_patterns = patterns
        return self
    
    def build(self) -> WatcherSettings:
        """Build the final configuration."""
        issues = self.settings.validate_configuration()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
        
        return self.settings


# Factory functions for common configurations

def create_development_config(watch_paths: List[Union[str, Path]]) -> WatcherSettings:
    """Create development-optimized watcher configuration."""
    builder = WatcherConfigBuilder(WatcherMode.DEVELOPMENT)
    
    for path in watch_paths:
        builder.add_path(path, recursive=True)
    
    return (builder
            .with_performance(batch_size=50, batch_timeout=1.0)
            .with_debouncing(base_period=0.3)
            .with_filtering(level=EventFilterLevel.NORMAL)
            .build())


def create_production_config(watch_paths: List[Union[str, Path]]) -> WatcherSettings:
    """Create production-optimized watcher configuration."""
    builder = WatcherConfigBuilder(WatcherMode.PRODUCTION)
    
    for path in watch_paths:
        builder.add_path(path, recursive=True)
    
    return (builder
            .with_performance(batch_size=200, batch_timeout=3.0, max_memory_mb=1000)
            .with_debouncing(base_period=1.0)
            .with_filtering(level=EventFilterLevel.STRICT)
            .build())


def create_testing_config(watch_paths: List[Union[str, Path]]) -> WatcherSettings:
    """Create testing-optimized watcher configuration."""
    builder = WatcherConfigBuilder(WatcherMode.TESTING)
    
    for path in watch_paths:
        builder.add_path(path, recursive=True)
    
    return (builder
            .with_performance(batch_size=10, batch_timeout=0.5)
            .with_debouncing(base_period=0.1)
            .with_filtering(level=EventFilterLevel.PERMISSIVE)
            .build())


def create_high_performance_config(watch_paths: List[Union[str, Path]]) -> WatcherSettings:
    """Create high-performance watcher configuration."""
    builder = WatcherConfigBuilder(WatcherMode.HIGH_PERFORMANCE)
    
    for path in watch_paths:
        builder.add_path(path, recursive=True, priority=1)
    
    return (builder
            .with_performance(batch_size=500, batch_timeout=0.5, max_memory_mb=2000)
            .with_debouncing(base_period=0.1)
            .with_filtering(level=EventFilterLevel.NORMAL)
            .build())


def load_config_from_extract(extract_config: ExtractConfig) -> WatcherSettings:
    """Create watcher configuration from extract configuration."""
    mode = WatcherMode.DEVELOPMENT  # Default mode
    
    # Use extract config paths
    watch_paths = []
    if hasattr(extract_config, 'source_paths') and extract_config.source_paths:
        watch_paths = extract_config.source_paths
    
    if not watch_paths:
        logger.warning("No source paths in extract config, using current directory")
        watch_paths = [Path.cwd()]
    
    return create_development_config(watch_paths)
