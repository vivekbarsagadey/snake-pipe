"""Configuration management for batch processing operations.

This module defines configuration classes and enums for batch processing
engine operations following Clean Architecture principles.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import multiprocessing


class ProcessingMode(Enum):
    """ETL processing modes for different throughput requirements."""
    BATCH = "batch"
    STREAMING = "streaming"
    REAL_TIME = "real_time"


class BatchStrategy(Enum):
    """Batch processing strategies for different scenarios."""
    SIZE_BASED = "size_based"
    COUNT_BASED = "count_based"
    TIME_BASED = "time_based"
    ADAPTIVE = "adaptive"


class ErrorHandlingMode(Enum):
    """Error handling modes for batch processing."""
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"
    RETRY = "retry"
    QUARANTINE = "quarantine"


class BatchPriority(Enum):
    """Priority levels for batch processing."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ErrorCode(Enum):
    """Error codes for batch processing operations."""
    BATCH_VALIDATION_FAILED = "BATCH_VALIDATION_FAILED"
    FILE_ACCESS_ERROR = "FILE_ACCESS_ERROR"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    WORKER_FAILURE = "WORKER_FAILURE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    CHECKPOINT_ERROR = "CHECKPOINT_ERROR"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"


@dataclass
class BatchConfig:
    """Configuration for batch processing operations."""
    
    # Core batch settings
    batch_size: int = 100
    max_batch_size: int = 1000
    min_batch_size: int = 10
    batch_strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    
    # Processing settings
    max_workers: int = field(default_factory=lambda: min(multiprocessing.cpu_count(), 8))
    processing_timeout: float = 300.0  # seconds per batch
    memory_limit_mb: int = 2048
    
    # Error handling
    error_handling_mode: ErrorHandlingMode = ErrorHandlingMode.CONTINUE
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Performance optimization
    enable_checkpointing: bool = True
    checkpoint_interval: int = 10  # batches
    enable_progress_tracking: bool = True
    enable_compression: bool = False
    
    # File processing
    max_file_size_mb: int = 50
    supported_extensions: List[str] = field(default_factory=lambda: [".json"])
    batch_size_mb: int = 10  # For size-based batching
    
    # Priority and scheduling
    priority: BatchPriority = BatchPriority.MEDIUM
    max_queue_size: int = 1000
    
    # Resource management
    cpu_limit_percent: float = 80.0
    memory_warning_threshold: float = 0.8
    disk_space_threshold_gb: float = 1.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Auto-adjust min_batch_size if it's greater than batch_size
        if self.min_batch_size > self.batch_size:
            self.min_batch_size = max(1, self.batch_size // 2)
        
        # Validation
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_batch_size < self.batch_size:
            raise ValueError("max_batch_size must be >= batch_size")
        if self.min_batch_size > self.batch_size:
            raise ValueError("min_batch_size must be <= batch_size")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.processing_timeout < 1:
            raise ValueError("processing_timeout must be at least 1")
        if self.memory_limit_mb < 100:
            raise ValueError("memory_limit_mb must be at least 100")
        if not 0 < self.cpu_limit_percent <= 100:
            raise ValueError("cpu_limit_percent must be between 0 and 100")

    @classmethod
    def high_throughput(cls) -> "BatchConfig":
        """Create configuration optimized for high throughput."""
        return cls(
            batch_size=1000,
            max_workers=8,
            processing_timeout=60.0,
            batch_strategy=BatchStrategy.ADAPTIVE,
            error_handling_mode=ErrorHandlingMode.CONTINUE,
            enable_compression=True,
            memory_limit_mb=4096
        )
    
    @classmethod
    def memory_optimized(cls) -> "BatchConfig":
        """Create configuration optimized for memory efficiency."""
        return cls(
            batch_size=50,
            max_workers=2,
            processing_timeout=120.0,
            batch_strategy=BatchStrategy.SIZE_BASED,
            error_handling_mode=ErrorHandlingMode.CONTINUE,
            enable_compression=True,
            memory_limit_mb=512
        )
    
    @classmethod
    def reliable(cls) -> "BatchConfig":
        """Create configuration optimized for reliability."""
        return cls(
            batch_size=100,
            max_workers=4,
            processing_timeout=300.0,
            batch_strategy=BatchStrategy.COUNT_BASED,
            error_handling_mode=ErrorHandlingMode.RETRY,
            retry_attempts=5,
            enable_checkpointing=True,
            checkpoint_interval=50
        )


@dataclass
class CheckpointConfig:
    """Configuration for batch processing checkpoints."""
    
    enabled: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    checkpoint_format: str = "json"  # json, pickle, custom
    retention_hours: int = 24
    compression_enabled: bool = True
    auto_cleanup: bool = True
    
    def __post_init__(self):
        """Ensure checkpoint directory exists."""
        if self.enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and optimization."""
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_interval: int = 10  # seconds
    enable_profiling: bool = False
    
    # Optimization settings
    adaptive_batch_sizing: bool = True
    memory_optimization: bool = True
    io_optimization: bool = True
    
    # Thresholds
    target_throughput: int = 10000  # files per minute
    max_latency_ms: int = 1000
    min_cpu_efficiency: float = 0.7
    
    # Adaptive behavior
    batch_size_adjustment_factor: float = 1.2
    performance_window_minutes: int = 5


@dataclass
class ResourceLimits:
    """Resource limits for batch processing operations."""
    
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    max_disk_usage_gb: float = 10.0
    max_open_files: int = 1000
    max_processing_time_hours: int = 24
    
    def __post_init__(self):
        """Validate resource limits."""
        if self.max_memory_mb < 256:
            raise ValueError("max_memory_mb must be at least 256")
        if not 0 < self.max_cpu_percent <= 100:
            raise ValueError("max_cpu_percent must be between 0 and 100")


@dataclass
class BatchProcessingConfig:
    """Comprehensive configuration for the batch processing engine."""
    
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    
    # Additional settings
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    enable_detailed_logging: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def high_throughput(cls) -> "BatchProcessingConfig":
        """Create configuration optimized for high throughput."""
        batch_config = BatchConfig(
            batch_size=500,
            max_batch_size=1000,
            max_workers=multiprocessing.cpu_count(),
            batch_strategy=BatchStrategy.ADAPTIVE,
            error_handling_mode=ErrorHandlingMode.SKIP_ERRORS
        )
        
        performance_config = PerformanceConfig(
            target_throughput=15000,
            adaptive_batch_sizing=True,
            memory_optimization=True,
            io_optimization=True
        )
        
        resource_limits = ResourceLimits(
            max_memory_mb=4096,
            max_cpu_percent=90.0
        )
        
        return cls(
            batch_config=batch_config,
            performance_config=performance_config,
            resource_limits=resource_limits,
            processing_mode=ProcessingMode.BATCH
        )
    
    @classmethod
    def memory_optimized(cls) -> "BatchProcessingConfig":
        """Create configuration optimized for memory efficiency."""
        batch_config = BatchConfig(
            batch_size=50,
            max_batch_size=100,
            max_workers=4,
            batch_strategy=BatchStrategy.SIZE_BASED,
            memory_limit_mb=1024
        )
        
        performance_config = PerformanceConfig(
            target_throughput=5000,
            adaptive_batch_sizing=False,
            memory_optimization=True
        )
        
        resource_limits = ResourceLimits(
            max_memory_mb=1024,
            max_cpu_percent=60.0
        )
        
        return cls(
            batch_config=batch_config,
            performance_config=performance_config,
            resource_limits=resource_limits,
            processing_mode=ProcessingMode.BATCH
        )
    
    @classmethod
    def reliable(cls) -> "BatchProcessingConfig":
        """Create configuration optimized for reliability."""
        batch_config = BatchConfig(
            batch_size=100,
            max_batch_size=200,
            batch_strategy=BatchStrategy.COUNT_BASED,
            error_handling_mode=ErrorHandlingMode.RETRY_FAILED,
            max_retries=5,
            enable_checkpointing=True,
            checkpoint_interval=5
        )
        
        checkpoint_config = CheckpointConfig(
            enabled=True,
            retention_hours=72,
            compression_enabled=True
        )
        
        performance_config = PerformanceConfig(
            enable_metrics=True,
            enable_profiling=True,
            adaptive_batch_sizing=True
        )
        
        return cls(
            batch_config=batch_config,
            checkpoint_config=checkpoint_config,
            performance_config=performance_config,
            processing_mode=ProcessingMode.BATCH
        )
