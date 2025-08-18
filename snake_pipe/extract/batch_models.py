"""Data models for batch processing operations.

This module contains the core data models used in batch processing,
including batch definitions, results, and processing metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
import uuid

from snake_pipe.extract.models import ASTFile, DiscoveryResult
from snake_pipe.extract.batch_config import BatchPriority, ErrorCode, ProcessingMode


class BatchStatus(Enum):
    """Status of batch processing operations."""
    PENDING = "pending"
    RUNNING = "running"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    CHECKPOINTED = "checkpointed"


class WorkerStatus(Enum):
    """Status of batch processing workers."""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    TERMINATED = "terminated"


class BatchType(Enum):
    """Types of batch processing operations."""
    COUNT_BASED = "count_based"
    SIZE_BASED = "size_based"
    TIME_BASED = "time_based"
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CUSTOM = "custom"


@dataclass
class BatchFile:
    """Represents a file within a batch for processing."""
    file_path: Path
    file_size: int
    priority: BatchPriority = BatchPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_attempts: int = 0
    last_error: Optional[str] = None
    processing_time: Optional[float] = None
    status: BatchStatus = BatchStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return self.file_path.name
    
    @property
    def has_failed(self) -> bool:
        """Check if file processing has failed."""
        return self.last_error is not None
    
    @property
    def attempts(self) -> int:
        """Alias for processing_attempts (backward compatibility)."""
        return self.processing_attempts
    
    @property
    def end_time(self) -> Optional[datetime]:
        """Alias for completed_at (backward compatibility)."""
        return self.completed_at
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Alias for started_at (backward compatibility)."""
        return self.started_at
    
    def mark_failed(self, error: str) -> None:
        """Mark file as failed with error message."""
        self.last_error = error
        self.processing_attempts += 1
        self.status = BatchStatus.FAILED
        self.completed_at = datetime.now()
    
    def reset_error(self) -> None:
        """Reset error state for retry."""
        self.last_error = None
        self.status = BatchStatus.PENDING
        self.started_at = None
        self.completed_at = None
    
    def start_processing(self) -> None:
        """Mark file as started processing."""
        self.status = BatchStatus.PROCESSING
        self.started_at = datetime.now()
        self.processing_attempts += 1
    
    def complete_processing(self, success: bool = True) -> None:
        """Mark file processing as completed."""
        self.status = BatchStatus.COMPLETED if success else BatchStatus.FAILED
        self.completed_at = datetime.now()
        if self.started_at and self.completed_at:
            self.processing_time = (self.completed_at - self.started_at).total_seconds()


@dataclass
class FileBatch:
    """Represents a batch of files for processing."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    files: List[BatchFile] = field(default_factory=list)
    batch_type: BatchType = BatchType.DISCOVERY
    priority: BatchPriority = BatchPriority.MEDIUM
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    retry_count: int = 0
    checkpoint_data: Optional[Dict[str, Any]] = None
    
    @property
    def file_count(self) -> int:
        """Get the number of files in the batch."""
        return len(self.files)
    
    @property
    def total_size(self) -> int:
        """Get the total size of all files in the batch."""
        return sum(file.file_size for file in self.files)
    
    @property
    def failed_files(self) -> List[BatchFile]:
        """Get list of files that failed processing."""
        return [file for file in self.files if file.has_failed]
    
    @property
    def successful_files(self) -> List[BatchFile]:
        """Get list of files that processed successfully."""
        return [file for file in self.files if not file.has_failed]
    
    @property
    def processing_time(self) -> Optional[float]:
        """Get total processing time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def assigned_worker(self) -> Optional[str]:
        """Alias for worker_id (backward compatibility)."""
        return self.worker_id
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Alias for started_at (backward compatibility)."""
        return self.started_at
    
    @property
    def end_time(self) -> Optional[datetime]:
        """Alias for completed_at (backward compatibility)."""
        return self.completed_at
    
    @property
    def is_complete(self) -> bool:
        """Check if batch processing is complete."""
        return self.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if not self.files:
            return 100.0
        successful = len(self.successful_files)
        return (successful / len(self.files)) * 100
    
    def add_file(self, file_path: Path, file_size: int, priority: BatchPriority = BatchPriority.MEDIUM) -> None:
        """Add a file to the batch."""
        batch_file = BatchFile(
            file_path=file_path,
            file_size=file_size,
            priority=priority
        )
        self.files.append(batch_file)
    
    def start_processing(self, worker_id: str) -> None:
        """Mark batch as started."""
        self.status = BatchStatus.PROCESSING
        self.started_at = datetime.now()
        self.worker_id = worker_id
    
    def complete_processing(self) -> None:
        """Mark batch as completed."""
        self.status = BatchStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def fail_processing(self, error: str) -> None:
        """Mark batch as failed."""
        self.status = BatchStatus.FAILED
        self.completed_at = datetime.now()
        # Store error in checkpoint data for debugging
        if self.checkpoint_data is None:
            self.checkpoint_data = {}
        self.checkpoint_data["error"] = error


@dataclass
class BatchResult:
    """Result of processing a single batch."""
    batch_id: str
    files_processed: int
    files_successful: int
    files_failed: int
    processing_time: float
    total_size_processed: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_created: bool = False
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.files_processed == 0:
            return 100.0
        return (self.files_successful / self.files_processed) * 100
    
    @property
    def throughput_files_per_second(self) -> float:
        """Calculate throughput in files per second."""
        if self.processing_time == 0:
            return 0.0
        return self.files_processed / self.processing_time
    
    @property
    def throughput_mb_per_second(self) -> float:
        """Calculate throughput in MB per second."""
        if self.processing_time == 0:
            return 0.0
        size_mb = self.total_size_processed / (1024 * 1024)
        return size_mb / self.processing_time
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.files_failed += 1
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


@dataclass
class BatchExecutionResult:
    """Result of executing multiple batches."""
    # Core execution tracking
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    
    # File processing statistics  
    files_discovered: int = 0
    files_processed: int = 0
    files_successful: int = 0
    files_failed: int = 0
    
    # Batch processing statistics
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    
    # Performance metrics
    total_processing_time: float = 0.0
    throughput_files_per_second: float = 0.0
    
    # Results and configuration
    batch_results: List[BatchResult] = field(default_factory=list)
    overall_errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    config: Optional[Any] = None  # BatchConfig type, avoiding circular import
    
    @property
    def batch_success_rate(self) -> float:
        """Calculate batch success rate as percentage."""
        if self.total_batches == 0:
            return 100.0
        return (self.successful_batches / self.total_batches) * 100
    
    @property
    def file_success_rate(self) -> float:
        """Calculate file success rate as percentage."""
        if self.files_processed == 0:
            return 100.0
        return (self.files_successful / self.files_processed) * 100
    
    @property
    def overall_throughput(self) -> float:
        """Calculate overall throughput in files per second."""
        if self.total_processing_time == 0:
            return 0.0
        return self.files_processed / self.total_processing_time
    
    @property
    def average_batch_time(self) -> float:
        """Calculate average time per batch."""
        if self.total_batches == 0:
            return 0.0
        return self.total_processing_time / self.total_batches
    
    def add_batch_result(self, result: BatchResult) -> None:
        """Add a batch result to the execution result."""
        self.batch_results.append(result)
        self.files_processed += result.files_processed
        self.files_successful += result.files_successful
        self.files_failed += result.files_failed
        self.total_processing_time += result.processing_time
        self.total_batches += 1
        
        if result.success_rate > 0:
            self.successful_batches += 1
        else:
            self.failed_batches += 1


@dataclass
class WorkerInfo:
    """Information about a batch processing worker."""
    worker_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    current_batch_id: Optional[str] = None
    processed_batches: int = 0
    failed_batches: int = 0
    total_files_processed: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    @property
    def batches_completed(self) -> int:
        """Get total completed batches (alias for processed_batches)."""
        return self.processed_batches
    
    @property
    def uptime_seconds(self) -> float:
        """Get worker uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def idle_time_seconds(self) -> float:
        """Get time since last activity."""
        return (datetime.now() - self.last_activity).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Calculate worker success rate."""
        total_batches = self.processed_batches + self.failed_batches
        if total_batches == 0:
            return 100.0
        return (self.processed_batches / total_batches) * 100
    
    def start_batch(self, batch_id: str) -> None:
        """Mark worker as processing a batch."""
        self.status = WorkerStatus.RUNNING
        self.current_batch_id = batch_id
        self.last_activity = datetime.now()
    
    def complete_batch(self, success: bool = True) -> None:
        """Mark worker as completing a batch."""
        self.status = WorkerStatus.IDLE
        self.current_batch_id = None
        self.last_activity = datetime.now()
        
        if success:
            self.processed_batches += 1
        else:
            self.failed_batches += 1


@dataclass
class CheckpointData:
    """Checkpoint data for resumable batch processing."""
    job_id: str
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    completed_batches: List[str] = field(default_factory=list)
    failed_batches: List[str] = field(default_factory=list)
    pending_batches: List[str] = field(default_factory=list)
    processing_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        total = len(self.completed_batches) + len(self.failed_batches) + len(self.pending_batches)
        if total == 0:
            return 100.0
        completed = len(self.completed_batches)
        return (completed / total) * 100
    
    def mark_batch_completed(self, batch_id: str) -> None:
        """Mark a batch as completed."""
        if batch_id in self.pending_batches:
            self.pending_batches.remove(batch_id)
        if batch_id not in self.completed_batches:
            self.completed_batches.append(batch_id)
    
    def mark_batch_failed(self, batch_id: str) -> None:
        """Mark a batch as failed."""
        if batch_id in self.pending_batches:
            self.pending_batches.remove(batch_id)
        if batch_id not in self.failed_batches:
            self.failed_batches.append(batch_id)


@dataclass
class BatchProcessingMetrics:
    """Metrics for batch processing performance monitoring."""
    
    # Throughput metrics
    files_per_minute: float = 0.0
    batches_per_minute: float = 0.0
    mb_per_second: float = 0.0
    
    # Performance metrics
    average_batch_time: float = 0.0
    average_file_time: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Quality metrics
    success_rate: float = 100.0
    error_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Resource metrics
    active_workers: int = 0
    queue_size: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    
    # Timing metrics
    measurement_time: datetime = field(default_factory=datetime.now)
    measurement_window_seconds: float = 60.0
    
    def update_throughput(self, files_processed: int, time_elapsed: float, size_processed: int) -> None:
        """Update throughput metrics."""
        if time_elapsed > 0:
            self.files_per_minute = (files_processed / time_elapsed) * 60
            self.mb_per_second = (size_processed / (1024 * 1024)) / time_elapsed
    
    def update_performance(self, batch_time: float, file_count: int, cpu: float, memory: float) -> None:
        """Update performance metrics."""
        self.average_batch_time = batch_time
        if file_count > 0:
            self.average_file_time = batch_time / file_count
        self.cpu_utilization = cpu
        self.memory_utilization = memory
    
    def update_quality(self, successful: int, total: int, retries: int) -> None:
        """Update quality metrics."""
        if total > 0:
            self.success_rate = (successful / total) * 100
            self.error_rate = ((total - successful) / total) * 100
            self.retry_rate = (retries / total) * 100


# Type aliases for better code readability
BatchList = List[FileBatch]
BatchProcessor = Callable[[FileBatch], BatchResult]
AsyncBatchProcessor = Callable[[FileBatch], AsyncIterator[BatchResult]]
ProgressCallback = Callable[[BatchProcessingMetrics], None]
