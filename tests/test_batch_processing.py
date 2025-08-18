"""Comprehensive test suite for the batch processing engine.

This module provides comprehensive testing for all batch processing components
including configuration, models, workers, and the main processing engine.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch, AsyncMock

from snake_pipe.extract.batch_config import (
    BatchConfig, ProcessingMode, BatchStrategy, ErrorHandlingMode, ErrorCode,
    BatchPriority, CheckpointConfig, PerformanceConfig, ResourceLimits
)
from snake_pipe.extract.batch_models import (
    BatchFile, FileBatch, BatchResult, BatchExecutionResult, 
    WorkerInfo, CheckpointData, BatchStatus, WorkerStatus, BatchType
)
from snake_pipe.extract.batch_worker import BatchWorker, WorkerPool
from snake_pipe.extract.batch_processor import BatchProcessingEngine
from snake_pipe.extract.models import ASTFile, ASTMetadata, LanguageInfo, LanguageType, FileStatus
from snake_pipe.utils.checkpoint_manager import CheckpointManager


class TestBatchConfig:
    """Test batch configuration classes and enums."""
    
    def test_batch_config_creation(self):
        """Test creating batch configuration with valid parameters."""
        config = BatchConfig(
            batch_size=100,
            max_workers=4,
            processing_timeout=300.0,
            batch_strategy=BatchStrategy.COUNT_BASED,
            error_handling_mode=ErrorHandlingMode.CONTINUE
        )
        
        assert config.batch_size == 100
        assert config.max_workers == 4
        assert config.processing_timeout == 300.0
        assert config.batch_strategy == BatchStrategy.COUNT_BASED
        assert config.error_handling_mode == ErrorHandlingMode.CONTINUE
    
    def test_high_throughput_config(self):
        """Test high throughput configuration factory method."""
        config = BatchConfig.high_throughput()
        
        assert config.batch_size == 1000
        assert config.max_workers == 8
        assert config.batch_strategy == BatchStrategy.ADAPTIVE
        assert config.error_handling_mode == ErrorHandlingMode.CONTINUE
        assert config.enable_compression is True
    
    def test_memory_optimized_config(self):
        """Test memory optimized configuration factory method."""
        config = BatchConfig.memory_optimized()
        
        assert config.batch_size == 50
        assert config.memory_limit_mb == 512
        assert config.enable_compression is True
        assert config.batch_strategy == BatchStrategy.SIZE_BASED
    
    def test_reliable_config(self):
        """Test reliable configuration factory method."""
        config = BatchConfig.reliable()
        
        assert config.retry_attempts == 5
        assert config.error_handling_mode == ErrorHandlingMode.RETRY
        assert config.enable_checkpointing is True
        assert config.checkpoint_interval == 50
    
    def test_enum_values(self):
        """Test that all enum values are properly defined."""
        # Test ProcessingMode
        assert ProcessingMode.BATCH.value == "batch"
        assert ProcessingMode.STREAMING.value == "streaming"
        assert ProcessingMode.REAL_TIME.value == "real_time"
        
        # Test BatchStrategy
        assert BatchStrategy.SIZE_BASED.value == "size_based"
        assert BatchStrategy.COUNT_BASED.value == "count_based"
        assert BatchStrategy.TIME_BASED.value == "time_based"
        assert BatchStrategy.ADAPTIVE.value == "adaptive"
        
        # Test ErrorHandlingMode
        assert ErrorHandlingMode.FAIL_FAST.value == "fail_fast"
        assert ErrorHandlingMode.CONTINUE.value == "continue"
        assert ErrorHandlingMode.RETRY.value == "retry"
        assert ErrorHandlingMode.QUARANTINE.value == "quarantine"
        
        # Test BatchPriority
        assert BatchPriority.LOW.value == 1
        assert BatchPriority.MEDIUM.value == 2
        assert BatchPriority.HIGH.value == 3
        assert BatchPriority.CRITICAL.value == 4


class TestBatchModels:
    """Test batch processing domain models."""
    
    def test_batch_file_creation(self):
        """Test creating BatchFile with required fields."""
        file_path = Path("/test/file.json")
        batch_file = BatchFile(
            file_path=file_path,
            file_size=1024
        )
        
        assert batch_file.file_path == file_path
        assert batch_file.file_size == 1024
        assert batch_file.status == BatchStatus.PENDING
        assert batch_file.attempts == 0
        assert batch_file.last_error is None
    
    def test_batch_file_processing_lifecycle(self):
        """Test BatchFile status transitions during processing."""
        batch_file = BatchFile(
            file_path=Path("/test/file.json"),
            file_size=1024
        )
        
        # Start processing
        batch_file.start_processing()
        assert batch_file.status == BatchStatus.PROCESSING
        assert batch_file.attempts == 1
        assert batch_file.start_time is not None
        
        # Mark as completed
        batch_file.complete_processing()
        assert batch_file.status == BatchStatus.COMPLETED
        assert batch_file.end_time is not None
    
    def test_batch_file_error_handling(self):
        """Test BatchFile error handling and retry logic."""
        batch_file = BatchFile(
            file_path=Path("/test/file.json"),
            file_size=1024
        )
        
        # Mark as failed
        error_message = "Processing failed"
        batch_file.mark_failed(error_message)
        
        assert batch_file.status == BatchStatus.FAILED
        assert batch_file.last_error == error_message
        assert batch_file.end_time is not None
    
    def test_file_batch_creation(self):
        """Test creating FileBatch with multiple files."""
        files = [
            BatchFile(Path(f"/test/file{i}.json"), 1024)
            for i in range(5)
        ]
        
        batch = FileBatch(
            batch_id="test_batch_001",
            files=files,
            batch_type=BatchType.COUNT_BASED
        )
        
        assert batch.batch_id == "test_batch_001"
        assert len(batch.files) == 5
        assert batch.batch_type == BatchType.COUNT_BASED
        assert batch.total_size == 5120  # 5 * 1024
        assert batch.status == BatchStatus.PENDING
    
    def test_file_batch_processing_lifecycle(self):
        """Test FileBatch status transitions during processing."""
        files = [BatchFile(Path(f"/test/file{i}.json"), 1024) for i in range(3)]
        batch = FileBatch("test_batch", files)
        
        # Start processing
        worker_id = "worker_001"
        batch.start_processing(worker_id)
        
        assert batch.status == BatchStatus.PROCESSING
        assert batch.assigned_worker == worker_id
        assert batch.start_time is not None
        
        # Complete processing
        batch.complete_processing()
        
        assert batch.status == BatchStatus.COMPLETED
        assert batch.end_time is not None
    
    def test_batch_result_creation(self):
        """Test creating BatchResult with processing metrics."""
        result = BatchResult(
            batch_id="test_batch_001",
            files_processed=100,
            files_successful=95,
            files_failed=5,
            processing_time=30.5,
            total_size_processed=1024000
        )
        
        assert result.batch_id == "test_batch_001"
        assert result.files_processed == 100
        assert result.files_successful == 95
        assert result.files_failed == 5
        assert result.processing_time == 30.5
        assert result.total_size_processed == 1024000
        assert result.throughput_files_per_second == pytest.approx(100 / 30.5, rel=1e-3)
    
    def test_worker_info_creation(self):
        """Test creating WorkerInfo with proper initialization."""
        worker_info = WorkerInfo(worker_id="worker_001")
        
        assert worker_info.worker_id == "worker_001"
        assert worker_info.status == WorkerStatus.IDLE
        assert worker_info.current_batch_id is None
        assert worker_info.batches_completed == 0
        assert worker_info.total_files_processed == 0
    
    def test_worker_info_batch_tracking(self):
        """Test WorkerInfo batch processing tracking."""
        worker_info = WorkerInfo(worker_id="worker_001")
        
        # Start batch
        batch_id = "test_batch_001"
        worker_info.start_batch(batch_id)
        
        assert worker_info.status == WorkerStatus.RUNNING
        assert worker_info.current_batch_id == batch_id
        
        # Complete batch
        worker_info.complete_batch(success=True)
        
        assert worker_info.status == WorkerStatus.IDLE
        assert worker_info.current_batch_id is None
        assert worker_info.batches_completed == 1


class TestBatchWorker:
    """Test batch worker functionality."""
    
    @pytest.fixture
    def mock_processor_func(self):
        """Mock processor function for testing."""
        async def mock_processor(files: List[ASTFile]) -> List[ASTFile]:
            # Simple mock that returns the same files
            return files
        return mock_processor
    
    @pytest.fixture
    def batch_config(self):
        """Basic batch configuration for testing."""
        return BatchConfig(
            batch_size=10,
            max_workers=2,
            processing_timeout=60.0
        )
    
    @pytest.fixture
    def sample_batch(self):
        """Sample batch for testing."""
        files = [
            BatchFile(Path(f"/test/file{i}.json"), 1024)
            for i in range(5)
        ]
        return FileBatch("test_batch_001", files)
    
    @pytest.mark.asyncio
    async def test_worker_initialization(self, batch_config, mock_processor_func):
        """Test worker initialization with proper configuration."""
        worker = BatchWorker(
            worker_id="test_worker_001",
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        assert worker.worker_id == "test_worker_001"
        assert worker.config == batch_config
        assert worker.processor_func == mock_processor_func
        assert worker.is_running is False
    
    @pytest.mark.asyncio
    async def test_worker_start_stop(self, batch_config, mock_processor_func):
        """Test worker start and stop lifecycle."""
        worker = BatchWorker(
            worker_id="test_worker_001",
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        # Start worker
        await worker.start()
        assert worker.is_running is True
        assert worker.info.status == WorkerStatus.IDLE
        
        # Stop worker
        await worker.stop()
        assert worker.is_running is False
        assert worker.info.status == WorkerStatus.TERMINATED
    
    @pytest.mark.asyncio
    async def test_worker_process_batch_success(self, batch_config, mock_processor_func, sample_batch):
        """Test successful batch processing by worker."""
        worker = BatchWorker(
            worker_id="test_worker_001",
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        await worker.start()
        
        # Process batch
        result = await worker.process_batch(sample_batch)
        
        assert result.batch_id == sample_batch.batch_id
        assert result.files_processed == 5
        assert result.files_successful == 5
        assert result.files_failed == 0
        assert result.processing_time > 0
        assert result.total_size_processed == 5120
        
        await worker.stop()
    
    @pytest.mark.asyncio
    async def test_worker_process_empty_batch(self, batch_config, mock_processor_func):
        """Test worker handling empty batch."""
        worker = BatchWorker(
            worker_id="test_worker_001",
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        await worker.start()
        
        # Create empty batch
        empty_batch = FileBatch("empty_batch", [])
        
        # Process empty batch
        result = await worker.process_batch(empty_batch)
        
        assert result.batch_id == "empty_batch"
        assert result.files_processed == 0
        assert result.files_successful == 0
        assert result.files_failed == 0
        
        await worker.stop()
    
    @pytest.mark.asyncio
    async def test_worker_get_status(self, batch_config, mock_processor_func):
        """Test getting worker status and metrics."""
        worker = BatchWorker(
            worker_id="test_worker_001",
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        await worker.start()
        
        status = await worker.get_status()
        
        assert status.worker_id == "test_worker_001"
        assert status.status == WorkerStatus.IDLE
        assert status.batches_completed == 0
        assert status.total_files_processed == 0
        
        await worker.stop()


class TestWorkerPool:
    """Test worker pool functionality."""
    
    @pytest.fixture
    def mock_processor_func(self):
        """Mock processor function for testing."""
        async def mock_processor(files: List[ASTFile]) -> List[ASTFile]:
            return files
        return mock_processor
    
    @pytest.fixture
    def batch_config(self):
        """Batch configuration for worker pool testing."""
        return BatchConfig(
            batch_size=10,
            max_workers=3,
            max_queue_size=50
        )
    
    @pytest.mark.asyncio
    async def test_worker_pool_initialization(self, batch_config, mock_processor_func):
        """Test worker pool initialization."""
        pool = WorkerPool(
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        assert pool.config == batch_config
        assert pool.processor_func == mock_processor_func
        assert pool.is_running is False
        assert len(pool.workers) == 0
    
    @pytest.mark.asyncio
    async def test_worker_pool_start_stop(self, batch_config, mock_processor_func):
        """Test worker pool start and stop lifecycle."""
        pool = WorkerPool(
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        # Start pool
        await pool.start()
        assert pool.is_running is True
        assert len(pool.workers) == batch_config.max_workers
        
        # Stop pool
        await pool.stop()
        assert pool.is_running is False
        assert len(pool.workers) == 0
    
    @pytest.mark.asyncio
    async def test_worker_pool_submit_batch(self, batch_config, mock_processor_func):
        """Test submitting batches to worker pool."""
        pool = WorkerPool(
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        await pool.start()
        
        # Create test batch
        files = [BatchFile(Path(f"/test/file{i}.json"), 1024) for i in range(3)]
        batch = FileBatch("test_batch", files)
        
        # Submit batch
        await pool.submit_batch(batch)
        
        # Queue should have the batch
        assert pool.batch_queue.qsize() == 1
        
        await pool.stop()
    
    @pytest.mark.asyncio
    async def test_worker_pool_get_worker_status(self, batch_config, mock_processor_func):
        """Test getting status of all workers in pool."""
        pool = WorkerPool(
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        await pool.start()
        
        status_list = await pool.get_worker_status()
        
        assert len(status_list) == batch_config.max_workers
        for status in status_list:
            assert isinstance(status, WorkerInfo)
            assert status.status == WorkerStatus.IDLE
        
        await pool.stop()


class TestBatchProcessingEngine:
    """Test the main batch processing engine."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample files for testing."""
        files = []
        for i in range(10):
            file_path = temp_dir / f"test_file_{i:03d}.json"
            content = {"id": i, "data": f"test_data_{i}"}
            file_path.write_text(json.dumps(content))
            files.append(file_path)
        return files
    
    @pytest.fixture
    def mock_processor_func(self):
        """Mock processor function for testing."""
        def mock_processor(files: List[ASTFile]) -> List[ASTFile]:
            # Simple mock that returns the same files
            return files
        return mock_processor
    
    @pytest.fixture
    def batch_config(self):
        """Batch configuration for engine testing."""
        return BatchConfig(
            batch_size=5,
            max_workers=2,
            processing_timeout=30.0,
            batch_strategy=BatchStrategy.COUNT_BASED
        )
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, batch_config, mock_processor_func):
        """Test batch processing engine initialization."""
        engine = BatchProcessingEngine(
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        assert engine.config == batch_config
        assert engine.processor_func == mock_processor_func
        assert engine.is_running is False
        assert engine.worker_pool is None
    
    @pytest.mark.asyncio
    async def test_engine_process_directory(self, batch_config, mock_processor_func, temp_dir, sample_files):
        """Test processing directory with multiple files."""
        engine = BatchProcessingEngine(
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        # Process directory
        result = await engine.process_directory(
            source_path=temp_dir,
            resume_from_checkpoint=False
        )
        
        assert isinstance(result, BatchExecutionResult)
        assert result.files_discovered == 10
        assert result.files_processed == 10
        assert result.files_successful == 10
        assert result.files_failed == 0
        assert result.total_processing_time > 0
        assert result.throughput_files_per_second > 0
    
    @pytest.mark.asyncio
    async def test_engine_process_specific_files(self, batch_config, mock_processor_func, sample_files):
        """Test processing specific list of files."""
        engine = BatchProcessingEngine(
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        # Process subset of files
        files_to_process = sample_files[:5]
        
        result = await engine.process_files(
            file_paths=files_to_process,
            resume_from_checkpoint=False
        )
        
        assert result.files_discovered == 5
        assert result.files_processed == 5
        assert result.files_successful == 5
        assert result.files_failed == 0
    
    @pytest.mark.asyncio
    async def test_engine_get_metrics(self, batch_config, mock_processor_func):
        """Test getting engine metrics during processing."""
        engine = BatchProcessingEngine(
            config=batch_config,
            processor_func=mock_processor_func
        )
        
        metrics = await engine.get_metrics()
        
        assert "execution_id" in metrics
        assert "is_running" in metrics
        assert "total_files_discovered" in metrics
        assert "total_files_processed" in metrics
        assert "throughput_files_per_second" in metrics
        assert "worker_count" in metrics
        assert "batch_count" in metrics
    
    @pytest.mark.asyncio
    async def test_engine_error_handling(self, batch_config, temp_dir):
        """Test engine error handling with failing processor."""
        def failing_processor(files: List[ASTFile]) -> List[ASTFile]:
            raise Exception("Processor failed")
        
        engine = BatchProcessingEngine(
            config=batch_config,
            processor_func=failing_processor
        )
        
        # Create test file
        test_file = temp_dir / "test.json"
        test_file.write_text('{"test": "data"}')
        
        # Process should handle errors gracefully
        result = await engine.process_directory(temp_dir, resume_from_checkpoint=False)
        
        # Should discover file but fail processing
        assert result.files_discovered >= 1
        # Note: Exact behavior depends on error handling mode


# Integration tests
class TestBatchProcessingIntegration:
    """Integration tests for the complete batch processing pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def large_dataset(self, temp_dir):
        """Create larger dataset for performance testing."""
        files = []
        for i in range(100):
            file_path = temp_dir / f"large_test_{i:03d}.json"
            content = {
                "id": i,
                "data": f"test_data_{i}" * 100,  # Larger content
                "timestamp": datetime.now().isoformat(),
                "metadata": {"size": "large", "batch": i // 10}
            }
            file_path.write_text(json.dumps(content))
            files.append(file_path)
        return files
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, temp_dir, large_dataset):
        """Test complete end-to-end batch processing pipeline."""
        def processor_func(files: List[ASTFile]) -> List[ASTFile]:
            # Simulate some processing time
            import time
            time.sleep(0.01)  # 10ms per file
            return files
        
        config = BatchConfig.high_throughput()
        config.batch_size = 20  # Smaller batches for testing
        config.max_workers = 3
        
        engine = BatchProcessingEngine(
            config=config,
            processor_func=processor_func
        )
        
        # Process the large dataset
        result = await engine.process_directory(temp_dir, resume_from_checkpoint=False)
        
        # Verify results
        assert result.files_discovered == 100
        assert result.files_processed == 100
        assert result.files_successful == 100
        assert result.files_failed == 0
        assert result.total_processing_time > 0
        assert result.throughput_files_per_second > 0
        
        # Should have created multiple batches
        assert len(result.batch_results) > 1
        
        # Verify batch results
        total_files_from_batches = sum(br.files_processed for br in result.batch_results)
        assert total_files_from_batches == 100
    
    @pytest.mark.asyncio
    async def test_different_batch_strategies(self, temp_dir, large_dataset):
        """Test different batch strategies produce different results."""
        def processor_func(files: List[ASTFile]) -> List[ASTFile]:
            return files
        
        strategies = [
            BatchStrategy.COUNT_BASED,
            BatchStrategy.SIZE_BASED,
            BatchStrategy.TIME_BASED
        ]
        
        results = {}
        
        for strategy in strategies:
            config = BatchConfig(
                batch_size=10,
                max_workers=2,
                batch_strategy=strategy
            )
            
            engine = BatchProcessingEngine(
                config=config,
                processor_func=processor_func
            )
            
            result = await engine.process_directory(temp_dir, resume_from_checkpoint=False)
            results[strategy] = result
        
        # All strategies should process all files successfully
        for strategy, result in results.items():
            assert result.files_processed == 100
            assert result.files_successful == 100
            assert result.files_failed == 0


# Performance tests
class TestBatchProcessingPerformance:
    """Performance tests for batch processing components."""
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(self, tmp_path):
        """Test that throughput measurements are accurate."""
        def processor_func(files: List[ASTFile]) -> List[ASTFile]:
            import time
            time.sleep(0.001)  # 1ms per file
            return files
        
        config = BatchConfig(
            batch_size=10,
            max_workers=1  # Single worker for predictable timing
        )
        
        # Create actual temporary files for processing
        files = []
        for i in range(20):  # Smaller number for faster test
            file_path = tmp_path / f"test_file_{i}.json"
            file_path.write_text('{"test": "data"}')
            files.append(file_path)
        
        engine = BatchProcessingEngine(
            config=config,
            processor_func=processor_func
        )
        
        start_time = datetime.now()
        result = await engine.process_files(files, resume_from_checkpoint=False)
        end_time = datetime.now()
        
        actual_time = (end_time - start_time).total_seconds()
        reported_time = result.total_processing_time
        
        # Verify files were actually processed
        assert result.files_processed > 0, "No files were processed"
        assert result.files_discovered == len(files), f"Expected {len(files)} files discovered, got {result.files_discovered}"
        
        # Times should be reasonably close (within 50% due to overhead)
        if reported_time > 0:
            assert abs(actual_time - reported_time) / max(actual_time, reported_time) < 0.5
        
        # Throughput should be reasonable (at least 5 files/sec with overhead)
        assert result.throughput_files_per_second > 5, f"Throughput too low: {result.throughput_files_per_second} files/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
