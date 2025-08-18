"""Worker implementation for parallel batch processing.

This module provides the worker implementation for processing batches
in parallel using async/await patterns and proper resource management.
"""

import asyncio
import time
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
import logging
import uuid

from snake_pipe.extract.batch_config import BatchConfig, ErrorHandlingMode, ErrorCode
from snake_pipe.extract.batch_models import (
    FileBatch, BatchResult, WorkerInfo, WorkerStatus, BatchFile, BatchStatus
)
from snake_pipe.extract.models import ASTFile
from snake_pipe.utils.checkpoint_manager import CheckpointManager


logger = logging.getLogger(__name__)


class WorkerError(Exception):
    """Base exception for worker operations."""
    pass


class WorkerResourceError(WorkerError):
    """Raised when worker encounters resource limitations."""
    pass


class WorkerTimeoutError(WorkerError):
    """Raised when worker operation times out."""
    pass


class BatchWorker:
    """Worker for processing batches of AST files in parallel.
    
    This class implements a worker that can process batches of files
    asynchronously with proper error handling and resource management.
    """
    
    def __init__(
        self,
        worker_id: str,
        config: BatchConfig,
        processor_func: Callable[[List[ASTFile]], List[ASTFile]],
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """Initialize the batch worker.
        
        Args:
            worker_id: Unique identifier for this worker
            config: Batch processing configuration
            processor_func: Function to process a list of AST files
            checkpoint_manager: Optional checkpoint manager for persistence
        """
        self.worker_id = worker_id
        self.config = config
        self.processor_func = processor_func
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(f"{__name__}.{worker_id}")
        
        # Worker state
        self.info = WorkerInfo(worker_id=worker_id)
        self.is_running = False
        self.current_batch: Optional[FileBatch] = None
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_processing_time = 0.0
        self.last_memory_check = 0.0
        
        # Process monitoring
        self.process = psutil.Process()
    
    async def process_batch(self, batch: FileBatch) -> BatchResult:
        """Process a single batch of files.
        
        Args:
            batch: The batch to process
            
        Returns:
            BatchResult: Result of the batch processing
            
        Raises:
            WorkerError: If processing fails
            WorkerResourceError: If resource limits are exceeded
            WorkerTimeoutError: If processing times out
        """
        # Validate batch
        if not batch.files:
            self.logger.warning(f"Received empty batch {batch.batch_id}")
            return BatchResult(
                batch_id=batch.batch_id,
                files_processed=0,
                files_successful=0,
                files_failed=0,
                processing_time=0.0,
                total_size_processed=0
            )
        
        # Start processing
        self.current_batch = batch
        self.info.start_batch(batch.batch_id)
        batch.start_processing(self.worker_id)
        
        start_time = time.time()
        files_processed = 0
        files_successful = 0
        files_failed = 0
        total_size = 0
        errors = []
        warnings = []
        
        try:
            self.logger.info(
                f"Starting batch {batch.batch_id} with {len(batch.files)} files "
                f"({batch.total_size / 1024 / 1024:.1f} MB)"
            )
            
            # Check resource limits before starting
            await self._check_resource_limits()
            
            # Process files in the batch
            for i, batch_file in enumerate(batch.files):
                try:
                    # Check for cancellation
                    if not self.is_running:
                        break
                    
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > self.config.processing_timeout:
                        raise WorkerTimeoutError(
                            f"Batch processing timed out after {elapsed:.1f}s"
                        )
                    
                    # Check memory periodically
                    if i % 10 == 0:
                        await self._check_memory_usage()
                    
                    # Process individual file
                    file_start_time = time.time()
                    success = await self._process_file(batch_file)
                    file_processing_time = time.time() - file_start_time
                    
                    # Update metrics
                    batch_file.processing_time = file_processing_time
                    files_processed += 1
                    total_size += batch_file.file_size
                    
                    if success:
                        files_successful += 1
                    else:
                        files_failed += 1
                        if batch_file.last_error:
                            errors.append(
                                f"File {batch_file.filename}: {batch_file.last_error}"
                            )
                    
                    # Log progress periodically
                    if files_processed % 100 == 0:
                        progress = (files_processed / len(batch.files)) * 100
                        self.logger.info(
                            f"Batch {batch.batch_id} progress: {progress:.1f}% "
                            f"({files_processed}/{len(batch.files)} files)"
                        )
                
                except Exception as e:
                    self.logger.error(f"Error processing file {batch_file.filename}: {e}")
                    batch_file.mark_failed(str(e))
                    files_failed += 1
                    errors.append(f"File {batch_file.filename}: {str(e)}")
                    
                    # Handle error based on configuration
                    if self.config.error_handling_mode == ErrorHandlingMode.FAIL_FAST:
                        raise WorkerError(f"Batch processing failed on file {batch_file.filename}: {e}")
            
            # Calculate final metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Mark batch as completed
            if files_failed == 0:
                batch.complete_processing()
            else:
                if self.config.error_handling_mode == ErrorHandlingMode.FAIL_FAST and files_failed > 0:
                    batch.fail_processing(f"Failed {files_failed} files")
                else:
                    batch.complete_processing()  # Partial success is still completion
            
            # Update worker info
            self.info.complete_batch(success=(files_failed == 0))
            self.info.total_files_processed += files_processed
            
            # Create result
            result = BatchResult(
                batch_id=batch.batch_id,
                files_processed=files_processed,
                files_successful=files_successful,
                files_failed=files_failed,
                processing_time=processing_time,
                total_size_processed=total_size,
                errors=errors,
                warnings=warnings,
                metrics={
                    "worker_id": self.worker_id,
                    "throughput_files_per_second": files_processed / processing_time if processing_time > 0 else 0,
                    "average_file_time": processing_time / files_processed if files_processed > 0 else 0,
                    "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": self.process.cpu_percent()
                }
            )
            
            self.logger.info(
                f"Completed batch {batch.batch_id}: {files_successful}/{files_processed} successful "
                f"({result.throughput_files_per_second:.1f} files/sec, {processing_time:.1f}s)"
            )
            
            return result
            
        except Exception as e:
            # Mark batch as failed
            batch.fail_processing(str(e))
            self.info.complete_batch(success=False)
            
            self.logger.error(f"Batch {batch.batch_id} failed: {e}")
            
            # Create error result
            processing_time = time.time() - start_time
            return BatchResult(
                batch_id=batch.batch_id,
                files_processed=files_processed,
                files_successful=files_successful,
                files_failed=files_failed,
                processing_time=processing_time,
                total_size_processed=total_size,
                errors=errors + [str(e)],
                warnings=warnings
            )
        
        finally:
            self.current_batch = None
    
    async def start(self) -> None:
        """Start the worker."""
        self.is_running = True
        self.info.status = WorkerStatus.IDLE
        self.logger.info(f"Worker {self.worker_id} started")
    
    async def stop(self) -> None:
        """Stop the worker."""
        self.is_running = False
        self.info.status = WorkerStatus.TERMINATED
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    async def get_status(self) -> WorkerInfo:
        """Get current worker status and metrics."""
        # Update runtime metrics
        if self.info.status == WorkerStatus.RUNNING and self.current_batch:
            self.info.last_activity = datetime.now()
        
        return self.info
    
    async def _process_file(self, batch_file: BatchFile) -> bool:
        """Process a single file within a batch.
        
        Args:
            batch_file: The file to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create ASTFile object for processing
            # Note: This is a simplified conversion - in real implementation,
            # you would properly convert BatchFile to ASTFile with metadata
            ast_files = [self._convert_to_ast_file(batch_file)]
            
            # Call the processor function (check if it's async)
            if asyncio.iscoroutinefunction(self.processor_func):
                processed_files = await self.processor_func(ast_files)
            else:
                processed_files = self.processor_func(ast_files)
            
            # Check if processing was successful
            if processed_files and len(processed_files) > 0:
                return True
            else:
                batch_file.mark_failed("Processing returned empty result")
                return False
                
        except Exception as e:
            batch_file.mark_failed(str(e))
            return False
    
    def _convert_to_ast_file(self, batch_file: BatchFile) -> ASTFile:
        """Convert BatchFile to ASTFile for processing.
        
        This is a simplified conversion. In a real implementation,
        you would properly populate all ASTFile fields.
        """
        from snake_pipe.extract.models import (
            ASTFile, ASTMetadata, LanguageInfo, LanguageType, FileStatus
        )
        
        # Create minimal metadata
        metadata = ASTMetadata(
            file_size=batch_file.file_size,
            modified_time=datetime.now()
        )
        
        # Create language info (simplified)
        language_info = LanguageInfo(
            language=LanguageType.UNKNOWN,
            confidence=0.5,
            detection_method="batch_processing"
        )
        
        return ASTFile(
            path=batch_file.file_path,
            language_info=language_info,
            metadata=metadata,
            status=FileStatus.DISCOVERED
        )
    
    async def _check_resource_limits(self) -> None:
        """Check if resource limits are being exceeded."""
        try:
            # Check memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > self.config.memory_limit_mb:
                raise WorkerResourceError(
                    f"Memory limit exceeded: {memory_mb:.1f}MB > {self.config.memory_limit_mb}MB"
                )
            
            # Check CPU usage (averaged over time)
            cpu_percent = self.process.cpu_percent()
            # Note: CPU limit checking would need more sophisticated logic in production
            
        except psutil.Error as e:
            self.logger.warning(f"Failed to check resource limits: {e}")
    
    async def _check_memory_usage(self) -> None:
        """Check current memory usage and log if high."""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Update last memory check
            self.last_memory_check = memory_mb
            
            # Log warning if memory usage is high
            warning_threshold = self.config.memory_limit_mb * 0.8
            if memory_mb > warning_threshold:
                self.logger.warning(
                    f"High memory usage: {memory_mb:.1f}MB "
                    f"({memory_mb/self.config.memory_limit_mb*100:.1f}% of limit)"
                )
                
        except psutil.Error as e:
            self.logger.warning(f"Failed to check memory usage: {e}")


class WorkerPool:
    """Pool of workers for parallel batch processing."""
    
    def __init__(
        self,
        config: BatchConfig,
        processor_func: Callable[[List[ASTFile]], List[ASTFile]],
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """Initialize the worker pool.
        
        Args:
            config: Batch processing configuration
            processor_func: Function to process AST files
            checkpoint_manager: Optional checkpoint manager
        """
        self.config = config
        self.processor_func = processor_func
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(__name__)
        
        # Worker management
        self.workers: Dict[str, BatchWorker] = {}
        self.worker_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
        # Queue for batch processing
        self.batch_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.result_queue: asyncio.Queue = asyncio.Queue()
    
    async def start(self) -> None:
        """Start the worker pool."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create workers
        for i in range(self.config.max_workers):
            worker_id = f"worker_{i:03d}_{uuid.uuid4().hex[:8]}"
            worker = BatchWorker(
                worker_id=worker_id,
                config=self.config,
                processor_func=self.processor_func,
                checkpoint_manager=self.checkpoint_manager
            )
            
            self.workers[worker_id] = worker
            await worker.start()
        
        self.logger.info(f"Started worker pool with {len(self.workers)} workers")
    
    async def stop(self) -> None:
        """Stop the worker pool."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all workers
        for worker in self.workers.values():
            await worker.stop()
        
        # Cancel any running tasks
        for task in self.worker_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks.values(), return_exceptions=True)
        
        self.workers.clear()
        self.worker_tasks.clear()
        
        self.logger.info("Stopped worker pool")
    
    async def submit_batch(self, batch: FileBatch) -> None:
        """Submit a batch for processing.
        
        Args:
            batch: The batch to process
            
        Raises:
            asyncio.QueueFull: If the queue is full
        """
        if not self.is_running:
            raise RuntimeError("Worker pool is not running")
        
        await self.batch_queue.put(batch)
        self.logger.debug(f"Submitted batch {batch.batch_id} to queue")
    
    async def get_result(self, timeout: Optional[float] = None) -> BatchResult:
        """Get a batch processing result.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            BatchResult: The result of batch processing
            
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        if timeout:
            return await asyncio.wait_for(self.result_queue.get(), timeout)
        else:
            return await self.result_queue.get()
    
    async def process_batches(self) -> AsyncIterator[BatchResult]:
        """Process batches continuously and yield results.
        
        Yields:
            BatchResult: Results of batch processing
        """
        # Start worker tasks
        worker_tasks = []
        for worker in self.workers.values():
            task = asyncio.create_task(self._worker_loop(worker))
            worker_tasks.append(task)
            self.worker_tasks[worker.worker_id] = task
        
        try:
            # Yield results as they become available
            while self.is_running:
                try:
                    result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                    yield result
                except asyncio.TimeoutError:
                    continue
                    
        finally:
            # Cancel worker tasks
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    async def get_worker_status(self) -> List[WorkerInfo]:
        """Get status of all workers.
        
        Returns:
            List[WorkerInfo]: Status of all workers
        """
        status_list = []
        for worker in self.workers.values():
            status = await worker.get_status()
            status_list.append(status)
        return status_list
    
    async def _worker_loop(self, worker: BatchWorker) -> None:
        """Main loop for a worker to process batches."""
        while self.is_running:
            try:
                # Get batch from queue with timeout
                batch = await asyncio.wait_for(self.batch_queue.get(), timeout=1.0)
                
                # Process the batch
                result = await worker.process_batch(batch)
                
                # Put result in result queue
                await self.result_queue.put(result)
                
                # Mark task as done
                self.batch_queue.task_done()
                
            except asyncio.TimeoutError:
                # No batch available, continue
                continue
            except asyncio.CancelledError:
                # Worker is being cancelled
                break
            except Exception as e:
                self.logger.error(f"Worker {worker.worker_id} error: {e}")
                # Continue processing other batches
