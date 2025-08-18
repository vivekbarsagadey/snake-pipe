"""Main batch processing engine for AST file processing.

This module provides the core batch processing engine that coordinates
file discovery, batching, parallel processing, and result aggregation.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, AsyncIterator, Set
import logging
import uuid

from snake_pipe.extract.batch_config import (
    BatchConfig, ProcessingMode, BatchStrategy, ErrorHandlingMode, 
    ErrorCode, BatchPriority
)
from snake_pipe.extract.batch_models import (
    FileBatch, BatchResult, BatchExecutionResult, BatchFile, 
    BatchStatus, BatchType
)
from snake_pipe.extract.batch_worker import WorkerPool
from snake_pipe.extract.models import ASTFile
from snake_pipe.utils.checkpoint_manager import CheckpointManager


logger = logging.getLogger(__name__)


class BatchProcessingError(Exception):
    """Base exception for batch processing operations."""
    pass


class BatchDiscoveryError(BatchProcessingError):
    """Raised when file discovery fails."""
    pass


class BatchCoordinationError(BatchProcessingError):
    """Raised when batch coordination fails."""
    pass


class BatchProcessingEngine:
    """Main engine for batch processing of AST files.
    
    This engine coordinates the entire batch processing pipeline:
    1. File discovery and filtering
    2. Batch creation with adaptive sizing
    3. Parallel processing using worker pool
    4. Result aggregation and checkpointing
    5. Performance monitoring and optimization
    """
    
    def __init__(
        self,
        config: BatchConfig,
        processor_func: Callable[[List[ASTFile]], List[ASTFile]],
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """Initialize the batch processing engine.
        
        Args:
            config: Batch processing configuration
            processor_func: Function to process AST files
            checkpoint_manager: Optional checkpoint manager for persistence
        """
        self.config = config
        self.processor_func = processor_func
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(__name__)
        
        # Engine state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.execution_id = str(uuid.uuid4())
        
        # Processing components
        self.worker_pool: Optional[WorkerPool] = None
        
        # Metrics and monitoring
        self.total_files_discovered = 0
        self.total_files_processed = 0
        self.total_files_successful = 0
        self.total_files_failed = 0
        self.total_processing_time = 0.0
        self.batch_results: List[BatchResult] = []
        
        # File tracking
        self.discovered_files: Set[Path] = set()
        self.processed_files: Set[Path] = set()
        self.failed_files: Set[Path] = set()
        
        # Adaptive batching
        self.current_batch_size = config.batch_size
        self.recent_throughput_history: List[float] = []
    
    async def process_directory(
        self,
        source_path: Path,
        output_path: Optional[Path] = None,
        resume_from_checkpoint: bool = True
    ) -> BatchExecutionResult:
        """Process all AST files in a directory using batch processing.
        
        Args:
            source_path: Directory containing AST files to process
            output_path: Optional output directory for processed files
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            BatchExecutionResult: Results of the batch processing execution
            
        Raises:
            BatchProcessingError: If processing fails
        """
        if self.is_running:
            raise BatchProcessingError("Engine is already running")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            self.logger.info(
                f"Starting batch processing for directory: {source_path} "
                f"(execution_id: {self.execution_id})"
            )
            
            # Initialize worker pool
            self.worker_pool = WorkerPool(
                config=self.config,
                processor_func=self.processor_func,
                checkpoint_manager=self.checkpoint_manager
            )
            await self.worker_pool.start()
            
            # Load checkpoint if requested
            checkpoint_data = None
            if resume_from_checkpoint and self.checkpoint_manager:
                checkpoint_data = await self._load_checkpoint(source_path)
            
            # Discover files
            discovered_files = await self._discover_files(source_path, checkpoint_data)
            
            if not discovered_files:
                self.logger.warning(f"No files discovered in {source_path}")
                return self._create_execution_result()
            
            self.logger.info(f"Discovered {len(discovered_files)} files for processing")
            
            # Create batches
            batches = await self._create_batches(discovered_files)
            self.logger.info(f"Created {len(batches)} batches for processing")
            
            # Process batches
            await self._process_batches(batches)
            
            # Save final checkpoint
            if self.checkpoint_manager:
                await self._save_checkpoint(source_path)
            
            # Create execution result
            result = self._create_execution_result()
            
            self.logger.info(
                f"Batch processing completed: {result.files_successful}/{result.files_processed} successful "
                f"({result.throughput_files_per_second:.1f} files/sec, {result.total_processing_time:.1f}s)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise BatchProcessingError(f"Processing failed: {e}")
        
        finally:
            self.is_running = False
            if self.worker_pool:
                await self.worker_pool.stop()
    
    async def process_files(
        self,
        file_paths: List[Path],
        resume_from_checkpoint: bool = True
    ) -> BatchExecutionResult:
        """Process a specific list of files using batch processing.
        
        Args:
            file_paths: List of file paths to process
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            BatchExecutionResult: Results of the batch processing execution
        """
        if self.is_running:
            raise BatchProcessingError("Engine is already running")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing {len(file_paths)} specific files")
            
            # Initialize worker pool
            self.worker_pool = WorkerPool(
                config=self.config,
                processor_func=self.processor_func,
                checkpoint_manager=self.checkpoint_manager
            )
            await self.worker_pool.start()
            
            # Convert paths to batch files
            batch_files = []
            for file_path in file_paths:
                if file_path.exists():
                    batch_file = BatchFile(
                        file_path=file_path,
                        file_size=file_path.stat().st_size
                    )
                    batch_files.append(batch_file)
                    self.discovered_files.add(file_path)
                else:
                    self.logger.warning(f"File not found: {file_path}")
            
            if not batch_files:
                self.logger.warning("No valid files found for processing")
                return self._create_execution_result()
            
            self.total_files_discovered = len(batch_files)
            
            # Create batches
            batches = await self._create_batches(batch_files)
            
            # Process batches
            await self._process_batches(batches)
            
            return self._create_execution_result()
            
        except Exception as e:
            self.logger.error(f"File processing failed: {e}")
            raise BatchProcessingError(f"Processing failed: {e}")
        
        finally:
            self.is_running = False
            if self.worker_pool:
                await self.worker_pool.stop()
    
    async def _discover_files(
        self,
        source_path: Path,
        checkpoint_data: Optional[Dict[str, Any]] = None
    ) -> List[BatchFile]:
        """Discover AST files in the source directory.
        
        Args:
            source_path: Directory to search for files
            checkpoint_data: Optional checkpoint data for resuming
            
        Returns:
            List[BatchFile]: Discovered files ready for processing
        """
        try:
            discovered_files = []
            processed_files_set = set()
            
            # Load processed files from checkpoint
            if checkpoint_data:
                processed_files_set = set(
                    Path(p) for p in checkpoint_data.get("processed_files", [])
                )
                self.logger.info(f"Resuming: {len(processed_files_set)} files already processed")
            
            # File patterns to search for
            patterns = ["*.json", "*.ast", "*.txt"]  # Add more patterns as needed
            
            for pattern in patterns:
                for file_path in source_path.rglob(pattern):
                    if file_path.is_file() and file_path not in processed_files_set:
                        try:
                            file_size = file_path.stat().st_size
                            
                            # Skip empty files
                            if file_size == 0:
                                self.logger.debug(f"Skipping empty file: {file_path}")
                                continue
                            
                            # Skip very large files if configured
                            max_file_size = getattr(self.config, 'max_file_size_mb', None)
                            if max_file_size and file_size > max_file_size * 1024 * 1024:
                                self.logger.warning(
                                    f"Skipping large file: {file_path} "
                                    f"({file_size / 1024 / 1024:.1f}MB > {max_file_size}MB)"
                                )
                                continue
                            
                            batch_file = BatchFile(
                                file_path=file_path,
                                file_size=file_size
                            )
                            discovered_files.append(batch_file)
                            self.discovered_files.add(file_path)
                            
                        except OSError as e:
                            self.logger.warning(f"Error accessing file {file_path}: {e}")
                            continue
            
            self.total_files_discovered = len(discovered_files)
            self.logger.info(f"Discovered {len(discovered_files)} files for processing")
            
            return discovered_files
            
        except Exception as e:
            raise BatchDiscoveryError(f"File discovery failed: {e}")
    
    async def _create_batches(self, files: List[BatchFile]) -> List[FileBatch]:
        """Create batches from discovered files using the configured strategy.
        
        Args:
            files: List of files to batch
            
        Returns:
            List[FileBatch]: Created batches ready for processing
        """
        if not files:
            return []
        
        batches = []
        
        if self.config.batch_strategy == BatchStrategy.SIZE_BASED:
            batches = await self._create_size_based_batches(files)
        elif self.config.batch_strategy == BatchStrategy.COUNT_BASED:
            batches = await self._create_count_based_batches(files)
        elif self.config.batch_strategy == BatchStrategy.TIME_BASED:
            batches = await self._create_time_based_batches(files)
        elif self.config.batch_strategy == BatchStrategy.ADAPTIVE:
            batches = await self._create_adaptive_batches(files)
        else:
            # Default to count-based
            batches = await self._create_count_based_batches(files)
        
        # Set batch priorities
        for i, batch in enumerate(batches):
            if i < len(batches) * 0.1:  # First 10% high priority
                batch.priority = BatchPriority.HIGH
            elif i < len(batches) * 0.3:  # Next 20% medium priority
                batch.priority = BatchPriority.MEDIUM
            else:
                batch.priority = BatchPriority.LOW
        
        self.logger.info(
            f"Created {len(batches)} batches using {self.config.batch_strategy.value} strategy"
        )
        
        return batches
    
    async def _create_count_based_batches(self, files: List[BatchFile]) -> List[FileBatch]:
        """Create batches based on file count."""
        batches = []
        batch_size = self.current_batch_size
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch = FileBatch(
                batch_id=f"batch_{len(batches):04d}_{uuid.uuid4().hex[:8]}",
                files=batch_files,
                batch_type=BatchType.COUNT_BASED
            )
            batches.append(batch)
        
        return batches
    
    async def _create_size_based_batches(self, files: List[BatchFile]) -> List[FileBatch]:
        """Create batches based on total file size."""
        batches = []
        target_size = self.config.batch_size_mb * 1024 * 1024  # Convert to bytes
        current_batch_files = []
        current_size = 0
        
        for file in files:
            if current_size + file.file_size > target_size and current_batch_files:
                # Create batch
                batch = FileBatch(
                    batch_id=f"batch_{len(batches):04d}_{uuid.uuid4().hex[:8]}",
                    files=current_batch_files.copy(),
                    batch_type=BatchType.SIZE_BASED
                )
                batches.append(batch)
                
                # Start new batch
                current_batch_files = []
                current_size = 0
            
            current_batch_files.append(file)
            current_size += file.file_size
        
        # Add remaining files as final batch
        if current_batch_files:
            batch = FileBatch(
                batch_id=f"batch_{len(batches):04d}_{uuid.uuid4().hex[:8]}",
                files=current_batch_files,
                batch_type=BatchType.SIZE_BASED
            )
            batches.append(batch)
        
        return batches
    
    async def _create_time_based_batches(self, files: List[BatchFile]) -> List[FileBatch]:
        """Create batches that can be processed within time limits."""
        # Estimate processing time per file based on size
        estimated_time_per_mb = 0.1  # seconds per MB (adjust based on profiling)
        target_time = 60.0  # Target 60 seconds per batch
        
        batches = []
        current_batch_files = []
        current_estimated_time = 0.0
        
        for file in files:
            file_time = (file.file_size / 1024 / 1024) * estimated_time_per_mb
            
            if current_estimated_time + file_time > target_time and current_batch_files:
                # Create batch
                batch = FileBatch(
                    batch_id=f"batch_{len(batches):04d}_{uuid.uuid4().hex[:8]}",
                    files=current_batch_files.copy(),
                    batch_type=BatchType.TIME_BASED
                )
                batches.append(batch)
                
                # Start new batch
                current_batch_files = []
                current_estimated_time = 0.0
            
            current_batch_files.append(file)
            current_estimated_time += file_time
        
        # Add remaining files
        if current_batch_files:
            batch = FileBatch(
                batch_id=f"batch_{len(batches):04d}_{uuid.uuid4().hex[:8]}",
                files=current_batch_files,
                batch_type=BatchType.TIME_BASED
            )
            batches.append(batch)
        
        return batches
    
    async def _create_adaptive_batches(self, files: List[BatchFile]) -> List[FileBatch]:
        """Create batches using adaptive sizing based on performance history."""
        # Start with configured batch size
        if not self.recent_throughput_history:
            return await self._create_count_based_batches(files)
        
        # Calculate optimal batch size based on recent performance
        avg_throughput = sum(self.recent_throughput_history) / len(self.recent_throughput_history)
        
        # Adjust batch size based on throughput
        if avg_throughput > 100:  # High throughput - increase batch size
            self.current_batch_size = min(self.config.batch_size * 2, 1000)
        elif avg_throughput < 10:  # Low throughput - decrease batch size
            self.current_batch_size = max(self.config.batch_size // 2, 10)
        else:
            self.current_batch_size = self.config.batch_size
        
        return await self._create_count_based_batches(files)
    
    async def _process_batches(self, batches: List[FileBatch]) -> None:
        """Process all batches using the worker pool.
        
        Args:
            batches: List of batches to process
        """
        if not self.worker_pool:
            raise BatchCoordinationError("Worker pool not initialized")
        
        # Submit all batches to the worker pool
        for batch in batches:
            await self.worker_pool.submit_batch(batch)
        
        # Process results as they become available
        processed_batches = 0
        total_batches = len(batches)
        
        async for result in self.worker_pool.process_batches():
            processed_batches += 1
            self.batch_results.append(result)
            
            # Update metrics
            self.total_files_processed += result.files_processed
            self.total_files_successful += result.files_successful
            self.total_files_failed += result.files_failed
            self.total_processing_time += result.processing_time
            
            # Update throughput history for adaptive batching
            if result.processing_time > 0:
                throughput = result.files_processed / result.processing_time
                self.recent_throughput_history.append(throughput)
                # Keep only recent history
                if len(self.recent_throughput_history) > 10:
                    self.recent_throughput_history.pop(0)
            
            # Log progress
            progress_percent = (processed_batches / total_batches) * 100
            self.logger.info(
                f"Batch progress: {progress_percent:.1f}% "
                f"({processed_batches}/{total_batches} batches, "
                f"{self.total_files_successful}/{self.total_files_processed} files successful)"
            )
            
            # Save checkpoint periodically
            if self.checkpoint_manager and processed_batches % 10 == 0:
                await self._save_partial_checkpoint()
            
            # Check if all batches are processed
            if processed_batches >= total_batches:
                break
    
    async def _save_checkpoint(self, source_path: Path) -> None:
        """Save processing checkpoint."""
        if not self.checkpoint_manager:
            return
        
        checkpoint_data = {
            "execution_id": self.execution_id,
            "source_path": str(source_path),
            "timestamp": datetime.now().isoformat(),
            "total_files_discovered": self.total_files_discovered,
            "total_files_processed": self.total_files_processed,
            "total_files_successful": self.total_files_successful,
            "total_files_failed": self.total_files_failed,
            "processed_files": [str(p) for p in self.processed_files],
            "failed_files": [str(p) for p in self.failed_files],
            "batch_results": [result.__dict__ for result in self.batch_results],
            "config": self.config.__dict__
        }
        
        await self.checkpoint_manager.save_checkpoint(
            f"batch_processing_{self.execution_id}",
            checkpoint_data
        )
    
    async def _save_partial_checkpoint(self) -> None:
        """Save partial checkpoint during processing."""
        if not self.checkpoint_manager:
            return
        
        # This could save a lightweight checkpoint
        # with just the essential progress information
        pass
    
    async def _load_checkpoint(self, source_path: Path) -> Optional[Dict[str, Any]]:
        """Load processing checkpoint."""
        if not self.checkpoint_manager:
            return None
        
        try:
            # Look for recent checkpoints for this source path
            checkpoints = await self.checkpoint_manager.list_checkpoints()
            
            for checkpoint_name in checkpoints:
                if "batch_processing" in checkpoint_name:
                    checkpoint_data = await self.checkpoint_manager.load_checkpoint(checkpoint_name)
                    if checkpoint_data and checkpoint_data.get("source_path") == str(source_path):
                        self.logger.info(f"Loaded checkpoint: {checkpoint_name}")
                        return checkpoint_data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def _create_execution_result(self) -> BatchExecutionResult:
        """Create the final execution result."""
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds() if self.start_time else 0.0
        
        return BatchExecutionResult(
            execution_id=self.execution_id,
            start_time=self.start_time or datetime.now(),
            end_time=end_time,
            total_processing_time=total_time,
            files_discovered=self.total_files_discovered,
            files_processed=self.total_files_processed,
            files_successful=self.total_files_successful,
            files_failed=self.total_files_failed,
            throughput_files_per_second=self.total_files_processed / total_time if total_time > 0 else 0,
            batch_results=self.batch_results,
            config=self.config
        )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics."""
        worker_status = []
        if self.worker_pool:
            worker_status = await self.worker_pool.get_worker_status()
        
        return {
            "execution_id": self.execution_id,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "total_files_discovered": self.total_files_discovered,
            "total_files_processed": self.total_files_processed,
            "total_files_successful": self.total_files_successful,
            "total_files_failed": self.total_files_failed,
            "throughput_files_per_second": (
                self.total_files_processed / self.total_processing_time 
                if self.total_processing_time > 0 else 0
            ),
            "worker_count": len(worker_status),
            "worker_status": [status.__dict__ for status in worker_status],
            "batch_count": len(self.batch_results),
            "current_batch_size": self.current_batch_size
        }
