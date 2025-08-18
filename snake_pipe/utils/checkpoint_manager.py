"""Checkpoint manager for resumable batch processing.

This module provides checkpoint management functionality to enable
resumable batch processing operations with state persistence.
"""

import asyncio
import json
import pickle
import gzip
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator
import logging

from snake_pipe.extract.batch_config import CheckpointConfig, ErrorCode
from snake_pipe.extract.batch_models import CheckpointData, FileBatch, BatchStatus


logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint is not found."""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when a checkpoint is corrupted or invalid."""
    pass


class CheckpointManager:
    """Manages checkpoint creation and recovery for batch processing operations.
    
    This class provides functionality to create, save, load, and manage
    checkpoints for resumable batch processing operations.
    """
    
    def __init__(self, config: CheckpointConfig):
        """Initialize the checkpoint manager.
        
        Args:
            config: Checkpoint configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Ensure checkpoint directory exists
        if self.config.enabled:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_checkpoint(
        self,
        job_id: str,
        completed_batches: List[str],
        failed_batches: List[str],
        pending_batches: List[str],
        processing_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CheckpointData:
        """Create a new checkpoint for the given job.
        
        Args:
            job_id: Unique identifier for the processing job
            completed_batches: List of completed batch IDs
            failed_batches: List of failed batch IDs
            pending_batches: List of pending batch IDs
            processing_state: Additional processing state to save
            metadata: Additional metadata to save
            
        Returns:
            CheckpointData: The created checkpoint data
            
        Raises:
            CheckpointError: If checkpoint creation fails
        """
        if not self.config.enabled:
            self.logger.debug("Checkpointing is disabled")
            return CheckpointData(job_id=job_id)
        
        try:
            checkpoint_data = CheckpointData(
                job_id=job_id,
                completed_batches=completed_batches.copy(),
                failed_batches=failed_batches.copy(),
                pending_batches=pending_batches.copy(),
                processing_state=processing_state or {},
                metadata=metadata or {}
            )
            
            # Save checkpoint to disk
            await self._save_checkpoint(checkpoint_data)
            
            self.logger.info(
                f"Created checkpoint {checkpoint_data.checkpoint_id} for job {job_id} "
                f"(progress: {checkpoint_data.progress_percentage:.1f}%)"
            )
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint for job {job_id}: {e}")
            raise CheckpointError(f"Checkpoint creation failed: {e}")
    
    async def save_checkpoint(self, checkpoint_data: CheckpointData) -> None:
        """Save checkpoint data to disk.
        
        Args:
            checkpoint_data: The checkpoint data to save
            
        Raises:
            CheckpointError: If saving fails
        """
        if not self.config.enabled:
            return
        
        try:
            await self._save_checkpoint(checkpoint_data)
            self.logger.debug(f"Saved checkpoint {checkpoint_data.checkpoint_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_data.checkpoint_id}: {e}")
            raise CheckpointError(f"Checkpoint save failed: {e}")
    
    async def load_checkpoint(self, job_id: str, checkpoint_id: Optional[str] = None) -> CheckpointData:
        """Load checkpoint data from disk.
        
        Args:
            job_id: Job ID to load checkpoint for
            checkpoint_id: Specific checkpoint ID (optional, loads latest if None)
            
        Returns:
            CheckpointData: The loaded checkpoint data
            
        Raises:
            CheckpointNotFoundError: If checkpoint is not found
            CheckpointCorruptedError: If checkpoint is corrupted
            CheckpointError: If loading fails
        """
        if not self.config.enabled:
            raise CheckpointNotFoundError("Checkpointing is disabled")
        
        try:
            if checkpoint_id:
                checkpoint_path = self._get_checkpoint_path(job_id, checkpoint_id)
            else:
                checkpoint_path = await self._get_latest_checkpoint_path(job_id)
            
            if not checkpoint_path.exists():
                raise CheckpointNotFoundError(f"Checkpoint not found for job {job_id}")
            
            checkpoint_data = await self._load_checkpoint_file(checkpoint_path)
            
            self.logger.info(
                f"Loaded checkpoint {checkpoint_data.checkpoint_id} for job {job_id} "
                f"(progress: {checkpoint_data.progress_percentage:.1f}%)"
            )
            
            return checkpoint_data
            
        except CheckpointNotFoundError:
            raise
        except CheckpointCorruptedError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for job {job_id}: {e}")
            raise CheckpointError(f"Checkpoint load failed: {e}")
    
    async def list_checkpoints(self, job_id: str) -> List[CheckpointData]:
        """List all checkpoints for a given job.
        
        Args:
            job_id: Job ID to list checkpoints for
            
        Returns:
            List[CheckpointData]: List of available checkpoints
        """
        if not self.config.enabled:
            return []
        
        try:
            checkpoints = []
            job_pattern = f"{job_id}_*.checkpoint"
            
            for checkpoint_path in self.config.checkpoint_dir.glob(job_pattern):
                try:
                    checkpoint_data = await self._load_checkpoint_file(checkpoint_path)
                    checkpoints.append(checkpoint_data)
                except Exception as e:
                    self.logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            
            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints for job {job_id}: {e}")
            return []
    
    async def delete_checkpoint(self, job_id: str, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.
        
        Args:
            job_id: Job ID
            checkpoint_id: Checkpoint ID to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if not self.config.enabled:
            return False
        
        try:
            checkpoint_path = self._get_checkpoint_path(job_id, checkpoint_id)
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                self.logger.info(f"Deleted checkpoint {checkpoint_id} for job {job_id}")
                return True
            else:
                self.logger.warning(f"Checkpoint {checkpoint_id} not found for job {job_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_id} for job {job_id}: {e}")
            return False
    
    async def cleanup_old_checkpoints(self) -> int:
        """Clean up old checkpoints based on retention policy.
        
        Returns:
            int: Number of checkpoints cleaned up
        """
        if not self.config.enabled or not self.config.auto_cleanup:
            return 0
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config.retention_hours)
            cleanup_count = 0
            
            for checkpoint_path in self.config.checkpoint_dir.glob("*.checkpoint"):
                try:
                    # Check file modification time
                    mtime = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
                    
                    if mtime < cutoff_time:
                        checkpoint_path.unlink()
                        cleanup_count += 1
                        self.logger.debug(f"Cleaned up old checkpoint: {checkpoint_path.name}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup checkpoint {checkpoint_path}: {e}")
            
            if cleanup_count > 0:
                self.logger.info(f"Cleaned up {cleanup_count} old checkpoints")
            
            return cleanup_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0
    
    async def get_recovery_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get recovery information for a job.
        
        Args:
            job_id: Job ID to get recovery info for
            
        Returns:
            Optional[Dict[str, Any]]: Recovery information or None if not available
        """
        try:
            checkpoints = await self.list_checkpoints(job_id)
            
            if not checkpoints:
                return None
            
            latest_checkpoint = checkpoints[0]  # Already sorted by timestamp
            
            return {
                "job_id": job_id,
                "latest_checkpoint_id": latest_checkpoint.checkpoint_id,
                "progress_percentage": latest_checkpoint.progress_percentage,
                "timestamp": latest_checkpoint.timestamp.isoformat(),
                "completed_batches": len(latest_checkpoint.completed_batches),
                "failed_batches": len(latest_checkpoint.failed_batches),
                "pending_batches": len(latest_checkpoint.pending_batches),
                "available_checkpoints": len(checkpoints)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get recovery info for job {job_id}: {e}")
            return None
    
    async def _save_checkpoint(self, checkpoint_data: CheckpointData) -> None:
        """Save checkpoint data to disk."""
        checkpoint_path = self._get_checkpoint_path(
            checkpoint_data.job_id, 
            checkpoint_data.checkpoint_id
        )
        
        # Prepare data for serialization
        data = {
            "job_id": checkpoint_data.job_id,
            "checkpoint_id": checkpoint_data.checkpoint_id,
            "timestamp": checkpoint_data.timestamp.isoformat(),
            "completed_batches": checkpoint_data.completed_batches,
            "failed_batches": checkpoint_data.failed_batches,
            "pending_batches": checkpoint_data.pending_batches,
            "processing_state": checkpoint_data.processing_state,
            "metadata": checkpoint_data.metadata
        }
        
        # Serialize based on format
        if self.config.checkpoint_format == "json":
            content = json.dumps(data, indent=2).encode('utf-8')
        elif self.config.checkpoint_format == "pickle":
            content = pickle.dumps(data)
        else:
            raise CheckpointError(f"Unsupported checkpoint format: {self.config.checkpoint_format}")
        
        # Compress if enabled
        if self.config.compression_enabled:
            content = gzip.compress(content)
        
        # Write atomically using temporary file
        temp_path = checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            # Atomic move
            temp_path.replace(checkpoint_path)
            
        except Exception:
            # Cleanup temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    async def _load_checkpoint_file(self, checkpoint_path: Path) -> CheckpointData:
        """Load checkpoint data from a file."""
        try:
            with open(checkpoint_path, 'rb') as f:
                content = f.read()
            
            # Decompress if needed
            if self.config.compression_enabled:
                try:
                    content = gzip.decompress(content)
                except gzip.BadGzipFile:
                    # Try without decompression (backward compatibility)
                    pass
            
            # Deserialize based on format
            if self.config.checkpoint_format == "json":
                data = json.loads(content.decode('utf-8'))
            elif self.config.checkpoint_format == "pickle":
                data = pickle.loads(content)
            else:
                raise CheckpointCorruptedError(f"Unsupported checkpoint format: {self.config.checkpoint_format}")
            
            # Validate required fields
            required_fields = ["job_id", "checkpoint_id", "timestamp"]
            for field in required_fields:
                if field not in data:
                    raise CheckpointCorruptedError(f"Missing required field: {field}")
            
            # Create CheckpointData object
            checkpoint_data = CheckpointData(
                job_id=data["job_id"],
                checkpoint_id=data["checkpoint_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                completed_batches=data.get("completed_batches", []),
                failed_batches=data.get("failed_batches", []),
                pending_batches=data.get("pending_batches", []),
                processing_state=data.get("processing_state", {}),
                metadata=data.get("metadata", {})
            )
            
            return checkpoint_data
            
        except json.JSONDecodeError as e:
            raise CheckpointCorruptedError(f"Invalid JSON in checkpoint: {e}")
        except pickle.PickleError as e:
            raise CheckpointCorruptedError(f"Invalid pickle data in checkpoint: {e}")
        except Exception as e:
            raise CheckpointCorruptedError(f"Failed to load checkpoint: {e}")
    
    def _get_checkpoint_path(self, job_id: str, checkpoint_id: str) -> Path:
        """Get the file path for a checkpoint."""
        filename = f"{job_id}_{checkpoint_id}.checkpoint"
        return self.config.checkpoint_dir / filename
    
    async def _get_latest_checkpoint_path(self, job_id: str) -> Path:
        """Get the path to the latest checkpoint for a job."""
        pattern = f"{job_id}_*.checkpoint"
        checkpoint_paths = list(self.config.checkpoint_dir.glob(pattern))
        
        if not checkpoint_paths:
            raise CheckpointNotFoundError(f"No checkpoints found for job {job_id}")
        
        # Find the newest checkpoint by modification time
        latest_path = max(checkpoint_paths, key=lambda p: p.stat().st_mtime)
        return latest_path


async def create_checkpoint_manager(config: CheckpointConfig) -> CheckpointManager:
    """Factory function to create a checkpoint manager.
    
    Args:
        config: Checkpoint configuration
        
    Returns:
        CheckpointManager: Configured checkpoint manager instance
    """
    manager = CheckpointManager(config)
    
    # Perform initial cleanup if enabled
    if config.auto_cleanup:
        await manager.cleanup_old_checkpoints()
    
    return manager
