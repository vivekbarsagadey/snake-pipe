"""Real-time file watcher service for AST JSON files.

This module provides real-time file system monitoring capabilities using the watchdog library
for immediate detection and processing of AST file changes in development workflows.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from snake_pipe.extract.models import LanguageType, FileStatus
from snake_pipe.extract.event_processor import FileEventProcessor, FileEvent
from snake_pipe.config.extract_config import ExtractConfig

logger = logging.getLogger(__name__)


@dataclass
class WatcherStats:
    """Statistics for file watcher monitoring."""
    events_processed: int = 0
    batches_processed: int = 0
    processing_errors: int = 0
    files_detected: int = 0
    start_time: float = field(default_factory=time.time)
    
    def get_runtime_seconds(self) -> float:
        """Get total runtime in seconds."""
        return time.time() - self.start_time
    
    def get_events_per_second(self) -> float:
        """Calculate events processed per second."""
        runtime = self.get_runtime_seconds()
        return self.events_processed / runtime if runtime > 0 else 0.0


@dataclass
class WatcherConfig:
    """Configuration for real-time file watcher."""
    ast_file_patterns: List[str] = field(default_factory=lambda: [
        '*.json', '**/*.ast.json', '**/ast/*.json'
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '**/.git/**', '**/node_modules/**', '**/build/**',
        '**/dist/**', '**/__pycache__/**', '**/*.tmp', 
        '**/*.swp', '**/*.log', '**/coverage/**'
    ])
    recursive_monitoring: bool = True
    debounce_period: float = 0.5
    batch_size: int = 100
    batch_timeout: float = 2.0
    queue_size: int = 10000
    max_queue_wait: float = 5.0
    
    @classmethod
    def from_extract_config(cls, extract_config: ExtractConfig) -> 'WatcherConfig':
        """Create watcher config from extract configuration."""
        return cls(
            ast_file_patterns=['*.json'],  # Focus on JSON files
            recursive_monitoring=True,
            debounce_period=0.3,  # Faster response for real-time
            batch_size=50,  # Smaller batches for responsiveness
            batch_timeout=1.0  # Shorter timeout for real-time processing
        )


class ASTFileEventHandler(FileSystemEventHandler):
    """Handle file system events and filter for relevant AST files."""
    
    def __init__(self, event_queue: asyncio.Queue, config: WatcherConfig):
        super().__init__()
        self.event_queue = event_queue
        self.config = config
        self.debounce_cache: Dict[str, float] = {}
        self.event_processor = FileEventProcessor(config)
        self._loop = None
        
    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio event loop for async operations."""
        self._loop = loop
    
    def on_any_event(self, event: FileSystemEvent) -> None:
        """Handle any file system event with filtering and debouncing."""
        try:
            # Skip directory events
            if event.is_directory:
                return
            
            # Filter for relevant AST files
            if not self._is_relevant_file(event.src_path):
                return
            
            # Apply debouncing
            if self._should_debounce(event.src_path):
                return
            
            # Create file event
            file_event = FileEvent(
                path=Path(event.src_path),
                event_type=self._normalize_event_type(event.event_type),
                timestamp=time.time(),
                file_size=self._get_file_size(event.src_path)
            )
            
            # Queue event asynchronously if loop is available
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._queue_event(file_event), 
                    self._loop
                )
            
        except Exception as e:
            logger.error(f"Error handling file system event for {event.src_path}: {e}")
    
    def _normalize_event_type(self, event_type: str) -> str:
        """Normalize watchdog event types to standard names."""
        event_mapping = {
            'created': 'created',
            'modified': 'modified', 
            'deleted': 'deleted',
            'moved': 'moved'
        }
        
        # Extract event type from watchdog's format
        clean_type = event_type.lower()
        for key, value in event_mapping.items():
            if key in clean_type:
                return value
        
        return 'modified'  # Default fallback
    
    def _is_relevant_file(self, file_path: str) -> bool:
        """Check if file matches AST patterns and isn't excluded."""
        path_obj = Path(file_path)
        
        # Check file extension and patterns
        if not any(path_obj.match(pattern) for pattern in self.config.ast_file_patterns):
            return False
        
        # Check exclusion patterns
        if any(path_obj.match(pattern) for pattern in self.config.exclude_patterns):
            return False
        
        return True
    
    def _should_debounce(self, file_path: str) -> bool:
        """Apply debouncing to prevent rapid duplicate events."""
        current_time = time.time()
        last_event_time = self.debounce_cache.get(file_path, 0)
        
        if current_time - last_event_time < self.config.debounce_period:
            return True  # Skip this event
        
        self.debounce_cache[file_path] = current_time
        
        # Clean old entries from debounce cache periodically
        if len(self.debounce_cache) > 10000:
            cutoff_time = current_time - (self.config.debounce_period * 2)
            self.debounce_cache = {
                path: timestamp for path, timestamp in self.debounce_cache.items()
                if timestamp > cutoff_time
            }
        
        return False
    
    def _get_file_size(self, file_path: str) -> Optional[int]:
        """Get file size safely."""
        try:
            return Path(file_path).stat().st_size
        except (OSError, FileNotFoundError):
            return None
    
    async def _queue_event(self, event: FileEvent) -> None:
        """Queue file event for processing."""
        try:
            # Use put_nowait to avoid blocking the file system thread
            if self.event_queue.qsize() < self.config.queue_size:
                self.event_queue.put_nowait(event)
                logger.debug(f"Queued file event: {event.path} ({event.event_type})")
            else:
                logger.warning(f"Event queue full, dropping event for {event.path}")
                
        except Exception as e:
            logger.error(f"Failed to queue file event {event.path}: {e}")


class RealtimeFileWatcher:
    """Real-time file watcher for AST JSON files with async processing."""
    
    def __init__(self, config: WatcherConfig):
        self.config = config
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[ASTFileEventHandler] = None
        self.running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.stats = WatcherStats()
        self.watch_paths: List[Path] = []
    
    async def start_monitoring(
        self, 
        watch_paths: List[Path], 
        event_callback: Callable[[List[FileEvent]], None]
    ) -> None:
        """Start real-time monitoring of specified directories."""
        logger.info(f"Starting file watcher for {len(watch_paths)} paths")
        
        self.watch_paths = watch_paths
        
        # Initialize event handler with current event loop
        self.event_handler = ASTFileEventHandler(self.event_queue, self.config)
        self.event_handler.set_event_loop(asyncio.get_event_loop())
        
        # Create and configure observer
        self.observer = Observer()
        
        # Add watch paths
        monitored_paths = 0
        for watch_path in watch_paths:
            if watch_path.exists() and watch_path.is_dir():
                self.observer.schedule(
                    self.event_handler,
                    str(watch_path),
                    recursive=self.config.recursive_monitoring
                )
                logger.info(f"Monitoring path: {watch_path}")
                monitored_paths += 1
            else:
                logger.warning(f"Watch path does not exist or is not a directory: {watch_path}")
        
        if monitored_paths == 0:
            raise ValueError("No valid watch paths found")
        
        # Start observer
        self.observer.start()
        self.running = True
        
        # Start event processing task
        self.processing_task = asyncio.create_task(
            self._process_events_loop(event_callback)
        )
        
        logger.info(f"File watcher started successfully, monitoring {monitored_paths} paths")
    
    async def _process_events_loop(self, event_callback: Callable[[List[FileEvent]], None]) -> None:
        """Process file events in batches with timeout handling."""
        batch = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=self.config.batch_timeout
                    )
                    batch.append(event)
                    self.stats.events_processed += 1
                    
                except asyncio.TimeoutError:
                    # Process batch on timeout
                    if batch:
                        await self._process_batch(batch, event_callback)
                        batch.clear()
                        last_batch_time = time.time()
                    continue
                
                # Process batch when full or timeout reached
                current_time = time.time()
                if (len(batch) >= self.config.batch_size or 
                    current_time - last_batch_time >= self.config.batch_timeout):
                    
                    await self._process_batch(batch, event_callback)
                    batch.clear()
                    last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                self.stats.processing_errors += 1
                await asyncio.sleep(0.1)  # Brief delay before retry
        
        # Process remaining events on shutdown
        if batch:
            await self._process_batch(batch, event_callback)
        
        logger.info("Event processing loop stopped")
    
    async def _process_batch(
        self, 
        events: List[FileEvent], 
        callback: Callable[[List[FileEvent]], None]
    ) -> None:
        """Process a batch of file events."""
        if not events:
            return
        
        try:
            logger.debug(f"Processing batch of {len(events)} file events")
            
            # Remove duplicates and filter events
            filtered_events = self._deduplicate_and_filter_events(events)
            self.stats.files_detected += len(filtered_events)
            
            # Call event callback in executor to avoid blocking
            if filtered_events:
                await asyncio.get_event_loop().run_in_executor(
                    None, callback, filtered_events
                )
                self.stats.batches_processed += 1
                logger.debug(f"Successfully processed batch of {len(filtered_events)} events")
                
        except Exception as e:
            logger.error(f"Failed to process event batch: {e}")
            self.stats.processing_errors += 1
    
    def _deduplicate_and_filter_events(self, events: List[FileEvent]) -> List[FileEvent]:
        """Remove duplicate events and apply additional filtering."""
        # Keep only the most recent event per file path
        path_to_event = {}
        for event in events:
            key = str(event.path)
            if key not in path_to_event or event.timestamp > path_to_event[key].timestamp:
                path_to_event[key] = event
        
        # Filter out deleted files that don't exist
        filtered_events = []
        for event in path_to_event.values():
            if event.event_type == 'deleted' or event.path.exists():
                filtered_events.append(event)
        
        return filtered_events
    
    async def stop_monitoring(self) -> None:
        """Stop file monitoring and clean up resources."""
        logger.info("Stopping file watcher")
        self.running = False
        
        # Stop observer first
        if self.observer:
            self.observer.stop()
            # Use run_in_executor to avoid blocking async context
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.observer.join(timeout=5)
            )
        
        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Log final statistics
        runtime = self.stats.get_runtime_seconds()
        logger.info(
            f"File watcher stopped. Runtime: {runtime:.2f}s, "
            f"Events: {self.stats.events_processed}, "
            f"Files detected: {self.stats.files_detected}, "
            f"Batches: {self.stats.batches_processed}, "
            f"Errors: {self.stats.processing_errors}, "
            f"Events/sec: {self.stats.get_events_per_second():.2f}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        return {
            'events_processed': self.stats.events_processed,
            'batches_processed': self.stats.batches_processed,
            'processing_errors': self.stats.processing_errors,
            'files_detected': self.stats.files_detected,
            'queue_size': self.event_queue.qsize(),
            'runtime_seconds': self.stats.get_runtime_seconds(),
            'events_per_second': self.stats.get_events_per_second(),
            'watch_paths': [str(path) for path in self.watch_paths],
            'is_running': self.running
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        health_status = {
            'status': 'healthy' if self.running else 'stopped',
            'observer_alive': self.observer.is_alive() if self.observer else False,
            'queue_size': self.event_queue.qsize(),
            'queue_capacity': self.config.queue_size,
            'processing_task_done': self.processing_task.done() if self.processing_task else True,
            'stats': self.get_stats()
        }
        
        # Check for potential issues
        if self.event_queue.qsize() > self.config.queue_size * 0.8:
            health_status['warnings'] = ['Event queue is near capacity']
        
        if self.stats.processing_errors > 10:
            health_status['warnings'] = health_status.get('warnings', []) + ['High error count']
        
        return health_status


def create_file_watcher(config: Optional[WatcherConfig] = None) -> RealtimeFileWatcher:
    """Factory function to create a configured file watcher."""
    if config is None:
        config = WatcherConfig()
    
    return RealtimeFileWatcher(config)


def create_watcher_from_extract_config(extract_config: ExtractConfig) -> RealtimeFileWatcher:
    """Create file watcher from extract configuration."""
    watcher_config = WatcherConfig.from_extract_config(extract_config)
    return RealtimeFileWatcher(watcher_config)
