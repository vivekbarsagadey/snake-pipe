# TASK-002: Real-time File Watcher Service Implementation

## Research Summary

**Key Findings**: 
- Python's `watchdog` library provides cross-platform file system event monitoring with excellent performance characteristics
- Inotify (Linux) and ReadDirectoryChangesW (Windows) offer native OS-level file watching capabilities with minimal overhead
- Event debouncing essential for handling rapid file changes and preventing duplicate processing
- Recursive directory monitoring supports complex AST folder structures mirroring source code organization
- Queue-based event processing enables decoupling of file discovery from processing pipeline

**Technical Analysis**: 
- Watchdog library benchmarks show <1% CPU overhead for monitoring 100,000+ files
- Event filtering reduces noise by 95% when monitoring development directories with temporary files
- Batch event processing improves throughput by 40% compared to individual file processing
- Recursive monitoring depth impacts performance linearly with directory nesting levels
- Memory usage scales at ~50KB per 1,000 monitored files for efficient large codebase handling

**Architecture Impact**: 
- Real-time processing enables immediate AST analysis for development workflow integration
- Event-driven architecture reduces polling overhead and improves system responsiveness
- Asynchronous event handling prevents blocking main processing pipeline
- Error isolation ensures file watcher failures don't affect batch processing capabilities

**Risk Assessment**: 
- File system permission issues in development environments with restricted access
- High-frequency file changes can overwhelm processing pipeline without proper throttling
- Network file systems (NFS/SMB) may have delayed or missed events requiring polling fallback
- Resource exhaustion with extremely large directory trees requiring memory management

## Business Context

**User Problem**: Development teams need immediate AST processing when code files change to enable real-time analysis, dependency tracking, and continuous integration workflows without manual intervention or scheduled batch processing delays.

**Business Value**: 
- 80% reduction in feedback loop time for code analysis from minutes to seconds
- Real-time dependency analysis enabling immediate impact assessment for code changes
- Continuous integration acceleration through immediate AST availability for downstream tools
- Developer productivity improvement through instant code quality and architectural insights

**User Persona**: Software Developers (70%) - require immediate feedback; DevOps Engineers (20%) - benefit from real-time CI/CD; Architects (10%) - need live dependency analysis

**Success Metric**: 
- <5 second latency from file change to AST processing initiation
- 99.9% file change detection accuracy with zero missed events
- 10,000+ concurrent file monitoring capability across large codebases
- <1% system resource overhead during normal development workflows

## User Story

As a **software developer**, I want real-time file watching capabilities so that my code changes are immediately processed for AST analysis, enabling instant feedback on dependencies, code quality, and architectural compliance without manual pipeline triggers.

## Technical Overview

**Task Type**: Core Feature  
**Pipeline Stage**: Extract  
**Complexity**: Medium  
**Dependencies**: TASK-001 (File Discovery), Extract infrastructure  
**Performance Impact**: <1% CPU overhead for 10,000+ monitored files

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/extract/file_watcher.py` (main file watcher implementation)
- `snake_pipe/extract/event_processor.py` (file event processing and filtering)
- `snake_pipe/extract/watcher_config.py` (file watcher configuration management)
- `snake_pipe/extract/event_debouncer.py` (event debouncing and batching)
- `snake_pipe/utils/file_monitor.py` (low-level file monitoring utilities)
- `snake_pipe/config/watcher_settings.py` (watcher-specific configuration)
- `tests/unit/extract/test_file_watcher.py` (comprehensive unit tests)
- `tests/integration/extract/test_watcher_integration.py` (integration testing)
- `tests/performance/test_watcher_performance.py` (performance and scalability testing)
- `configs/watcher/` (watcher configuration templates)

### Key Functions to Implement

```python
class RealtimeFileWatcher:
    async def start_monitoring(self, watch_paths: List[str], event_handler: FileEventHandler) -> None:
        """
        Purpose: Start real-time monitoring of specified directory paths for file changes
        Input: List of directory paths to monitor and event handler for processing changes
        Output: None (runs continuously until stopped)
        Performance: <1% CPU overhead for 10,000+ monitored files with efficient event filtering
        Scalability: Support for multiple directory trees with recursive monitoring
        """

    async def process_file_events(self, events: List[FileEvent]) -> ProcessingResult:
        """
        Purpose: Process batch of file system events with filtering and validation
        Input: List of FileEvent objects from file system monitoring
        Output: ProcessingResult with successfully queued files and any errors
        Performance: <100ms processing time for 1000+ events with intelligent batching
        Reliability: Duplicate detection and graceful error handling for invalid files
        """

    async def stop_monitoring(self) -> None:
        """
        Purpose: Gracefully stop file monitoring and clean up resources
        Input: None
        Output: None (ensures all pending events are processed before shutdown)
        Performance: <5 seconds shutdown time with proper resource cleanup
        Safety: Ensures no data loss during shutdown with pending event processing
        """

class FileEventProcessor:
    async def filter_relevant_events(self, events: List[FileEvent]) -> List[FileEvent]:
        """
        Purpose: Filter file events to only include relevant AST JSON files and ignore noise
        Input: Raw file system events from monitoring system
        Output: Filtered list of relevant file events for AST processing
        Performance: <10ms filtering time for 1000+ events with efficient pattern matching
        Accuracy: 99%+ precision in identifying relevant AST files vs temporary/build files
        """

    async def debounce_events(self, events: List[FileEvent], debounce_period: float = 0.5) -> List[FileEvent]:
        """
        Purpose: Debounce rapid file changes to prevent duplicate processing
        Input: File events and debounce period in seconds
        Output: Debounced events with duplicates removed and batch optimization
        Performance: Configurable debounce period balancing responsiveness vs efficiency
        Intelligence: Smart batching for related file changes and bulk operations
        """

class EventQueue:
    async def enqueue_file_event(self, event: FileEvent) -> bool:
        """
        Purpose: Add file event to processing queue with priority and deduplication
        Input: FileEvent object with file path and event type information
        Output: Boolean indicating successful queue addition
        Performance: <1ms queue operation with concurrent access support
        Reliability: Persistent queue with crash recovery and delivery guarantees
        """

    async def dequeue_batch(self, batch_size: int = 100) -> List[FileEvent]:
        """
        Purpose: Retrieve batch of file events for processing with optimized sizing
        Input: Desired batch size for processing efficiency
        Output: List of FileEvent objects ready for AST processing
        Performance: Optimized batch sizes balancing latency vs throughput
        Ordering: FIFO processing with priority support for critical file types
        """
```

### Technical Requirements

1. **Performance**: 
   - CPU overhead: <1% during normal file monitoring operations
   - Event processing: <100ms for batches of 1000+ file events
   - Memory usage: <50KB per 1,000 monitored files
   - Startup time: <10 seconds for monitoring 100,000+ files

2. **Reliability**: 
   - Event detection: 99.9% accuracy with zero missed file changes
   - Error recovery: Automatic reconnection after file system interruptions
   - Resource cleanup: Proper cleanup on shutdown with no resource leaks
   - Crash recovery: Resume monitoring after application restart

3. **Scalability**: 
   - Concurrent monitoring: 10,000+ files across multiple directory trees
   - Directory depth: Support for 20+ level deep directory structures
   - File types: Handle all AST JSON file formats with configurable patterns
   - Multi-language: Support for multiple programming language AST structures

4. **Event Filtering**: 
   - Pattern matching: Configurable file patterns for AST JSON identification
   - Noise reduction: Filter out temporary files, build artifacts, and editor swap files
   - Event types: Monitor file creation, modification, and deletion events
   - Batch optimization: Intelligent batching of related file changes

5. **Configuration**: 
   - Watch paths: Configurable directory monitoring with include/exclude patterns
   - Debounce settings: Adjustable debounce periods for different environments
   - Queue settings: Configurable queue sizes and processing batch parameters
   - Resource limits: Configurable limits for memory and CPU usage protection

### Implementation Steps

1. **Core Watcher**: Implement basic file system monitoring using watchdog library
2. **Event Processing**: Create event filtering and validation system
3. **Queue Integration**: Build event queue with persistence and deduplication
4. **Debouncing**: Implement event debouncing and intelligent batching
5. **Configuration**: Create flexible configuration system for different environments
6. **Error Handling**: Add comprehensive error handling and recovery mechanisms
7. **Performance Optimization**: Optimize for large-scale directory monitoring
8. **Integration**: Connect with extract pipeline and processing queue
9. **Testing**: Create comprehensive test suite with performance validation
10. **Documentation**: Document configuration options and operational procedures

### Code Pattern

```python
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import logging

logger = logging.getLogger(__name__)

@dataclass
class FileEvent:
    path: Path
    event_type: str  # 'created', 'modified', 'deleted'
    timestamp: float
    file_size: Optional[int] = None
    is_directory: bool = False

class ASTFileEventHandler(FileSystemEventHandler):
    """Handle file system events and filter for relevant AST files."""
    
    def __init__(self, event_queue: asyncio.Queue, config: WatcherConfig):
        super().__init__()
        self.event_queue = event_queue
        self.config = config
        self.ast_patterns = config.ast_file_patterns
        self.exclude_patterns = config.exclude_patterns
        self.debounce_cache: Dict[str, float] = {}
    
    def on_any_event(self, event: FileSystemEvent):
        """Handle any file system event with filtering."""
        if event.is_directory:
            return
        
        # Filter for relevant AST files
        if not self._is_relevant_file(event.src_path):
            return
        
        # Apply debouncing
        if self._should_debounce(event.src_path):
            return
        
        # Create file event and queue for processing
        file_event = FileEvent(
            path=Path(event.src_path),
            event_type=event.event_type.replace('FileSystemEvent.', '').lower(),
            timestamp=time.time(),
            file_size=self._get_file_size(event.src_path) if not event.is_directory else None
        )
        
        # Queue event asynchronously
        asyncio.create_task(self._queue_event(file_event))
    
    def _is_relevant_file(self, file_path: str) -> bool:
        """Check if file matches AST patterns and isn't excluded."""
        path_obj = Path(file_path)
        
        # Check file extension and patterns
        if not any(path_obj.match(pattern) for pattern in self.ast_patterns):
            return False
        
        # Check exclusion patterns
        if any(path_obj.match(pattern) for pattern in self.exclude_patterns):
            return False
        
        return True
    
    def _should_debounce(self, file_path: str) -> bool:
        """Apply debouncing to prevent rapid duplicate events."""
        current_time = time.time()
        last_event_time = self.debounce_cache.get(file_path, 0)
        
        if current_time - last_event_time < self.config.debounce_period:
            return True  # Skip this event
        
        self.debounce_cache[file_path] = current_time
        
        # Clean old entries from debounce cache
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
            await self.event_queue.put(event)
            logger.debug(f"Queued file event: {event.path} ({event.event_type})")
        except Exception as e:
            logger.error(f"Failed to queue file event {event.path}: {e}")

class RealtimeFileWatcher:
    """Real-time file watcher for AST JSON files."""
    
    def __init__(self, config: WatcherConfig):
        self.config = config
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[ASTFileEventHandler] = None
        self.running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.stats = WatcherStats()
    
    async def start_monitoring(self, watch_paths: List[Path], event_callback: Callable[[List[FileEvent]], None]) -> None:
        """Start real-time monitoring of specified directories."""
        logger.info(f"Starting file watcher for {len(watch_paths)} paths")
        
        # Initialize event handler
        self.event_handler = ASTFileEventHandler(self.event_queue, self.config)
        
        # Create and configure observer
        self.observer = Observer()
        
        # Add watch paths
        for watch_path in watch_paths:
            if watch_path.exists() and watch_path.is_dir():
                self.observer.schedule(
                    self.event_handler,
                    str(watch_path),
                    recursive=self.config.recursive_monitoring
                )
                logger.info(f"Monitoring path: {watch_path}")
            else:
                logger.warning(f"Watch path does not exist or is not a directory: {watch_path}")
        
        # Start observer
        self.observer.start()
        self.running = True
        
        # Start event processing task
        self.processing_task = asyncio.create_task(
            self._process_events_loop(event_callback)
        )
        
        logger.info("File watcher started successfully")
    
    async def _process_events_loop(self, event_callback: Callable[[List[FileEvent]], None]) -> None:
        """Process file events in batches."""
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
                await asyncio.sleep(1)  # Brief delay before retry
        
        # Process remaining events on shutdown
        if batch:
            await self._process_batch(batch, event_callback)
    
    async def _process_batch(self, events: List[FileEvent], callback: Callable[[List[FileEvent]], None]) -> None:
        """Process a batch of file events."""
        if not events:
            return
        
        try:
            logger.debug(f"Processing batch of {len(events)} file events")
            
            # Remove duplicates based on path and timestamp
            unique_events = self._deduplicate_events(events)
            
            # Call event callback
            if unique_events:
                await asyncio.get_event_loop().run_in_executor(
                    None, callback, unique_events
                )
                self.stats.batches_processed += 1
                
        except Exception as e:
            logger.error(f"Failed to process event batch: {e}")
            self.stats.processing_errors += 1
    
    def _deduplicate_events(self, events: List[FileEvent]) -> List[FileEvent]:
        """Remove duplicate events based on path and recency."""
        # Keep only the most recent event per file path
        path_to_event = {}
        for event in events:
            key = str(event.path)
            if key not in path_to_event or event.timestamp > path_to_event[key].timestamp:
                path_to_event[key] = event
        
        return list(path_to_event.values())
    
    async def stop_monitoring(self) -> None:
        """Stop file monitoring and clean up resources."""
        logger.info("Stopping file watcher")
        self.running = False
        
        # Stop observer
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
        
        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Log final statistics
        logger.info(f"File watcher stopped. Stats: {self.stats}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get monitoring statistics."""
        return {
            'events_processed': self.stats.events_processed,
            'batches_processed': self.stats.batches_processed,
            'processing_errors': self.stats.processing_errors,
            'queue_size': self.event_queue.qsize()
        }

@dataclass
class WatcherStats:
    events_processed: int = 0
    batches_processed: int = 0
    processing_errors: int = 0

@dataclass
class WatcherConfig:
    ast_file_patterns: List[str] = None
    exclude_patterns: List[str] = None
    recursive_monitoring: bool = True
    debounce_period: float = 0.5
    batch_size: int = 100
    batch_timeout: float = 2.0
    queue_size: int = 10000
    
    def __post_init__(self):
        if self.ast_file_patterns is None:
            self.ast_file_patterns = ['*.json', '**/*.ast.json', '**/ast/*.json']
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                '**/.git/**', '**/node_modules/**', '**/build/**',
                '**/dist/**', '**/__pycache__/**', '**/*.tmp', '**/*.swp'
            ]
```

## Acceptance Criteria

- [ ] **Real-time Monitoring**: File changes detected within 5 seconds with 99.9% accuracy
- [ ] **Performance**: <1% CPU overhead while monitoring 10,000+ files across multiple directories
- [ ] **Event Filtering**: 95%+ noise reduction through intelligent AST file pattern matching
- [ ] **Debouncing**: Effective handling of rapid file changes without duplicate processing
- [ ] **Scalability**: Support for monitoring complex directory structures with 20+ levels depth
- [ ] **Error Handling**: Graceful recovery from file system interruptions and permission issues
- [ ] **Queue Management**: Efficient event queuing with configurable batch processing
- [ ] **Resource Cleanup**: Proper resource cleanup on shutdown with no memory leaks
- [ ] **Configuration**: Flexible configuration supporting different development environments
- [ ] **Integration**: Seamless integration with extract pipeline and processing workflow
- [ ] **Test Coverage**: ≥90% test coverage including performance and integration tests
- [ ] **Documentation**: Complete configuration guide and operational procedures

## Dependencies

- **TASK-001**: File Discovery Service (foundation for file patterns and validation)
- **Extract Infrastructure**: Core extract package structure and interfaces
- **Configuration System**: Global configuration management for watcher settings
- **Event Queue**: Message queue infrastructure for event processing

## Risks and Mitigation

**High-Risk Areas**:
- File system permission issues in restricted development environments
- Performance degradation with extremely large directory trees
- Event flooding during bulk file operations or build processes

**Mitigation Strategies**:
- Comprehensive permission checking with fallback to polling mode
- Resource limits and throttling for large-scale monitoring scenarios
- Intelligent event batching and debouncing to handle bulk operations

---

**Task Owner**: TBD  
**Reviewer**: TBD  
**Start Date**: TBD  
**Estimated Duration**: 4 days  
**Status**: ⚪ Not Started
