"""File event processing and filtering for real-time AST monitoring.

This module handles file system events, applies filtering logic, and manages
event debouncing for efficient real-time AST file processing.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, TYPE_CHECKING
from enum import Enum, auto

from snake_pipe.extract.models import LanguageType, FileStatus

if TYPE_CHECKING:
    from snake_pipe.extract.file_watcher import WatcherConfig

logger = logging.getLogger(__name__)


class EventType(Enum):
    """File system event types."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


class EventPriority(Enum):
    """Event processing priority levels."""
    HIGH = auto()      # New AST files, critical changes
    MEDIUM = auto()    # Updates to existing files
    LOW = auto()       # Bulk operations, minor changes


@dataclass
class FileEvent:
    """Represents a file system event for AST processing."""
    path: Path
    event_type: str
    timestamp: float
    file_size: Optional[int] = None
    is_directory: bool = False
    priority: EventPriority = EventPriority.MEDIUM
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}
        
        # Set priority based on event type and file characteristics
        self._set_priority()
    
    def _set_priority(self) -> None:
        """Determine event priority based on characteristics."""
        if self.event_type == EventType.CREATED.value:
            self.priority = EventPriority.HIGH
        elif self.event_type == EventType.DELETED.value:
            self.priority = EventPriority.LOW
        elif self.file_size and self.file_size > 1024 * 1024:  # Files > 1MB
            self.priority = EventPriority.LOW
        else:
            self.priority = EventPriority.MEDIUM
    
    def is_valid(self) -> bool:
        """Check if the event represents a valid file operation."""
        try:
            if self.event_type == EventType.DELETED.value:
                return True  # Deleted files don't need to exist
            
            return self.path.exists() and self.path.is_file()
        except (OSError, PermissionError):
            return False
    
    def get_language_hint(self) -> LanguageType:
        """Get language hint from file path and metadata."""
        # Extract language from path patterns
        path_str = str(self.path).lower()
        
        if 'java' in path_str or self.path.suffix == '.java':
            return LanguageType.JAVA
        elif 'python' in path_str or self.path.suffix == '.py':
            return LanguageType.PYTHON
        elif 'javascript' in path_str or 'js' in path_str:
            return LanguageType.JAVASCRIPT
        elif 'typescript' in path_str or 'ts' in path_str:
            return LanguageType.TYPESCRIPT
        
        return LanguageType.UNKNOWN


@dataclass
class EventBatch:
    """Batch of file events for processing."""
    events: List[FileEvent]
    created_at: float
    priority: EventPriority
    
    def __post_init__(self) -> None:
        """Initialize batch priority based on constituent events."""
        if not self.events:
            self.priority = EventPriority.LOW
            return
        
        # Set batch priority to highest priority event
        priorities = [event.priority for event in self.events]
        if EventPriority.HIGH in priorities:
            self.priority = EventPriority.HIGH
        elif EventPriority.MEDIUM in priorities:
            self.priority = EventPriority.MEDIUM
        else:
            self.priority = EventPriority.LOW
    
    def size(self) -> int:
        """Get number of events in batch."""
        return len(self.events)
    
    def get_unique_files(self) -> Set[Path]:
        """Get set of unique file paths in batch."""
        return {event.path for event in self.events}


class FileEventProcessor:
    """Processes and filters file system events for AST processing."""
    
    def __init__(self, config: 'WatcherConfig'):
        self.config = config
        self.processed_events: Dict[str, float] = {}
        self.stats = {
            'events_processed': 0,
            'events_filtered': 0,
            'events_debounced': 0,
            'batches_created': 0
        }
    
    async def filter_relevant_events(self, events: List[FileEvent]) -> List[FileEvent]:
        """Filter file events to only include relevant AST JSON files."""
        relevant_events = []
        
        for event in events:
            try:
                if await self._is_relevant_event(event):
                    relevant_events.append(event)
                    self.stats['events_processed'] += 1
                else:
                    self.stats['events_filtered'] += 1
                    
            except Exception as e:
                logger.error(f"Error filtering event {event.path}: {e}")
        
        logger.debug(f"Filtered {len(events)} events to {len(relevant_events)} relevant events")
        return relevant_events
    
    async def _is_relevant_event(self, event: FileEvent) -> bool:
        """Check if a single event is relevant for AST processing."""
        # Basic path filtering
        if not self._matches_ast_patterns(event.path):
            return False
        
        # Skip excluded patterns
        if self._matches_exclude_patterns(event.path):
            return False
        
        # Skip directories
        if event.is_directory:
            return False
        
        # For created/modified events, verify it's a valid JSON file
        if event.event_type in [EventType.CREATED.value, EventType.MODIFIED.value]:
            if not await self._is_valid_ast_file(event.path):
                return False
        
        return True
    
    def _matches_ast_patterns(self, path: Path) -> bool:
        """Check if path matches AST file patterns."""
        return any(path.match(pattern) for pattern in self.config.ast_file_patterns)
    
    def _matches_exclude_patterns(self, path: Path) -> bool:
        """Check if path matches exclusion patterns."""
        return any(path.match(pattern) for pattern in self.config.exclude_patterns)
    
    async def _is_valid_ast_file(self, path: Path) -> bool:
        """Validate that file is a proper AST JSON file."""
        try:
            # Check file size
            if not path.exists():
                return False
            
            file_size = path.stat().st_size
            if file_size == 0:
                return False
            
            # For large files, do a lightweight check
            if file_size > 10 * 1024 * 1024:  # 10MB
                return self._quick_json_check(path)
            
            # For smaller files, validate JSON structure
            return await self._validate_json_structure(path)
            
        except (OSError, PermissionError) as e:
            logger.debug(f"Cannot access file {path}: {e}")
            return False
    
    def _quick_json_check(self, path: Path) -> bool:
        """Quick JSON validation for large files."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # Read first 1KB to check if it looks like JSON
                chunk = f.read(1024)
                return chunk.strip().startswith(('{', '['))
        except Exception:
            return False
    
    async def _validate_json_structure(self, path: Path) -> bool:
        """Validate JSON structure asynchronously."""
        try:
            def _load_json():
                with open(path, 'r', encoding='utf-8') as f:
                    json.load(f)
                return True
            
            # Run JSON parsing in executor to avoid blocking
            return await asyncio.get_event_loop().run_in_executor(None, _load_json)
            
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug(f"Invalid JSON file: {path}")
            return False
        except Exception as e:
            logger.debug(f"Error validating JSON {path}: {e}")
            return False
    
    async def debounce_events(
        self, 
        events: List[FileEvent], 
        debounce_period: Optional[float] = None
    ) -> List[FileEvent]:
        """Debounce rapid file changes to prevent duplicate processing."""
        if debounce_period is None:
            debounce_period = self.config.debounce_period
        
        current_time = time.time()
        debounced_events = []
        
        for event in events:
            event_key = f"{event.path}:{event.event_type}"
            last_processed = self.processed_events.get(event_key, 0)
            
            if current_time - last_processed >= debounce_period:
                debounced_events.append(event)
                self.processed_events[event_key] = current_time
                self.stats['events_processed'] += 1
            else:
                self.stats['events_debounced'] += 1
        
        # Clean old entries periodically
        if len(self.processed_events) > 50000:
            cutoff_time = current_time - (debounce_period * 10)
            self.processed_events = {
                key: timestamp for key, timestamp in self.processed_events.items()
                if timestamp > cutoff_time
            }
        
        logger.debug(f"Debounced {len(events)} events to {len(debounced_events)}")
        return debounced_events
    
    async def create_event_batches(
        self, 
        events: List[FileEvent], 
        max_batch_size: Optional[int] = None
    ) -> List[EventBatch]:
        """Create optimized batches from file events."""
        if not events:
            return []
        
        if max_batch_size is None:
            max_batch_size = self.config.batch_size
        
        # Sort events by priority and timestamp
        sorted_events = sorted(
            events, 
            key=lambda e: (e.priority.value, e.timestamp)
        )
        
        batches = []
        current_batch = []
        
        for event in sorted_events:
            current_batch.append(event)
            
            # Create batch when size limit reached or priority changes
            if (len(current_batch) >= max_batch_size or 
                (len(current_batch) > 1 and 
                 current_batch[-1].priority != current_batch[-2].priority)):
                
                batch = EventBatch(
                    events=current_batch.copy(),
                    created_at=time.time(),
                    priority=current_batch[0].priority
                )
                batches.append(batch)
                current_batch.clear()
                self.stats['batches_created'] += 1
        
        # Add remaining events as final batch
        if current_batch:
            batch = EventBatch(
                events=current_batch,
                created_at=time.time(),
                priority=current_batch[0].priority
            )
            batches.append(batch)
            self.stats['batches_created'] += 1
        
        logger.debug(f"Created {len(batches)} batches from {len(events)} events")
        return batches
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get event processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'events_processed': 0,
            'events_filtered': 0,
            'events_debounced': 0,
            'batches_created': 0
        }


class EventQueue:
    """Async queue for file events with priority support."""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self._high_priority: asyncio.Queue = asyncio.Queue(maxsize=maxsize // 3)
        self._medium_priority: asyncio.Queue = asyncio.Queue(maxsize=maxsize // 3)
        self._low_priority: asyncio.Queue = asyncio.Queue(maxsize=maxsize // 3)
        self._stats = {
            'enqueued': 0,
            'dequeued': 0,
            'dropped': 0
        }
    
    async def enqueue_file_event(self, event: FileEvent) -> bool:
        """Add file event to priority queue."""
        try:
            queue = self._get_queue_for_priority(event.priority)
            
            if queue.qsize() >= queue.maxsize:
                # Drop low priority events when queue is full
                if event.priority == EventPriority.LOW:
                    self._stats['dropped'] += 1
                    return False
                else:
                    # For high/medium priority, try to drop a low priority event
                    if not self._low_priority.empty():
                        try:
                            self._low_priority.get_nowait()
                            self._stats['dropped'] += 1
                        except asyncio.QueueEmpty:
                            pass
            
            queue.put_nowait(event)
            self._stats['enqueued'] += 1
            return True
            
        except asyncio.QueueFull:
            self._stats['dropped'] += 1
            return False
    
    async def dequeue_batch(self, batch_size: int = 100) -> List[FileEvent]:
        """Retrieve batch of file events with priority ordering."""
        events = []
        
        # Process high priority first
        events.extend(await self._dequeue_from_queue(self._high_priority, batch_size))
        
        # Then medium priority
        remaining = batch_size - len(events)
        if remaining > 0:
            events.extend(await self._dequeue_from_queue(self._medium_priority, remaining))
        
        # Finally low priority
        remaining = batch_size - len(events)
        if remaining > 0:
            events.extend(await self._dequeue_from_queue(self._low_priority, remaining))
        
        self._stats['dequeued'] += len(events)
        return events
    
    async def _dequeue_from_queue(self, queue: asyncio.Queue, max_items: int) -> List[FileEvent]:
        """Dequeue items from specific priority queue."""
        events = []
        for _ in range(min(max_items, queue.qsize())):
            try:
                event = queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break
        return events
    
    def _get_queue_for_priority(self, priority: EventPriority) -> asyncio.Queue:
        """Get appropriate queue for event priority."""
        if priority == EventPriority.HIGH:
            return self._high_priority
        elif priority == EventPriority.MEDIUM:
            return self._medium_priority
        else:
            return self._low_priority
    
    def qsize(self) -> int:
        """Get total queue size across all priorities."""
        return (self._high_priority.qsize() + 
                self._medium_priority.qsize() + 
                self._low_priority.qsize())
    
    def empty(self) -> bool:
        """Check if all queues are empty."""
        return (self._high_priority.empty() and 
                self._medium_priority.empty() and 
                self._low_priority.empty())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            'total_size': self.qsize(),
            'high_priority_size': self._high_priority.qsize(),
            'medium_priority_size': self._medium_priority.qsize(),
            'low_priority_size': self._low_priority.qsize(),
        }
