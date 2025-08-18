"""Event debouncing and batching utilities for file watcher.

This module provides advanced event debouncing capabilities to handle
rapid file changes efficiently and prevent duplicate processing.
"""

import asyncio
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any, Deque
from enum import Enum, auto

from snake_pipe.extract.event_processor import FileEvent, EventPriority

logger = logging.getLogger(__name__)


class DebounceStrategy(Enum):
    """Different debouncing strategies for file events."""
    SIMPLE = "simple"        # Basic time-based debouncing
    ADAPTIVE = "adaptive"      # Adaptive debouncing based on file characteristics
    INTELLIGENT = "intelligent"   # Smart batching for related files


@dataclass
class DebounceConfig:
    """Configuration for event debouncing."""
    strategy: DebounceStrategy = DebounceStrategy.ADAPTIVE
    base_debounce_period: float = 0.5
    max_debounce_period: float = 5.0
    min_debounce_period: float = 0.1
    batch_window: float = 2.0
    max_batch_size: int = 100
    adaptive_factor: float = 0.2
    related_file_threshold: float = 1.0


@dataclass
class PendingEvent:
    """Event waiting in debounce queue."""
    event: FileEvent
    scheduled_time: float
    retry_count: int = 0
    related_events: Set[str] = field(default_factory=set)


class EventDebouncer:
    """Advanced event debouncing with adaptive strategies."""
    
    def __init__(self, config: DebounceConfig):
        self.config = config
        self.pending_events: Dict[str, PendingEvent] = {}
        self.file_activity: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10))
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False
        self.event_callback: Optional[Callable[[List[FileEvent]], None]] = None
        self.stats = {
            'events_debounced': 0,
            'events_processed': 0,
            'batches_created': 0,
            'adaptive_adjustments': 0
        }
    
    async def start(self, event_callback: Callable[[List[FileEvent]], None]) -> None:
        """Start the debouncer processing loop."""
        self.event_callback = event_callback
        self.running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info(f"Event debouncer started with strategy: {self.config.strategy.name}")
    
    async def stop(self) -> None:
        """Stop debouncer and process remaining events."""
        logger.info("Stopping event debouncer")
        self.running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Process any remaining events
        await self._process_all_pending()
        logger.info(f"Event debouncer stopped. Final stats: {self.stats}")
    
    async def add_event(self, event: FileEvent) -> None:
        """Add event to debouncer for processing."""
        event_key = self._get_event_key(event)
        current_time = time.time()
        
        # Update file activity tracking
        self.file_activity[str(event.path)].append(current_time)
        
        # Calculate debounce period based on strategy
        debounce_period = self._calculate_debounce_period(event)
        scheduled_time = current_time + debounce_period
        
        # Check if event already exists
        if event_key in self.pending_events:
            pending = self.pending_events[event_key]
            # Update with newer event and extend debounce time
            pending.event = event
            pending.scheduled_time = max(pending.scheduled_time, scheduled_time)
            pending.retry_count += 1
        else:
            # Create new pending event
            pending = PendingEvent(
                event=event,
                scheduled_time=scheduled_time,
                related_events=self._find_related_events(event)
            )
            self.pending_events[event_key] = pending
        
        self.stats['events_debounced'] += 1
        logger.debug(f"Added event {event_key} with debounce period {debounce_period:.2f}s")
    
    def _get_event_key(self, event: FileEvent) -> str:
        """Generate unique key for event deduplication."""
        return f"{event.path}:{event.event_type}"
    
    def _calculate_debounce_period(self, event: FileEvent) -> float:
        """Calculate appropriate debounce period based on strategy."""
        if self.config.strategy == DebounceStrategy.SIMPLE:
            return self.config.base_debounce_period
        
        elif self.config.strategy == DebounceStrategy.ADAPTIVE:
            return self._calculate_adaptive_period(event)
        
        elif self.config.strategy == DebounceStrategy.INTELLIGENT:
            return self._calculate_intelligent_period(event)
        
        return self.config.base_debounce_period
    
    def _calculate_adaptive_period(self, event: FileEvent) -> float:
        """Calculate adaptive debounce period based on file activity."""
        file_path = str(event.path)
        activity = self.file_activity[file_path]
        
        if len(activity) < 2:
            return self.config.base_debounce_period
        
        # Calculate average time between recent events
        recent_intervals = []
        for i in range(1, len(activity)):
            interval = activity[i] - activity[i-1]
            recent_intervals.append(interval)
        
        if not recent_intervals:
            return self.config.base_debounce_period
        
        avg_interval = sum(recent_intervals) / len(recent_intervals)
        
        # Adaptive calculation: longer periods for frequently changing files
        if avg_interval < 1.0:  # Very frequent changes
            adaptive_period = self.config.base_debounce_period * (1 + self.config.adaptive_factor * 3)
        elif avg_interval < 5.0:  # Moderate changes
            adaptive_period = self.config.base_debounce_period * (1 + self.config.adaptive_factor)
        else:  # Infrequent changes
            adaptive_period = self.config.base_debounce_period * (1 - self.config.adaptive_factor)
        
        # Clamp to configured bounds
        adaptive_period = max(self.config.min_debounce_period, 
                            min(self.config.max_debounce_period, adaptive_period))
        
        self.stats['adaptive_adjustments'] += 1
        return adaptive_period
    
    def _calculate_intelligent_period(self, event: FileEvent) -> float:
        """Calculate intelligent debounce period considering related files."""
        base_period = self._calculate_adaptive_period(event)
        
        # Find related files that might be changing together
        related_count = len(self._find_related_events(event))
        
        if related_count > 0:
            # Longer debounce for files that are part of bulk operations
            intelligent_period = base_period * (1 + (related_count * 0.1))
            return min(self.config.max_debounce_period, intelligent_period)
        
        return base_period
    
    def _find_related_events(self, event: FileEvent) -> Set[str]:
        """Find events for files that might be related to current event."""
        related = set()
        current_time = time.time()
        
        # Look for files in same directory that changed recently
        parent_dir = event.path.parent
        
        for event_key, pending in self.pending_events.items():
            other_event = pending.event
            
            # Skip same file
            if other_event.path == event.path:
                continue
            
            # Check if in same directory
            if other_event.path.parent == parent_dir:
                # Check if event is recent
                if current_time - other_event.timestamp < self.config.related_file_threshold:
                    related.add(event_key)
        
        return related
    
    async def _processing_loop(self) -> None:
        """Main processing loop for debounced events."""
        while self.running:
            try:
                current_time = time.time()
                ready_events = []
                
                # Find events ready for processing
                for event_key, pending in list(self.pending_events.items()):
                    if current_time >= pending.scheduled_time:
                        ready_events.append(pending.event)
                        del self.pending_events[event_key]
                
                # Process ready events in batches
                if ready_events:
                    await self._process_event_batch(ready_events)
                
                # Sleep briefly before next check
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in debouncer processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_event_batch(self, events: List[FileEvent]) -> None:
        """Process a batch of ready events."""
        if not events or not self.event_callback:
            return
        
        # Group events by directory for intelligent batching
        if self.config.strategy == DebounceStrategy.INTELLIGENT:
            batches = self._create_intelligent_batches(events)
        else:
            batches = [events[i:i + self.config.max_batch_size] 
                      for i in range(0, len(events), self.config.max_batch_size)]
        
        for batch in batches:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.event_callback, batch
                )
                self.stats['events_processed'] += len(batch)
                self.stats['batches_created'] += 1
                logger.debug(f"Processed batch of {len(batch)} events")
                
            except Exception as e:
                logger.error(f"Error processing event batch: {e}")
    
    def _create_intelligent_batches(self, events: List[FileEvent]) -> List[List[FileEvent]]:
        """Create intelligent batches based on file relationships."""
        # Group by directory
        dir_groups: Dict[Path, List[FileEvent]] = defaultdict(list)
        for event in events:
            dir_groups[event.path.parent].append(event)
        
        batches = []
        current_batch = []
        
        for directory, dir_events in dir_groups.items():
            # Sort by priority and timestamp
            dir_events.sort(key=lambda e: (e.priority.value, e.timestamp))
            
            for event in dir_events:
                current_batch.append(event)
                
                # Create batch when size limit reached
                if len(current_batch) >= self.config.max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _process_all_pending(self) -> None:
        """Process all remaining pending events during shutdown."""
        if not self.pending_events:
            return
        
        remaining_events = [pending.event for pending in self.pending_events.values()]
        await self._process_event_batch(remaining_events)
        self.pending_events.clear()
        logger.info(f"Processed {len(remaining_events)} remaining events during shutdown")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get debouncer statistics."""
        return {
            **self.stats,
            'pending_events': len(self.pending_events),
            'tracked_files': len(self.file_activity),
            'strategy': self.config.strategy.name
        }
    
    def get_pending_count(self) -> int:
        """Get number of pending events."""
        return len(self.pending_events)
    
    async def flush_events(self, timeout: float = 5.0) -> int:
        """Flush all pending events with timeout."""
        start_time = time.time()
        initial_count = len(self.pending_events)
        
        while self.pending_events and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        processed = initial_count - len(self.pending_events)
        logger.info(f"Flushed {processed} events in {time.time() - start_time:.2f}s")
        return processed


def create_debouncer(
    strategy: DebounceStrategy = DebounceStrategy.ADAPTIVE,
    base_period: float = 0.5
) -> EventDebouncer:
    """Factory function to create configured event debouncer."""
    config = DebounceConfig(
        strategy=strategy,
        base_debounce_period=base_period
    )
    return EventDebouncer(config)


def create_high_performance_debouncer() -> EventDebouncer:
    """Create debouncer optimized for high-performance scenarios."""
    config = DebounceConfig(
        strategy=DebounceStrategy.INTELLIGENT,
        base_debounce_period=0.2,
        max_debounce_period=2.0,
        min_debounce_period=0.05,
        batch_window=1.0,
        max_batch_size=200,
        adaptive_factor=0.3
    )
    return EventDebouncer(config)


def create_development_debouncer() -> EventDebouncer:
    """Create debouncer optimized for development environments."""
    config = DebounceConfig(
        strategy=DebounceStrategy.ADAPTIVE,
        base_debounce_period=0.8,
        max_debounce_period=10.0,
        min_debounce_period=0.2,
        batch_window=3.0,
        max_batch_size=50,
        adaptive_factor=0.4
    )
    return EventDebouncer(config)
