"""Unit tests for real-time file watcher implementation.

Tests cover file watcher functionality, event processing, and configuration
with comprehensive mocking and isolation.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest

from snake_pipe.extract.file_watcher import (
    RealtimeFileWatcher, ASTFileEventHandler, WatcherConfig, WatcherStats,
    create_file_watcher, create_watcher_from_extract_config
)
from snake_pipe.extract.event_processor import FileEvent, EventPriority
from snake_pipe.config.extract_config import create_default_config


class TestWatcherConfig:
    """Test watcher configuration functionality."""
    
    def test_default_config_creation(self):
        """Test creating default watcher configuration."""
        config = WatcherConfig()
        
        assert config.recursive_monitoring is True
        assert config.debounce_period == 0.5
        assert config.batch_size == 100
        assert config.queue_size == 10000
        assert '*.json' in config.ast_file_patterns
        assert '**/.git/**' in config.exclude_patterns
    
    def test_config_from_extract_config(self):
        """Test creating watcher config from extract configuration."""
        extract_config = create_default_config(Path('/tmp'))
        watcher_config = WatcherConfig.from_extract_config(extract_config)
        
        assert watcher_config.debounce_period == 0.3  # Faster for real-time
        assert watcher_config.batch_size == 50  # Smaller batches
        assert watcher_config.batch_timeout == 1.0  # Shorter timeout
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = WatcherConfig(
            debounce_period=-1.0,  # Invalid
            batch_size=0,          # Invalid
            queue_size=-100        # Invalid
        )
        
        # Config should still work with defaults
        assert config.ast_file_patterns is not None
        assert config.exclude_patterns is not None


class TestWatcherStats:
    """Test watcher statistics functionality."""
    
    def test_stats_initialization(self):
        """Test stats initialization."""
        stats = WatcherStats()
        
        assert stats.events_processed == 0
        assert stats.batches_processed == 0
        assert stats.processing_errors == 0
        assert stats.files_detected == 0
        assert stats.start_time > 0
    
    def test_runtime_calculation(self):
        """Test runtime calculation."""
        stats = WatcherStats()
        time.sleep(0.1)  # Brief delay
        
        runtime = stats.get_runtime_seconds()
        assert runtime >= 0.1
        assert runtime < 1.0  # Should be well under a second
    
    def test_events_per_second(self):
        """Test events per second calculation."""
        stats = WatcherStats()
        stats.events_processed = 100
        time.sleep(0.1)
        
        eps = stats.get_events_per_second()
        assert eps > 0
        assert eps > 100  # Should be >1000 events/sec with 0.1s runtime


class TestASTFileEventHandler:
    """Test file system event handler."""
    
    @pytest.fixture
    def mock_event_queue(self):
        """Mock event queue for testing."""
        return Mock(spec=asyncio.Queue)
    
    @pytest.fixture
    def watcher_config(self):
        """Test watcher configuration."""
        return WatcherConfig(
            ast_file_patterns=['*.json'],
            exclude_patterns=['*.tmp'],
            debounce_period=0.1
        )
    
    @pytest.fixture
    def event_handler(self, mock_event_queue, watcher_config):
        """Create event handler for testing."""
        handler = ASTFileEventHandler(mock_event_queue, watcher_config)
        # Mock the event loop
        loop_mock = Mock()
        loop_mock.is_closed.return_value = False
        handler.set_event_loop(loop_mock)
        return handler
    
    def test_relevant_file_detection(self, event_handler):
        """Test detection of relevant AST files."""
        # Test relevant files
        assert event_handler._is_relevant_file('/path/to/file.json') is True
        assert event_handler._is_relevant_file('/path/to/ast.json') is True
        
        # Test excluded files
        assert event_handler._is_relevant_file('/path/to/file.tmp') is False
        assert event_handler._is_relevant_file('/path/to/file.py') is False
    
    def test_debouncing_logic(self, event_handler):
        """Test event debouncing logic."""
        file_path = '/test/file.json'
        
        # First event should not be debounced
        assert event_handler._should_debounce(file_path) is False
        
        # Immediate second event should be debounced
        assert event_handler._should_debounce(file_path) is True
        
        # After debounce period, should not be debounced
        time.sleep(0.15)  # Longer than debounce period
        assert event_handler._should_debounce(file_path) is False
    
    def test_event_type_normalization(self, event_handler):
        """Test normalization of event types."""
        assert event_handler._normalize_event_type('created') == 'created'
        assert event_handler._normalize_event_type('MODIFIED') == 'modified'
        assert event_handler._normalize_event_type('file_created') == 'created'
        assert event_handler._normalize_event_type('unknown_event') == 'modified'
    
    @patch('snake_pipe.extract.file_watcher.Path')
    def test_file_size_retrieval(self, mock_path, event_handler):
        """Test safe file size retrieval."""
        # Mock successful stat
        mock_stat = Mock()
        mock_stat.st_size = 1024
        mock_path.return_value.stat.return_value = mock_stat
        
        size = event_handler._get_file_size('/test/file.json')
        assert size == 1024
        
        # Mock file not found
        mock_path.return_value.stat.side_effect = FileNotFoundError()
        size = event_handler._get_file_size('/nonexistent/file.json')
        assert size is None
    
    def test_debounce_cache_cleanup(self, event_handler):
        """Test debounce cache cleanup."""
        # Fill cache with many entries
        for i in range(15000):
            event_handler._should_debounce(f'/test/file_{i}.json')
        
        # Cache should be cleaned up when it gets too large
        assert len(event_handler.debounce_cache) <= 10000


class TestRealtimeFileWatcher:
    """Test real-time file watcher functionality."""
    
    @pytest.fixture
    def watcher_config(self):
        """Test watcher configuration."""
        return WatcherConfig(
            batch_size=10,
            batch_timeout=0.5,
            debounce_period=0.1,
            queue_size=100
        )
    
    @pytest.fixture
    def file_watcher(self, watcher_config):
        """Create file watcher for testing."""
        return RealtimeFileWatcher(watcher_config)
    
    def test_watcher_initialization(self, file_watcher):
        """Test watcher initialization."""
        assert file_watcher.running is False
        assert file_watcher.observer is None
        assert file_watcher.event_handler is None
        assert file_watcher.processing_task is None
        assert isinstance(file_watcher.stats, WatcherStats)
    
    @pytest.mark.asyncio
    async def test_start_monitoring_invalid_paths(self, file_watcher):
        """Test starting monitoring with invalid paths."""
        invalid_paths = [Path('/nonexistent/path')]
        
        with pytest.raises(ValueError, match="No valid watch paths found"):
            await file_watcher.start_monitoring(invalid_paths, Mock())
    
    @pytest.mark.asyncio
    async def test_start_monitoring_valid_paths(self, file_watcher):
        """Test starting monitoring with valid paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_callback = Mock()
            
            with patch('snake_pipe.extract.file_watcher.Observer') as mock_observer_class:
                mock_observer = Mock()
                mock_observer_class.return_value = mock_observer
                
                await file_watcher.start_monitoring([temp_path], mock_callback)
                
                assert file_watcher.running is True
                assert file_watcher.observer is not None
                assert file_watcher.event_handler is not None
                assert file_watcher.processing_task is not None
                
                # Cleanup
                await file_watcher.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, file_watcher):
        """Test stopping file monitoring."""
        # Mock observer and processing task
        mock_observer = Mock()
        mock_task = Mock()
        mock_task.cancel = Mock()
        mock_task.done.return_value = True
        
        file_watcher.observer = mock_observer
        file_watcher.processing_task = mock_task
        file_watcher.running = True
        
        await file_watcher.stop_monitoring()
        
        assert file_watcher.running is False
        mock_observer.stop.assert_called_once()
        mock_task.cancel.assert_called_once()
    
    def test_event_deduplication(self, file_watcher):
        """Test event deduplication logic."""
        events = [
            FileEvent(Path('/test/file1.json'), 'modified', time.time()),
            FileEvent(Path('/test/file1.json'), 'modified', time.time() + 1),  # Newer
            FileEvent(Path('/test/file2.json'), 'created', time.time()),
        ]
        
        deduplicated = file_watcher._deduplicate_and_filter_events(events)
        
        assert len(deduplicated) == 2
        # Should keep newer event for file1.json
        file1_events = [e for e in deduplicated if e.path.name == 'file1.json']
        assert len(file1_events) == 1
        assert file1_events[0].timestamp == events[1].timestamp
    
    def test_stats_collection(self, file_watcher):
        """Test statistics collection."""
        file_watcher.stats.events_processed = 100
        file_watcher.stats.batches_processed = 10
        file_watcher.stats.files_detected = 50
        
        stats = file_watcher.get_stats()
        
        assert stats['events_processed'] == 100
        assert stats['batches_processed'] == 10
        assert stats['files_detected'] == 50
        assert 'runtime_seconds' in stats
        assert 'events_per_second' in stats
        assert 'queue_size' in stats
    
    @pytest.mark.asyncio
    async def test_health_check(self, file_watcher):
        """Test health check functionality."""
        health = await file_watcher.health_check()
        
        assert 'status' in health
        assert 'observer_alive' in health
        assert 'queue_size' in health
        assert 'stats' in health
        
        # Test with queue near capacity
        file_watcher.config.queue_size = 100
        for _ in range(85):  # 85% of capacity
            await file_watcher.event_queue.put(Mock())
        
        health = await file_watcher.health_check()
        assert 'warnings' in health


class TestFactoryFunctions:
    """Test factory functions for creating watchers."""
    
    def test_create_file_watcher_default(self):
        """Test creating file watcher with default config."""
        watcher = create_file_watcher()
        
        assert isinstance(watcher, RealtimeFileWatcher)
        assert isinstance(watcher.config, WatcherConfig)
    
    def test_create_file_watcher_custom_config(self):
        """Test creating file watcher with custom config."""
        config = WatcherConfig(batch_size=200)
        watcher = create_file_watcher(config)
        
        assert watcher.config.batch_size == 200
    
    def test_create_watcher_from_extract_config(self):
        """Test creating watcher from extract configuration."""
        extract_config = create_default_config(Path('/tmp'))
        watcher = create_watcher_from_extract_config(extract_config)
        
        assert isinstance(watcher, RealtimeFileWatcher)
        assert watcher.config.batch_size == 50  # Optimized for real-time


class TestIntegrationScenarios:
    """Integration test scenarios for file watcher."""
    
    @pytest.mark.asyncio
    async def test_file_creation_detection(self):
        """Test detection of file creation events."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = WatcherConfig(batch_timeout=0.1, debounce_period=0.05)
            watcher = RealtimeFileWatcher(config)
            
            detected_events = []
            
            def event_callback(events):
                detected_events.extend(events)
            
            # Create test file after starting watcher
            test_file = temp_path / 'test.json'
            test_file.write_text('{"test": "data"}')
            
            # Simulate file event
            event = FileEvent(test_file, 'created', time.time())
            await watcher.event_queue.put(event)
            
            # Process one batch
            await watcher._process_batch([event], event_callback)
            
            assert len(detected_events) == 1
            assert detected_events[0].path == test_file
    
    @pytest.mark.asyncio
    async def test_batch_processing_timeout(self):
        """Test batch processing with timeout."""
        config = WatcherConfig(batch_size=100, batch_timeout=0.1)
        watcher = RealtimeFileWatcher(config)
        
        processed_batches = []
        
        def event_callback(events):
            processed_batches.append(events)
        
        # Add a single event (less than batch size)
        event = FileEvent(Path('/test/file.json'), 'modified', time.time())
        await watcher.event_queue.put(event)
        
        # Start processing loop briefly
        watcher.running = True
        task = asyncio.create_task(watcher._process_events_loop(event_callback))
        
        # Wait for timeout processing
        await asyncio.sleep(0.2)
        
        watcher.running = False
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should have processed the single event due to timeout
        assert len(processed_batches) >= 1
        if processed_batches:
            assert len(processed_batches[0]) == 1


@pytest.mark.performance
class TestPerformanceCharacteristics:
    """Performance tests for file watcher."""
    
    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self):
        """Test processing large number of events."""
        config = WatcherConfig(batch_size=1000, batch_timeout=1.0)
        watcher = RealtimeFileWatcher(config)
        
        # Generate large number of events
        events = []
        for i in range(5000):
            event = FileEvent(Path(f'/test/file_{i}.json'), 'modified', time.time())
            events.append(event)
        
        processed_events = []
        
        def event_callback(batch_events):
            processed_events.extend(batch_events)
        
        start_time = time.time()
        
        # Process events in batches
        batch_size = 1000
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            await watcher._process_batch(batch, event_callback)
        
        processing_time = time.time() - start_time
        
        assert len(processed_events) == 5000
        assert processing_time < 5.0  # Should process 5000 events in under 5 seconds
        
        # Calculate throughput
        throughput = len(processed_events) / processing_time
        assert throughput > 1000  # Should handle >1000 events/second
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage remains stable during operation."""
        config = WatcherConfig(queue_size=50000)
        watcher = RealtimeFileWatcher(config)
        
        # Fill queue to capacity
        for i in range(1000):
            event = FileEvent(Path(f'/test/file_{i}.json'), 'modified', time.time())
            await watcher.event_queue.put(event)
        
        initial_queue_size = watcher.event_queue.qsize()
        
        # Process all events
        processed_count = 0
        
        def event_callback(events):
            nonlocal processed_count
            processed_count += len(events)
        
        while not watcher.event_queue.empty():
            try:
                event = await asyncio.wait_for(watcher.event_queue.get(), timeout=0.1)
                await watcher._process_batch([event], event_callback)
            except asyncio.TimeoutError:
                break
        
        assert processed_count >= initial_queue_size * 0.9  # Should process most events
        assert watcher.event_queue.qsize() < initial_queue_size * 0.1  # Queue should be mostly empty
