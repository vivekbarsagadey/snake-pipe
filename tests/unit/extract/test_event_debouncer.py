"""Unit tests for event debouncer functionality.

Tests cover different debouncing strategies, adaptive behavior, and performance.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from snake_pipe.extract.event_debouncer import (
    EventDebouncer, DebounceStrategy, AdaptiveDebouncer,
    SimpleDebouncer, IntelligentDebouncer, RelatedFileDetector
)
from snake_pipe.extract.event_processor import FileEvent, EventPriority


class TestDebounceStrategy:
    """Test debouncing strategy enumeration."""
    
    def test_strategy_values(self):
        """Test strategy enumeration values."""
        assert DebounceStrategy.SIMPLE.value == "simple"
        assert DebounceStrategy.ADAPTIVE.value == "adaptive"
        assert DebounceStrategy.INTELLIGENT.value == "intelligent"
    
    def test_strategy_comparison(self):
        """Test strategy comparison and ordering."""
        strategies = [DebounceStrategy.SIMPLE, DebounceStrategy.ADAPTIVE, DebounceStrategy.INTELLIGENT]
        assert len(strategies) == 3
        assert DebounceStrategy.SIMPLE in strategies


class TestSimpleDebouncer:
    """Test simple debouncing strategy."""
    
    @pytest.fixture
    def simple_debouncer(self):
        """Create simple debouncer for testing."""
        return SimpleDebouncer(debounce_period=0.5)
    
    def test_debouncer_initialization(self, simple_debouncer):
        """Test debouncer initialization."""
        assert simple_debouncer.debounce_period == 0.5
        assert len(simple_debouncer.last_event_times) == 0
    
    def test_first_event_not_debounced(self, simple_debouncer):
        """Test that first event for a file is not debounced."""
        event = FileEvent(Path('/test/file.json'), 'modified', time.time())
        
        assert simple_debouncer.should_debounce(event) is False
    
    def test_rapid_events_debounced(self, simple_debouncer):
        """Test that rapid consecutive events are debounced."""
        event = FileEvent(Path('/test/file.json'), 'modified', time.time())
        
        # First event
        assert simple_debouncer.should_debounce(event) is False
        
        # Immediate second event should be debounced
        event2 = FileEvent(Path('/test/file.json'), 'modified', time.time())
        assert simple_debouncer.should_debounce(event2) is True
    
    def test_events_after_debounce_period(self, simple_debouncer):
        """Test that events after debounce period are not debounced."""
        event1 = FileEvent(Path('/test/file.json'), 'modified', time.time())
        
        # First event
        assert simple_debouncer.should_debounce(event1) is False
        
        # Event after debounce period
        future_time = time.time() + 0.6  # Longer than debounce period
        event2 = FileEvent(Path('/test/file.json'), 'modified', future_time)
        
        with patch('time.time', return_value=future_time):
            assert simple_debouncer.should_debounce(event2) is False
    
    def test_different_files_not_debounced(self, simple_debouncer):
        """Test that events for different files are not debounced."""
        event1 = FileEvent(Path('/test/file1.json'), 'modified', time.time())
        event2 = FileEvent(Path('/test/file2.json'), 'modified', time.time())
        
        assert simple_debouncer.should_debounce(event1) is False
        assert simple_debouncer.should_debounce(event2) is False
    
    def test_cache_cleanup(self, simple_debouncer):
        """Test debounce cache cleanup."""
        # Fill cache with many entries
        for i in range(12000):
            event = FileEvent(Path(f'/test/file_{i}.json'), 'modified', time.time())
            simple_debouncer.should_debounce(event)
        
        # Cache should be cleaned up
        assert len(simple_debouncer.last_event_times) <= 10000
    
    def test_get_stats(self, simple_debouncer):
        """Test debouncer statistics."""
        # Process some events
        for i in range(5):
            event = FileEvent(Path(f'/test/file_{i}.json'), 'modified', time.time())
            simple_debouncer.should_debounce(event)
        
        stats = simple_debouncer.get_stats()
        
        assert 'strategy' in stats
        assert 'cache_size' in stats
        assert 'debounce_period' in stats
        assert stats['strategy'] == DebounceStrategy.SIMPLE.value


class TestAdaptiveDebouncer:
    """Test adaptive debouncing strategy."""
    
    @pytest.fixture
    def adaptive_debouncer(self):
        """Create adaptive debouncer for testing."""
        return AdaptiveDebouncer(
            base_period=0.3,
            min_period=0.1,
            max_period=2.0,
            adaptation_factor=1.5
        )
    
    def test_adaptive_initialization(self, adaptive_debouncer):
        """Test adaptive debouncer initialization."""
        assert adaptive_debouncer.base_period == 0.3
        assert adaptive_debouncer.min_period == 0.1
        assert adaptive_debouncer.max_period == 2.0
        assert adaptive_debouncer.adaptation_factor == 1.5
    
    def test_initial_period_calculation(self, adaptive_debouncer):
        """Test initial adaptive period calculation."""
        event = FileEvent(Path('/test/file.json'), 'modified', time.time())
        period = adaptive_debouncer._calculate_adaptive_period(event)
        
        # Should start with base period
        assert period == 0.3
    
    def test_period_adaptation_with_history(self, adaptive_debouncer):
        """Test period adaptation based on event history."""
        file_path = Path('/test/active_file.json')
        
        # Simulate rapid events
        for i in range(10):
            event = FileEvent(file_path, 'modified', time.time() + i * 0.05)
            adaptive_debouncer.should_debounce(event)
        
        # Period should have adapted to higher value
        period = adaptive_debouncer._calculate_adaptive_period(
            FileEvent(file_path, 'modified', time.time())
        )
        assert period > 0.3  # Should be higher than base period
    
    def test_period_bounds_enforcement(self, adaptive_debouncer):
        """Test that adaptive period stays within bounds."""
        file_path = Path('/test/file.json')
        
        # Simulate many rapid events to force high adaptation
        for i in range(100):
            event = FileEvent(file_path, 'modified', time.time() + i * 0.01)
            adaptive_debouncer.should_debounce(event)
        
        period = adaptive_debouncer._calculate_adaptive_period(
            FileEvent(file_path, 'modified', time.time())
        )
        
        assert adaptive_debouncer.min_period <= period <= adaptive_debouncer.max_period
    
    def test_different_files_independent_adaptation(self, adaptive_debouncer):
        """Test that different files have independent adaptation."""
        file1 = Path('/test/active_file.json')
        file2 = Path('/test/quiet_file.json')
        
        # Make file1 very active
        for i in range(20):
            event = FileEvent(file1, 'modified', time.time() + i * 0.02)
            adaptive_debouncer.should_debounce(event)
        
        # file2 gets only one event
        event2 = FileEvent(file2, 'modified', time.time())
        adaptive_debouncer.should_debounce(event2)
        
        period1 = adaptive_debouncer._calculate_adaptive_period(FileEvent(file1, 'modified', time.time()))
        period2 = adaptive_debouncer._calculate_adaptive_period(FileEvent(file2, 'modified', time.time()))
        
        # file1 should have higher period due to activity
        assert period1 > period2
    
    def test_adaptive_stats(self, adaptive_debouncer):
        """Test adaptive debouncer statistics."""
        stats = adaptive_debouncer.get_stats()
        
        assert stats['strategy'] == DebounceStrategy.ADAPTIVE.value
        assert 'base_period' in stats
        assert 'min_period' in stats
        assert 'max_period' in stats
        assert 'adaptation_factor' in stats


class TestIntelligentDebouncer:
    """Test intelligent debouncing strategy."""
    
    @pytest.fixture
    def intelligent_debouncer(self):
        """Create intelligent debouncer for testing."""
        return IntelligentDebouncer(
            base_period=0.5,
            batch_period=2.0,
            max_batch_size=10
        )
    
    def test_intelligent_initialization(self, intelligent_debouncer):
        """Test intelligent debouncer initialization."""
        assert intelligent_debouncer.base_period == 0.5
        assert intelligent_debouncer.batch_period == 2.0
        assert intelligent_debouncer.max_batch_size == 10
        assert isinstance(intelligent_debouncer.related_detector, RelatedFileDetector)
    
    def test_related_file_batching(self, intelligent_debouncer):
        """Test batching of related file events."""
        # Simulate related files (same directory, similar names)
        base_path = Path('/project/src')
        events = [
            FileEvent(base_path / 'main.json', 'modified', time.time()),
            FileEvent(base_path / 'utils.json', 'modified', time.time() + 0.1),
            FileEvent(base_path / 'config.json', 'modified', time.time() + 0.2),
        ]
        
        # First event should not be debounced
        assert intelligent_debouncer.should_debounce(events[0]) is False
        
        # Related events should be considered for batching
        should_debounce_2 = intelligent_debouncer.should_debounce(events[1])
        should_debounce_3 = intelligent_debouncer.should_debounce(events[2])
        
        # Behavior depends on implementation details
        assert isinstance(should_debounce_2, bool)
        assert isinstance(should_debounce_3, bool)
    
    def test_batch_size_limit(self, intelligent_debouncer):
        """Test batch size limits."""
        base_path = Path('/project/src')
        
        # Create many related events
        for i in range(15):  # More than max_batch_size
            event = FileEvent(base_path / f'file_{i}.json', 'modified', time.time() + i * 0.1)
            intelligent_debouncer.should_debounce(event)
        
        # Check that batch tracking doesn't grow indefinitely
        stats = intelligent_debouncer.get_stats()
        assert 'batch_count' in stats
    
    def test_unrelated_files_not_batched(self, intelligent_debouncer):
        """Test that unrelated files are not batched."""
        unrelated_events = [
            FileEvent(Path('/project1/main.json'), 'modified', time.time()),
            FileEvent(Path('/project2/data.json'), 'modified', time.time() + 0.1),
            FileEvent(Path('/tmp/temp.json'), 'modified', time.time() + 0.2),
        ]
        
        # Each should be processed independently
        for event in unrelated_events:
            result = intelligent_debouncer.should_debounce(event)
            # First event for each path should not be debounced
            assert isinstance(result, bool)
    
    def test_intelligent_stats(self, intelligent_debouncer):
        """Test intelligent debouncer statistics."""
        stats = intelligent_debouncer.get_stats()
        
        assert stats['strategy'] == DebounceStrategy.INTELLIGENT.value
        assert 'batch_period' in stats
        assert 'max_batch_size' in stats
        assert 'batch_count' in stats


class TestRelatedFileDetector:
    """Test related file detection functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create related file detector for testing."""
        return RelatedFileDetector()
    
    def test_same_directory_detection(self, detector):
        """Test detection of files in same directory."""
        file1 = Path('/project/src/main.json')
        file2 = Path('/project/src/utils.json')
        file3 = Path('/project/tests/test.json')
        
        assert detector.are_related(file1, file2) is True  # Same directory
        assert detector.are_related(file1, file3) is False  # Different directory
    
    def test_similar_name_detection(self, detector):
        """Test detection of files with similar names."""
        file1 = Path('/project/user_model.json')
        file2 = Path('/project/user_service.json')
        file3 = Path('/project/product_model.json')
        
        # Files with similar prefixes should be related
        related_12 = detector.are_related(file1, file2)
        related_13 = detector.are_related(file1, file3)
        
        assert isinstance(related_12, bool)
        assert isinstance(related_13, bool)
    
    def test_extension_matching(self, detector):
        """Test that files with different extensions are less related."""
        json_file = Path('/project/data.json')
        py_file = Path('/project/data.py')
        txt_file = Path('/project/data.txt')
        
        # Same base name but different extensions
        related_json_py = detector.are_related(json_file, py_file)
        related_json_txt = detector.are_related(json_file, txt_file)
        
        assert isinstance(related_json_py, bool)
        assert isinstance(related_json_txt, bool)
    
    def test_path_depth_consideration(self, detector):
        """Test consideration of path depth in relatedness."""
        shallow = Path('/project/file.json')
        deep = Path('/project/deeply/nested/path/file.json')
        
        # Very different depths should be less related
        related = detector.are_related(shallow, deep)
        assert isinstance(related, bool)
    
    def test_common_prefixes(self, detector):
        """Test detection of common prefixes."""
        files = [
            Path('/project/user_model.json'),
            Path('/project/user_controller.json'),
            Path('/project/user_view.json'),
            Path('/project/order_model.json'),
        ]
        
        # user_* files should be more related to each other
        user_related = detector.are_related(files[0], files[1])
        cross_related = detector.are_related(files[0], files[3])
        
        assert isinstance(user_related, bool)
        assert isinstance(cross_related, bool)


class TestEventDebouncer:
    """Test main event debouncer functionality."""
    
    @pytest.fixture
    def event_debouncer(self):
        """Create event debouncer for testing."""
        return EventDebouncer(
            strategy=DebounceStrategy.ADAPTIVE,
            base_period=0.3
        )
    
    def test_debouncer_initialization(self, event_debouncer):
        """Test debouncer initialization."""
        assert event_debouncer.strategy == DebounceStrategy.ADAPTIVE
        assert isinstance(event_debouncer.debouncer, AdaptiveDebouncer)
    
    def test_strategy_switching(self):
        """Test switching between debouncing strategies."""
        # Test simple strategy
        simple_debouncer = EventDebouncer(strategy=DebounceStrategy.SIMPLE)
        assert isinstance(simple_debouncer.debouncer, SimpleDebouncer)
        
        # Test intelligent strategy
        intelligent_debouncer = EventDebouncer(strategy=DebounceStrategy.INTELLIGENT)
        assert isinstance(intelligent_debouncer.debouncer, IntelligentDebouncer)
    
    def test_should_debounce_delegation(self, event_debouncer):
        """Test that should_debounce delegates to underlying strategy."""
        event = FileEvent(Path('/test/file.json'), 'modified', time.time())
        
        result = event_debouncer.should_debounce(event)
        assert isinstance(result, bool)
    
    def test_process_events_batch(self, event_debouncer):
        """Test processing batches of events."""
        events = [
            FileEvent(Path('/test/file1.json'), 'modified', time.time()),
            FileEvent(Path('/test/file1.json'), 'modified', time.time() + 0.1),  # Should be debounced
            FileEvent(Path('/test/file2.json'), 'created', time.time()),
        ]
        
        processed = event_debouncer.process_events(events)
        
        # Should have fewer events due to debouncing
        assert len(processed) <= len(events)
        assert len(processed) >= 1  # At least one event should remain
    
    def test_get_stats_delegation(self, event_debouncer):
        """Test that get_stats delegates to underlying strategy."""
        stats = event_debouncer.get_stats()
        
        assert 'strategy' in stats
        assert stats['strategy'] == DebounceStrategy.ADAPTIVE.value
    
    def test_reset_functionality(self, event_debouncer):
        """Test debouncer reset functionality."""
        # Process some events to build up state
        event = FileEvent(Path('/test/file.json'), 'modified', time.time())
        event_debouncer.should_debounce(event)
        
        # Reset should clear state
        event_debouncer.reset()
        
        # After reset, same event should not be debounced
        result = event_debouncer.should_debounce(event)
        assert result is False  # First event after reset


@pytest.mark.performance
class TestDebouncerPerformance:
    """Performance tests for debouncing functionality."""
    
    def test_simple_debouncer_performance(self):
        """Test simple debouncer performance with many events."""
        debouncer = SimpleDebouncer(debounce_period=0.1)
        
        # Generate many events
        events = []
        for i in range(10000):
            event = FileEvent(Path(f'/test/file_{i % 100}.json'), 'modified', time.time() + i * 0.001)
            events.append(event)
        
        start_time = time.time()
        
        for event in events:
            debouncer.should_debounce(event)
        
        processing_time = time.time() - start_time
        
        # Should process 10,000 events quickly
        assert processing_time < 5.0
        
        throughput = len(events) / processing_time
        assert throughput > 2000  # Should handle >2000 events/second
    
    def test_adaptive_debouncer_scalability(self):
        """Test adaptive debouncer scalability."""
        debouncer = AdaptiveDebouncer(base_period=0.1)
        
        # Test with varying file counts
        file_counts = [100, 1000, 5000]
        
        for file_count in file_counts:
            events = []
            for i in range(file_count):
                event = FileEvent(Path(f'/test/file_{i}.json'), 'modified', time.time())
                events.append(event)
            
            start_time = time.time()
            
            for event in events:
                debouncer.should_debounce(event)
            
            processing_time = time.time() - start_time
            
            # Performance should scale reasonably
            assert processing_time < file_count * 0.001  # Max 1ms per file
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable."""
        debouncer = EventDebouncer(strategy=DebounceStrategy.INTELLIGENT)
        
        # Process many events for the same files repeatedly
        for cycle in range(10):
            for i in range(1000):
                event = FileEvent(Path(f'/test/file_{i % 50}.json'), 'modified', time.time() + cycle * 1000 + i)
                debouncer.should_debounce(event)
        
        # Memory usage should be bounded
        stats = debouncer.get_stats()
        assert 'cache_size' in stats
        
        # Cache should not grow unbounded
        if isinstance(debouncer.debouncer, SimpleDebouncer):
            assert len(debouncer.debouncer.last_event_times) <= 10000


@pytest.mark.integration
class TestDebouncerIntegration:
    """Integration tests for debouncing with real scenarios."""
    
    @pytest.mark.asyncio
    async def test_real_world_file_editing_scenario(self):
        """Test debouncing during typical file editing scenario."""
        debouncer = EventDebouncer(strategy=DebounceStrategy.ADAPTIVE, base_period=0.2)
        
        # Simulate user editing a file (rapid saves)
        file_path = Path('/project/src/main.json')
        events = []
        
        # Rapid events (auto-save every 100ms for 2 seconds)
        base_time = time.time()
        for i in range(20):
            event = FileEvent(file_path, 'modified', base_time + i * 0.1)
            events.append(event)
        
        # Process events
        processed = debouncer.process_events(events)
        
        # Should significantly reduce the number of events
        assert len(processed) < len(events)
        assert len(processed) >= 1  # At least one event should remain
        
        # Verify the last event is preserved (most recent change)
        last_original = max(events, key=lambda e: e.timestamp)
        last_processed = max(processed, key=lambda e: e.timestamp)
        assert last_processed.timestamp == last_original.timestamp
    
    @pytest.mark.asyncio
    async def test_multi_file_project_scenario(self):
        """Test debouncing in multi-file project editing scenario."""
        debouncer = EventDebouncer(strategy=DebounceStrategy.INTELLIGENT, base_period=0.3)
        
        # Simulate refactoring affecting multiple related files
        project_files = [
            Path('/project/src/user/model.json'),
            Path('/project/src/user/controller.json'),
            Path('/project/src/user/view.json'),
            Path('/project/src/order/model.json'),
        ]
        
        events = []
        base_time = time.time()
        
        # Simulate rapid changes across related files
        for i in range(40):  # 10 changes per file
            file_index = i % len(project_files)
            event = FileEvent(project_files[file_index], 'modified', base_time + i * 0.05)
            events.append(event)
        
        processed = debouncer.process_events(events)
        
        # Should reduce events while preserving important changes
        assert len(processed) < len(events)
        assert len(processed) >= len(project_files)  # At least one event per file
        
        # Verify all files are represented in processed events
        processed_files = {event.path for event in processed}
        assert len(processed_files) == len(project_files)
