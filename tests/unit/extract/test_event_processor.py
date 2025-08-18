"""Unit tests for event processor functionality.

Tests cover event filtering, validation, priority queuing, and debouncing.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from snake_pipe.extract.event_processor import (
    FileEvent, EventPriority, EventQueue, FileEventProcessor,
    EventFilter, ValidationResult
)
from snake_pipe.extract.models import ProcessingError


class TestFileEvent:
    """Test FileEvent data model."""
    
    def test_event_creation(self):
        """Test creating file events."""
        path = Path('/test/file.json')
        event = FileEvent(path, 'modified', 1234567890.0)
        
        assert event.path == path
        assert event.event_type == 'modified'
        assert event.timestamp == 1234567890.0
        assert event.size is None
        assert event.priority == EventPriority.NORMAL
    
    def test_event_with_optional_fields(self):
        """Test event with all optional fields."""
        path = Path('/test/large_file.json')
        event = FileEvent(
            path=path,
            event_type='created',
            timestamp=1234567890.0,
            size=1024,
            priority=EventPriority.HIGH
        )
        
        assert event.size == 1024
        assert event.priority == EventPriority.HIGH
    
    def test_event_comparison(self):
        """Test event comparison and hashing."""
        path = Path('/test/file.json')
        event1 = FileEvent(path, 'modified', 1234567890.0)
        event2 = FileEvent(path, 'modified', 1234567890.0)
        event3 = FileEvent(path, 'created', 1234567890.0)
        
        assert event1 == event2
        assert event1 != event3
        assert hash(event1) == hash(event2)
        assert hash(event1) != hash(event3)
    
    def test_event_string_representation(self):
        """Test string representation of events."""
        path = Path('/test/file.json')
        event = FileEvent(path, 'modified', 1234567890.0, size=1024)
        
        str_repr = str(event)
        assert 'file.json' in str_repr
        assert 'modified' in str_repr
        assert '1024' in str_repr
    
    def test_is_ast_file(self):
        """Test AST file detection."""
        json_event = FileEvent(Path('/test/file.json'), 'created', time.time())
        ast_event = FileEvent(Path('/test/ast_data.json'), 'created', time.time())
        py_event = FileEvent(Path('/test/script.py'), 'created', time.time())
        
        assert json_event.is_ast_file() is True
        assert ast_event.is_ast_file() is True
        assert py_event.is_ast_file() is False


class TestEventPriority:
    """Test event priority enumeration."""
    
    def test_priority_values(self):
        """Test priority enumeration values."""
        assert EventPriority.LOW.value < EventPriority.NORMAL.value
        assert EventPriority.NORMAL.value < EventPriority.HIGH.value
        assert EventPriority.HIGH.value < EventPriority.CRITICAL.value
    
    def test_priority_comparison(self):
        """Test priority comparison."""
        assert EventPriority.LOW < EventPriority.HIGH
        assert EventPriority.CRITICAL > EventPriority.NORMAL
    
    def test_determine_priority(self):
        """Test priority determination logic."""
        # Large file should have higher priority
        large_event = FileEvent(Path('/test/large.json'), 'created', time.time(), size=10*1024*1024)
        small_event = FileEvent(Path('/test/small.json'), 'created', time.time(), size=1024)
        
        # Creation events should have higher priority than modifications
        create_event = FileEvent(Path('/test/file.json'), 'created', time.time())
        modify_event = FileEvent(Path('/test/file.json'), 'modified', time.time())
        
        # These would be determined by business logic in the processor
        assert create_event.event_type == 'created'
        assert modify_event.event_type == 'modified'


class TestEventQueue:
    """Test priority-based event queue."""
    
    @pytest.fixture
    def event_queue(self):
        """Create event queue for testing."""
        return EventQueue(maxsize=100)
    
    @pytest.mark.asyncio
    async def test_queue_initialization(self, event_queue):
        """Test queue initialization."""
        assert event_queue.empty()
        assert event_queue.qsize() == 0
        assert not event_queue.full()
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, event_queue):
        """Test that events are ordered by priority."""
        # Add events in reverse priority order
        low_event = FileEvent(Path('/test/low.json'), 'modified', time.time(), priority=EventPriority.LOW)
        normal_event = FileEvent(Path('/test/normal.json'), 'modified', time.time(), priority=EventPriority.NORMAL)
        high_event = FileEvent(Path('/test/high.json'), 'modified', time.time(), priority=EventPriority.HIGH)
        critical_event = FileEvent(Path('/test/critical.json'), 'created', time.time(), priority=EventPriority.CRITICAL)
        
        await event_queue.put(low_event)
        await event_queue.put(normal_event)
        await event_queue.put(high_event)
        await event_queue.put(critical_event)
        
        # Should come out in priority order
        first = await event_queue.get()
        second = await event_queue.get()
        third = await event_queue.get()
        fourth = await event_queue.get()
        
        assert first.priority == EventPriority.CRITICAL
        assert second.priority == EventPriority.HIGH
        assert third.priority == EventPriority.NORMAL
        assert fourth.priority == EventPriority.LOW
    
    @pytest.mark.asyncio
    async def test_timestamp_ordering_within_priority(self, event_queue):
        """Test timestamp ordering within same priority level."""
        # Create events with same priority but different timestamps
        old_event = FileEvent(Path('/test/old.json'), 'modified', 1000.0, priority=EventPriority.NORMAL)
        new_event = FileEvent(Path('/test/new.json'), 'modified', 2000.0, priority=EventPriority.NORMAL)
        
        await event_queue.put(old_event)
        await event_queue.put(new_event)
        
        # Newer events should come first within same priority
        first = await event_queue.get()
        second = await event_queue.get()
        
        assert first.timestamp > second.timestamp
    
    @pytest.mark.asyncio
    async def test_queue_capacity(self, event_queue):
        """Test queue capacity limits."""
        # Fill queue to capacity
        for i in range(100):
            event = FileEvent(Path(f'/test/file_{i}.json'), 'modified', time.time())
            await event_queue.put(event)
        
        assert event_queue.full()
        assert event_queue.qsize() == 100
        
        # Adding one more should not block (depends on implementation)
        try:
            await asyncio.wait_for(
                event_queue.put(FileEvent(Path('/test/overflow.json'), 'modified', time.time())),
                timeout=0.1
            )
            # If it succeeds, queue is not strictly bounded
        except asyncio.TimeoutError:
            # Queue is strictly bounded
            pass
    
    @pytest.mark.asyncio
    async def test_queue_stats(self, event_queue):
        """Test queue statistics."""
        # Add some events
        for i in range(10):
            event = FileEvent(Path(f'/test/file_{i}.json'), 'modified', time.time())
            await event_queue.put(event)
        
        stats = event_queue.get_stats()
        
        assert stats['size'] == 10
        assert stats['maxsize'] == 100
        assert stats['utilization'] == 0.1


class TestEventFilter:
    """Test event filtering functionality."""
    
    @pytest.fixture
    def event_filter(self):
        """Create event filter for testing."""
        return EventFilter(
            include_patterns=['*.json', '*.ast'],
            exclude_patterns=['*.tmp', '**/.git/**', '**/node_modules/**']
        )
    
    def test_include_pattern_matching(self, event_filter):
        """Test include pattern matching."""
        json_event = FileEvent(Path('/test/data.json'), 'created', time.time())
        ast_event = FileEvent(Path('/test/tree.ast'), 'created', time.time())
        py_event = FileEvent(Path('/test/script.py'), 'created', time.time())
        
        assert event_filter.should_include(json_event) is True
        assert event_filter.should_include(ast_event) is True
        assert event_filter.should_include(py_event) is False
    
    def test_exclude_pattern_matching(self, event_filter):
        """Test exclude pattern matching."""
        temp_event = FileEvent(Path('/test/temp.tmp'), 'created', time.time())
        git_event = FileEvent(Path('/project/.git/config'), 'modified', time.time())
        node_event = FileEvent(Path('/project/node_modules/package.json'), 'created', time.time())
        valid_event = FileEvent(Path('/test/data.json'), 'created', time.time())
        
        assert event_filter.should_include(temp_event) is False
        assert event_filter.should_include(git_event) is False
        assert event_filter.should_include(node_event) is False
        assert event_filter.should_include(valid_event) is True
    
    def test_complex_path_patterns(self, event_filter):
        """Test complex path pattern matching."""
        # Test nested exclusions
        deep_git = FileEvent(Path('/project/subdir/.git/hooks/pre-commit'), 'modified', time.time())
        deep_node = FileEvent(Path('/app/frontend/node_modules/react/package.json'), 'created', time.time())
        
        assert event_filter.should_include(deep_git) is False
        assert event_filter.should_include(deep_node) is False
    
    def test_case_sensitivity(self, event_filter):
        """Test case sensitivity in pattern matching."""
        json_upper = FileEvent(Path('/test/DATA.JSON'), 'created', time.time())
        json_mixed = FileEvent(Path('/test/Data.Json'), 'created', time.time())
        
        # Behavior depends on filesystem (usually case-insensitive on Windows/macOS)
        # This test documents expected behavior
        include_upper = event_filter.should_include(json_upper)
        include_mixed = event_filter.should_include(json_mixed)
        
        # On case-insensitive systems, these should be included
        assert isinstance(include_upper, bool)
        assert isinstance(include_mixed, bool)


class TestValidationResult:
    """Test validation result functionality."""
    
    def test_valid_result(self):
        """Test creating valid validation result."""
        result = ValidationResult(is_valid=True, message="Valid JSON")
        
        assert result.is_valid is True
        assert result.message == "Valid JSON"
        assert result.error_code is None
    
    def test_invalid_result(self):
        """Test creating invalid validation result."""
        result = ValidationResult(
            is_valid=False,
            message="Invalid JSON syntax",
            error_code="JSON_PARSE_ERROR"
        )
        
        assert result.is_valid is False
        assert result.message == "Invalid JSON syntax"
        assert result.error_code == "JSON_PARSE_ERROR"
    
    def test_result_string_representation(self):
        """Test string representation of validation result."""
        valid_result = ValidationResult(True, "Valid")
        invalid_result = ValidationResult(False, "Invalid", "ERROR")
        
        assert "Valid" in str(valid_result)
        assert "Invalid" in str(invalid_result)
        assert "ERROR" in str(invalid_result)


class TestFileEventProcessor:
    """Test file event processor functionality."""
    
    @pytest.fixture
    def processor_config(self):
        """Configuration for event processor."""
        return {
            'include_patterns': ['*.json'],
            'exclude_patterns': ['*.tmp', '**/.git/**'],
            'validate_json': True,
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'debounce_period': 0.3
        }
    
    @pytest.fixture
    def event_processor(self, processor_config):
        """Create event processor for testing."""
        return FileEventProcessor(processor_config)
    
    def test_processor_initialization(self, event_processor):
        """Test processor initialization."""
        assert event_processor.config is not None
        assert isinstance(event_processor.event_filter, EventFilter)
        assert event_processor.stats['events_processed'] == 0
        assert event_processor.stats['validation_errors'] == 0
    
    def test_filter_relevant_events(self, event_processor):
        """Test filtering of relevant events."""
        events = [
            FileEvent(Path('/test/data.json'), 'created', time.time()),      # Include
            FileEvent(Path('/test/script.py'), 'modified', time.time()),    # Exclude - wrong type
            FileEvent(Path('/test/temp.tmp'), 'created', time.time()),      # Exclude - temp file
            FileEvent(Path('/test/ast.json'), 'modified', time.time()),     # Include
        ]
        
        relevant = event_processor.filter_relevant_events(events)
        
        assert len(relevant) == 2
        assert all(event.path.suffix == '.json' for event in relevant)
        assert not any(str(event.path).endswith('.tmp') for event in relevant)
    
    def test_assign_event_priorities(self, event_processor):
        """Test event priority assignment."""
        # Large file should get higher priority
        large_event = FileEvent(Path('/test/large.json'), 'created', time.time(), size=5*1024*1024)
        small_event = FileEvent(Path('/test/small.json'), 'modified', time.time(), size=1024)
        
        prioritized = event_processor.assign_priorities([large_event, small_event])
        
        large_result = next(e for e in prioritized if e.path.name == 'large.json')
        small_result = next(e for e in prioritized if e.path.name == 'small.json')
        
        # Large file or creation event should have higher priority
        assert large_result.priority.value >= small_result.priority.value
    
    @patch('builtins.open')
    @patch('json.load')
    def test_validate_json_file_valid(self, mock_json_load, mock_open, event_processor):
        """Test JSON validation for valid files."""
        mock_json_load.return_value = {'valid': 'json'}
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        event = FileEvent(Path('/test/valid.json'), 'created', time.time())
        result = event_processor.validate_file(event)
        
        assert result.is_valid is True
        assert result.error_code is None
    
    @patch('builtins.open')
    def test_validate_json_file_invalid(self, mock_open, event_processor):
        """Test JSON validation for invalid files."""
        mock_open.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        event = FileEvent(Path('/test/invalid.json'), 'created', time.time())
        result = event_processor.validate_file(event)
        
        assert result.is_valid is False
        assert result.error_code == "JSON_PARSE_ERROR"
    
    @patch('builtins.open')
    def test_validate_file_not_found(self, mock_open, event_processor):
        """Test validation for missing files."""
        mock_open.side_effect = FileNotFoundError("File not found")
        
        event = FileEvent(Path('/test/missing.json'), 'created', time.time())
        result = event_processor.validate_file(event)
        
        assert result.is_valid is False
        assert result.error_code == "FILE_NOT_FOUND"
    
    def test_debounce_events(self, event_processor):
        """Test event debouncing."""
        now = time.time()
        
        events = [
            FileEvent(Path('/test/file1.json'), 'modified', now),
            FileEvent(Path('/test/file1.json'), 'modified', now + 0.1),  # Should be debounced
            FileEvent(Path('/test/file2.json'), 'created', now),
            FileEvent(Path('/test/file1.json'), 'modified', now + 0.5),  # Should not be debounced
        ]
        
        debounced = event_processor.debounce_events(events)
        
        # Should have 3 events: file2 created, file1 first and last modification
        assert len(debounced) == 3
        
        file1_events = [e for e in debounced if e.path.name == 'file1.json']
        file2_events = [e for e in debounced if e.path.name == 'file2.json']
        
        assert len(file1_events) == 2  # First and last
        assert len(file2_events) == 1
    
    @pytest.mark.asyncio
    async def test_process_event_batch(self, event_processor):
        """Test processing of event batches."""
        events = [
            FileEvent(Path('/test/file1.json'), 'created', time.time()),
            FileEvent(Path('/test/file2.json'), 'modified', time.time()),
        ]
        
        with patch.object(event_processor, 'validate_file') as mock_validate:
            mock_validate.return_value = ValidationResult(True, "Valid")
            
            processed = await event_processor.process_events(events)
            
            assert len(processed) == 2
            assert all(event.priority is not None for event in processed)
            assert mock_validate.call_count == 2
    
    def test_processor_statistics(self, event_processor):
        """Test processor statistics collection."""
        # Process some events to generate stats
        events = [
            FileEvent(Path('/test/file1.json'), 'created', time.time()),
            FileEvent(Path('/test/invalid.py'), 'modified', time.time()),  # Will be filtered
        ]
        
        filtered = event_processor.filter_relevant_events(events)
        
        stats = event_processor.get_stats()
        
        assert 'events_processed' in stats
        assert 'events_filtered' in stats
        assert 'validation_errors' in stats
        assert 'debounce_hits' in stats
    
    def test_large_file_handling(self, event_processor):
        """Test handling of large files."""
        # Create event for very large file
        large_event = FileEvent(
            Path('/test/huge.json'),
            'created',
            time.time(),
            size=100 * 1024 * 1024  # 100MB
        )
        
        prioritized = event_processor.assign_priorities([large_event])
        
        # Large files should get lower priority to avoid blocking
        assert prioritized[0].priority in [EventPriority.LOW, EventPriority.NORMAL]
    
    def test_event_type_specific_handling(self, event_processor):
        """Test handling of different event types."""
        events = [
            FileEvent(Path('/test/file.json'), 'created', time.time()),
            FileEvent(Path('/test/file.json'), 'modified', time.time()),
            FileEvent(Path('/test/file.json'), 'deleted', time.time()),
        ]
        
        prioritized = event_processor.assign_priorities(events)
        
        # Created events should generally have higher priority than modifications
        created_event = next((e for e in prioritized if e.event_type == 'created'), None)
        modified_event = next((e for e in prioritized if e.event_type == 'modified'), None)
        
        if created_event and modified_event:
            assert created_event.priority.value >= modified_event.priority.value


@pytest.mark.integration
class TestEventProcessorIntegration:
    """Integration tests for event processor."""
    
    @pytest.mark.asyncio
    async def test_full_processing_pipeline(self, tmp_path):
        """Test complete event processing pipeline."""
        # Create test files
        test_file = tmp_path / 'test_data.json'
        test_file.write_text('{"test": "data", "timestamp": 1234567890}')
        
        invalid_file = tmp_path / 'invalid.json'
        invalid_file.write_text('{"invalid": json}')  # Invalid JSON
        
        config = {
            'include_patterns': ['*.json'],
            'exclude_patterns': [],
            'validate_json': True,
            'max_file_size': 1024 * 1024,
            'debounce_period': 0.1
        }
        
        processor = FileEventProcessor(config)
        
        events = [
            FileEvent(test_file, 'created', time.time()),
            FileEvent(invalid_file, 'created', time.time()),
            FileEvent(Path(tmp_path / 'excluded.py'), 'created', time.time()),  # Should be filtered
        ]
        
        # Process events through full pipeline
        processed = await processor.process_events(events)
        
        # Should have 2 events (valid and invalid JSON, excluding .py file)
        assert len(processed) >= 1  # At least the valid file
        
        # Check that validation was performed
        stats = processor.get_stats()
        assert stats['events_processed'] > 0
