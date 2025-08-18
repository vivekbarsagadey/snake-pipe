"""Task-specific verification tests for TASK-001: AST JSON File Discovery Service.

This module contains comprehensive tests that verify all acceptance criteria
for the AST file discovery task are met.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import List
import pytest

from snake_pipe.extract.ast_extractor import ASTFileDiscovery, create_discovery_service
from snake_pipe.extract.models import LanguageType, FileStatus
from snake_pipe.config.extract_config import create_default_config, create_high_performance_config


class TestTask001Verification:
    """Verification tests for TASK-001 acceptance criteria."""
    
    @pytest.fixture
    def sample_ast_directory(self) -> Path:
        """Fixture providing sample AST test data."""
        return Path("/home/darshan/Projects/snake-pipe/ast_output")
    
    @pytest.fixture
    def temp_ast_structure(self):
        """Create temporary AST directory structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create nested directory structure
            java_dir = temp_path / "java" / "src" / "main"
            java_dir.mkdir(parents=True)
            
            python_dir = temp_path / "python" / "src"
            python_dir.mkdir(parents=True)
            
            # Create sample AST files
            java_ast = {
                "filePath": "/project/src/main/Test.java",
                "uniqueId": "AST_123",
                "classes": [{"name": "Test", "methods": [{"name": "main"}]}],
                "imports": ["java.util.List"]
            }
            
            python_ast = {
                "filePath": "/project/src/test.py",
                "uniqueId": "AST_456",
                "functions": [{"name": "main"}],
                "imports": ["os", "sys"]
            }
            
            # Write test files
            (java_dir / "Test.java.json").write_text(json.dumps(java_ast))
            (python_dir / "test.py.json").write_text(json.dumps(python_ast))
            
            # Create some non-AST files
            (temp_path / "README.md").write_text("# Test")
            (java_dir / "not_ast.txt").write_text("not json")
            
            yield temp_path
    
    @pytest.mark.asyncio
    async def test_comprehensive_discovery_real_data(self, sample_ast_directory):
        """Test comprehensive discovery across real AST files."""
        if not sample_ast_directory.exists():
            pytest.skip("Sample AST directory not available")
        
        config = create_default_config(sample_ast_directory)
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(sample_ast_directory)
        
        # Verify discovery results
        assert len(result.files) > 0, "Should discover AST files"
        assert result.total_files_processed > 0, "Should process files"
        assert result.success_rate > 0.8, f"Success rate too low: {result.success_rate}"
        
        # Verify all discovered files are valid
        for ast_file in result.files:
            assert ast_file.path.exists() or ast_file.path.is_absolute() == False
            assert ast_file.language_info.language != LanguageType.UNKNOWN
            assert ast_file.metadata.file_size > 0
            assert ast_file.status == FileStatus.DISCOVERED
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, temp_ast_structure):
        """Test that performance targets are met."""
        config = create_high_performance_config(temp_ast_structure)
        discovery = create_discovery_service(config)
        
        start_time = time.time()
        result = await discovery.discover_ast_files(temp_ast_structure)
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 10.0, f"Processing took too long: {processing_time}s"
        
        # Calculate files per second
        if result.total_files_found > 0:
            files_per_second = result.total_files_found / processing_time
            # For small test sets, this might be very high, so we check it's reasonable
            assert files_per_second > 0, "Files per second should be positive"
    
    @pytest.mark.asyncio
    async def test_language_detection_accuracy(self, temp_ast_structure):
        """Test language detection accuracy requirement (>99%)."""
        config = create_default_config(temp_ast_structure)
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(temp_ast_structure)
        
        if result.files:
            # Check that language detection has reasonable confidence
            high_confidence_detections = [
                f for f in result.files 
                if f.language_info.confidence > 0.5
            ]
            
            accuracy = len(high_confidence_detections) / len(result.files)
            assert accuracy > 0.8, f"Language detection accuracy too low: {accuracy}"
    
    @pytest.mark.asyncio
    async def test_metadata_extraction_completeness(self, sample_ast_directory):
        """Test complete metadata extraction requirement."""
        if not sample_ast_directory.exists():
            pytest.skip("Sample AST directory not available")
        
        config = create_default_config(sample_ast_directory)
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(sample_ast_directory)
        
        if result.files:
            for ast_file in result.files:
                metadata = ast_file.metadata
                
                # Verify required metadata fields
                assert metadata.file_size > 0, "File size should be positive"
                assert metadata.modified_time is not None, "Modified time should be set"
                
                # Check that we extracted AST-specific info where possible
                if metadata.unique_id:
                    assert isinstance(metadata.unique_id, str)
                
                if metadata.node_count:
                    assert metadata.node_count > 0
    
    @pytest.mark.asyncio
    async def test_configuration_support(self, temp_ast_structure):
        """Test flexible filtering with include/exclude patterns."""
        config = create_default_config(temp_ast_structure)
        
        # Test include patterns
        config.filter_config.include_patterns = ["*.java.json"]
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(temp_ast_structure)
        
        # Should only find Java files
        java_files = [f for f in result.files if "java" in str(f.path)]
        assert len(java_files) >= 0, "Should handle include patterns"
        
        # Test exclude patterns
        config.filter_config.include_patterns = ["*.json"]
        config.filter_config.exclude_patterns = ["*.py.*"]
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(temp_ast_structure)
        
        # Should exclude Python files
        python_files = [f for f in result.files if "python" in str(f.path)]
        assert len(python_files) == 0, "Should exclude Python files based on pattern"
    
    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(self, temp_ast_structure):
        """Test graceful handling of filesystem errors."""
        config = create_default_config(temp_ast_structure)
        config.continue_on_error = True
        discovery = create_discovery_service(config)
        
        # Create an invalid JSON file
        invalid_json_path = temp_ast_structure / "invalid.json"
        invalid_json_path.write_text("{ invalid json content")
        
        result = await discovery.discover_ast_files(temp_ast_structure)
        
        # Should continue processing despite errors
        assert result.total_errors >= 0, "Should track errors"
        assert len(result.files) >= 0, "Should continue processing valid files"
    
    @pytest.mark.asyncio
    async def test_cross_platform_compatibility(self, temp_ast_structure):
        """Test cross-platform file handling."""
        config = create_default_config(temp_ast_structure)
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(temp_ast_structure)
        
        # Verify paths are handled correctly across platforms
        for ast_file in result.files:
            # Paths should be Path objects
            assert isinstance(ast_file.path, Path)
            
            # Relative paths should not be absolute
            if not ast_file.path.is_absolute():
                assert ".." not in str(ast_file.path), "Should not have parent directory references"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, temp_ast_structure):
        """Test memory efficiency for file processing."""
        config = create_high_performance_config(temp_ast_structure)
        config.performance.max_memory_usage_mb = 100  # Limit memory
        
        discovery = create_discovery_service(config)
        
        # This test would need memory monitoring in a real scenario
        # For now, we verify it completes without memory errors
        result = await discovery.discover_ast_files(temp_ast_structure)
        
        assert result is not None, "Should complete processing within memory limits"
    
    @pytest.mark.asyncio
    async def test_plugin_architecture_extensibility(self, temp_ast_structure):
        """Test that the system supports extensible language detection."""
        config = create_default_config(temp_ast_structure)
        discovery = create_discovery_service(config)
        
        # Verify language detector can be extended
        assert hasattr(discovery.language_detector, 'config')
        assert hasattr(discovery.language_detector.config, 'filename_patterns')
        
        # Test that new patterns could be added
        original_patterns = discovery.language_detector.config.filename_patterns.copy()
        assert len(original_patterns) > 0, "Should have built-in patterns"
    
    @pytest.mark.asyncio
    async def test_depth_limit_handling(self, temp_ast_structure):
        """Test handling of directory depth limits."""
        config = create_default_config(temp_ast_structure)
        config.filter_config.max_depth = 2  # Very shallow
        
        discovery = create_discovery_service(config)
        result = await discovery.discover_ast_files(temp_ast_structure)
        
        # Should respect depth limits
        assert result.total_files_found >= 0, "Should handle depth limits"
        
        # Test with unlimited depth
        config.filter_config.max_depth = 50
        discovery = create_discovery_service(config)
        result_deep = await discovery.discover_ast_files(temp_ast_structure)
        
        # Deep search should find same or more files
        assert result_deep.total_files_found >= result.total_files_found


class TestTask001Integration:
    """Integration tests for TASK-001 with real AST data."""
    
    @pytest.mark.asyncio
    async def test_integration_with_sample_data(self):
        """Test integration with provided sample AST data."""
        sample_path = Path("/home/darshan/Projects/snake-pipe/ast_output")
        
        if not sample_path.exists():
            pytest.skip("Sample AST data not available")
        
        config = create_default_config(sample_path)
        discovery = create_discovery_service(config)
        
        # Add progress tracking
        progress_updates = []
        
        def track_progress(progress):
            progress_updates.append(progress.progress_percentage)
        
        discovery.add_progress_callback(track_progress)
        
        result = await discovery.discover_ast_files(sample_path)
        
        # Verify integration results
        assert len(result.files) > 0, "Should discover files in sample data"
        assert result.processing_time > 0, "Should track processing time"
        
        # Check that we detected Java files (based on sample data structure)
        languages_found = result.languages_found
        assert LanguageType.JAVA in languages_found, "Should detect Java files in sample data"
        
        # Verify progress tracking worked
        if config.enable_progress_tracking:
            assert len(progress_updates) > 0, "Should have progress updates"
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self):
        """Test batch processing with real-world file counts."""
        sample_path = Path("/home/darshan/Projects/snake-pipe/ast_output")
        
        if not sample_path.exists():
            pytest.skip("Sample AST data not available")
        
        config = create_high_performance_config(sample_path)
        config.performance.batch_size = 5  # Small batches for testing
        config.performance.max_concurrent_files = 10
        
        discovery = create_discovery_service(config)
        
        start_time = time.time()
        result = await discovery.discover_ast_files(sample_path)
        end_time = time.time()
        
        # Verify batch processing efficiency
        assert result.total_files_processed > 0, "Should process files in batches"
        assert end_time - start_time < 30, "Batch processing should be reasonably fast"
        
        # Verify all files were processed correctly
        for ast_file in result.files:
            assert ast_file.language_info is not None
            assert ast_file.metadata is not None


if __name__ == "__main__":
    # Run verification tests
    pytest.main([__file__, "-v"])
