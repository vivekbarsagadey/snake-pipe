"""Unit tests for AST extractor module.

This module contains comprehensive unit tests for the AST file discovery
and extraction functionality with mocks and isolated testing.
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from snake_pipe.extract.ast_extractor import (
    ASTFileDiscovery, LanguageDetector, FileFilter, DirectoryIndexer,
    validate_discovery_config, create_discovery_service
)
from snake_pipe.extract.models import (
    LanguageType, LanguageInfo, FilterConfig, ASTMetadata, ASTFile,
    ValidationLevel, FileStatus
)
from snake_pipe.config.extract_config import (
    ExtractConfig, LanguageDetectionConfig, PerformanceConfig,
    create_default_config
)
from snake_pipe.utils.file_utils import FileSystemError


class TestLanguageDetector:
    """Unit tests for LanguageDetector class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = LanguageDetectionConfig()
        self.detector = LanguageDetector(self.config)
    
    @pytest.mark.asyncio
    async def test_detect_language_by_filename(self):
        """Test language detection based on filename patterns."""
        # Test Java detection
        java_path = Path("Test.java.json")
        with patch.object(self.detector, '_detect_by_content', return_value=LanguageInfo.unknown()):
            with patch.object(self.detector, '_detect_by_directory', return_value=LanguageInfo.unknown()):
                result = await self.detector.detect_language(java_path)
        
        assert result.language == LanguageType.JAVA
        assert result.confidence >= 0.7
        assert "filename" in result.detection_method
    
    @pytest.mark.asyncio
    async def test_detect_language_by_content(self):
        """Test language detection based on JSON content."""
        test_path = Path("test.json")
        
        # Mock JSON content with Java indicators
        java_content = {
            "classes": [{"name": "TestClass"}],
            "imports": ["java.util.List"],
            "qualifiedName": "com.example.Test"
        }
        
        with patch('snake_pipe.utils.file_utils.read_json_metadata', return_value=java_content):
            with patch.object(self.detector, '_detect_by_filename', return_value=LanguageInfo.unknown()):
                with patch.object(self.detector, '_detect_by_directory', return_value=LanguageInfo.unknown()):
                    result = await self.detector.detect_language(test_path)
        
        assert result.language == LanguageType.JAVA
        assert result.confidence > 0
        assert "content" in result.detection_method
    
    @pytest.mark.asyncio
    async def test_detect_language_by_directory(self):
        """Test language detection based on directory structure."""
        java_path = Path("/project/src/main/java/com/example/Test.json")
        
        with patch.object(self.detector, '_detect_by_filename', return_value=LanguageInfo.unknown()):
            with patch.object(self.detector, '_detect_by_content', return_value=LanguageInfo.unknown()):
                result = await self.detector.detect_language(java_path)
        
        assert result.language == LanguageType.JAVA
        assert result.confidence > 0
        assert "directory" in result.detection_method
    
    @pytest.mark.asyncio
    async def test_detect_language_combined_strategies(self):
        """Test combining multiple detection strategies."""
        java_path = Path("Test.java.json")
        java_content = {"classes": [{"name": "Test"}]}
        
        with patch('snake_pipe.utils.file_utils.read_json_metadata', return_value=java_content):
            result = await self.detector.detect_language(java_path)
        
        # Should have high confidence from combined detection
        assert result.language == LanguageType.JAVA
        assert result.confidence >= 0.8
    
    @pytest.mark.asyncio
    async def test_detect_language_unknown(self):
        """Test handling of unknown languages."""
        unknown_path = Path("unknown_file.unknown")
        
        with patch('snake_pipe.utils.file_utils.read_json_metadata', return_value={}):
            result = await self.detector.detect_language(unknown_path)
        
        assert result.language == LanguageType.UNKNOWN
        assert result.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_language_error_handling(self):
        """Test error handling in language detection."""
        test_path = Path("error_file.json")
        
        with patch('snake_pipe.utils.file_utils.read_json_metadata', side_effect=Exception("Read error")):
            result = await self.detector.detect_language(test_path)
        
        # Should return unknown but not raise exception
        assert result.language == LanguageType.UNKNOWN


class TestFileFilter:
    """Unit tests for FileFilter class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = FilterConfig(
            include_patterns=["*.json"],
            exclude_patterns=["*test*"],
            max_depth=10,
            min_file_size=1,
            max_file_size=1024 * 1024
        )
        self.filter = FileFilter(self.config)
    
    @pytest.mark.asyncio
    async def test_should_include_file_valid(self):
        """Test including valid files."""
        test_path = Path("valid.json")
        
        with patch('snake_pipe.utils.file_utils.is_file_json', return_value=True):
            with patch('snake_pipe.utils.file_utils.get_file_stats', return_value=(1024, datetime.now(), None)):
                result = await self.filter.should_include_file(test_path, depth=5)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_should_exclude_file_pattern(self):
        """Test excluding files based on patterns."""
        test_path = Path("test_file.json")
        
        with patch('snake_pipe.utils.file_utils.is_file_json', return_value=True):
            with patch('snake_pipe.utils.file_utils.get_file_stats', return_value=(1024, datetime.now(), None)):
                result = await self.filter.should_include_file(test_path, depth=5)
        
        assert result is False  # Should be excluded due to "test" pattern
    
    @pytest.mark.asyncio
    async def test_should_exclude_file_depth(self):
        """Test excluding files based on depth limit."""
        test_path = Path("valid.json")
        
        with patch('snake_pipe.utils.file_utils.is_file_json', return_value=True):
            result = await self.filter.should_include_file(test_path, depth=15)  # Exceeds max_depth
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_should_exclude_file_size(self):
        """Test excluding files based on size limits."""
        test_path = Path("large.json")
        
        with patch('snake_pipe.utils.file_utils.is_file_json', return_value=True):
            with patch('snake_pipe.utils.file_utils.get_file_stats', return_value=(2 * 1024 * 1024, datetime.now(), None)):
                result = await self.filter.should_include_file(test_path, depth=5)
        
        assert result is False  # Too large
    
    @pytest.mark.asyncio
    async def test_should_exclude_non_json(self):
        """Test excluding non-JSON files."""
        test_path = Path("not_json.txt")
        
        with patch('snake_pipe.utils.file_utils.is_file_json', return_value=False):
            result = await self.filter.should_include_file(test_path, depth=5)
        
        assert result is False


class TestDirectoryIndexer:
    """Unit tests for DirectoryIndexer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.indexer = DirectoryIndexer()
    
    @pytest.mark.asyncio
    async def test_build_index(self):
        """Test building directory index."""
        test_dir = Path("/test/directory")
        
        # Mock file iteration
        mock_files = [
            Path("/test/directory/file1.json"),
            Path("/test/directory/file2.json"),
            Path("/test/directory/subdir")
        ]
        
        async def mock_scan_directory(directory):
            for file_path in mock_files:
                yield file_path
        
        with patch('snake_pipe.extract.ast_extractor.scan_directory_async', mock_scan_directory):
            with patch('snake_pipe.utils.file_utils.get_file_stats', return_value=(1024, datetime.now(), None)):
                with patch.object(Path, 'is_file', side_effect=lambda: True if 'file' in str(mock_files[0]) else False):
                    with patch.object(Path, 'is_dir', side_effect=lambda: 'subdir' in str(mock_files[0])):
                        index = await self.indexer.build_index(test_dir)
        
        assert index.base_path == test_dir
        assert index.file_count >= 0
        assert index.total_size >= 0
    
    def test_clear_cache(self):
        """Test clearing directory cache."""
        # Add something to cache
        test_path = Path("/test")
        self.indexer._cache[test_path] = MagicMock()
        
        assert len(self.indexer._cache) > 0
        
        self.indexer.clear_cache()
        
        assert len(self.indexer._cache) == 0


class TestASTFileDiscovery:
    """Unit tests for ASTFileDiscovery class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = create_default_config(Path("/test"))
        self.discovery = ASTFileDiscovery(self.config)
    
    @pytest.mark.asyncio
    async def test_discover_ast_files_success(self):
        """Test successful file discovery."""
        test_dir = Path("/test/directory")
        
        # Mock the file scanning
        mock_files = [Path("/test/directory/test.java.json")]
        
        async def mock_scan_directory(directory, max_depth=None, follow_symlinks=False):
            for file_path in mock_files:
                yield file_path
        
        # Mock file processing
        mock_ast_file = ASTFile(
            path=Path("test.java.json"),
            language_info=LanguageInfo(LanguageType.JAVA, 0.9, "test"),
            metadata=ASTMetadata(1024, datetime.now())
        )
        
        with patch('snake_pipe.extract.ast_extractor.scan_directory_async', mock_scan_directory):
            with patch.object(self.discovery, '_create_ast_file', return_value=mock_ast_file):
                with patch.object(self.discovery.file_filter, 'should_include_file', return_value=True):
                    with patch.object(Path, 'is_file', return_value=True):
                        result = await self.discovery.discover_ast_files(test_dir)
        
        assert len(result.files) > 0
        assert result.total_files_processed > 0
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_create_ast_file(self):
        """Test AST file creation."""
        test_path = Path("/test/file.java.json")
        base_path = Path("/test")
        
        mock_language_info = LanguageInfo(LanguageType.JAVA, 0.9, "test")
        mock_metadata = ASTMetadata(1024, datetime.now())
        
        with patch.object(self.discovery.language_detector, 'detect_language', return_value=mock_language_info):
            with patch('snake_pipe.utils.file_utils.extract_ast_metadata', return_value=mock_metadata):
                ast_file = await self.discovery._create_ast_file(test_path, base_path)
        
        assert ast_file.language_info.language == LanguageType.JAVA
        assert ast_file.metadata.file_size == 1024
        assert ast_file.status == FileStatus.DISCOVERED
    
    @pytest.mark.asyncio
    async def test_error_handling_continue_on_error(self):
        """Test error handling with continue_on_error=True."""
        test_dir = Path("/test/directory")
        self.config.continue_on_error = True
        
        mock_files = [
            Path("/test/directory/good.json"),
            Path("/test/directory/bad.json")
        ]
        
        async def mock_scan_directory(directory, max_depth=None, follow_symlinks=False):
            for file_path in mock_files:
                yield file_path
        
        def mock_create_ast_file(file_path, base_path):
            if "bad" in str(file_path):
                raise Exception("Processing error")
            return ASTFile(
                path=Path("good.json"),
                language_info=LanguageInfo(LanguageType.JAVA, 0.9, "test"),
                metadata=ASTMetadata(1024, datetime.now())
            )
        
        with patch('snake_pipe.extract.ast_extractor.scan_directory_async', mock_scan_directory):
            with patch.object(self.discovery, '_create_ast_file', side_effect=mock_create_ast_file):
                with patch.object(self.discovery.file_filter, 'should_include_file', return_value=True):
                    with patch.object(Path, 'is_file', return_value=True):
                        result = await self.discovery.discover_ast_files(test_dir)
        
        # Should process good files despite errors
        assert result.total_errors > 0
        assert len(result.files) > 0
    
    def test_add_progress_callback(self):
        """Test adding progress callbacks."""
        callback = MagicMock()
        
        self.discovery.add_progress_callback(callback)
        
        assert callback in self.discovery.progress_callbacks


class TestValidateDiscoveryConfig:
    """Unit tests for configuration validation."""
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        config = FilterConfig(
            include_patterns=["*.json"],
            exclude_patterns=["*test*"],
            max_depth=10,
            min_file_size=1,
            max_file_size=1024
        )
        
        result = validate_discovery_config(config)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.normalized_config is config
    
    def test_validate_config_invalid_patterns(self):
        """Test validation with invalid regex patterns."""
        config = FilterConfig(
            include_patterns=["[invalid"],  # Invalid regex
            max_depth=10
        )
        
        result = validate_discovery_config(config)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validate_config_invalid_limits(self):
        """Test validation with invalid numeric limits."""
        config = FilterConfig(
            max_depth=0,  # Invalid depth
            min_file_size=-1,  # Invalid size
            max_file_size=100,
            include_patterns=["*.json"]
        )
        
        result = validate_discovery_config(config)
        
        assert result.is_valid is False
        assert len(result.errors) >= 2  # At least depth and file size errors


class TestCreateDiscoveryService:
    """Unit tests for discovery service factory."""
    
    def test_create_discovery_service_valid_config(self):
        """Test creating discovery service with valid configuration."""
        config = create_default_config(Path("/test"))
        
        service = create_discovery_service(config)
        
        assert isinstance(service, ASTFileDiscovery)
        assert service.config == config
    
    def test_create_discovery_service_invalid_config(self):
        """Test creating discovery service with invalid configuration."""
        config = create_default_config(Path("/test"))
        config.filter_config.max_depth = 0  # Invalid
        
        with pytest.raises(Exception):  # Should raise FilterValidationError
            create_discovery_service(config)


class TestPerformanceOptimizations:
    """Unit tests for performance-related functionality."""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent file processing."""
        config = create_default_config(Path("/test"))
        config.performance.max_concurrent_files = 2
        discovery = ASTFileDiscovery(config)
        
        # Mock multiple files
        mock_files = [
            Path("/test/file1.json"),
            Path("/test/file2.json"),
            Path("/test/file3.json")
        ]
        
        async def mock_scan_directory(directory, max_depth=None, follow_symlinks=False):
            for file_path in mock_files:
                yield file_path
        
        call_count = 0
        
        async def mock_create_ast_file(file_path, base_path):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate processing time
            return ASTFile(
                path=file_path,
                language_info=LanguageInfo(LanguageType.JAVA, 0.9, "test"),
                metadata=ASTMetadata(1024, datetime.now())
            )
        
        with patch('snake_pipe.extract.ast_extractor.scan_directory_async', mock_scan_directory):
            with patch.object(discovery, '_create_ast_file', side_effect=mock_create_ast_file):
                with patch.object(discovery.file_filter, 'should_include_file', return_value=True):
                    with patch.object(Path, 'is_file', return_value=True):
                        result = await discovery.discover_ast_files(Path("/test"))
        
        # Should have processed files concurrently
        assert call_count == len(mock_files)
        assert len(result.files) == len(mock_files)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing functionality."""
        config = create_default_config(Path("/test"))
        config.performance.batch_size = 2  # Small batch size for testing
        discovery = ASTFileDiscovery(config)
        
        # Test is implicitly covered by the main discovery logic
        # This verifies the batch processing doesn't break anything
        mock_files = [Path(f"/test/file{i}.json") for i in range(5)]
        
        async def mock_scan_directory(directory, max_depth=None, follow_symlinks=False):
            for file_path in mock_files:
                yield file_path
        
        with patch('snake_pipe.extract.ast_extractor.scan_directory_async', mock_scan_directory):
            with patch.object(discovery, '_create_ast_file', return_value=MagicMock()):
                with patch.object(discovery.file_filter, 'should_include_file', return_value=True):
                    with patch.object(Path, 'is_file', return_value=True):
                        result = await discovery.discover_ast_files(Path("/test"))
        
        # Should handle batching without errors
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
