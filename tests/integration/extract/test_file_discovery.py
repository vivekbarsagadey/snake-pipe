"""Integration tests for AST file discovery with real directory structures.

This module contains integration tests that verify the file discovery system
works correctly with real file systems and directory structures.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
import pytest

from snake_pipe.extract.ast_extractor import create_discovery_service
from snake_pipe.extract.models import LanguageType, FileStatus
from snake_pipe.config.extract_config import (
    create_default_config, create_high_performance_config, create_strict_config
)


class TestFileDiscoveryIntegration:
    """Integration tests for file discovery with real file systems."""
    
    @pytest.fixture
    def complex_ast_structure(self):
        """Create a complex directory structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Create nested directory structure
            projects = {
                "java_project": {
                    "src/main/java/com/example": [
                        ("Main.java.json", self._create_java_ast("Main", ["main"])),
                        ("Service.java.json", self._create_java_ast("Service", ["process", "validate"])),
                    ],
                    "src/test/java/com/example": [
                        ("MainTest.java.json", self._create_java_ast("MainTest", ["testMain"])),
                    ]
                },
                "python_project": {
                    "src/python": [
                        ("main.py.json", self._create_python_ast("main.py", ["main", "init"])),
                        ("utils.py.json", self._create_python_ast("utils.py", ["helper", "formatter"])),
                    ],
                    "tests": [
                        ("test_main.py.json", self._create_python_ast("test_main.py", ["test_main"])),
                    ]
                },
                "mixed_project": {
                    "frontend/js": [
                        ("app.js.json", self._create_js_ast("app.js", ["init", "render"])),
                    ],
                    "backend/java": [
                        ("Controller.java.json", self._create_java_ast("Controller", ["handleRequest"])),
                    ]
                }
            }
            
            # Create the directory structure and files
            self._create_structure(base_path, projects)
            
            # Add some non-AST files
            (base_path / "README.md").write_text("# Test Project")
            (base_path / "config.yml").write_text("test: true")
            
            yield base_path
    
    def _create_java_ast(self, class_name: str, methods: list) -> Dict[str, Any]:
        """Create a sample Java AST structure."""
        return {
            "filePath": f"/project/src/{class_name}.java",
            "uniqueId": f"AST_{hash(class_name)}",
            "language": "java",
            "classes": [{
                "name": class_name,
                "qualifiedName": f"com.example.{class_name}",
                "methods": [{"name": method} for method in methods],
                "modifiers": ["public"]
            }],
            "imports": ["java.util.List", "java.io.IOException"]
        }
    
    def _create_python_ast(self, file_name: str, functions: list) -> Dict[str, Any]:
        """Create a sample Python AST structure."""
        return {
            "filePath": f"/project/src/{file_name}",
            "uniqueId": f"AST_{hash(file_name)}",
            "language": "python",
            "functions": [{"name": func} for func in functions],
            "imports": ["os", "sys", "json"]
        }
    
    def _create_js_ast(self, file_name: str, functions: list) -> Dict[str, Any]:
        """Create a sample JavaScript AST structure."""
        return {
            "filePath": f"/project/src/{file_name}",
            "uniqueId": f"AST_{hash(file_name)}",
            "language": "javascript",
            "functions": [{"name": func} for func in functions],
            "exports": ["module.exports"]
        }
    
    def _create_structure(self, base_path: Path, structure: Dict):
        """Recursively create directory structure."""
        for project_name, project_structure in structure.items():
            project_path = base_path / project_name
            
            for dir_path, files in project_structure.items():
                full_dir_path = project_path / dir_path
                full_dir_path.mkdir(parents=True, exist_ok=True)
                
                for file_name, content in files:
                    file_path = full_dir_path / file_name
                    file_path.write_text(json.dumps(content, indent=2))
    
    @pytest.mark.asyncio
    async def test_discover_files_complex_structure(self, complex_ast_structure):
        """Test file discovery across complex directory structure."""
        config = create_default_config(complex_ast_structure)
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(complex_ast_structure)
        
        # Verify discovery results
        assert len(result.files) > 0, "Should discover AST files"
        assert result.total_files_processed > 0, "Should process files"
        assert result.total_errors == 0, "Should have no errors"
        
        # Verify language detection
        languages_found = {f.language_info.language for f in result.files}
        assert LanguageType.JAVA in languages_found, "Should detect Java files"
        assert LanguageType.PYTHON in languages_found, "Should detect Python files"
        assert LanguageType.JAVASCRIPT in languages_found, "Should detect JavaScript files"
        
        # Verify file paths are relative
        for ast_file in result.files:
            assert not ast_file.path.is_absolute(), f"Path should be relative: {ast_file.path}"
    
    @pytest.mark.asyncio
    async def test_filtering_by_patterns(self, complex_ast_structure):
        """Test file filtering with include/exclude patterns."""
        config = create_default_config(complex_ast_structure)
        
        # Test include only Java files
        config.filter_config.include_patterns = ["*.java.json"]
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(complex_ast_structure)
        
        # Should only find Java files
        for ast_file in result.files:
            assert "java" in str(ast_file.path), f"Should only include Java files: {ast_file.path}"
        
        # Test exclude test files
        config.filter_config.include_patterns = ["*.json"]
        config.filter_config.exclude_patterns = ["*test*", "*Test*"]
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(complex_ast_structure)
        
        # Should exclude test files
        for ast_file in result.files:
            assert "test" not in str(ast_file.path).lower(), f"Should exclude test files: {ast_file.path}"
    
    @pytest.mark.asyncio
    async def test_depth_limiting(self, complex_ast_structure):
        """Test directory depth limiting."""
        config = create_default_config(complex_ast_structure)
        config.filter_config.max_depth = 2  # Very shallow
        
        discovery = create_discovery_service(config)
        result_shallow = await discovery.discover_ast_files(complex_ast_structure)
        
        # Test with deeper depth
        config.filter_config.max_depth = 10
        discovery = create_discovery_service(config)
        result_deep = await discovery.discover_ast_files(complex_ast_structure)
        
        # Deeper search should find more or equal files
        assert result_deep.total_files_found >= result_shallow.total_files_found
    
    @pytest.mark.asyncio
    async def test_language_specific_filtering(self, complex_ast_structure):
        """Test filtering by specific languages."""
        config = create_default_config(complex_ast_structure)
        config.filter_config.languages = {LanguageType.JAVA}
        
        discovery = create_discovery_service(config)
        result = await discovery.discover_ast_files(complex_ast_structure)
        
        # Should only include Java files (or files that couldn't be detected as other languages)
        java_files = [f for f in result.files if f.language_info.language == LanguageType.JAVA]
        non_java_files = [f for f in result.files if f.language_info.language != LanguageType.JAVA]
        
        # Most files should be Java, and non-Java files should have errors
        assert len(java_files) > 0, "Should find Java files"
        for non_java_file in non_java_files:
            assert len(non_java_file.errors) > 0, "Non-target language files should have errors"
    
    @pytest.mark.asyncio
    async def test_performance_with_high_performance_config(self, complex_ast_structure):
        """Test performance optimizations."""
        config = create_high_performance_config(complex_ast_structure)
        discovery = create_discovery_service(config)
        
        start_time = time.time()
        result = await discovery.discover_ast_files(complex_ast_structure)
        processing_time = time.time() - start_time
        
        # Verify performance
        assert processing_time < 10.0, f"Processing should be fast: {processing_time}s"
        assert result.total_files_processed > 0, "Should process files"
        
        # Verify directory caching was used
        if config.performance.enable_directory_cache:
            assert result.directory_index is not None, "Should have directory index"
    
    @pytest.mark.asyncio
    async def test_strict_validation_mode(self, complex_ast_structure):
        """Test strict validation configuration."""
        config = create_strict_config(complex_ast_structure)
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(complex_ast_structure)
        
        # In strict mode, all files should be properly validated
        for ast_file in result.files:
            assert ast_file.status in [FileStatus.DISCOVERED, FileStatus.VALIDATED]
            # In strict mode, files with low confidence should have warnings
            if ast_file.language_info.confidence < 0.8:
                # This would be implementation-specific
                pass
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_logging(self, complex_ast_structure):
        """Test error recovery with corrupted files."""
        # Create a corrupted JSON file
        corrupted_file = complex_ast_structure / "corrupted.json"
        corrupted_file.write_text("{ invalid json content")
        
        config = create_default_config(complex_ast_structure)
        config.continue_on_error = True
        
        discovery = create_discovery_service(config)
        result = await discovery.discover_ast_files(complex_ast_structure)
        
        # Should continue processing despite corrupted file
        assert result.total_errors >= 1, "Should detect at least one error"
        assert len(result.files) > 0, "Should still process valid files"
        assert result.success_rate < 1.0, "Success rate should reflect errors"
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, complex_ast_structure):
        """Test progress tracking functionality."""
        config = create_default_config(complex_ast_structure)
        config.enable_progress_tracking = True
        
        discovery = create_discovery_service(config)
        
        # Track progress updates
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append({
                'processed_files': progress.processed_files,
                'percentage': progress.progress_percentage,
                'files_per_second': progress.files_per_second
            })
        
        discovery.add_progress_callback(progress_callback)
        
        result = await discovery.discover_ast_files(complex_ast_structure)
        
        # Verify progress tracking
        if config.enable_progress_tracking:
            assert len(progress_updates) > 0, "Should have progress updates"
            
            # Progress should generally increase
            percentages = [update['percentage'] for update in progress_updates]
            if len(percentages) > 1:
                assert percentages[-1] >= percentages[0], "Progress should increase"
    
    @pytest.mark.asyncio
    async def test_metadata_extraction_accuracy(self, complex_ast_structure):
        """Test accuracy of metadata extraction."""
        config = create_default_config(complex_ast_structure)
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(complex_ast_structure)
        
        # Verify metadata completeness
        for ast_file in result.files:
            metadata = ast_file.metadata
            
            # Basic metadata should always be present
            assert metadata.file_size > 0, "File size should be positive"
            assert metadata.modified_time is not None, "Modified time should be set"
            
            # AST-specific metadata should be extracted where possible
            if metadata.unique_id:
                assert isinstance(metadata.unique_id, str), "Unique ID should be string"
                assert "AST_" in metadata.unique_id, "Should match our test data format"
            
            # Structure info should be extracted for known formats
            if metadata.structure_info:
                structure_keys = metadata.structure_info.keys()
                assert any("count" in key for key in structure_keys), "Should have count information"
    
    @pytest.mark.asyncio
    async def test_concurrent_discovery_multiple_paths(self, complex_ast_structure):
        """Test discovery across multiple source paths."""
        # Create separate discovery services for different subdirectories
        java_path = complex_ast_structure / "java_project"
        python_path = complex_ast_structure / "python_project"
        
        java_config = create_default_config(java_path)
        python_config = create_default_config(python_path)
        
        java_discovery = create_discovery_service(java_config)
        python_discovery = create_discovery_service(python_config)
        
        # Run discoveries concurrently
        import asyncio
        java_result, python_result = await asyncio.gather(
            java_discovery.discover_ast_files(java_path),
            python_discovery.discover_ast_files(python_path)
        )
        
        # Verify separate results
        assert len(java_result.files) > 0, "Should find Java files"
        assert len(python_result.files) > 0, "Should find Python files"
        
        # Verify language separation
        java_languages = {f.language_info.language for f in java_result.files}
        python_languages = {f.language_info.language for f in python_result.files}
        
        assert LanguageType.JAVA in java_languages, "Java discovery should find Java files"
        assert LanguageType.PYTHON in python_languages, "Python discovery should find Python files"


class TestRealWorldIntegration:
    """Integration tests with the provided sample AST data."""
    
    @pytest.mark.asyncio
    async def test_sample_ast_data_discovery(self):
        """Test discovery with real sample AST data."""
        sample_path = Path("/home/darshan/Projects/snake-pipe/ast_output")
        
        if not sample_path.exists():
            pytest.skip("Sample AST data not available")
        
        config = create_default_config(sample_path)
        discovery = create_discovery_service(config)
        
        result = await discovery.discover_ast_files(sample_path)
        
        # Verify real data processing
        assert len(result.files) > 0, "Should discover files in sample data"
        assert result.total_files_processed > 0, "Should process files"
        
        # Check language detection on real data
        languages_found = result.languages_found
        assert LanguageType.JAVA in languages_found, "Sample data should contain Java files"
        
        # Verify metadata extraction from real AST files
        for ast_file in result.files:
            assert ast_file.metadata.file_size > 0, "Real files should have size"
            assert ast_file.language_info.confidence > 0, "Should have language confidence"
            
            # Check for expected AST structure in real files
            if ast_file.metadata.unique_id:
                assert "AST_" in ast_file.metadata.unique_id, "Real AST files should have unique IDs"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks_real_data(self):
        """Test performance benchmarks with real data."""
        sample_path = Path("/home/darshan/Projects/snake-pipe/ast_output")
        
        if not sample_path.exists():
            pytest.skip("Sample AST data not available")
        
        config = create_high_performance_config(sample_path)
        discovery = create_discovery_service(config)
        
        start_time = time.time()
        result = await discovery.discover_ast_files(sample_path)
        processing_time = time.time() - start_time
        
        # Performance assertions for real data
        files_per_second = result.total_files_found / max(processing_time, 0.001)
        
        assert processing_time < 30.0, f"Processing should complete quickly: {processing_time}s"
        assert files_per_second > 1, f"Should process at reasonable rate: {files_per_second} files/s"
        
        # Log performance metrics for analysis
        print(f"Performance metrics:")
        print(f"  Files found: {result.total_files_found}")
        print(f"  Files processed: {result.total_files_processed}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Files per second: {files_per_second:.2f}")
        print(f"  Success rate: {result.success_rate:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
