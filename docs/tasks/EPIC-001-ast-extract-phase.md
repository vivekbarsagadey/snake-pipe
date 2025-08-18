# EPIC-001: AST Processing Extract Phase Implementation

## Research Summary

**Key Findings**: 
- AST JSON files require hierarchical directory monitoring mirroring source code structure
- Multi-language parsers generate varying file sizes (1KB-50MB) requiring adaptive processing strategies  
- Real-time file watching critical for developer workflow integration and immediate feedback
- Batch processing essential for initial large-scale codebase ingestion (millions of files)
- File integrity validation prevents downstream pipeline corruption from malformed JSON

**Technical Analysis**: 
- Modern file watching APIs (inotify/kqueue) provide sub-millisecond change detection
- Async I/O patterns enable concurrent processing of thousands of files
- Memory-mapped file reading optimizes large AST JSON processing
- Event-driven architecture scales better than polling for real-time monitoring
- Backpressure handling prevents memory exhaustion during batch ingestion

**Architecture Impact**: 
- Extract phase serves as pipeline entry point requiring robust error handling
- Plugin architecture enables language-specific extraction strategies
- Clean separation between file discovery and content processing maintains modularity
- Async processing foundation supports entire pipeline scalability

**Risk Assessment**: 
- File system permissions and access control complexity
- Memory usage optimization for large file processing  
- Race conditions in concurrent file access scenarios
- Network file system latency and reliability issues

## Business Context

**User Problem**: Development teams need immediate AST processing when code changes occur, plus ability to ingest large existing codebases efficiently without overwhelming system resources or blocking development workflows.

**Business Value**: 
- 95% reduction in time-to-insight for code analysis (real-time vs. batch processing)
- Support for codebases with 1,000,000+ files through optimized batch processing
- Developer productivity improvement through non-blocking file processing
- Foundation enabling all downstream AST analysis and code intelligence features

**User Persona**: Data Engineers (40%) - require reliable, high-throughput file ingestion; Software Architects (30%) - need real-time code change analysis; Development Teams (20%) - immediate feedback on code modifications

**Success Metric**: 
- Process 10,000+ AST JSON files per minute during batch ingestion
- Sub-second detection and processing of individual file changes
- 99.9% successful file discovery and integrity validation
- Zero data loss during high-volume processing scenarios

## User Story

As a **data engineer**, I want a robust AST file extraction system so that I can reliably ingest code analysis data from multiple language parsers in both real-time and batch modes while maintaining data integrity and system performance.

## Technical Overview

**Task Type**: Epic  
**Pipeline Stage**: Extract  
**Complexity**: High  
**Dependencies**: Project setup, development environment configuration  
**Performance Impact**: Foundation for entire pipeline throughput - directly impacts 10,000+ files/min target

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/extract/ast_extractor.py` (core AST file discovery and extraction logic)
- `snake_pipe/extract/file_watcher.py` (real-time file monitoring with inotify/kqueue)
- `snake_pipe/extract/batch_processor.py` (high-throughput batch file processing)
- `snake_pipe/extract/file_validator.py` (JSON integrity and schema validation)
- `snake_pipe/config/extract_config.py` (extraction configuration management)
- `snake_pipe/utils/file_utils.py` (shared file handling utilities)
- `tests/tasks/test_epic001_verification.py` (epic verification tests)
- `tests/tasks/test_epic001_integration.py` (epic integration tests)
- `tests/unit/extract/test_ast_extractor.py` (unit tests with mocks)
- `tests/integration/extract/test_file_processing.py` (integration tests with real files)
- `tests/performance/extract/test_throughput.py` (performance and load testing)

### Key Functions to Implement

```python
async def discover_ast_files(source_path: Path, filter_config: FilterConfig) -> List[ASTFile]:
    """
    Purpose: Discover all AST JSON files in directory structure recursively
    Input: Source directory path and filtering configuration
    Output: List of ASTFile objects with metadata (path, size, modified time, language)
    Performance: Process 100,000+ files in <10 seconds
    """

async def watch_directory_changes(source_path: Path, callback: Callable) -> AsyncIterator[FileEvent]:
    """
    Purpose: Monitor directory for real-time file changes using OS-level events
    Input: Directory path and change notification callback
    Output: Stream of file events (created, modified, deleted)
    Performance: Sub-second change detection, handle 1000+ concurrent changes
    """

async def validate_ast_file_integrity(file_path: Path) -> ValidationResult:
    """
    Purpose: Validate AST JSON file structure and basic schema compliance
    Input: File path to AST JSON file
    Output: ValidationResult with status, errors, and file metadata
    Performance: <50ms per file validation, memory-efficient for large files
    """

async def extract_batch_files(file_paths: List[Path], config: BatchConfig) -> BatchResult:
    """
    Purpose: Process multiple AST files in optimized batches with concurrent I/O
    Input: List of file paths and batch processing configuration
    Output: BatchResult with processed files, errors, and performance metrics
    Performance: Process 10,000+ files per minute with <2GB memory usage
    """

async def extract_ast_metadata(file_path: Path) -> ASTMetadata:
    """
    Purpose: Extract language, parser version, and structural metadata from AST
    Input: Path to validated AST JSON file
    Output: ASTMetadata object with language, version, node counts, complexity
    Performance: <10ms per file metadata extraction
    """
```

### Technical Requirements

1. **Performance**: 
   - File discovery: Process 100,000+ files in <10 seconds
   - Real-time monitoring: Sub-second change detection and processing
   - Batch processing: 10,000+ files per minute sustained throughput
   - Memory efficiency: <2GB RAM usage for 100,000 concurrent files

2. **Error Handling**: 
   - File permission errors with graceful degradation
   - Malformed JSON detection and quarantine
   - Network file system timeouts and retries
   - Concurrent access conflicts resolution

3. **Scalability**: 
   - Horizontal scaling across multiple worker processes
   - Async/await patterns for I/O-bound operations
   - Backpressure handling for downstream pipeline protection
   - Memory-mapped file access for large AST files

4. **Integration**: 
   - Plugin architecture for language-specific extraction
   - Configuration-driven behavior without code changes
   - Event-driven architecture for real-time processing
   - Clean interfaces for transformation phase handoff

5. **Data Quality**: 
   - 99.9% successful file discovery accuracy
   - JSON schema validation before pipeline entry
   - Duplicate detection and handling strategies
   - Comprehensive audit logging for data lineage

6. **Reliability**: 
   - Automatic retry mechanisms for transient failures
   - Graceful degradation when file systems unavailable
   - Circuit breaker patterns for overload protection
   - Comprehensive monitoring and health checks

### Implementation Steps

1. **Core Domain Models**: Define ASTFile, FileEvent, ValidationResult, BatchResult entities following domain-driven design
2. **File Discovery Service**: Implement recursive directory traversal with filtering and metadata extraction
3. **Real-time Monitoring**: Integrate OS-level file watching (inotify/kqueue) with async event processing
4. **Batch Processing Engine**: Develop concurrent file processing with memory optimization and backpressure handling
5. **Validation Framework**: Create JSON schema validation with error reporting and quarantine management
6. **Configuration System**: Implement flexible configuration management for extraction behavior
7. **Plugin Architecture**: Design extensible system for language-specific extraction strategies
8. **Monitoring Integration**: Add comprehensive metrics, logging, and health check endpoints
9. **Performance Testing**: Benchmark throughput with real-world AST file datasets
10. **Integration Testing**: Validate end-to-end extraction with downstream transformation phase

### Code Patterns

```python
# Plugin Architecture Pattern for Language-Specific Extraction
class ASTExtractorFactory:
    @staticmethod
    def create_extractor(language: str, config: ExtractConfig) -> ASTExtractor:
        if language == "python":
            return PythonASTExtractor(config)
        elif language == "java":
            return JavaASTExtractor(config)
        else:
            return GenericASTExtractor(config)

# Event-Driven File Processing Pattern
class FileWatcher:
    async def start_watching(self, source_path: Path) -> None:
        async for event in self._watch_directory(source_path):
            await self._process_file_event(event)
    
    async def _process_file_event(self, event: FileEvent) -> None:
        if event.event_type == "created":
            await self._handle_file_creation(event.file_path)
        elif event.event_type == "modified":
            await self._handle_file_modification(event.file_path)

# Async Batch Processing Pattern
class BatchProcessor:
    async def process_batch(self, file_paths: List[Path]) -> BatchResult:
        semaphore = asyncio.Semaphore(50)  # Limit concurrent processing
        tasks = [self._process_single_file(path, semaphore) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._aggregate_results(results)
```

## Epic Acceptance Criteria

- [ ] **File Discovery**: Complete recursive AST file discovery with language detection and metadata extraction
- [ ] **Real-time Monitoring**: OS-level file watching with sub-second change detection and processing
- [ ] **Batch Processing**: High-throughput batch processing achieving 10,000+ files per minute
- [ ] **Validation Framework**: JSON schema validation with error reporting and quarantine management
- [ ] **Plugin Architecture**: Extensible extraction system supporting multiple programming languages
- [ ] **Performance Benchmarks**: All throughput and latency targets met under load testing
- [ ] **Integration Testing**: Seamless handoff to transformation phase with comprehensive error handling
- [ ] **Documentation**: Complete API documentation and operational runbooks
- [ ] **Monitoring**: Health checks, metrics collection, and performance monitoring operational
- [ ] **Test Coverage**: ≥90% test coverage with unit, integration, and performance tests

## Sub-Tasks

1. **TASK-001**: AST JSON File Discovery Service (Critical - 3 days)
2. **TASK-002**: Real-time File Watcher Implementation (High - 5 days)  
3. **TASK-003**: Batch Processing Engine (High - 4 days)
4. **TASK-004**: File Integrity Validation (Medium - 2 days)

## Dependencies

- Project development environment setup
- Base infrastructure for logging and configuration
- Test framework and coverage tooling
- Performance testing infrastructure

## Risks and Mitigation

**High-Risk Areas**:
- File system performance variations across operating systems
- Memory usage optimization for processing large AST files
- Race conditions in concurrent file access scenarios

**Mitigation Strategies**:
- Comprehensive cross-platform testing (Linux, macOS, Windows)
- Memory profiling and optimization with streaming file processing
- File locking mechanisms and retry strategies for concurrent access
- Circuit breaker patterns for overload protection

---

**Epic Owner**: TBD  
**Start Date**: TBD  
**Target Completion**: TBD  
**Status**: ⚪ Not Started
