# TASK-001: AST JSON File Discovery Service Implementation

## Research Summary

**Key Findings**: 
- AST JSON files follow hierarchical directory structures mirroring source code organization (src/main/java → ast/java/main)
- Language parsers generate different file naming conventions (*.ast.json, *.syntax.json, *_ast.json)
- File sizes vary dramatically: Python AST (1-10KB), Java AST (5-50KB), C++ AST (10-100KB+)
- Metadata extraction crucial for downstream processing decisions (language detection, parser version, complexity metrics)
- Modern file systems support efficient recursive traversal with OS-optimized APIs (os.scandir, pathlib)

**Technical Analysis**: 
- Recursive directory traversal with async I/O enables processing 100,000+ files in seconds
- Pathlib provides cross-platform path handling superior to string manipulation
- File metadata caching reduces repeated filesystem calls during batch processing
- Language detection via file naming patterns and JSON content analysis
- Memory-efficient iteration using generators prevents memory exhaustion

**Architecture Impact**: 
- File discovery serves as pipeline entry point - performance impacts entire system throughput
- Clean separation between discovery and content processing enables independent optimization
- Plugin architecture allows language-specific discovery strategies
- Metadata standardization enables consistent downstream processing

**Risk Assessment**: 
- File system permission restrictions in enterprise environments
- Network filesystem latency and reliability issues
- Memory usage growth with very large codebases (1M+ files)
- Cross-platform compatibility challenges (Windows vs. Unix paths)

## Business Context

**User Problem**: Data engineers need reliable discovery of AST JSON files across complex directory structures with varying naming conventions from different language parsers, enabling consistent processing regardless of source code organization.

**Business Value**: 
- 100% file discovery accuracy ensuring no AST data loss during ingestion
- Language-agnostic processing supporting polyglot development environments
- 10x faster file discovery than naive recursive approaches through optimized algorithms
- Foundation enabling reliable data pipeline operation at enterprise scale

**User Persona**: Data Engineers (Primary) - require comprehensive file discovery for ETL pipeline reliability; Software Architects (Secondary) - need visibility into codebase structure and organization

**Success Metric**: 
- Discover 100,000+ AST files in <10 seconds
- 100% accuracy in file detection across all supported languages
- Language detection accuracy >99% based on file analysis
- Support directory structures up to 20 levels deep without performance degradation

## User Story

As a **data engineer**, I want a reliable AST file discovery service so that I can automatically locate and catalog all AST JSON files across complex directory structures, ensuring comprehensive data ingestion regardless of language or parser variations.

## Technical Overview

**Task Type**: Story  
**Pipeline Stage**: Extract  
**Complexity**: Medium  
**Dependencies**: Base project structure, logging framework  
**Performance Impact**: Critical path for pipeline initialization - affects overall throughput

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/extract/ast_extractor.py` (core file discovery implementation)
- `snake_pipe/extract/models.py` (data models for ASTFile, metadata)
- `snake_pipe/config/extract_config.py` (discovery configuration settings)
- `snake_pipe/utils/file_utils.py` (shared file system utilities)
- `tests/tasks/test_task001_verification.py` (task-specific verification tests)
- `tests/tasks/test_task001_integration.py` (task-specific integration tests)
- `tests/unit/extract/test_ast_extractor.py` (comprehensive unit tests with mocks)
- `tests/integration/extract/test_file_discovery.py` (integration tests with real directory structures)
- `tests/fixtures/sample_ast_files/` (test data with various AST formats)

### Key Functions to Implement

```python
async def discover_ast_files(source_path: Path, filter_config: FilterConfig) -> List[ASTFile]:
    """
    Purpose: Recursively discover all AST JSON files in directory structure
    Input: Source directory path and filtering configuration (patterns, languages, depth limits)
    Output: List of ASTFile objects with complete metadata (path, language, size, modified_time)
    Performance: Process 100,000+ files in <10 seconds using async I/O
    Error Handling: Permission errors, broken symlinks, inaccessible directories
    """

async def detect_file_language(file_path: Path) -> LanguageInfo:
    """
    Purpose: Identify programming language from AST file analysis
    Input: Path to AST JSON file
    Output: LanguageInfo with detected language, confidence score, parser details
    Performance: <5ms per file using pattern matching and JSON header analysis
    Accuracy: >99% language detection accuracy across supported languages
    """

async def extract_file_metadata(file_path: Path) -> ASTMetadata:
    """
    Purpose: Extract comprehensive metadata from AST file without full parsing
    Input: Path to validated AST JSON file
    Output: ASTMetadata with file stats, JSON structure info, language details
    Performance: <10ms per file using streaming JSON analysis
    Memory: Process files up to 100MB without loading full content
    """

def validate_discovery_config(config: FilterConfig) -> ValidationResult:
    """
    Purpose: Validate file discovery configuration before processing
    Input: FilterConfig with patterns, exclusions, depth limits
    Output: ValidationResult indicating configuration validity and normalized settings
    Performance: <1ms configuration validation
    Error Prevention: Catch invalid regex patterns, conflicting rules early
    """

async def build_directory_index(source_path: Path, cache_duration: int = 3600) -> DirectoryIndex:
    """
    Purpose: Create indexed view of directory structure for optimized repeated access
    Input: Source directory path and cache duration in seconds
    Output: DirectoryIndex with hierarchical file organization and fast lookup
    Performance: 90% speedup for repeated directory scans using smart caching
    Memory: Efficient storage using compressed directory trees
    """
```

### Technical Requirements

1. **Performance**: 
   - File discovery rate: >10,000 files per second on SSD storage
   - Memory usage: <100MB for indexing 1,000,000 files
   - Language detection: <5ms per file average processing time
   - Directory traversal: Handle 20+ nested directory levels efficiently

2. **Error Handling**: 
   - Permission denied errors with graceful skip and logging
   - Broken symbolic links detection and handling
   - Network filesystem timeout recovery
   - Malformed directory structures and path handling

3. **Scalability**: 
   - Async I/O patterns for concurrent file system operations
   - Memory-efficient generators for large directory trees
   - Configurable concurrency limits to prevent system overload
   - Streaming processing for directories with millions of files

4. **Integration**: 
   - Plugin interface for language-specific discovery rules
   - Configuration-driven filtering without code modification
   - Event emission for real-time discovery progress tracking
   - Clean handoff to file validation and processing stages

5. **Data Quality**: 
   - 100% file discovery accuracy with comprehensive validation
   - Language detection confidence scoring and fallback strategies
   - Duplicate detection across directory structures
   - Metadata consistency validation and error reporting

6. **Reliability**: 
   - Atomic operations for directory indexing and caching
   - Graceful degradation when portions of filesystem unavailable
   - Comprehensive logging for troubleshooting and auditing
   - Recovery mechanisms for interrupted discovery processes

### Implementation Steps

1. **Core Domain Models**: Define ASTFile, LanguageInfo, ASTMetadata, FilterConfig entities with proper validation
2. **Directory Traversal Engine**: Implement async recursive traversal using pathlib and asyncio.gather for concurrency
3. **Language Detection System**: Create pattern-based and content-based language identification with confidence scoring
4. **Metadata Extraction**: Develop streaming JSON analysis for efficient metadata extraction from large files
5. **Configuration Framework**: Build flexible filtering system with regex patterns, exclusions, and depth controls
6. **Caching and Indexing**: Implement directory index with smart caching for repeated access optimization
7. **Error Handling**: Add comprehensive error handling with logging, recovery, and graceful degradation
8. **Performance Optimization**: Profile and optimize for target throughput using async patterns and efficient data structures
9. **Testing Infrastructure**: Create comprehensive test suite with mock filesystems and real directory structures
10. **Documentation**: Write API documentation and usage examples for discovery service integration

### Code Patterns

```python
# Async File Discovery Pattern
class ASTFileDiscovery:
    async def discover_files(self, source_path: Path, config: FilterConfig) -> List[ASTFile]:
        """Main discovery coordination with error handling and progress tracking."""
        discovered_files = []
        
        async for file_path in self._scan_directory_async(source_path, config):
            try:
                if await self._should_process_file(file_path, config):
                    ast_file = await self._create_ast_file(file_path)
                    discovered_files.append(ast_file)
            except Exception as e:
                await self._handle_file_error(file_path, e)
        
        return discovered_files

# Language Detection Strategy Pattern
class LanguageDetector:
    def __init__(self):
        self.strategies = [
            FilenamePatternStrategy(),
            JSONContentStrategy(),
            DirectoryStructureStrategy()
        ]
    
    async def detect_language(self, file_path: Path) -> LanguageInfo:
        for strategy in self.strategies:
            result = await strategy.detect(file_path)
            if result.confidence > 0.8:
                return result
        return LanguageInfo.unknown()

# Configuration-Driven Filtering Pattern
class FileFilter:
    def __init__(self, config: FilterConfig):
        self.include_patterns = [re.compile(p) for p in config.include_patterns]
        self.exclude_patterns = [re.compile(p) for p in config.exclude_patterns]
        self.max_depth = config.max_depth
        self.min_file_size = config.min_file_size
    
    async def should_include_file(self, file_path: Path, depth: int) -> bool:
        if depth > self.max_depth:
            return False
        
        if file_path.stat().st_size < self.min_file_size:
            return False
        
        filename = file_path.name
        if not any(pattern.match(filename) for pattern in self.include_patterns):
            return False
        
        if any(pattern.match(filename) for pattern in self.exclude_patterns):
            return False
        
        return True
```

## Acceptance Criteria

- [ ] **Comprehensive Discovery**: Recursively discover all AST JSON files in complex directory structures up to 20 levels deep
- [ ] **Performance Targets**: Process 100,000+ files in <10 seconds with <100MB memory usage
- [ ] **Language Detection**: Achieve >99% accuracy in programming language identification from AST files
- [ ] **Metadata Extraction**: Extract complete file metadata (size, modified time, language, complexity) in <10ms per file
- [ ] **Configuration Support**: Flexible filtering with include/exclude patterns, depth limits, and size constraints
- [ ] **Error Handling**: Graceful handling of permission errors, broken symlinks, and filesystem issues
- [ ] **Cross-Platform**: Full compatibility across Linux, macOS, and Windows filesystems
- [ ] **Memory Efficiency**: Stream large files without loading full content into memory
- [ ] **Plugin Architecture**: Extensible system for language-specific discovery strategies
- [ ] **Test Coverage**: ≥90% test coverage with unit tests, integration tests, and performance benchmarks
- [ ] **Documentation**: Complete API documentation with usage examples and performance characteristics
- [ ] **Integration**: Seamless handoff to file validation and batch processing components

## Dependencies

- Base project logging and configuration infrastructure
- Pathlib and asyncio for file system operations
- JSON parsing capabilities for metadata extraction
- Test framework setup for comprehensive validation

## Estimated Effort

**3 days** (24 hours)
- Day 1: Core discovery engine and language detection (8 hours)
- Day 2: Metadata extraction, filtering, and caching (8 hours)  
- Day 3: Error handling, testing, and performance optimization (8 hours)

## Acceptance Tests

```python
def test_discovers_all_ast_files_in_complex_structure():
    """Verify complete discovery across nested directory structure."""
    
def test_language_detection_accuracy():
    """Validate >99% accuracy in language identification."""
    
def test_performance_benchmarks():
    """Ensure 100,000+ files processed in <10 seconds."""
    
def test_memory_efficiency():
    """Confirm <100MB memory usage for large file sets."""
    
def test_error_handling_graceful_degradation():
    """Verify graceful handling of filesystem errors."""
```

---

**Task Owner**: TBD  
**Start Date**: TBD  
**Due Date**: TBD  
**Status**: ⚪ Not Started  
**Priority**: Critical
