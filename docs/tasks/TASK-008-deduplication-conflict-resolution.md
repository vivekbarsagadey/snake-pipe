# Task 008: Deduplication and Conflict Resolution Implementation

## Research Summary

**Key Findings**: 
- AST files can have duplicates due to multiple parser runs, CI/CD rebuilds, and incremental processing workflows
- Content-based deduplication using hash comparison is more reliable than filename-based deduplication
- Conflict resolution requires timestamp-based prioritization and metadata comparison for determining canonical versions
- Large codebases may contain 10-30% duplicate AST files, significantly impacting storage and processing efficiency
- Smart deduplication can reduce downstream processing load and improve data quality metrics

**Technical Analysis**: 
- Fast content hashing using xxhash or blake3 provides 10x performance improvement over SHA-256
- Two-phase deduplication: fast size/timestamp filtering followed by content hash comparison
- Conflict resolution strategies: newest-first, largest-first, parser-version-priority, manual resolution
- Incremental deduplication tracking prevents re-processing of previously deduplicated files
- Memory-efficient deduplication using bloom filters for large-scale duplicate detection

**Architecture Impact**: 
- Deduplication improves overall pipeline efficiency by reducing redundant processing
- Clean separation between duplicate detection and conflict resolution enables flexible strategies
- Integration with file discovery and validation ensures deduplicated files maintain quality standards
- Plugin-compatible design enables custom deduplication strategies for specific AST generators

**Risk Assessment**: 
- **False Positive Risk**: Overly aggressive deduplication may remove legitimate file variations - mitigated by configurable similarity thresholds
- **Performance Risk**: Hash calculation overhead may impact processing speed - addressed with fast hashing algorithms and caching
- **Data Loss Risk**: Incorrect conflict resolution may discard important file versions - handled with backup and audit logging

## Business Context

**User Problem**: Development teams need efficient deduplication and conflict resolution for AST files to eliminate redundant processing, reduce storage costs, and ensure data consistency across the pipeline.

**Business Value**: 
- **Storage Efficiency**: Reduce storage requirements by 20-30% through intelligent deduplication
- **Processing Optimization**: Eliminate redundant processing of duplicate files, improving throughput by 15-25%
- **Data Quality**: Ensure consistent data by resolving conflicts and maintaining canonical file versions
- **Cost Reduction**: Lower infrastructure costs through reduced storage and compute requirements

**User Persona**: Data Engineers (50%) who need efficient data processing, Software Architects (30%) requiring data consistency, DevOps Engineers (20%) managing storage and infrastructure costs.

**Success Metric**: 
- Deduplication accuracy: >99% detection of true duplicates with <0.1% false positives
- Processing efficiency: 20-30% reduction in processing time for repositories with duplicates
- Storage optimization: 25% average reduction in storage requirements
- Conflict resolution: 95% automated resolution of file conflicts with audit logging

## User Story

As a **data engineer**, I want **intelligent deduplication and conflict resolution for AST files** so that **I can eliminate redundant processing, reduce storage costs, and ensure data consistency across the ETL pipeline**.

## Technical Overview

**Task Type**: Story Task (Data Quality Component)
**Pipeline Stage**: Transform (Deduplication Phase)
**Complexity**: High
**Dependencies**: File validation (TASK-004), metadata extraction
**Performance Impact**: Deduplication must not introduce significant overhead while providing substantial processing savings

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/transform/deduplicator.py` (main deduplication engine implementation)
- `snake_pipe/transform/conflict_resolver.py` (conflict resolution strategies and logic)
- `snake_pipe/transform/dedup_config.py` (configuration management for deduplication parameters)
- `snake_pipe/transform/dedup_models.py` (data models for duplicates, conflicts, and resolution results)
- `snake_pipe/transform/hash_manager.py` (content hashing and similarity detection)
- `snake_pipe/utils/bloom_filter.py` (memory-efficient duplicate detection for large datasets)
- `snake_pipe/utils/similarity_utils.py` (content similarity and comparison utilities)
- `tests/unit/transform/test_deduplicator.py` (comprehensive unit tests with synthetic duplicates)
- `tests/integration/transform/test_dedup_integration.py` (integration tests with real duplicate scenarios)
- `tests/performance/test_dedup_performance.py` (performance testing with large-scale duplicate datasets)
- `tests/tasks/test_task008_verification.py` (task-specific verification and acceptance tests)
- `tests/tasks/test_task008_integration.py` (end-to-end deduplication integration tests)

### Key Functions to Implement

```python
async def detect_duplicates(
    file_list: List[ValidatedFile], 
    config: DeduplicationConfig
) -> DuplicateDetectionResult:
    """
    Purpose: Detect duplicate AST files using content hashing and similarity analysis
    Input: List of validated files and deduplication configuration
    Output: Detection result with duplicate groups and similarity scores
    Performance: Process 10,000+ files in <60 seconds with <1GB memory usage
    """

async def calculate_content_hash(
    file_path: Path, 
    algorithm: HashAlgorithm = HashAlgorithm.XXHASH
) -> ContentHash:
    """
    Purpose: Calculate fast content hash for duplicate detection
    Input: File path and hashing algorithm selection
    Output: Content hash with metadata and calculation time
    Performance: <5ms per file for standard AST file sizes
    """

async def resolve_conflicts(
    duplicate_group: DuplicateGroup, 
    resolution_strategy: ResolutionStrategy
) -> ConflictResolutionResult:
    """
    Purpose: Resolve conflicts between duplicate files using specified strategy
    Input: Group of duplicate files and resolution strategy
    Output: Resolution result with canonical file selection and rationale
    Performance: <10ms per conflict group with comprehensive audit logging
    """

async def create_deduplication_report(
    dedup_result: DeduplicationResult, 
    include_details: bool = True
) -> DeduplicationReport:
    """
    Purpose: Generate comprehensive deduplication report with statistics and recommendations
    Input: Deduplication result and detail level configuration
    Output: Detailed report with savings analysis and quality metrics
    Performance: <1 second for reports covering 100,000+ files
    """

class FileDeduplicationEngine:
    """
    Purpose: Comprehensive deduplication engine with multiple detection strategies
    Features: Content hashing, similarity detection, conflict resolution, incremental processing
    Performance: Support for large-scale deduplication without memory exhaustion
    """
    
    async def deduplicate_repository(
        self, 
        files: List[ValidatedFile], 
        config: DeduplicationConfig
    ) -> DeduplicationResult:
        """Complete repository deduplication with conflict resolution and reporting"""
    
    async def incremental_deduplication(
        self, 
        new_files: List[ValidatedFile], 
        existing_hashes: HashDatabase
    ) -> IncrementalDeduplicationResult:
        """Incremental deduplication for new files against existing dataset"""
    
    async def get_deduplication_statistics(self) -> DeduplicationStatistics:
        """Retrieve deduplication performance metrics and efficiency statistics"""
```

### Technical Requirements

1. **Performance**: 
   - Duplicate detection: Process 10,000+ files in <60 seconds
   - Content hashing: <5ms per file using fast hashing algorithms
   - Memory efficiency: <1GB RAM for deduplicating 100,000+ files
   - Conflict resolution: <10ms per duplicate group

2. **Error Handling**: 
   - Graceful handling of files that cannot be hashed or compared
   - Comprehensive error reporting with file-level error details
   - Fallback strategies when primary deduplication methods fail
   - Audit logging for all deduplication and conflict resolution decisions

3. **Scalability**: 
   - Memory-efficient processing using streaming and bloom filters
   - Incremental deduplication for continuous processing workflows
   - Horizontal scaling support for distributed deduplication
   - Configurable similarity thresholds and detection strategies

4. **Integration**: 
   - Integration with file validation to ensure deduplicated files maintain quality
   - Plugin interface for custom deduplication and conflict resolution strategies
   - Configuration-driven behavior with multiple resolution strategies
   - Comprehensive logging and monitoring for operational tracking

5. **Data Quality**: 
   - >99% accuracy in duplicate detection with <0.1% false positives
   - Consistent conflict resolution with audit trails and justification
   - Preservation of important file metadata during deduplication
   - Quality metrics tracking for continuous improvement

6. **Reliability**: 
   - Consistent deduplication results across different environments
   - Backup and recovery capabilities for deduplication metadata
   - Health checks and deduplication service monitoring
   - Graceful degradation when deduplication resources are limited

### Implementation Steps

1. **Core Models**: Define data models for duplicates, conflicts, resolutions, and statistics following domain-driven design
2. **Hash Engine**: Implement high-performance content hashing with multiple algorithm support
3. **Duplicate Detection**: Create sophisticated duplicate detection using hashing and similarity analysis
4. **Conflict Resolution**: Develop flexible conflict resolution strategies with configurable prioritization
5. **Similarity Analysis**: Build content similarity detection for near-duplicate identification
6. **Incremental Processing**: Add support for incremental deduplication in continuous workflows
7. **Reporting**: Create comprehensive reporting with statistics and savings analysis
8. **Performance Optimization**: Optimize memory usage and processing speed for large-scale deduplication
9. **Testing**: Create comprehensive unit and integration tests with various duplicate scenarios
10. **Monitoring**: Add deduplication metrics, efficiency tracking, and operational monitoring

### Code Patterns

```python
# Deduplication Engine Pattern (following project conventions)
@dataclass
class DeduplicationConfig:
    """Configuration for deduplication operations"""
    hash_algorithm: HashAlgorithm = HashAlgorithm.XXHASH
    similarity_threshold: float = 0.95
    resolution_strategy: ResolutionStrategy = ResolutionStrategy.NEWEST_FIRST
    enable_incremental: bool = True
    max_memory_mb: int = 1000
    enable_bloom_filter: bool = True
    audit_logging: bool = True

@dataclass
class DuplicateGroup:
    """Group of duplicate files with similarity information"""
    group_id: str
    files: List[ValidatedFile]
    similarity_scores: Dict[str, float]
    detection_method: DetectionMethod
    canonical_file: Optional[ValidatedFile]
    resolution_rationale: Optional[str]

class FileDeduplicationEngine:
    """High-performance deduplication with multiple strategies"""
    
    def __init__(self, config: DeduplicationConfig, hash_manager: HashManager):
        self.config = config
        self.hash_manager = hash_manager
        self.conflict_resolver = ConflictResolver(config)
        self.bloom_filter = BloomFilter() if config.enable_bloom_filter else None
        self.stats = DeduplicationStatistics()
    
    async def deduplicate_repository(self, files: List[ValidatedFile]) -> DeduplicationResult:
        """Complete repository deduplication with multi-phase processing"""
        # Phase 1: Fast filtering by size and basic metadata
        size_groups = await self._group_by_size(files)
        
        # Phase 2: Content hash calculation and comparison
        duplicate_candidates = await self._find_hash_duplicates(size_groups)
        
        # Phase 3: Similarity analysis for near-duplicates
        if self.config.similarity_threshold < 1.0:
            similarity_duplicates = await self._find_similarity_duplicates(files)
            duplicate_candidates.extend(similarity_duplicates)
        
        # Phase 4: Conflict resolution
        resolved_groups = await self._resolve_all_conflicts(duplicate_candidates)
        
        return DeduplicationResult(
            original_count=len(files),
            duplicate_groups=resolved_groups,
            canonical_files=self._extract_canonical_files(resolved_groups),
            savings_analysis=self._calculate_savings(files, resolved_groups)
        )
    
    async def _find_hash_duplicates(self, size_groups: Dict[int, List[ValidatedFile]]) -> List[DuplicateGroup]:
        """Find exact duplicates using content hashing"""
        # Implementation with parallel hashing and efficient grouping

# Conflict Resolution Strategy Pattern
from abc import ABC, abstractmethod

class ResolutionStrategy(ABC):
    """Abstract base class for conflict resolution strategies"""
    
    @abstractmethod
    async def resolve_conflict(self, duplicate_group: DuplicateGroup) -> ConflictResolutionResult:
        """Resolve conflict and select canonical file"""

class NewestFirstStrategy(ResolutionStrategy):
    """Resolution strategy prioritizing newest files"""
    
    async def resolve_conflict(self, duplicate_group: DuplicateGroup) -> ConflictResolutionResult:
        newest_file = max(duplicate_group.files, key=lambda f: f.metadata.modified_time)
        return ConflictResolutionResult(
            canonical_file=newest_file,
            rationale=f"Selected newest file from {duplicate_group.files[0].metadata.modified_time}",
            confidence=0.9
        )

class LargestFirstStrategy(ResolutionStrategy):
    """Resolution strategy prioritizing largest files"""
    
    async def resolve_conflict(self, duplicate_group: DuplicateGroup) -> ConflictResolutionResult:
        largest_file = max(duplicate_group.files, key=lambda f: f.metadata.size)
        return ConflictResolutionResult(
            canonical_file=largest_file,
            rationale=f"Selected largest file ({largest_file.metadata.size} bytes)",
            confidence=0.8
        )

# Hash Management Pattern
class HashManager:
    """Manages content hashing with multiple algorithms and caching"""
    
    def __init__(self, cache_size: int = 10000):
        self.hash_cache: Dict[str, ContentHash] = {}
        self.cache_size = cache_size
    
    async def calculate_hash(self, file_path: Path, algorithm: HashAlgorithm) -> ContentHash:
        """Calculate file hash with caching and performance optimization"""
        cache_key = f"{file_path}:{algorithm.value}:{file_path.stat().st_mtime}"
        
        if cache_key in self.hash_cache:
            return self.hash_cache[cache_key]
        
        # Calculate hash using optimized algorithms
        if algorithm == HashAlgorithm.XXHASH:
            hash_value = await self._calculate_xxhash(file_path)
        elif algorithm == HashAlgorithm.BLAKE3:
            hash_value = await self._calculate_blake3(file_path)
        else:
            hash_value = await self._calculate_sha256(file_path)
        
        content_hash = ContentHash(
            algorithm=algorithm,
            value=hash_value,
            file_path=file_path,
            calculation_time=time.time()
        )
        
        # Cache management with LRU eviction
        if len(self.hash_cache) >= self.cache_size:
            oldest_key = min(self.hash_cache.keys(), key=lambda k: self.hash_cache[k].calculation_time)
            del self.hash_cache[oldest_key]
        
        self.hash_cache[cache_key] = content_hash
        return content_hash
```

## Acceptance Criteria

### Functional Requirements
- [ ] **Duplicate Detection**: Accurately detect exact duplicates using content hashing
- [ ] **Similarity Detection**: Identify near-duplicates using configurable similarity thresholds
- [ ] **Conflict Resolution**: Automatically resolve conflicts using multiple resolution strategies
- [ ] **Incremental Processing**: Support incremental deduplication for continuous workflows
- [ ] **Audit Logging**: Maintain comprehensive audit trails for all deduplication decisions
- [ ] **Reporting**: Generate detailed reports with statistics and savings analysis
- [ ] **Configuration**: Support configurable strategies, thresholds, and processing options
- [ ] **Integration**: Seamless integration with validation and batch processing systems

### Performance Requirements
- [ ] **Detection Speed**: Process 10,000+ files for duplicate detection in <60 seconds
- [ ] **Hashing Performance**: Calculate content hashes in <5ms per file
- [ ] **Memory Efficiency**: Use <1GB RAM for deduplicating 100,000+ files
- [ ] **Conflict Resolution Speed**: Resolve conflicts in <10ms per duplicate group
- [ ] **Scalability**: Support horizontal scaling for distributed deduplication

### Quality Requirements
- [ ] **Test Coverage**: Achieve 90%+ test coverage with unit, integration, and performance tests
- [ ] **Code Quality**: Pass all linting and type checking requirements
- [ ] **Documentation**: Complete API documentation with strategy guides and examples
- [ ] **Logging**: Comprehensive logging with deduplication metrics and decision rationale
- [ ] **Monitoring**: Deduplication efficiency metrics and performance monitoring

### Integration Requirements
- [ ] **Validation Integration**: Ensure deduplicated files maintain validation status
- [ ] **Batch Processing Integration**: Support batch deduplication for TASK-003 batch processing
- [ ] **Configuration System**: Integration with project-wide configuration management
- [ ] **Plugin Architecture**: Extensible design for custom deduplication strategies
- [ ] **Pipeline Integration**: Ready for integration with load phase and downstream processing

## Priority Guidelines

**Critical**: Exact duplicate detection, conflict resolution, performance optimization, audit logging
**High**: Similarity detection, incremental processing, reporting capabilities, configuration flexibility
**Medium**: Advanced resolution strategies, custom similarity algorithms, distributed processing, analytics
**Low**: Advanced reporting features, deduplication UI, developer tooling, optimization edge cases

**Focus**: Create a reliable, high-performance deduplication system that significantly improves pipeline efficiency while maintaining data quality and providing comprehensive audit capabilities for operational excellence.
