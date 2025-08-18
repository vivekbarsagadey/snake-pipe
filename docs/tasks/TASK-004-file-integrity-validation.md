# Task 004: File Integrity Validation Implementation

## Research Summary

**Key Findings**: 
- AST JSON files can be corrupted during generation, transfer, or storage leading to downstream processing failures
- Different AST generators produce varying JSON structures requiring flexible validation approaches
- File integrity includes JSON syntax validation, schema compliance, and content consistency checks
- Early validation prevents corrupt data from entering the ETL pipeline and corrupting analysis results
- Validation performance is critical as it's performed on every file during discovery and processing

**Technical Analysis**: 
- JSON syntax validation using fast parsers (orjson, ujson) provides 10x performance improvement over standard json
- Schema validation using jsonschema with compiled schemas reduces validation time by 50%
- Content integrity checks include required field validation, value range validation, and cross-reference consistency
- Checksum validation ensures file integrity during transfer and storage operations
- Parallel validation across multiple files enables high-throughput validation without bottlenecks

**Architecture Impact**: 
- Validation serves as quality gate preventing corrupt data from entering transform phase
- Early failure detection reduces downstream processing costs and improves error reporting
- Plugin-compatible design enables language-specific validation rules and custom checks
- Clean separation between syntax, schema, and content validation enables selective validation modes

**Risk Assessment**: 
- **Performance Risk**: Validation overhead may impact processing throughput - mitigated by fast parsers and caching
- **False Positive Risk**: Overly strict validation may reject valid AST files - handled with configurable validation levels
- **Schema Evolution Risk**: AST format changes may break validation - addressed with versioned schema support

## Business Context

**User Problem**: Development teams need reliable validation of AST JSON files to ensure data quality and prevent processing failures due to corrupted or malformed files.

**Business Value**: 
- **Data Quality**: Achieve 99.9% data quality through early detection of corrupted or malformed AST files
- **Processing Reliability**: Reduce downstream processing failures by 95% through pre-validation
- **Error Diagnosis**: Provide detailed validation error reports for faster issue resolution
- **Cost Optimization**: Prevent waste of processing resources on invalid data

**User Persona**: Data Engineers (60%) who need reliable data validation, Software Architects (25%) requiring data quality assurance, Development Teams (15%) needing error diagnostics for AST generation issues.

**Success Metric**: 
- Validation accuracy: >99.9% detection of corrupted or invalid AST files
- Validation performance: <10ms per file for JSON syntax validation, <50ms for schema validation
- Error reporting: Detailed error messages with suggested fixes for 90% of validation failures
- Processing reliability: <0.1% downstream failures due to undetected validation issues

## User Story

As a **data engineer**, I want **comprehensive AST file validation** so that **I can ensure data quality and prevent processing failures due to corrupted or malformed files**.

## Technical Overview

**Task Type**: Story Task (Quality Assurance Component)
**Pipeline Stage**: Extract (Validation Phase)
**Complexity**: Medium
**Dependencies**: File discovery (TASK-001), JSON schema definitions
**Performance Impact**: Validation overhead must not reduce overall processing throughput below target rates

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/extract/file_validator.py` (main validation engine implementation)
- `snake_pipe/extract/validation_config.py` (configuration management for validation parameters)
- `snake_pipe/extract/validation_models.py` (data models for validation results and errors)
- `snake_pipe/extract/schema_manager.py` (schema loading and management for different AST formats)
- `snake_pipe/extract/validation_rules.py` (custom validation rules and content checks)
- `snake_pipe/utils/json_utils.py` (optimized JSON parsing and validation utilities)
- `snake_pipe/utils/checksum_utils.py` (file integrity and checksum validation utilities)
- `schemas/ast/` (directory for JSON schema definitions for different languages)
- `tests/unit/extract/test_file_validator.py` (comprehensive unit tests with mock data)
- `tests/integration/extract/test_validation_integration.py` (integration tests with real AST files)
- `tests/tasks/test_task004_verification.py` (task-specific verification and acceptance tests)
- `tests/tasks/test_task004_integration.py` (end-to-end validation integration tests)

### Key Functions to Implement

```python
async def validate_json_syntax(
    file_path: Path, 
    use_fast_parser: bool = True
) -> SyntaxValidationResult:
    """
    Purpose: Validate JSON syntax of AST file using optimized parsers
    Input: File path and parser selection option
    Output: Syntax validation result with error details
    Performance: <10ms per file, support for 1000+ files per second
    """

async def validate_ast_schema(
    file_path: Path, 
    language: str, 
    schema_version: Optional[str] = None
) -> SchemaValidationResult:
    """
    Purpose: Validate AST file against language-specific JSON schema
    Input: File path, language identifier, and optional schema version
    Output: Schema validation result with detailed error reporting
    Performance: <50ms per file with compiled schema caching
    """

async def validate_content_integrity(
    ast_data: Dict[str, Any], 
    validation_rules: List[ValidationRule]
) -> ContentValidationResult:
    """
    Purpose: Perform content-level validation checks on AST data
    Input: Parsed AST data and list of validation rules
    Output: Content validation result with rule-specific errors
    Performance: <20ms per file for standard validation rule sets
    """

async def calculate_file_checksum(
    file_path: Path, 
    algorithm: str = "sha256"
) -> ChecksumResult:
    """
    Purpose: Calculate file checksum for integrity verification
    Input: File path and checksum algorithm
    Output: Checksum result with hash value and metadata
    Performance: <5ms per file for standard AST file sizes
    """

class FileValidationEngine:
    """
    Purpose: Comprehensive validation engine for AST files with multi-level validation
    Features: Syntax validation, schema validation, content validation, integrity checks
    Performance: Support for high-throughput validation without processing bottlenecks
    """
    
    async def validate_file(
        self, 
        file_path: Path, 
        config: ValidationConfig
    ) -> FileValidationResult:
        """Complete file validation with configurable validation levels"""
    
    async def validate_batch(
        self, 
        file_paths: List[Path], 
        config: ValidationConfig
    ) -> BatchValidationResult:
        """Parallel validation of multiple files with aggregated results"""
    
    async def get_validation_statistics(self) -> ValidationStatistics:
        """Retrieve validation performance metrics and error statistics"""
```

### Technical Requirements

1. **Performance**: 
   - JSON syntax validation: <10ms per file
   - Schema validation: <50ms per file with schema caching
   - Content validation: <20ms per file for standard rule sets
   - Parallel validation: Support 1000+ files per second validation rate

2. **Error Handling**: 
   - Detailed error reporting with line numbers and specific failure reasons
   - Suggested fixes for common validation failures
   - Graceful handling of corrupted files that cannot be parsed
   - Error categorization for filtering and reporting

3. **Scalability**: 
   - Parallel validation across multiple files and CPU cores
   - Schema caching to avoid repeated schema loading
   - Memory-efficient validation for large AST files
   - Streaming validation for extremely large files

4. **Integration**: 
   - Clean interfaces for file discovery and batch processing integration
   - Plugin architecture for custom validation rules and language-specific checks
   - Configuration-driven validation levels (strict, standard, lenient)
   - Comprehensive logging and monitoring integration

5. **Data Quality**: 
   - 99.9% accuracy in detecting corrupted or invalid files
   - Minimal false positives (< 0.1%) for valid AST files
   - Comprehensive coverage of JSON syntax, schema, and content validation
   - Support for multiple AST format versions and schema evolution

6. **Reliability**: 
   - Consistent validation results across different environments
   - Graceful degradation when validation resources are constrained
   - Error recovery for validation service failures
   - Health checks and validation service monitoring

### Implementation Steps

1. **Core Models**: Define data models for validation results, errors, and configuration following domain-driven design
2. **JSON Validation**: Implement high-performance JSON syntax validation with multiple parser options
3. **Schema Management**: Create schema loading and management system with caching and versioning
4. **Content Rules**: Develop flexible content validation rule system with custom rule support
5. **Parallel Processing**: Add parallel validation capabilities for high-throughput processing
6. **Error Reporting**: Build comprehensive error reporting with detailed diagnostics and suggestions
7. **Integration**: Integrate with file discovery and batch processing systems
8. **Performance Optimization**: Optimize validation performance through caching and efficient algorithms
9. **Testing**: Create comprehensive unit and integration tests with various file corruption scenarios
10. **Monitoring**: Add validation metrics, error tracking, and performance monitoring

### Code Patterns

```python
# Validation Engine Pattern (following project conventions)
@dataclass
class ValidationConfig:
    """Configuration for file validation operations"""
    syntax_validation: bool = True
    schema_validation: bool = True
    content_validation: bool = True
    checksum_validation: bool = False
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    max_file_size_mb: int = 100
    timeout_seconds: int = 30
    parallel_validation: bool = True

@dataclass
class FileValidationResult:
    """Comprehensive validation result for a single file"""
    file_path: Path
    is_valid: bool
    syntax_result: SyntaxValidationResult
    schema_result: Optional[SchemaValidationResult]
    content_result: Optional[ContentValidationResult]
    checksum_result: Optional[ChecksumResult]
    validation_time: float
    errors: List[ValidationError]

class FileValidationEngine:
    """High-performance validation engine with multi-level validation"""
    
    def __init__(self, config: ValidationConfig, schema_manager: SchemaManager):
        self.config = config
        self.schema_manager = schema_manager
        self.stats = ValidationStatistics()
        self.error_categorizer = ErrorCategorizer()
    
    async def validate_file(self, file_path: Path) -> FileValidationResult:
        """Comprehensive file validation with performance optimization"""
        result = FileValidationResult(file_path=file_path)
        
        # Syntax validation (always performed)
        result.syntax_result = await self._validate_syntax(file_path)
        if not result.syntax_result.is_valid:
            result.is_valid = False
            return result
        
        # Schema validation (if enabled and syntax is valid)
        if self.config.schema_validation:
            result.schema_result = await self._validate_schema(file_path)
            if not result.schema_result.is_valid:
                result.is_valid = False
        
        # Content validation (if enabled and previous validations passed)
        if self.config.content_validation and result.is_valid:
            result.content_result = await self._validate_content(file_path)
            result.is_valid = result.content_result.is_valid
        
        return result
    
    async def validate_batch(self, file_paths: List[Path]) -> BatchValidationResult:
        """Parallel validation of multiple files with aggregated results"""
        # Implementation with parallel processing and result aggregation

# Schema Management Pattern
class SchemaManager:
    """Manages JSON schemas for different AST formats and versions"""
    
    def __init__(self, schema_directory: Path):
        self.schema_directory = schema_directory
        self.schema_cache: Dict[str, JsonSchema] = {}
    
    async def get_schema(self, language: str, version: Optional[str] = None) -> JsonSchema:
        """Get compiled schema for language and version with caching"""
        cache_key = f"{language}:{version or 'latest'}"
        if cache_key not in self.schema_cache:
            schema_data = await self._load_schema(language, version)
            self.schema_cache[cache_key] = jsonschema.compile(schema_data)
        return self.schema_cache[cache_key]

# Validation Rule Pattern for Content Validation
from abc import ABC, abstractmethod

class ValidationRule(ABC):
    """Abstract base class for content validation rules"""
    
    @abstractmethod
    def validate(self, ast_data: Dict[str, Any]) -> RuleValidationResult:
        """Validate AST data against specific rule"""

class RequiredFieldsRule(ValidationRule):
    """Validation rule for required field presence"""
    
    def __init__(self, required_fields: List[str]):
        self.required_fields = required_fields
    
    def validate(self, ast_data: Dict[str, Any]) -> RuleValidationResult:
        # Implementation for required fields validation

class ValueRangeRule(ValidationRule):
    """Validation rule for numeric value ranges"""
    
    def __init__(self, field_path: str, min_value: float, max_value: float):
        self.field_path = field_path
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, ast_data: Dict[str, Any]) -> RuleValidationResult:
        # Implementation for value range validation
```

## Acceptance Criteria

### Functional Requirements
- [ ] **JSON Syntax Validation**: Detect and report JSON syntax errors with line numbers and error details
- [ ] **Schema Validation**: Validate AST files against language-specific JSON schemas
- [ ] **Content Validation**: Perform content-level validation using configurable rules
- [ ] **Checksum Validation**: Calculate and verify file checksums for integrity assurance
- [ ] **Error Reporting**: Provide detailed error messages with suggested fixes
- [ ] **Parallel Processing**: Support parallel validation of multiple files
- [ ] **Configuration**: Support configurable validation levels and rule sets
- [ ] **Integration**: Seamless integration with file discovery and batch processing

### Performance Requirements
- [ ] **Syntax Validation Speed**: <10ms per file for JSON syntax validation
- [ ] **Schema Validation Speed**: <50ms per file with compiled schema caching
- [ ] **Content Validation Speed**: <20ms per file for standard validation rules
- [ ] **Throughput**: Support validation of 1000+ files per second
- [ ] **Memory Efficiency**: <100MB RAM for validating batches of 10,000+ files

### Quality Requirements
- [ ] **Test Coverage**: Achieve 90%+ test coverage with unit and integration tests
- [ ] **Code Quality**: Pass all linting and type checking requirements
- [ ] **Documentation**: Complete API documentation with validation rule examples
- [ ] **Logging**: Comprehensive logging with validation metrics and error details
- [ ] **Monitoring**: Validation performance metrics and error rate monitoring

### Integration Requirements
- [ ] **File Discovery Integration**: Validate files discovered through TASK-001 file discovery
- [ ] **Batch Processing Integration**: Support batch validation for TASK-003 batch processing
- [ ] **Configuration System**: Integration with project-wide configuration management
- [ ] **Plugin Architecture**: Extensible design for custom validation rules and language-specific checks
- [ ] **Pipeline Integration**: Ready for integration with transform phase validation requirements

## Priority Guidelines

**Critical**: JSON syntax validation, schema validation, error reporting, performance optimization
**High**: Content validation rules, parallel processing, configuration flexibility, integration interfaces
**Medium**: Advanced validation rules, checksum validation, custom rule development, monitoring
**Low**: Validation analytics, advanced error categorization, developer tooling, validation UI components

**Focus**: Create a reliable, high-performance validation system that ensures data quality while maintaining processing throughput and providing detailed error diagnostics for operational excellence.
