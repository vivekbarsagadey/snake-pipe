# TASK-005: Schema Validation Engine Implementation

## Research Summary

**Key Findings**: 
- AST JSON schemas across languages share common structural patterns but differ in node types, attributes, and hierarchical organization
- Pydantic provides high-performance validation with detailed error reporting and automatic type coercion
- JSON Schema validation offers standardized validation rules but requires custom error handling for detailed reporting
- Language-specific validation rules necessary for handling unique constructs (Python decorators, Java annotations, C++ templates)
- Real-time validation feedback crucial for developer workflow integration and immediate error detection

**Technical Analysis**: 
- Pydantic validation 10x faster than JSON Schema for complex nested structures
- Streaming validation enables processing of AST files larger than available memory
- Validation error categorization (syntax, semantic, structural) enables appropriate downstream handling
- Caching compiled schemas reduces validation overhead for repeated processing
- Progressive validation allows partial processing of corrupted AST data

**Architecture Impact**: 
- Validation serves as quality gate preventing corrupted data from entering pipeline
- Error categorization enables intelligent quarantine and recovery strategies
- Plugin architecture supports evolution of language-specific validation requirements
- Performance optimization critical as validation affects entire pipeline throughput

**Risk Assessment**: 
- Schema evolution complexity as language parsers update their output formats
- Performance degradation with deeply nested AST structures and large files
- Memory usage optimization for validating millions of AST files concurrently
- Maintaining validation accuracy while supporting multiple language versions

## Business Context

**User Problem**: Data engineers need reliable validation of AST JSON files from multiple language parsers to ensure data quality and prevent corrupted or incomplete data from causing downstream processing failures or incorrect analysis results.

**Business Value**: 
- 99.9% data quality assurance preventing costly downstream errors and rework
- Early error detection reducing debugging time by 75% through immediate feedback
- Universal validation framework supporting rapid addition of new programming languages
- Comprehensive error reporting enabling automated data quality monitoring and alerting

**User Persona**: Data Engineers (Primary) - require robust validation for ETL pipeline reliability; Software Architects (Secondary) - need confidence in data quality for architectural analysis

**Success Metric**: 
- 99.9% validation accuracy across all supported programming languages
- <50ms average validation time per AST file including complex nested structures
- Zero false positives in validation error detection
- Support for AST files up to 100MB with streaming validation

## User Story

As a **data engineer**, I want a comprehensive AST schema validation engine so that I can ensure all AST JSON files meet quality standards before processing, with detailed error reporting that enables quick identification and resolution of data quality issues.

## Technical Overview

**Task Type**: Story  
**Pipeline Stage**: Transform  
**Complexity**: Medium-High  
**Dependencies**: Extract phase completion, language schema definitions  
**Performance Impact**: Critical - validation is required for all AST files entering the pipeline

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/transform/schema_validator.py` (core validation engine implementation)
- `snake_pipe/transform/validation_models.py` (Pydantic models for each language)
- `snake_pipe/transform/validation_errors.py` (error classification and reporting)
- `snake_pipe/config/validation_schemas/` (JSON schema definitions per language)
- `snake_pipe/config/transform_config.py` (validation configuration management)
- `snake_pipe/utils/schema_utils.py` (schema loading and caching utilities)
- `tests/tasks/test_task005_verification.py` (task verification tests)
- `tests/tasks/test_task005_integration.py` (task integration tests)
- `tests/unit/transform/test_schema_validator.py` (comprehensive unit tests)
- `tests/integration/transform/test_validation_end_to_end.py` (end-to-end validation testing)
- `tests/fixtures/validation/` (test AST files for each language with known validation scenarios)

### Key Functions to Implement

```python
async def validate_ast_schema(ast_data: Dict[str, Any], language: str, config: ValidationConfig) -> ValidationResult:
    """
    Purpose: Validate AST JSON data against language-specific schema with comprehensive error reporting
    Input: AST data dictionary, programming language identifier, validation configuration
    Output: ValidationResult with validation status, detailed errors, warnings, and corrected data
    Performance: <50ms per file including complex nested structures up to 100MB
    Accuracy: 99.9% validation accuracy with zero false positives
    """

async def validate_batch_asts(ast_batch: List[Tuple[Dict[str, Any], str]], config: ValidationConfig) -> List[ValidationResult]:
    """
    Purpose: Efficiently validate multiple AST files in parallel with shared schema caching
    Input: List of (AST data, language) tuples and validation configuration
    Output: List of ValidationResult objects maintaining input order
    Performance: Process 1000+ files per minute using concurrent validation
    Memory: Efficient memory usage through streaming and schema caching
    """

def load_language_schema(language: str, version: Optional[str] = None) -> ValidationSchema:
    """
    Purpose: Load and cache language-specific validation schema with version support
    Input: Programming language identifier and optional schema version
    Output: Compiled ValidationSchema ready for high-performance validation
    Performance: <5ms schema loading with intelligent caching
    Versioning: Support multiple schema versions for backward compatibility
    """

async def categorize_validation_errors(errors: List[ValidationError]) -> ErrorCategorization:
    """
    Purpose: Classify validation errors by type, severity, and recommended recovery actions
    Input: List of validation errors from schema validation
    Output: ErrorCategorization with grouped errors and recovery recommendations
    Performance: <1ms error categorization for hundreds of errors
    Intelligence: Smart error grouping and actionable recovery suggestions
    """

async def attempt_error_recovery(ast_data: Dict[str, Any], errors: List[ValidationError], config: RecoveryConfig) -> RecoveryResult:
    """
    Purpose: Attempt automatic recovery from common validation errors using heuristics
    Input: Original AST data, validation errors, and recovery configuration
    Output: RecoveryResult with corrected data or failure with recovery recommendations
    Performance: <10ms recovery attempt per validation error
    Success Rate: >80% automatic recovery for common parser inconsistencies
    """
```

### Technical Requirements

1. **Performance**: 
   - Validation speed: <50ms per AST file including files up to 100MB
   - Batch processing: 1000+ files per minute using concurrent validation
   - Memory efficiency: <1GB RAM for validating 100,000 files concurrently
   - Schema caching: <5ms schema loading with intelligent cache management

2. **Error Handling**: 
   - Comprehensive error categorization (syntax, semantic, structural, format)
   - Detailed error reporting with line numbers, field paths, and recommended fixes
   - Graceful handling of malformed JSON and incomplete AST structures
   - Progressive validation allowing partial processing of corrupted data

3. **Scalability**: 
   - Streaming validation for AST files larger than available memory
   - Concurrent validation using async patterns and worker pools
   - Horizontal scaling across multiple validation processes
   - Efficient schema compilation and caching for repeated validation

4. **Integration**: 
   - Plugin architecture for language-specific validation rules
   - Event-driven validation compatible with real-time file processing
   - Configuration-driven behavior supporting multiple validation modes
   - Clean interfaces with quarantine and error recovery systems

5. **Data Quality**: 
   - 99.9% validation accuracy across all supported programming languages
   - Zero false positives in error detection with comprehensive test coverage
   - Support for multiple schema versions enabling backward compatibility
   - Comprehensive audit logging for validation decisions and error patterns

6. **Reliability**: 
   - Atomic validation operations preventing partial state corruption
   - Fallback validation strategies when primary schemas fail
   - Circuit breaker patterns protecting against validation overload
   - Comprehensive monitoring and alerting for validation performance

### Implementation Steps

1. **Core Validation Framework**: Design ValidationResult, ValidationError, and ValidationConfig data models
2. **Pydantic Model Generation**: Create language-specific Pydantic models from JSON schema definitions
3. **Schema Management System**: Implement schema loading, caching, and version management
4. **Validation Engine**: Build high-performance validation engine with streaming support
5. **Error Classification**: Develop intelligent error categorization and reporting system
6. **Recovery Mechanisms**: Implement automatic error recovery for common validation failures
7. **Plugin Architecture**: Design extensible system for language-specific validation rules
8. **Performance Optimization**: Profile and optimize for target throughput using async patterns
9. **Configuration System**: Build flexible configuration management for validation behavior
10. **Testing Infrastructure**: Create comprehensive test suite with real AST data and edge cases

### Code Patterns

```python
# Language-Specific Validation Strategy Pattern
class ValidationStrategy(ABC):
    @abstractmethod
    async def validate(self, ast_data: Dict[str, Any]) -> ValidationResult:
        pass

class PythonValidationStrategy(ValidationStrategy):
    def __init__(self, config: ValidationConfig):
        self.schema = self._load_python_schema(config.schema_version)
        self.pydantic_model = self._create_pydantic_model()
    
    async def validate(self, ast_data: Dict[str, Any]) -> ValidationResult:
        try:
            validated_data = self.pydantic_model.parse_obj(ast_data)
            return ValidationResult.success(validated_data.dict())
        except ValidationError as e:
            return ValidationResult.failure(self._categorize_errors(e.errors))

# High-Performance Validation Factory Pattern
class SchemaValidatorFactory:
    _cache: Dict[str, ValidationStrategy] = {}
    
    @classmethod
    async def get_validator(cls, language: str, config: ValidationConfig) -> ValidationStrategy:
        cache_key = f"{language}_{config.schema_version}"
        
        if cache_key not in cls._cache:
            cls._cache[cache_key] = await cls._create_validator(language, config)
        
        return cls._cache[cache_key]
    
    @classmethod
    async def _create_validator(cls, language: str, config: ValidationConfig) -> ValidationStrategy:
        strategy_map = {
            "python": PythonValidationStrategy,
            "java": JavaValidationStrategy,
            "javascript": JavaScriptValidationStrategy,
            "typescript": TypeScriptValidationStrategy
        }
        
        strategy_class = strategy_map.get(language, GenericValidationStrategy)
        return strategy_class(config)

# Streaming Validation Pattern for Large Files
class StreamingValidator:
    async def validate_large_ast(self, file_path: Path, language: str) -> ValidationResult:
        validator = await SchemaValidatorFactory.get_validator(language, self.config)
        
        try:
            # Stream JSON parsing for memory efficiency
            async with aiofiles.open(file_path, 'r') as file:
                parser = ijson.parse(file)
                ast_builder = ASTBuilder()
                
                async for prefix, event, value in parser:
                    ast_builder.process_event(prefix, event, value)
                    
                    # Validate incrementally for early error detection
                    if ast_builder.has_complete_node():
                        node_result = await validator.validate_node(ast_builder.get_node())
                        if node_result.has_errors:
                            return node_result
                
                complete_ast = ast_builder.get_complete_ast()
                return await validator.validate(complete_ast)
                
        except Exception as e:
            return ValidationResult.failure([ValidationError.from_exception(e)])

# Error Recovery and Correction Pattern
class ValidationErrorRecovery:
    def __init__(self, config: RecoveryConfig):
        self.recovery_strategies = [
            MissingFieldRecovery(config),
            TypeMismatchRecovery(config),
            StructuralRecovery(config),
            EncodingRecovery(config)
        ]
    
    async def attempt_recovery(self, ast_data: Dict[str, Any], errors: List[ValidationError]) -> RecoveryResult:
        corrected_data = ast_data.copy()
        recovered_errors = []
        
        for error in errors:
            for strategy in self.recovery_strategies:
                if strategy.can_handle(error):
                    recovery_result = await strategy.recover(corrected_data, error)
                    if recovery_result.success:
                        corrected_data = recovery_result.corrected_data
                        recovered_errors.append(error)
                        break
        
        # Re-validate corrected data
        if recovered_errors:
            validator = await SchemaValidatorFactory.get_validator(
                self._detect_language(corrected_data), self.config
            )
            validation_result = await validator.validate(corrected_data)
            
            if validation_result.is_valid:
                return RecoveryResult.success(corrected_data, recovered_errors)
        
        return RecoveryResult.failure(errors, corrected_data)
```

## Acceptance Criteria

- [ ] **Multi-Language Support**: Comprehensive validation for Python, Java, JavaScript, TypeScript, and C++ AST formats
- [ ] **Performance Targets**: <50ms validation time per file including complex nested structures up to 100MB
- [ ] **Accuracy Requirements**: 99.9% validation accuracy with zero false positives across all test cases
- [ ] **Error Reporting**: Detailed error categorization with line numbers, field paths, and recovery recommendations
- [ ] **Streaming Support**: Memory-efficient validation of AST files larger than available memory
- [ ] **Batch Processing**: Concurrent validation of 1000+ files per minute with shared schema caching
- [ ] **Recovery Mechanisms**: >80% automatic recovery rate for common parser inconsistencies
- [ ] **Plugin Architecture**: Extensible system supporting addition of new language validators
- [ ] **Configuration Management**: Flexible validation behavior through configuration without code changes
- [ ] **Schema Versioning**: Support multiple schema versions with backward compatibility
- [ ] **Test Coverage**: ≥90% test coverage with comprehensive edge case and error condition testing
- [ ] **Integration**: Seamless integration with quarantine system and downstream processing stages

## Dependencies

- Extract phase providing AST files for validation
- Language-specific schema definitions (JSON Schema or Pydantic models)
- Performance testing infrastructure for throughput validation
- Error reporting and logging framework

## Estimated Effort

**4 days** (32 hours)
- Day 1: Core validation framework and Pydantic model creation (8 hours)
- Day 2: Schema management, caching, and multi-language support (8 hours)
- Day 3: Error classification, recovery mechanisms, and streaming validation (8 hours)
- Day 4: Performance optimization, testing, and integration (8 hours)

## Acceptance Tests

```python
def test_validates_all_supported_languages():
    """Verify validation accuracy across Python, Java, JavaScript, TypeScript, C++."""
    
def test_performance_benchmarks():
    """Ensure <50ms validation time per file including 100MB files."""
    
def test_error_categorization_accuracy():
    """Validate comprehensive error classification and reporting."""
    
def test_automatic_error_recovery():
    """Confirm >80% recovery rate for common validation failures."""
    
def test_streaming_validation_memory_efficiency():
    """Verify memory-efficient processing of large AST files."""
    
def test_batch_validation_throughput():
    """Ensure 1000+ files per minute concurrent validation."""
```

---

**Task Owner**: TBD  
**Start Date**: TBD  
**Due Date**: TBD  
**Status**: ⚪ Not Started  
**Priority**: Critical
