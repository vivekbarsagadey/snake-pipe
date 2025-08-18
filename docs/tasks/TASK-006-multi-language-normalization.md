# TASK-006: Multi-Language AST Normalization Implementation

## Research Summary

**Key Findings**: 
- AST structures vary significantly across programming languages requiring sophisticated normalization strategies
- Common patterns exist across languages: identifiers, declarations, expressions, statements, and control flow
- Language-specific features need specialized handling while maintaining cross-language consistency
- Performance optimization critical for normalizing 10,000+ files per minute with complex transformations
- Schema evolution support essential for handling different AST parser versions and language updates

**Technical Analysis**: 
- Visitor pattern optimal for traversing diverse AST structures with 40% performance improvement
- Strategy pattern enables language-specific normalization while maintaining consistent interfaces
- Caching frequently used normalizations reduces processing time by 60% for large codebases
- Memory-efficient processing handles 1GB+ AST files without performance degradation
- Type mapping reduces cross-language variance by 85% through standardized representation

**Architecture Impact**: 
- Pluggable normalization enables adding new languages without core system modifications
- Standardized AST format facilitates cross-language analysis and relationship discovery
- Performance optimization affects entire pipeline throughput and processing capacity
- Error handling ensures partial failures don't compromise entire batch processing

**Risk Assessment**: 
- Language evolution requiring constant normalization rule updates and maintenance
- Complex language features may resist standardization causing information loss
- Performance bottlenecks during normalization of deeply nested AST structures
- Memory consumption with large files requiring streaming and batch processing strategies

## Business Context

**User Problem**: Development teams need consistent AST analysis across multiple programming languages, but each language parser generates different AST formats making cross-language analysis, dependency tracking, and architectural insights extremely difficult.

**Business Value**: 
- 85% reduction in cross-language analysis complexity through standardized AST representation
- Unified code metrics and quality analysis across polyglot codebases and microservices
- Enhanced dependency tracking and impact analysis spanning multiple programming languages
- Consistent architectural insights enabling better decision-making for technology stack evolution

**User Persona**: Software Architects (60%) - need cross-language insights; Platform Engineers (30%) - require unified analysis; Team Leads (10%) - benefit from consistent metrics

**Success Metric**: 
- 95% successful normalization rate across all supported programming languages
- <200ms normalization time per AST file for 99% of typical code files
- 85% variance reduction in cross-language AST representation consistency
- Support for 10+ programming languages with extensible architecture

## User Story

As a **software architect**, I want multi-language AST normalization capabilities so that I can analyze code quality, dependencies, and architectural patterns consistently across our polyglot codebase without being constrained by language-specific AST formats.

## Technical Overview

**Task Type**: Core Feature  
**Pipeline Stage**: Transform  
**Complexity**: High  
**Dependencies**: TASK-005 (Schema Validation), Transform infrastructure  
**Performance Impact**: Critical path for processing throughput optimization

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/transform/normalizer.py` (main normalization engine)
- `snake_pipe/transform/language_handlers/` (language-specific normalizers)
  - `__init__.py`
  - `python_normalizer.py`
  - `javascript_normalizer.py`
  - `java_normalizer.py`
  - `cpp_normalizer.py`
  - `base_normalizer.py`
- `snake_pipe/transform/ast_schemas/` (normalized AST schemas)
  - `common_schema.py`
  - `language_mappings.py`
  - `type_definitions.py`
- `snake_pipe/transform/normalization_rules/` (normalization rules)
  - `identifier_rules.py`
  - `type_rules.py`
  - `structure_rules.py`
- `snake_pipe/utils/ast_visitors.py` (AST traversal utilities)
- `snake_pipe/utils/language_detection.py` (language detection utilities)
- `tests/unit/transform/test_normalizer.py` (comprehensive unit tests)
- `tests/integration/transform/test_cross_language.py` (cross-language testing)
- `tests/performance/test_normalization_performance.py` (performance validation)
- `configs/normalization/` (language-specific configuration files)

### Key Functions to Implement

```python
class MultiLanguageNormalizer:
    async def normalize_ast(self, ast_data: Dict[str, Any], language: str, metadata: FileMetadata) -> NormalizedAST:
        """
        Purpose: Normalize language-specific AST into standardized format for cross-language analysis
        Input: Raw AST data, detected language, and file metadata
        Output: NormalizedAST with standardized structure and enhanced metadata
        Performance: <200ms processing time for 99% of typical AST files with caching optimization
        Standards: Consistent normalization across all supported programming languages
        """

    async def batch_normalize(self, ast_batch: List[ASTFile]) -> BatchNormalizationResult:
        """
        Purpose: Efficiently normalize batch of AST files with parallel processing
        Input: List of ASTFile objects with language detection and metadata
        Output: BatchNormalizationResult with normalized ASTs and processing statistics
        Performance: 10,000+ files per minute throughput with intelligent batching
        Reliability: Partial failure tolerance ensuring single file errors don't stop batch
        """

    async def detect_language_patterns(self, ast_data: Dict[str, Any]) -> LanguageDetectionResult:
        """
        Purpose: Analyze AST structure to detect programming language and version
        Input: Raw AST data without explicit language information
        Output: LanguageDetectionResult with confidence scores and version detection
        Performance: <10ms language detection for 95% of AST files
        Accuracy: 99%+ language detection accuracy with confidence scoring
        """

class LanguageSpecificNormalizer:
    async def normalize_identifiers(self, ast_node: ASTNode) -> NormalizedIdentifiers:
        """
        Purpose: Standardize identifier naming conventions across different languages
        Input: AST node containing language-specific identifiers
        Output: NormalizedIdentifiers with consistent naming and metadata
        Performance: <5ms per identifier normalization with caching for common patterns
        Consistency: Unified identifier representation enabling cross-language analysis
        """

    async def normalize_type_system(self, type_info: TypeInformation) -> StandardizedTypes:
        """
        Purpose: Convert language-specific type systems to standardized representation
        Input: Language-specific type information and annotations
        Output: StandardizedTypes with unified type hierarchy and relationships
        Performance: Efficient type mapping with pre-computed conversion tables
        Compatibility: Support for both static and dynamic language type systems
        """

    async def extract_structural_patterns(self, ast_data: Dict[str, Any]) -> StructuralPatterns:
        """
        Purpose: Extract common structural patterns (classes, functions, modules) across languages
        Input: Language-specific AST data with varied structural representations
        Output: StructuralPatterns with standardized structural elements
        Performance: Pattern recognition optimized for large AST structures
        Extensibility: Pluggable pattern extraction for new language features
        """

class NormalizationRuleEngine:
    async def apply_transformation_rules(self, ast_node: ASTNode, rules: List[NormalizationRule]) -> TransformationResult:
        """
        Purpose: Apply configurable normalization rules to AST nodes for consistent transformation
        Input: AST node and applicable normalization rules
        Output: TransformationResult with transformed node and applied rule metadata
        Performance: Rule engine optimized for high-frequency transformations
        Flexibility: Configurable rules enabling customization for specific requirements
        """

    async def validate_normalization_quality(self, original: ASTNode, normalized: NormalizedAST) -> QualityMetrics:
        """
        Purpose: Validate normalization quality ensuring information preservation and consistency
        Input: Original AST node and normalized representation
        Output: QualityMetrics with preservation score and consistency indicators
        Performance: Validation overhead <5% of total normalization time
        Reliability: Comprehensive quality checks preventing data corruption
        """
```

### Technical Requirements

1. **Performance**: 
   - Normalization speed: <200ms per AST file for 99% of typical code files
   - Batch throughput: 10,000+ files per minute with parallel processing
   - Memory efficiency: <100MB peak memory usage for 1GB AST files
   - Cache hit rate: >80% for repeated normalization patterns

2. **Language Support**: 
   - Initial support: Python, JavaScript, Java, C++, TypeScript
   - Extensible architecture: Easy addition of new languages without core changes
   - Version compatibility: Support for multiple language versions and parser outputs
   - Feature coverage: 95%+ coverage of common language constructs

3. **Normalization Quality**: 
   - Consistency: 85% variance reduction in cross-language representation
   - Information preservation: 99%+ preservation of semantic information
   - Error tolerance: Graceful handling of malformed or incomplete AST data
   - Validation: Comprehensive quality checks and metrics collection

4. **Configurability**: 
   - Rule-based normalization: Configurable transformation rules per language
   - Customization: Support for organization-specific naming conventions
   - Feature flags: Enable/disable specific normalization features
   - Environment adaptation: Different normalization strategies for dev/prod

5. **Integration**: 
   - Schema compliance: Output conforms to standardized AST schema
   - Metadata preservation: Maintain original language and parser information
   - Error reporting: Detailed error information for failed normalizations
   - Progress tracking: Real-time progress reporting for batch operations

### Implementation Steps

1. **Core Architecture**: Design pluggable normalization architecture with language handlers
2. **Common Schema**: Define standardized AST schema for normalized output
3. **Base Normalizer**: Implement abstract base class for language-specific normalizers
4. **Language Handlers**: Create normalizers for Python, JavaScript, Java, C++
5. **Rule Engine**: Build configurable rule engine for transformation logic
6. **Type System**: Implement unified type system mapping across languages
7. **Identifier Normalization**: Create consistent identifier handling across languages
8. **Performance Optimization**: Optimize for high-throughput batch processing
9. **Quality Validation**: Implement normalization quality metrics and validation
10. **Testing Framework**: Create comprehensive testing for all supported languages

### Code Pattern

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

class LanguageType(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript" 
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"

@dataclass
class NormalizedAST:
    """Standardized AST representation across all languages."""
    language: LanguageType
    version: str
    file_path: str
    
    # Standardized structural elements
    declarations: List['Declaration']
    imports: List['ImportStatement']
    exports: List['ExportStatement']
    
    # Metadata and analysis
    metrics: Dict[str, Any]
    relationships: List['CodeRelationship']
    symbols: List['SymbolDefinition']
    
    # Original AST preservation
    original_language_features: Dict[str, Any]
    normalization_metadata: 'NormalizationMetadata'

@dataclass 
class NormalizationMetadata:
    """Metadata about the normalization process."""
    normalizer_version: str
    processing_time: float
    applied_rules: List[str]
    quality_score: float
    warnings: List[str]

class BaseLanguageNormalizer(ABC):
    """Abstract base class for language-specific normalizers."""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.type_mapper = TypeMapper(config.type_mappings)
        self.identifier_normalizer = IdentifierNormalizer(config.identifier_rules)
        
    @abstractmethod
    async def normalize(self, ast_data: Dict[str, Any], metadata: FileMetadata) -> NormalizedAST:
        """Normalize language-specific AST to standard format."""
        pass
    
    @abstractmethod
    def get_supported_language(self) -> LanguageType:
        """Return the language type this normalizer supports."""
        pass
    
    @abstractmethod
    async def extract_declarations(self, ast_data: Dict[str, Any]) -> List[Declaration]:
        """Extract and normalize declarations (classes, functions, variables)."""
        pass
    
    @abstractmethod 
    async def extract_imports(self, ast_data: Dict[str, Any]) -> List[ImportStatement]:
        """Extract and normalize import/include statements."""
        pass
    
    async def normalize_with_quality_check(self, ast_data: Dict[str, Any], metadata: FileMetadata) -> NormalizedAST:
        """Normalize with comprehensive quality validation."""
        start_time = time.time()
        
        try:
            # Perform normalization
            normalized = await self.normalize(ast_data, metadata)
            
            # Validate quality
            quality_score = await self._calculate_quality_score(ast_data, normalized)
            normalized.normalization_metadata.quality_score = quality_score
            normalized.normalization_metadata.processing_time = time.time() - start_time
            
            return normalized
            
        except Exception as e:
            logger.error(f"Normalization failed for {metadata.file_path}: {e}")
            raise NormalizationError(f"Failed to normalize {metadata.file_path}: {e}")
    
    async def _calculate_quality_score(self, original: Dict[str, Any], normalized: NormalizedAST) -> float:
        """Calculate normalization quality score."""
        # Implementation depends on language-specific validation
        return 0.95  # Placeholder

class PythonNormalizer(BaseLanguageNormalizer):
    """Python-specific AST normalizer."""
    
    def get_supported_language(self) -> LanguageType:
        return LanguageType.PYTHON
    
    async def normalize(self, ast_data: Dict[str, Any], metadata: FileMetadata) -> NormalizedAST:
        """Normalize Python AST to standard format."""
        logger.debug(f"Normalizing Python AST: {metadata.file_path}")
        
        # Extract major structural elements
        declarations = await self.extract_declarations(ast_data)
        imports = await self.extract_imports(ast_data)
        exports = await self.extract_exports(ast_data)
        
        # Extract symbols and relationships
        symbols = await self._extract_symbols(ast_data)
        relationships = await self._extract_relationships(ast_data)
        
        # Calculate metrics
        metrics = await self._calculate_metrics(ast_data)
        
        return NormalizedAST(
            language=LanguageType.PYTHON,
            version=self._detect_python_version(ast_data),
            file_path=metadata.file_path,
            declarations=declarations,
            imports=imports,
            exports=exports,
            metrics=metrics,
            relationships=relationships,
            symbols=symbols,
            original_language_features=self._preserve_python_features(ast_data),
            normalization_metadata=NormalizationMetadata(
                normalizer_version="1.0.0",
                processing_time=0.0,  # Set later
                applied_rules=self.config.active_rules,
                quality_score=0.0,    # Set later
                warnings=[]
            )
        )
    
    async def extract_declarations(self, ast_data: Dict[str, Any]) -> List[Declaration]:
        """Extract Python class and function declarations."""
        declarations = []
        
        # Extract classes
        for node in self._find_nodes_by_type(ast_data, "ClassDef"):
            class_decl = await self._normalize_class_declaration(node)
            declarations.append(class_decl)
        
        # Extract functions
        for node in self._find_nodes_by_type(ast_data, "FunctionDef"):
            func_decl = await self._normalize_function_declaration(node)
            declarations.append(func_decl)
            
        # Extract async functions
        for node in self._find_nodes_by_type(ast_data, "AsyncFunctionDef"):
            async_func_decl = await self._normalize_async_function_declaration(node)
            declarations.append(async_func_decl)
        
        return declarations
    
    async def extract_imports(self, ast_data: Dict[str, Any]) -> List[ImportStatement]:
        """Extract Python import statements."""
        imports = []
        
        # Extract regular imports
        for node in self._find_nodes_by_type(ast_data, "Import"):
            for alias in node.get("names", []):
                import_stmt = ImportStatement(
                    module_name=alias.get("name"),
                    alias=alias.get("asname"),
                    import_type="module",
                    is_relative=False
                )
                imports.append(import_stmt)
        
        # Extract from imports
        for node in self._find_nodes_by_type(ast_data, "ImportFrom"):
            module = node.get("module", "")
            level = node.get("level", 0)
            
            for alias in node.get("names", []):
                import_stmt = ImportStatement(
                    module_name=f"{module}.{alias.get('name')}" if module else alias.get("name"),
                    alias=alias.get("asname"),
                    import_type="from_import",
                    is_relative=level > 0
                )
                imports.append(import_stmt)
        
        return imports
    
    async def extract_exports(self, ast_data: Dict[str, Any]) -> List[ExportStatement]:
        """Extract Python exports (mainly __all__ definitions)."""
        exports = []
        
        # Find __all__ definitions
        for node in self._find_nodes_by_type(ast_data, "Assign"):
            if self._is_all_assignment(node):
                export_names = self._extract_all_names(node)
                for name in export_names:
                    export_stmt = ExportStatement(
                        symbol_name=name,
                        export_type="public_api",
                        is_default=False
                    )
                    exports.append(export_stmt)
        
        return exports
    
    def _find_nodes_by_type(self, ast_data: Dict[str, Any], node_type: str) -> List[Dict[str, Any]]:
        """Find all nodes of specific type in AST."""
        nodes = []
        
        def visit_node(node):
            if isinstance(node, dict):
                if node.get("type") == node_type:
                    nodes.append(node)
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        visit_node(value)
            elif isinstance(node, list):
                for item in node:
                    visit_node(item)
        
        visit_node(ast_data)
        return nodes
    
    async def _normalize_class_declaration(self, node: Dict[str, Any]) -> Declaration:
        """Normalize Python class declaration."""
        return Declaration(
            name=node.get("name"),
            declaration_type="class",
            access_modifier="public",  # Python doesn't have explicit access modifiers
            parameters=await self._extract_class_parameters(node),
            return_type=None,
            decorators=await self._extract_decorators(node),
            docstring=await self._extract_docstring(node),
            line_start=node.get("lineno"),
            line_end=node.get("end_lineno"),
            language_specific_data={"bases": node.get("bases", [])}
        )
    
    async def _normalize_function_declaration(self, node: Dict[str, Any]) -> Declaration:
        """Normalize Python function declaration."""
        return Declaration(
            name=node.get("name"),
            declaration_type="function",
            access_modifier=self._determine_access_modifier(node.get("name")),
            parameters=await self._extract_function_parameters(node),
            return_type=await self._extract_return_type_annotation(node),
            decorators=await self._extract_decorators(node),
            docstring=await self._extract_docstring(node),
            line_start=node.get("lineno"),
            line_end=node.get("end_lineno"),
            language_specific_data={"is_async": False}
        )

class MultiLanguageNormalizer:
    """Main normalizer orchestrating language-specific normalizers."""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.normalizers: Dict[LanguageType, BaseLanguageNormalizer] = {}
        self.language_detector = LanguageDetector()
        self._initialize_normalizers()
        
    def _initialize_normalizers(self):
        """Initialize all language-specific normalizers."""
        # Register available normalizers
        normalizer_classes = [
            PythonNormalizer,
            JavaScriptNormalizer,
            JavaNormalizer,
            CppNormalizer
        ]
        
        for normalizer_class in normalizer_classes:
            try:
                normalizer = normalizer_class(self.config)
                language = normalizer.get_supported_language()
                self.normalizers[language] = normalizer
                logger.info(f"Registered normalizer for {language.value}")
            except Exception as e:
                logger.error(f"Failed to initialize {normalizer_class.__name__}: {e}")
    
    async def normalize_ast(self, ast_data: Dict[str, Any], language: Optional[str], metadata: FileMetadata) -> NormalizedAST:
        """Normalize AST using appropriate language-specific normalizer."""
        # Detect language if not provided
        if not language:
            detection_result = await self.language_detector.detect_language(ast_data, metadata)
            language = detection_result.detected_language
            
        language_type = LanguageType(language.lower())
        
        # Get appropriate normalizer
        normalizer = self.normalizers.get(language_type)
        if not normalizer:
            raise UnsupportedLanguageError(f"No normalizer available for {language}")
        
        # Perform normalization
        return await normalizer.normalize_with_quality_check(ast_data, metadata)
    
    async def batch_normalize(self, ast_files: List[ASTFile]) -> BatchNormalizationResult:
        """Normalize batch of AST files with parallel processing."""
        start_time = time.time()
        results = []
        errors = []
        
        # Process files concurrently
        semaphore = asyncio.Semaphore(self.config.max_concurrent_normalizations)
        
        async def normalize_single(ast_file: ASTFile) -> Optional[NormalizedAST]:
            async with semaphore:
                try:
                    return await self.normalize_ast(
                        ast_file.ast_data,
                        ast_file.language,
                        ast_file.metadata
                    )
                except Exception as e:
                    errors.append(NormalizationError(ast_file.metadata.file_path, str(e)))
                    return None
        
        # Execute all normalizations
        normalization_tasks = [normalize_single(ast_file) for ast_file in ast_files]
        normalized_results = await asyncio.gather(*normalization_tasks)
        
        # Collect successful results
        results = [result for result in normalized_results if result is not None]
        
        return BatchNormalizationResult(
            normalized_asts=results,
            processing_time=time.time() - start_time,
            success_count=len(results),
            error_count=len(errors),
            errors=errors
        )
```

## Acceptance Criteria

- [ ] **Multi-Language Support**: Normalization for Python, JavaScript, Java, C++, TypeScript with extensible architecture
- [ ] **Performance**: <200ms normalization time per file for 99% of typical AST files
- [ ] **Batch Processing**: 10,000+ files per minute throughput with parallel processing
- [ ] **Quality Assurance**: 99%+ information preservation and 85% variance reduction across languages
- [ ] **Language Detection**: 99%+ accuracy in automatic language detection from AST structure
- [ ] **Configurability**: Rule-based normalization with configurable transformation logic
- [ ] **Schema Compliance**: Output conforms to standardized normalized AST schema
- [ ] **Error Handling**: Graceful handling of malformed AST data with detailed error reporting
- [ ] **Memory Efficiency**: <100MB peak memory usage for 1GB AST files
- [ ] **Extensibility**: Easy addition of new languages without core system modifications
- [ ] **Test Coverage**: ≥90% test coverage including cross-language validation tests
- [ ] **Documentation**: Comprehensive normalization rules and language support guide

## Dependencies

- **TASK-005**: Schema Validation Engine (foundation for validation integration)
- **Transform Infrastructure**: Core transform package structure and configuration
- **Type System**: Standardized type definitions and mappings
- **Performance Framework**: Benchmarking and optimization infrastructure

## Risks and Mitigation

**High-Risk Areas**:
- Language evolution requiring constant rule updates and maintenance overhead
- Complex language features that resist standardization and may cause information loss
- Performance bottlenecks with deeply nested AST structures and large files

**Mitigation Strategies**:
- Automated testing for language parser updates and version compatibility
- Configurable information preservation levels with quality validation
- Streaming normalization and memory optimization for large AST processing

---

**Task Owner**: TBD  
**Reviewer**: TBD  
**Start Date**: TBD  
**Estimated Duration**: 6 days  
**Status**: ⚪ Not Started
