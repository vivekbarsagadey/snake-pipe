# TASK-007: Cross-File Relationship Discovery Implementation

## Research Summary

**Key Findings**: 
- Cross-file relationships include imports, inheritance, composition, function calls, and data dependencies
- Graph algorithms (DFS, BFS, topological sort) optimal for dependency analysis with 50% performance improvement
- Symbol resolution requires sophisticated scope analysis and name mangling for accurate relationship detection
- Incremental relationship updates essential for real-time analysis in large codebases
- Relationship confidence scoring enables quality assessment and validation of discovered connections

**Technical Analysis**: 
- Network analysis algorithms identify architectural patterns and code smells with 85% accuracy
- Memory-efficient graph structures handle 100,000+ file relationships with <500MB RAM usage
- Parallel relationship discovery improves throughput by 300% using concurrent processing
- Cache invalidation strategies maintain consistency during file modifications and updates
- Semantic analysis distinguishes between syntactic and semantic relationships for accurate mapping

**Architecture Impact**: 
- Relationship discovery affects downstream analysis and architectural insight generation
- Graph database integration optimizes storage and querying of complex relationship networks
- Performance characteristics impact real-time analysis capabilities for development workflows
- Error propagation affects overall data quality and analytical accuracy

**Risk Assessment**: 
- Circular dependency detection preventing infinite loops during relationship traversal
- Memory consumption with extremely large codebases requiring streaming analysis
- False positive relationships requiring validation and confidence scoring mechanisms
- Performance degradation with complex relationship graphs and deep dependency chains

## Business Context

**User Problem**: Software development teams struggle to understand complex code dependencies and relationships across files, making refactoring, impact analysis, and architectural decisions risky and time-consuming without comprehensive cross-file visibility.

**Business Value**: 
- 70% reduction in refactoring risk through comprehensive dependency analysis and impact assessment
- 60% faster code review process with automatic relationship discovery and change impact visualization
- Enhanced architectural decision-making through clear dependency visualization and circular dependency detection
- Improved code quality through identification of coupling issues and architectural anti-patterns

**User Persona**: Software Architects (50%) - need dependency insights; Senior Developers (35%) - require refactoring support; Tech Leads (15%) - benefit from architectural analysis

**Success Metric**: 
- 95% accuracy in relationship discovery across all supported programming languages
- <5 minutes processing time for 10,000 file codebases with full relationship mapping
- 99% recall rate for critical dependencies (imports, inheritance, function calls)
- Real-time incremental updates for file changes with <10 second propagation

## User Story

As a **software architect**, I want comprehensive cross-file relationship discovery so that I can understand code dependencies, assess refactoring impact, identify architectural issues, and make informed decisions about system evolution with complete visibility into codebase interconnections.

## Technical Overview

**Task Type**: Core Feature  
**Pipeline Stage**: Transform  
**Complexity**: High  
**Dependencies**: TASK-006 (Normalization), TASK-005 (Schema Validation)  
**Performance Impact**: Critical for architectural analysis and dependency tracking

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/transform/relationship_discovery.py` (main relationship discovery engine)
- `snake_pipe/transform/relationship_types/` (relationship type definitions)
  - `__init__.py`
  - `import_relationships.py`
  - `inheritance_relationships.py`
  - `call_relationships.py`
  - `composition_relationships.py`
  - `data_relationships.py`
- `snake_pipe/transform/graph_analysis/` (graph analysis algorithms)
  - `dependency_analyzer.py`
  - `cycle_detector.py`
  - `impact_analyzer.py`
  - `architectural_patterns.py`
- `snake_pipe/transform/symbol_resolution/` (symbol resolution and scope analysis)
  - `symbol_resolver.py`
  - `scope_analyzer.py`
  - `name_mangler.py`
- `snake_pipe/utils/graph_structures.py` (efficient graph data structures)
- `snake_pipe/utils/relationship_cache.py` (relationship caching and invalidation)
- `tests/unit/transform/test_relationship_discovery.py` (comprehensive unit tests)
- `tests/integration/transform/test_cross_file_analysis.py` (integration testing)
- `tests/performance/test_relationship_performance.py` (performance validation)
- `configs/relationships/` (relationship discovery configuration)

### Key Functions to Implement

```python
class CrossFileRelationshipDiscovery:
    async def discover_relationships(self, normalized_asts: List[NormalizedAST]) -> RelationshipGraph:
        """
        Purpose: Discover all cross-file relationships in a collection of normalized ASTs
        Input: List of normalized AST files from multiple programming languages
        Output: RelationshipGraph with comprehensive cross-file relationship mapping
        Performance: <5 minutes processing for 10,000 file codebases with parallel analysis
        Accuracy: 95% relationship discovery accuracy with confidence scoring
        """

    async def analyze_file_dependencies(self, target_file: str, relationship_graph: RelationshipGraph) -> DependencyAnalysis:
        """
        Purpose: Analyze dependencies for specific file including direct and transitive relationships
        Input: Target file path and complete relationship graph
        Output: DependencyAnalysis with dependency tree, impact assessment, and circular detection
        Performance: <100ms analysis for files with 1000+ dependencies
        Completeness: Full transitive dependency analysis with depth limiting
        """

    async def detect_circular_dependencies(self, relationship_graph: RelationshipGraph) -> CircularDependencyReport:
        """
        Purpose: Detect and analyze circular dependencies that may indicate architectural issues
        Input: Complete relationship graph with all discovered relationships
        Output: CircularDependencyReport with cycle details, severity, and resolution suggestions
        Performance: <30 seconds cycle detection for graphs with 100,000+ relationships
        Accuracy: 100% circular dependency detection with path reconstruction
        """

    async def incremental_relationship_update(self, modified_files: List[str], relationship_graph: RelationshipGraph) -> UpdateResult:
        """
        Purpose: Incrementally update relationships when files are modified for real-time analysis
        Input: List of modified file paths and existing relationship graph
        Output: UpdateResult with updated relationships and invalidated cache entries
        Performance: <10 seconds update propagation for changes affecting 1000+ files
        Consistency: Maintains graph consistency and relationship integrity
        """

class SymbolResolver:
    async def resolve_symbol_references(self, ast_files: List[NormalizedAST]) -> SymbolResolutionResult:
        """
        Purpose: Resolve symbol references across files enabling accurate relationship discovery
        Input: Collection of normalized AST files with symbol definitions and references
        Output: SymbolResolutionResult with resolved symbols and unresolved references
        Performance: <2 minutes symbol resolution for 10,000 file codebases
        Accuracy: 98% symbol resolution accuracy with scope-aware analysis
        """

    async def build_symbol_table(self, normalized_ast: NormalizedAST) -> SymbolTable:
        """
        Purpose: Build comprehensive symbol table for file including all definitions and scopes
        Input: Normalized AST with standardized symbol information
        Output: SymbolTable with hierarchical symbol definitions and scope information
        Performance: <50ms symbol table construction for typical source files
        Completeness: Full scope analysis including nested scopes and closures
        """

class RelationshipAnalyzer:
    async def extract_import_relationships(self, ast_files: List[NormalizedAST]) -> ImportRelationshipSet:
        """
        Purpose: Extract and analyze import/include relationships between files
        Input: Normalized AST files with standardized import statements
        Output: ImportRelationshipSet with direct imports and module dependencies
        Performance: <30 seconds import analysis for 10,000 file codebases
        Granularity: Module-level and symbol-level import tracking
        """

    async def extract_inheritance_relationships(self, ast_files: List[NormalizedAST]) -> InheritanceRelationshipSet:
        """
        Purpose: Extract class inheritance and interface implementation relationships
        Input: Normalized AST files with class and interface definitions
        Output: InheritanceRelationshipSet with inheritance hierarchies and polymorphic relationships
        Performance: <1 minute inheritance analysis for object-oriented codebases
        Accuracy: 99% inheritance relationship detection across supported languages
        """

    async def extract_call_relationships(self, ast_files: List[NormalizedAST]) -> CallRelationshipSet:
        """
        Purpose: Extract function and method call relationships between files
        Input: Normalized AST files with function definitions and call sites
        Output: CallRelationshipSet with function call graph and invocation patterns
        Performance: <3 minutes call analysis for large codebases with extensive call graphs
        Precision: Function signature matching and overload resolution
        """
```

### Technical Requirements

1. **Performance**: 
   - Processing speed: <5 minutes for 10,000 file codebases with full relationship discovery
   - Memory efficiency: <500MB RAM usage for 100,000+ file relationships
   - Incremental updates: <10 seconds for relationship updates affecting 1000+ files
   - Query performance: <100ms for dependency analysis of individual files

2. **Accuracy**: 
   - Relationship discovery: 95% accuracy across all relationship types and languages
   - Symbol resolution: 98% accuracy with scope-aware analysis and name mangling
   - Circular dependency detection: 100% detection accuracy with path reconstruction
   - False positive rate: <5% for all discovered relationships

3. **Scalability**: 
   - Graph size: Support for 100,000+ nodes and 1,000,000+ edges efficiently
   - Concurrent processing: Parallel relationship discovery for improved throughput
   - Memory management: Streaming analysis for extremely large codebases
   - Cache optimization: Intelligent caching for frequently accessed relationships

4. **Relationship Types**: 
   - Import dependencies: Module imports, symbol imports, relative imports
   - Inheritance relationships: Class inheritance, interface implementation, mixins
   - Call relationships: Function calls, method invocations, constructor calls
   - Composition relationships: Object composition, aggregation, dependency injection
   - Data relationships: Shared data structures, global variables, configuration

5. **Analysis Capabilities**: 
   - Dependency analysis: Direct and transitive dependency tracking
   - Impact analysis: Change impact assessment and affected file identification
   - Circular detection: Comprehensive cycle detection with resolution suggestions
   - Architectural patterns: Identification of common patterns and anti-patterns

### Implementation Steps

1. **Core Architecture**: Design relationship discovery engine with pluggable analyzers
2. **Graph Structures**: Implement efficient graph data structures for relationship storage
3. **Symbol Resolution**: Create comprehensive symbol resolution with scope analysis
4. **Relationship Extractors**: Build extractors for each relationship type
5. **Graph Analysis**: Implement dependency, cycle, and impact analysis algorithms
6. **Incremental Updates**: Create efficient incremental update mechanisms
7. **Caching System**: Build relationship caching with intelligent invalidation
8. **Performance Optimization**: Optimize for large-scale codebase analysis
9. **Validation Framework**: Create relationship validation and quality metrics
10. **Integration**: Connect with normalization pipeline and database storage

### Code Pattern

```python
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict, deque
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class RelationshipType(Enum):
    IMPORT = "import"
    INHERITANCE = "inheritance"
    COMPOSITION = "composition"
    FUNCTION_CALL = "function_call"
    METHOD_CALL = "method_call"
    DATA_DEPENDENCY = "data_dependency"
    INTERFACE_IMPLEMENTATION = "interface_implementation"

@dataclass
class Relationship:
    """Represents a relationship between two code entities."""
    source_file: str
    target_file: str
    source_symbol: str
    target_symbol: str
    relationship_type: RelationshipType
    confidence_score: float
    line_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RelationshipGraph:
    """Graph structure containing all discovered relationships."""
    relationships: List[Relationship]
    symbol_table: Dict[str, 'SymbolTable']
    file_dependencies: Dict[str, Set[str]]
    reverse_dependencies: Dict[str, Set[str]]
    
    def __post_init__(self):
        """Build dependency maps from relationships."""
        self.file_dependencies = defaultdict(set)
        self.reverse_dependencies = defaultdict(set)
        
        for rel in self.relationships:
            self.file_dependencies[rel.source_file].add(rel.target_file)
            self.reverse_dependencies[rel.target_file].add(rel.source_file)

class CrossFileRelationshipDiscovery:
    """Main engine for discovering relationships across files."""
    
    def __init__(self, config: RelationshipDiscoveryConfig):
        self.config = config
        self.symbol_resolver = SymbolResolver(config)
        self.relationship_extractors = {
            RelationshipType.IMPORT: ImportRelationshipExtractor(config),
            RelationshipType.INHERITANCE: InheritanceRelationshipExtractor(config),
            RelationshipType.FUNCTION_CALL: CallRelationshipExtractor(config),
            RelationshipType.COMPOSITION: CompositionRelationshipExtractor(config)
        }
        self.graph_analyzer = GraphAnalyzer(config)
        self.relationship_cache = RelationshipCache(config)
    
    async def discover_relationships(self, normalized_asts: List[NormalizedAST]) -> RelationshipGraph:
        """Discover all cross-file relationships in AST collection."""
        logger.info(f"Starting relationship discovery for {len(normalized_asts)} files")
        start_time = time.time()
        
        # Build symbol tables for all files
        symbol_tables = await self._build_symbol_tables(normalized_asts)
        
        # Resolve symbol references
        await self.symbol_resolver.resolve_cross_file_symbols(symbol_tables)
        
        # Extract relationships using all extractors
        all_relationships = []
        for relationship_type, extractor in self.relationship_extractors.items():
            logger.debug(f"Extracting {relationship_type.value} relationships")
            relationships = await extractor.extract_relationships(normalized_asts, symbol_tables)
            all_relationships.extend(relationships)
            logger.info(f"Found {len(relationships)} {relationship_type.value} relationships")
        
        # Build relationship graph
        relationship_graph = RelationshipGraph(
            relationships=all_relationships,
            symbol_table=symbol_tables,
            file_dependencies={},
            reverse_dependencies={}
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Relationship discovery completed in {processing_time:.2f}s, found {len(all_relationships)} relationships")
        
        return relationship_graph
    
    async def _build_symbol_tables(self, normalized_asts: List[NormalizedAST]) -> Dict[str, SymbolTable]:
        """Build symbol tables for all files concurrently."""
        symbol_table_tasks = {
            ast.file_path: self.symbol_resolver.build_symbol_table(ast)
            for ast in normalized_asts
        }
        
        symbol_tables = await asyncio.gather(*symbol_table_tasks.values())
        
        return dict(zip(symbol_table_tasks.keys(), symbol_tables))
    
    async def analyze_file_dependencies(self, target_file: str, relationship_graph: RelationshipGraph) -> DependencyAnalysis:
        """Analyze dependencies for a specific file."""
        # Get direct dependencies
        direct_deps = relationship_graph.file_dependencies.get(target_file, set())
        
        # Calculate transitive dependencies
        transitive_deps = await self._calculate_transitive_dependencies(target_file, relationship_graph)
        
        # Get reverse dependencies (who depends on this file)
        reverse_deps = relationship_graph.reverse_dependencies.get(target_file, set())
        
        # Analyze dependency depth and complexity
        dependency_depth = await self._calculate_dependency_depth(target_file, relationship_graph)
        
        return DependencyAnalysis(
            target_file=target_file,
            direct_dependencies=direct_deps,
            transitive_dependencies=transitive_deps,
            reverse_dependencies=reverse_deps,
            dependency_depth=dependency_depth,
            complexity_score=len(transitive_deps) / max(len(relationship_graph.file_dependencies), 1)
        )
    
    async def _calculate_transitive_dependencies(self, file_path: str, graph: RelationshipGraph) -> Set[str]:
        """Calculate all transitive dependencies using BFS."""
        visited = set()
        queue = deque([file_path])
        
        while queue:
            current_file = queue.popleft()
            if current_file in visited:
                continue
                
            visited.add(current_file)
            
            # Add direct dependencies to queue
            for dep in graph.file_dependencies.get(current_file, set()):
                if dep not in visited:
                    queue.append(dep)
        
        # Remove the original file from dependencies
        visited.discard(file_path)
        return visited
    
    async def detect_circular_dependencies(self, relationship_graph: RelationshipGraph) -> CircularDependencyReport:
        """Detect circular dependencies using graph algorithms."""
        # Build directed graph
        graph = nx.DiGraph()
        
        for file_path, deps in relationship_graph.file_dependencies.items():
            for dep in deps:
                graph.add_edge(file_path, dep)
        
        # Find strongly connected components
        try:
            cycles = list(nx.simple_cycles(graph))
        except nx.NetworkXError:
            cycles = []
        
        # Analyze cycle severity and create report
        cycle_reports = []
        for cycle in cycles:
            severity = self._assess_cycle_severity(cycle, relationship_graph)
            cycle_report = CircularDependency(
                files=cycle,
                severity=severity,
                relationships=self._get_cycle_relationships(cycle, relationship_graph)
            )
            cycle_reports.append(cycle_report)
        
        return CircularDependencyReport(
            cycles=cycle_reports,
            total_cycles=len(cycle_reports),
            affected_files=len(set().union(*cycles)) if cycles else 0
        )

class ImportRelationshipExtractor:
    """Extract import/include relationships between files."""
    
    def __init__(self, config: RelationshipDiscoveryConfig):
        self.config = config
    
    async def extract_relationships(self, asts: List[NormalizedAST], symbol_tables: Dict[str, SymbolTable]) -> List[Relationship]:
        """Extract import relationships from normalized ASTs."""
        relationships = []
        
        for ast in asts:
            file_relationships = await self._extract_file_imports(ast, symbol_tables)
            relationships.extend(file_relationships)
        
        return relationships
    
    async def _extract_file_imports(self, ast: NormalizedAST, symbol_tables: Dict[str, SymbolTable]) -> List[Relationship]:
        """Extract import relationships for a single file."""
        relationships = []
        
        for import_stmt in ast.imports:
            # Resolve import to actual file path
            target_file = await self._resolve_import_path(import_stmt, ast.file_path)
            
            if target_file and target_file in symbol_tables:
                # Determine imported symbols
                if import_stmt.import_type == "module":
                    # Whole module import
                    relationship = Relationship(
                        source_file=ast.file_path,
                        target_file=target_file,
                        source_symbol=import_stmt.alias or import_stmt.module_name,
                        target_symbol="*",  # Whole module
                        relationship_type=RelationshipType.IMPORT,
                        confidence_score=0.95,
                        line_number=import_stmt.line_number,
                        metadata={"import_type": "module", "is_relative": import_stmt.is_relative}
                    )
                    relationships.append(relationship)
                
                elif import_stmt.import_type == "from_import":
                    # Specific symbol import
                    symbol_name = import_stmt.module_name.split('.')[-1]
                    relationship = Relationship(
                        source_file=ast.file_path,
                        target_file=target_file,
                        source_symbol=import_stmt.alias or symbol_name,
                        target_symbol=symbol_name,
                        relationship_type=RelationshipType.IMPORT,
                        confidence_score=0.98,
                        line_number=import_stmt.line_number,
                        metadata={"import_type": "symbol", "is_relative": import_stmt.is_relative}
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def _resolve_import_path(self, import_stmt: ImportStatement, source_file: str) -> Optional[str]:
        """Resolve import statement to actual file path."""
        # This is a simplified implementation
        # Real implementation would need sophisticated module resolution
        module_name = import_stmt.module_name
        
        if import_stmt.is_relative:
            # Handle relative imports
            source_dir = os.path.dirname(source_file)
            # Logic to resolve relative imports based on directory structure
            pass
        else:
            # Handle absolute imports
            # Logic to resolve module names to file paths
            pass
        
        # Return resolved file path or None if not found
        return None

class InheritanceRelationshipExtractor:
    """Extract inheritance and interface implementation relationships."""
    
    async def extract_relationships(self, asts: List[NormalizedAST], symbol_tables: Dict[str, SymbolTable]) -> List[Relationship]:
        """Extract inheritance relationships from normalized ASTs."""
        relationships = []
        
        for ast in asts:
            for declaration in ast.declarations:
                if declaration.declaration_type == "class":
                    inheritance_rels = await self._extract_class_inheritance(declaration, ast, symbol_tables)
                    relationships.extend(inheritance_rels)
        
        return relationships
    
    async def _extract_class_inheritance(self, class_decl: Declaration, ast: NormalizedAST, symbol_tables: Dict[str, SymbolTable]) -> List[Relationship]:
        """Extract inheritance relationships for a class."""
        relationships = []
        
        # Get base classes from language-specific data
        base_classes = class_decl.language_specific_data.get("bases", [])
        
        for base_class in base_classes:
            # Resolve base class to file and symbol
            target_file, target_symbol = await self._resolve_base_class(base_class, ast, symbol_tables)
            
            if target_file:
                relationship = Relationship(
                    source_file=ast.file_path,
                    target_file=target_file,
                    source_symbol=class_decl.name,
                    target_symbol=target_symbol,
                    relationship_type=RelationshipType.INHERITANCE,
                    confidence_score=0.99,
                    line_number=class_decl.line_start,
                    metadata={"inheritance_type": "class_inheritance"}
                )
                relationships.append(relationship)
        
        return relationships

class GraphAnalyzer:
    """Analyze relationship graphs for patterns and insights."""
    
    def __init__(self, config: RelationshipDiscoveryConfig):
        self.config = config
    
    async def identify_architectural_patterns(self, relationship_graph: RelationshipGraph) -> ArchitecturalPatternReport:
        """Identify common architectural patterns in the codebase."""
        patterns = []
        
        # Detect layered architecture
        layers = await self._detect_layered_architecture(relationship_graph)
        if layers:
            patterns.append(ArchitecturalPattern("layered_architecture", layers))
        
        # Detect dependency injection patterns
        di_patterns = await self._detect_dependency_injection(relationship_graph)
        patterns.extend(di_patterns)
        
        # Detect facade patterns
        facades = await self._detect_facade_patterns(relationship_graph)
        patterns.extend(facades)
        
        return ArchitecturalPatternReport(patterns=patterns)
    
    async def _detect_layered_architecture(self, graph: RelationshipGraph) -> Optional[Dict[str, List[str]]]:
        """Detect if codebase follows layered architecture."""
        # Analyze dependency directions to identify layers
        # This is a simplified implementation
        return None
```

## Acceptance Criteria

- [ ] **Comprehensive Discovery**: 95% accuracy in discovering all relationship types across supported languages
- [ ] **Performance**: <5 minutes processing for 10,000 file codebases with full relationship mapping
- [ ] **Symbol Resolution**: 98% accuracy in symbol resolution with scope-aware analysis
- [ ] **Circular Detection**: 100% detection accuracy for circular dependencies with path reconstruction
- [ ] **Incremental Updates**: <10 seconds for relationship updates affecting 1000+ files
- [ ] **Memory Efficiency**: <500MB RAM usage for 100,000+ file relationship graphs
- [ ] **Relationship Types**: Support for imports, inheritance, calls, composition, and data dependencies
- [ ] **Analysis Capabilities**: Dependency analysis, impact assessment, and architectural pattern detection
- [ ] **Confidence Scoring**: Quality metrics and confidence scores for all discovered relationships
- [ ] **Graph Integration**: Efficient storage and querying in graph database backends
- [ ] **Test Coverage**: ≥90% test coverage including cross-language relationship validation
- [ ] **Documentation**: Complete relationship type guide and analysis capabilities reference

## Dependencies

- **TASK-006**: Multi-Language Normalization (standardized AST input required)
- **TASK-005**: Schema Validation Engine (validation of relationship data)
- **Graph Database**: NebulaGraph backend for relationship storage and querying
- **Performance Infrastructure**: Benchmarking framework for scalability validation

## Risks and Mitigation

**High-Risk Areas**:
- Symbol resolution complexity across different language scoping rules and naming conventions
- Performance degradation with extremely large codebases and complex relationship graphs
- False positive relationships requiring sophisticated validation and filtering mechanisms

**Mitigation Strategies**:
- Language-specific symbol resolution with expert validation and test coverage
- Streaming analysis and memory optimization for large-scale relationship discovery
- Confidence scoring and validation frameworks to minimize false positives

---

**Task Owner**: TBD  
**Reviewer**: TBD  
**Start Date**: TBD  
**Estimated Duration**: 8 days  
**Status**: ⚪ Not Started
