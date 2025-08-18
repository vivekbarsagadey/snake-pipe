# Task 014: Elasticsearch Backend Implementation

## Research Summary

**Key Findings**: 
- Elasticsearch excels at full-text search, real-time analytics, and aggregation queries for large-scale code analysis
- Dynamic mapping and schema-less design enable flexible storage of varied AST structures across programming languages
- Inverted indexes provide sub-second search performance on code content, comments, and documentation
- Aggregation framework enables powerful analytics on code metrics, patterns, and evolution trends
- Elasticsearch's distributed architecture supports horizontal scaling for large enterprise codebases

**Technical Analysis**: 
- JSON document model naturally aligns with AST data structures and hierarchical code relationships
- Text analysis with custom analyzers optimizes search for code-specific patterns and identifiers
- Multi-field mapping enables both exact matching and full-text search on the same code elements
- Nested documents support complex code structures like classes with methods and inner classes
- Bulk indexing API provides high-throughput ingestion with automatic batching and error handling

**Architecture Impact**: 
- Elasticsearch backend enables powerful search and analytics capabilities for development teams
- Integration with Kibana provides rich dashboards and visualization for code metrics and trends
- Real-time indexing supports live code analysis and instant search across evolving codebases
- Aggregation queries enable advanced code analytics and reporting for architectural insights

**Risk Assessment**: 
- **Performance Risk**: Large document sizes may impact indexing speed - mitigated by document size optimization and field selection
- **Memory Risk**: Elasticsearch can be memory-intensive for large indexes - managed with proper heap sizing and cluster configuration
- **Query Risk**: Complex aggregations may impact cluster performance - addressed with query optimization and caching

## Business Context

**User Problem**: Development teams need powerful search, analytics, and real-time monitoring capabilities for code analysis, documentation discovery, and development workflow optimization.

**Business Value**: 
- **Code Discovery**: Fast full-text search across entire codebases including comments and documentation
- **Development Analytics**: Real-time insights into code changes, developer productivity, and quality metrics
- **Knowledge Management**: Searchable code documentation and architectural decision records
- **Quality Monitoring**: Real-time alerts and dashboards for code quality and technical debt trends

**User Persona**: Development Teams (50%) needing code search and discovery, Engineering Managers (30%) requiring analytics and reporting, Technical Writers (20%) managing searchable documentation.

**Success Metric**: 
- Search performance: <100ms for full-text search queries across millions of code documents
- Indexing throughput: >2000 AST documents per minute with real-time availability
- Analytics performance: <500ms for complex aggregation queries on large datasets
- Storage efficiency: <200MB index storage per 10,000 code files with full-text indexing

## User Story

As a **development team member**, I want **Elasticsearch-powered search and analytics for code analysis** so that **I can quickly discover code, analyze development patterns, and monitor code quality trends across our entire codebase**.

## Technical Overview

**Task Type**: Story Task (Database Backend Implementation)
**Pipeline Stage**: Load (Search & Analytics Database Integration)
**Complexity**: Medium-High
**Dependencies**: Backend plugin architecture (TASK-010), AST data transformation
**Performance Impact**: Elasticsearch operations must support high-throughput indexing and sub-second search

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/load/elasticsearch_backend.py` (main Elasticsearch backend implementation)
- `snake_pipe/load/elasticsearch_config.py` (Elasticsearch-specific configuration management)
- `snake_pipe/load/elasticsearch_mapping.py` (index mapping and schema definition)
- `snake_pipe/load/elasticsearch_queries.py` (search query builders and templates)
- `snake_pipe/load/search_mapper.py` (AST to search document mapping logic)
- `snake_pipe/utils/elasticsearch_client.py` (Elasticsearch client wrapper with connection management)
- `snake_pipe/utils/search_utils.py` (search utilities and query optimization helpers)
- `snake_pipe/analytics/elasticsearch_analytics.py` (analytics and aggregation engine)
- `tests/unit/load/test_elasticsearch_backend.py` (comprehensive unit tests with mocked Elasticsearch)
- `tests/integration/load/test_elasticsearch_integration.py` (integration tests with real Elasticsearch cluster)
- `tests/performance/test_elasticsearch_performance.py` (performance testing for indexing and search)
- `tests/tasks/test_task014_verification.py` (task-specific verification and acceptance tests)
- `tests/tasks/test_task014_integration.py` (end-to-end Elasticsearch integration tests)

### Key Functions to Implement

```python
async def index_ast_documents(
    ast_data: List[TransformedASTData], 
    config: ElasticsearchConfig
) -> IndexingResult:
    """
    Purpose: Index AST data as searchable documents in Elasticsearch
    Input: Transformed AST data and Elasticsearch configuration
    Output: Indexing result with document counts and performance metrics
    Performance: Index 500+ AST documents per minute with real-time availability
    """

async def bulk_index_documents(
    documents: List[SearchDocument], 
    client: ElasticsearchClient
) -> BulkIndexResult:
    """
    Purpose: Efficiently bulk index documents using Elasticsearch bulk API
    Input: List of search documents and Elasticsearch client
    Output: Bulk indexing result with success/failure statistics
    Performance: Index 10,000+ documents per minute with error handling
    """

async def execute_code_search(
    search_query: CodeSearchQuery, 
    client: ElasticsearchClient
) -> SearchResult:
    """
    Purpose: Execute full-text search queries on indexed code documents
    Input: Search query configuration and Elasticsearch client
    Output: Search results with ranking and highlighting
    Performance: <100ms for full-text search on millions of documents
    """

async def execute_analytics_aggregation(
    analytics_query: AnalyticsQuery, 
    client: ElasticsearchClient
) -> AnalyticsResult:
    """
    Purpose: Execute analytics and aggregation queries for code insights
    Input: Analytics query and Elasticsearch client
    Output: Aggregation results with metrics and trends
    Performance: <500ms for complex aggregations on large datasets
    """

class ElasticsearchBackend(DatabaseBackend):
    """
    Purpose: Elasticsearch implementation for search and analytics
    Features: Full-text search, real-time analytics, document indexing
    Performance: High-throughput indexing with sub-second search performance
    """
    
    async def connect(self, config: ElasticsearchConfig) -> None:
        """Initialize connection to Elasticsearch cluster with proper authentication"""
    
    async def write_batch(self, data: List[TransformedASTData]) -> WriteResult:
        """Index batch of AST data as searchable documents"""
    
    async def health_check(self) -> HealthCheckResult:
        """Check Elasticsearch cluster health including index status and search performance"""
    
    async def execute_search_query(self, query: SearchQuery) -> QueryResult:
        """Execute search and analytics queries on indexed documents"""
```

### Technical Requirements

1. **Performance**: 
   - Indexing throughput: >2000 AST documents per minute with real-time search availability
   - Search performance: <100ms for full-text search queries on millions of documents
   - Analytics performance: <500ms for complex aggregation queries on large datasets
   - Bulk operations: Index 10,000+ documents per minute with efficient batching

2. **Search Capabilities**: 
   - Full-text search across code content, comments, and documentation
   - Fuzzy search and typo tolerance for developer-friendly search experience
   - Multi-field search with boosting for relevance optimization
   - Faceted search with filters for language, file type, and metadata

3. **Analytics**: 
   - Real-time aggregations for code metrics and quality trends
   - Time-series analytics for code evolution and development patterns
   - Custom metrics calculation for architectural insights
   - Dashboard-ready data for visualization and reporting

4. **Integration**: 
   - Implementation of DatabaseBackend interface for plugin compatibility
   - Configuration-driven index settings and mapping management
   - Integration with monitoring for search operation metrics and cluster health
   - Support for multiple Elasticsearch versions and deployment types

5. **Scalability**: 
   - Horizontal scaling support for distributed Elasticsearch clusters
   - Index sharding and replication for performance and availability
   - Efficient memory usage for large document collections
   - Automatic index lifecycle management for storage optimization

6. **Reliability**: 
   - Robust handling of Elasticsearch cluster failures and node outages
   - Automatic retry mechanisms for transient indexing and search failures
   - Health monitoring and cluster state management
   - Backup and recovery integration with Elasticsearch snapshot APIs

### Implementation Steps

1. **Index Design**: Define comprehensive index mapping for AST documents with optimal field types
2. **Client Integration**: Implement Elasticsearch client wrapper with connection management and error handling
3. **Backend Implementation**: Create Elasticsearch backend following DatabaseBackend interface
4. **Document Mapping**: Develop sophisticated AST to search document transformation
5. **Search Engine**: Build comprehensive search query engine with full-text and analytics capabilities
6. **Bulk Operations**: Implement efficient bulk indexing with error handling and retry logic
7. **Analytics Engine**: Create analytics and aggregation engine for code insights
8. **Monitoring Integration**: Add comprehensive monitoring and performance metrics
9. **Testing**: Create extensive unit, integration, and performance tests
10. **Documentation**: Develop search query documentation and analytics examples

### Code Patterns

```python
# Elasticsearch Backend Pattern (following project conventions)
@dataclass
class ElasticsearchConfig:
    """Configuration for Elasticsearch backend operations"""
    hosts: List[str] = field(default_factory=lambda: ["localhost:9200"])
    username: Optional[str] = None
    password: Optional[str] = None
    index_prefix: str = "snake_pipe"
    shards: int = 1
    replicas: int = 0
    refresh_interval: str = "1s"
    timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_ssl: bool = False
    verify_certs: bool = True

@dataclass
class SearchDocument:
    """Representation of code document for Elasticsearch indexing"""
    document_id: str
    file_path: str
    language: str
    content: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    indexed_at: datetime

@dataclass
class SearchResult:
    """Result of search query execution"""
    hits: List[SearchHit]
    total_hits: int
    query_time: float
    aggregations: Optional[Dict[str, Any]]
    suggestions: Optional[List[str]]

class ElasticsearchBackend(DatabaseBackend):
    """High-performance Elasticsearch backend for search and analytics"""
    
    def __init__(self, config: ElasticsearchConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.client: Optional[ElasticsearchClient] = None
        self.mapping_manager = ElasticsearchMappingManager(config)
        self.search_mapper = SearchDocumentMapper()
        self.stats = ElasticsearchStatistics()
    
    async def connect(self, config: ElasticsearchConfig) -> None:
        """Initialize connection to Elasticsearch cluster"""
        try:
            self.client = ElasticsearchClient(config)
            await self.client.connect()
            
            # Ensure indexes exist with proper mappings
            await self.mapping_manager.ensure_indexes_exist(self.client)
            
            self.logger.info(f"Connected to Elasticsearch cluster: {config.hosts}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise DatabaseConnectionError(f"Elasticsearch connection failed: {e}")
    
    async def write_batch(self, data: List[TransformedASTData]) -> WriteResult:
        """Index batch of AST data as searchable documents"""
        if not self.client:
            raise DatabaseConnectionError("Elasticsearch client not connected")
        
        try:
            # Transform AST data to search documents
            search_documents = await self._transform_to_search_documents(data)
            
            # Bulk index documents
            index_result = await self._bulk_index_documents(search_documents)
            
            # Update statistics
            self.stats.total_writes += len(data)
            self.stats.documents_indexed += len(search_documents)
            
            return WriteResult(
                success=True,
                items_written=len(data),
                operation_time=index_result.operation_time,
                backend_specific_metrics=index_result.metrics
            )
            
        except Exception as e:
            self.logger.error(f"Failed to write batch to Elasticsearch: {e}")
            self.stats.write_errors += 1
            raise DatabaseWriteError(f"Elasticsearch write failed: {e}")
    
    async def _transform_to_search_documents(self, data: List[TransformedASTData]) -> List[SearchDocument]:
        """Transform AST data to search-optimized documents"""
        search_documents = []
        
        for ast_data in data:
            # Create main file document
            file_document = SearchDocument(
                document_id=f"file_{ast_data.file_id}",
                file_path=ast_data.file_path,
                language=ast_data.language,
                content=self._extract_searchable_content(ast_data),
                functions=self._extract_function_summaries(ast_data.functions),
                classes=self._extract_class_summaries(ast_data.classes),
                metadata={
                    "file_id": ast_data.file_id,
                    "size_bytes": ast_data.size_bytes,
                    "complexity": ast_data.complexity_metrics,
                    "dependencies": [dep.target_file_id for dep in ast_data.dependencies],
                    "quality_score": ast_data.quality_metrics.overall_score,
                    "last_modified": ast_data.last_modified.isoformat()
                },
                indexed_at=datetime.utcnow()
            )
            search_documents.append(file_document)
            
            # Create individual function documents for detailed search
            for function in ast_data.functions:
                function_document = SearchDocument(
                    document_id=f"function_{function.function_id}",
                    file_path=ast_data.file_path,
                    language=ast_data.language,
                    content=f"{function.name} {function.signature} {function.doc_string or ''}",
                    functions=[],
                    classes=[],
                    metadata={
                        "type": "function",
                        "file_id": ast_data.file_id,
                        "function_id": function.function_id,
                        "function_name": function.name,
                        "complexity": function.complexity,
                        "line_start": function.line_start,
                        "line_end": function.line_end,
                        "parameters": [p.name for p in function.parameters],
                        "return_type": function.return_type
                    },
                    indexed_at=datetime.utcnow()
                )
                search_documents.append(function_document)
        
        return search_documents
    
    def _extract_searchable_content(self, ast_data: TransformedASTData) -> str:
        """Extract searchable text content from AST data"""
        content_parts = []
        
        # File path components for path-based search
        content_parts.append(ast_data.file_path)
        
        # Function names and signatures
        for function in ast_data.functions:
            content_parts.append(function.name)
            content_parts.append(function.signature)
            if function.doc_string:
                content_parts.append(function.doc_string)
        
        # Class names and documentation
        for class_info in ast_data.classes:
            content_parts.append(class_info.name)
            if class_info.namespace:
                content_parts.append(class_info.namespace)
            if class_info.doc_string:
                content_parts.append(class_info.doc_string)
        
        # Comments and documentation
        for comment in ast_data.comments:
            content_parts.append(comment.text)
        
        # Import statements for dependency search
        for import_stmt in ast_data.imports:
            content_parts.append(import_stmt.module_name)
            if import_stmt.alias:
                content_parts.append(import_stmt.alias)
        
        return " ".join(content_parts)
    
    async def _bulk_index_documents(self, documents: List[SearchDocument]) -> BulkIndexResult:
        """Efficiently bulk index documents using Elasticsearch bulk API"""
        start_time = time.time()
        
        # Prepare bulk request body
        bulk_body = []
        for doc in documents:
            # Index action
            bulk_body.append({
                "index": {
                    "_index": self._get_index_name(doc.language),
                    "_id": doc.document_id
                }
            })
            # Document body
            bulk_body.append({
                "file_path": doc.file_path,
                "language": doc.language,
                "content": doc.content,
                "functions": doc.functions,
                "classes": doc.classes,
                "metadata": doc.metadata,
                "indexed_at": doc.indexed_at.isoformat()
            })
        
        # Execute bulk request
        response = await self.client.bulk(body=bulk_body)
        
        # Process response
        successful = 0
        errors = []
        for item in response["items"]:
            if "index" in item:
                if item["index"]["status"] in [200, 201]:
                    successful += 1
                else:
                    errors.append(item["index"]["error"])
        
        operation_time = time.time() - start_time
        
        return BulkIndexResult(
            documents_indexed=successful,
            errors=errors,
            operation_time=operation_time,
            throughput=successful / operation_time if operation_time > 0 else 0
        )

# Index Mapping Management Pattern
class ElasticsearchMappingManager:
    """Manages Elasticsearch index mappings for AST documents"""
    
    def __init__(self, config: ElasticsearchConfig):
        self.config = config
        self.mapping_version = "1.0"
    
    async def ensure_indexes_exist(self, client: ElasticsearchClient) -> None:
        """Ensure all required indexes exist with proper mappings"""
        # Create language-specific indexes
        languages = ["python", "java", "javascript", "typescript", "go", "rust"]
        
        for language in languages:
            index_name = self._get_index_name(language)
            await self._create_index_if_not_exists(client, index_name, language)
    
    async def _create_index_if_not_exists(
        self, 
        client: ElasticsearchClient, 
        index_name: str, 
        language: str
    ) -> None:
        """Create index with optimized mapping for code search"""
        if not await client.indices.exists(index=index_name):
            mapping = self._create_code_mapping(language)
            settings = self._create_index_settings()
            
            await client.indices.create(
                index=index_name,
                body={"mappings": mapping, "settings": settings}
            )
    
    def _create_code_mapping(self, language: str) -> Dict[str, Any]:
        """Create optimized mapping for code documents"""
        return {
            "properties": {
                "file_path": {
                    "type": "text",
                    "analyzer": "path_analyzer",
                    "fields": {
                        "keyword": {"type": "keyword"},
                        "hierarchy": {"type": "text", "analyzer": "path_hierarchy"}
                    }
                },
                "language": {
                    "type": "keyword"
                },
                "content": {
                    "type": "text",
                    "analyzer": "code_analyzer",
                    "search_analyzer": "code_search_analyzer"
                },
                "functions": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "text", "analyzer": "identifier_analyzer"},
                        "signature": {"type": "text", "analyzer": "code_analyzer"},
                        "complexity": {"type": "integer"},
                        "line_start": {"type": "integer"},
                        "line_end": {"type": "integer"}
                    }
                },
                "classes": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "text", "analyzer": "identifier_analyzer"},
                        "namespace": {"type": "keyword"},
                        "method_count": {"type": "integer"},
                        "is_abstract": {"type": "boolean"}
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "keyword"},
                        "size_bytes": {"type": "long"},
                        "complexity": {"type": "object"},
                        "dependencies": {"type": "keyword"},
                        "quality_score": {"type": "float"},
                        "last_modified": {"type": "date"}
                    }
                },
                "indexed_at": {
                    "type": "date"
                }
            }
        }
    
    def _create_index_settings(self) -> Dict[str, Any]:
        """Create index settings with custom analyzers for code search"""
        return {
            "number_of_shards": self.config.shards,
            "number_of_replicas": self.config.replicas,
            "refresh_interval": self.config.refresh_interval,
            "analysis": {
                "analyzer": {
                    "code_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "camelcase_filter", "underscore_filter"]
                    },
                    "code_search_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop"]
                    },
                    "identifier_analyzer": {
                        "type": "custom",
                        "tokenizer": "keyword",
                        "filter": ["lowercase", "camelcase_filter", "underscore_filter"]
                    },
                    "path_analyzer": {
                        "type": "custom",
                        "tokenizer": "path_hierarchy",
                        "filter": ["lowercase"]
                    }
                },
                "filter": {
                    "camelcase_filter": {
                        "type": "word_delimiter",
                        "split_on_case_change": True,
                        "split_on_numerics": True,
                        "stem_english_possessive": False
                    },
                    "underscore_filter": {
                        "type": "word_delimiter",
                        "split_on_case_change": False,
                        "split_on_numerics": True,
                        "stem_english_possessive": False
                    }
                }
            }
        }

# Search Query Engine Pattern
class SearchQueryEngine:
    """Provides comprehensive search capabilities for code discovery"""
    
    def __init__(self, client: ElasticsearchClient):
        self.client = client
        self.query_cache: Dict[str, SearchResult] = {}
    
    async def search_code(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        highlight: bool = True,
        limit: int = 10
    ) -> SearchResult:
        """Execute full-text search on code documents"""
        search_body = {
            "query": self._build_search_query(query, filters),
            "size": limit,
            "sort": [{"_score": {"order": "desc"}}]
        }
        
        if highlight:
            search_body["highlight"] = {
                "fields": {
                    "content": {"fragment_size": 200, "number_of_fragments": 3},
                    "functions.name": {},
                    "classes.name": {}
                }
            }
        
        response = await self.client.search(body=search_body)
        
        return SearchResult(
            hits=[self._convert_hit(hit) for hit in response["hits"]["hits"]],
            total_hits=response["hits"]["total"]["value"],
            query_time=response["took"],
            aggregations=response.get("aggregations"),
            suggestions=None
        )
    
    def _build_search_query(self, query: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build Elasticsearch query with proper boosting and filtering"""
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "content^1.0",
                        "functions.name^3.0",
                        "classes.name^3.0",
                        "file_path^2.0"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
        ]
        
        filter_clauses = []
        
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {field: value}})
                else:
                    filter_clauses.append({"term": {field: value}})
        
        return {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }
```

## Acceptance Criteria

### Functional Requirements
- [ ] **Document Indexing**: Index AST data as searchable documents with proper field mapping
- [ ] **Full-Text Search**: Provide fast full-text search across code content and documentation
- [ ] **Advanced Search**: Support fuzzy search, faceted search, and complex query combinations
- [ ] **Analytics**: Execute aggregation queries for code metrics and trend analysis
- [ ] **Real-Time Indexing**: Index documents with near real-time search availability
- [ ] **Bulk Operations**: Efficient bulk indexing using Elasticsearch bulk API
- [ ] **Integration**: Full implementation of DatabaseBackend interface for plugin compatibility
- [ ] **Custom Analyzers**: Code-specific text analysis for optimal search experience

### Performance Requirements
- [ ] **Indexing Throughput**: Index >2000 AST documents per minute with real-time availability
- [ ] **Search Performance**: Execute full-text searches in <100ms on millions of documents
- [ ] **Bulk Performance**: Index 10,000+ documents per minute using bulk operations
- [ ] **Analytics Performance**: Execute aggregation queries in <500ms on large datasets
- [ ] **Storage Efficiency**: <200MB index storage per 10,000 code files

### Quality Requirements
- [ ] **Test Coverage**: Achieve 90%+ test coverage with unit, integration, and performance tests
- [ ] **Code Quality**: Pass all linting and type checking requirements
- [ ] **Documentation**: Complete search query documentation and analytics examples
- [ ] **Logging**: Comprehensive logging with search operation metrics and cluster health
- [ ] **Monitoring**: Elasticsearch cluster health monitoring and performance metrics

### Integration Requirements
- [ ] **Backend Plugin Integration**: Full compliance with DatabaseBackend interface
- [ ] **Configuration System**: Integration with project-wide configuration management
- [ ] **Pipeline Integration**: Seamless integration with transform phase and multi-backend coordinator
- [ ] **Cluster Integration**: Support for distributed Elasticsearch deployments
- [ ] **Monitoring Integration**: Health checks and metrics for operational monitoring

## Priority Guidelines

**Critical**: Document indexing, full-text search, bulk operations, backend interface compliance
**High**: Performance optimization, analytics aggregations, custom analyzers, search relevance
**Medium**: Advanced search features, real-time analytics, dashboard integration, search suggestions
**Low**: Custom scoring, advanced aggregations, search UI components, optimization edge cases

**Focus**: Create a robust, high-performance Elasticsearch backend that enables powerful code search and analytics while maintaining the performance and reliability standards required for production ETL pipelines.
