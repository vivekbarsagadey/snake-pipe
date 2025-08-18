# Task 012: PostgreSQL Backend Implementation

## Research Summary

**Key Findings**: 
- PostgreSQL provides robust ACID transaction support and mature ecosystem for relational data storage
- JSON columns with GIN indexes enable efficient storage and querying of semi-structured AST data
- PostgreSQL's foreign key constraints and referential integrity ensure data consistency across related tables
- Connection pooling and prepared statements are essential for high-throughput write operations
- Partitioning strategies can optimize query performance for large datasets with time-series characteristics

**Technical Analysis**: 
- asyncpg driver provides high-performance async PostgreSQL operations with native protocol support
- Schema design must balance normalization for consistency with denormalization for query performance
- Table structure: Files, Functions, Classes, Dependencies, Metrics with appropriate indexes
- JSON storage for complex AST structures with relational metadata for efficient queries
- Bulk insert operations with COPY protocol for maximum throughput performance

**Architecture Impact**: 
- PostgreSQL backend provides ACID guarantees and referential integrity for critical AST metadata
- Relational structure enables complex analytical queries and reporting capabilities
- Integration with existing PostgreSQL infrastructure and tooling ecosystem
- Support for advanced features like materialized views and stored procedures for analytics

**Risk Assessment**: 
- **Performance Risk**: Large JSON documents may impact query performance - mitigated by selective indexing and partitioning
- **Scalability Risk**: Single-node PostgreSQL may become bottleneck - addressed with read replicas and connection pooling
- **Complexity Risk**: Schema evolution complexity for large datasets - managed with migration scripts and versioning

## Business Context

**User Problem**: Development teams and data analysts need reliable, queryable storage of AST metadata and metrics with strong consistency guarantees and analytical capabilities.

**Business Value**: 
- **Data Integrity**: ACID transactions ensure consistent AST metadata across complex operations
- **Analytical Queries**: SQL enables complex analysis and reporting on code metrics and trends
- **Ecosystem Integration**: Leverage existing PostgreSQL tools and infrastructure investments
- **Compliance**: Meet data governance and audit requirements with transaction logging

**User Persona**: Data Engineers (40%) requiring reliable data storage, Analysts (35%) needing SQL analytics capabilities, DevOps Teams (25%) managing PostgreSQL infrastructure.

**Success Metric**: 
- Write throughput: >5000 AST records per minute with full ACID guarantees
- Query performance: <100ms for typical analytical queries on indexed columns
- Data consistency: 100% referential integrity maintenance across all operations
- System availability: 99.9% uptime with automated failover and backup

## User Story

As a **data engineer**, I want **reliable PostgreSQL storage of AST metadata and metrics** so that **I can ensure data consistency and enable powerful SQL-based analysis of code evolution**.

## Technical Overview

**Task Type**: Story Task (Database Backend Implementation)
**Pipeline Stage**: Load (Relational Database Integration)
**Complexity**: Medium-High
**Dependencies**: Backend plugin architecture (TASK-010), AST data transformation
**Performance Impact**: PostgreSQL must support high-throughput writes without blocking pipeline

## Implementation Requirements

### Files to Modify/Create

- `snake_pipe/load/postgresql_backend.py` (main PostgreSQL backend implementation)
- `snake_pipe/load/postgresql_config.py` (PostgreSQL-specific configuration management)
- `snake_pipe/load/postgresql_schema.py` (database schema definition and migrations)
- `snake_pipe/load/postgresql_queries.py` (optimized query templates and builders)
- `snake_pipe/load/relational_mapper.py` (AST to relational structure mapping logic)
- `snake_pipe/utils/postgresql_client.py` (PostgreSQL client wrapper with connection pooling)
- `snake_pipe/utils/sql_utils.py` (SQL utilities and query optimization helpers)
- `migrations/postgresql/` (directory for PostgreSQL schema migrations and upgrades)
- `tests/unit/load/test_postgresql_backend.py` (comprehensive unit tests with mocked PostgreSQL)
- `tests/integration/load/test_postgresql_integration.py` (integration tests with real PostgreSQL instance)
- `tests/performance/test_postgresql_performance.py` (performance testing for bulk operations)
- `tests/tasks/test_task012_verification.py` (task-specific verification and acceptance tests)
- `tests/tasks/test_task012_integration.py` (end-to-end PostgreSQL integration tests)

### Key Functions to Implement

```python
async def write_ast_to_postgresql(
    ast_data: TransformedASTData, 
    config: PostgreSQLConfig
) -> RelationalWriteResult:
    """
    Purpose: Write AST data to PostgreSQL with proper normalization and referential integrity
    Input: Transformed AST data and PostgreSQL configuration
    Output: Write result with record counts and transaction details
    Performance: Process 200+ AST files per minute with full ACID guarantees
    """

async def bulk_insert_ast_records(
    records: List[ASTRecord], 
    client: PostgreSQLClient
) -> BulkInsertResult:
    """
    Purpose: Efficiently bulk insert AST records using COPY protocol
    Input: List of AST records and PostgreSQL client
    Output: Bulk insert result with performance metrics
    Performance: Insert 50,000+ records per minute with transaction safety
    """

async def execute_analytical_query(
    query_config: AnalyticalQueryConfig, 
    client: PostgreSQLClient
) -> QueryResult:
    """
    Purpose: Execute complex analytical queries on AST data
    Input: Query configuration and PostgreSQL client
    Output: Query result with analytical data and performance metrics
    Performance: <100ms for typical analytical queries on indexed data
    """

async def maintain_referential_integrity(
    operation: DatabaseOperation, 
    client: PostgreSQLClient
) -> IntegrityResult:
    """
    Purpose: Ensure referential integrity across related AST tables
    Input: Database operation and PostgreSQL client
    Output: Integrity validation result with constraint checks
    Performance: <50ms for integrity validation on typical operations
    """

class PostgreSQLBackend(DatabaseBackend):
    """
    Purpose: PostgreSQL implementation of database backend interface
    Features: ACID transactions, referential integrity, analytical queries
    Performance: High-throughput relational operations with connection pooling
    """
    
    async def connect(self, config: PostgreSQLConfig) -> None:
        """Initialize connection to PostgreSQL with proper authentication and pooling"""
    
    async def write_batch(self, data: List[TransformedASTData]) -> WriteResult:
        """Write batch of AST data to PostgreSQL with transaction safety"""
    
    async def health_check(self) -> HealthCheckResult:
        """Check PostgreSQL health including connection pool and transaction log"""
    
    async def execute_sql_query(self, query: SQLQuery) -> QueryResult:
        """Execute custom SQL queries for analysis and reporting"""
```

### Technical Requirements

1. **Performance**: 
   - Write throughput: >5000 AST records per minute with full ACID guarantees
   - Bulk insert performance: 50,000+ records per minute using COPY protocol
   - Query performance: <100ms for typical analytical queries on indexed columns
   - Connection efficiency: Connection pooling with <5ms connection acquisition

2. **Data Integrity**: 
   - 100% referential integrity maintenance across all table relationships
   - ACID transaction support for all write operations
   - Foreign key constraints enforcement for data consistency
   - Comprehensive data validation at database schema level

3. **Scalability**: 
   - Read replica support for analytical workloads and reporting
   - Connection pooling to optimize database connection utilization
   - Table partitioning for large datasets with time-series characteristics
   - Efficient indexing strategy for high-performance queries

4. **Integration**: 
   - Implementation of DatabaseBackend interface for plugin compatibility
   - Configuration-driven behavior with environment-specific database settings
   - Integration with monitoring for database operation metrics and health
   - Support for schema migrations and version management

5. **Query Optimization**: 
   - Optimized indexes for common query patterns and analytical workloads
   - Prepared statement caching for frequently executed queries
   - Query plan optimization and performance monitoring
   - Support for materialized views for complex analytical queries

6. **Reliability**: 
   - Automatic retry mechanisms for transient database connection issues
   - Health monitoring and connection pool management
   - Backup and recovery integration with PostgreSQL tooling
   - Transaction log monitoring and deadlock detection

### Implementation Steps

1. **Schema Design**: Define comprehensive relational schema for AST data with proper normalization
2. **Connection Management**: Implement PostgreSQL client wrapper with connection pooling and error handling
3. **Backend Implementation**: Create PostgreSQL backend following DatabaseBackend interface
4. **Data Mapping**: Develop sophisticated AST to relational structure mapping
5. **Query Optimization**: Build optimized query templates and index strategies
6. **Bulk Operations**: Implement efficient bulk insert operations using COPY protocol
7. **Migration System**: Create schema migration system for database evolution
8. **Monitoring Integration**: Add comprehensive monitoring and performance metrics
9. **Testing**: Create extensive unit, integration, and performance tests
10. **Documentation**: Develop schema documentation and query examples

### Code Patterns

```python
# PostgreSQL Backend Pattern (following project conventions)
@dataclass
class PostgreSQLConfig:
    """Configuration for PostgreSQL backend operations"""
    host: str = "localhost"
    port: int = 5432
    database: str = "ast_analysis"
    username: str = "postgres"
    password: str = ""
    connection_pool_size: int = 20
    max_overflow: int = 10
    timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_ssl: bool = True
    schema_name: str = "public"

@dataclass
class ASTRecord:
    """Representation of AST data for relational storage"""
    file_id: str
    file_path: str
    language: str
    ast_content: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class RelationalWriteResult:
    """Result of relational database write operations"""
    success: bool
    records_written: int
    tables_affected: List[str]
    operation_time: float
    transaction_id: Optional[str]

class PostgreSQLBackend(DatabaseBackend):
    """High-performance PostgreSQL backend with ACID guarantees"""
    
    def __init__(self, config: PostgreSQLConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.client: Optional[PostgreSQLClient] = None
        self.schema_manager = PostgreSQLSchemaManager(config)
        self.relational_mapper = RelationalMapper()
        self.stats = PostgreSQLStatistics()
    
    async def connect(self, config: PostgreSQLConfig) -> None:
        """Initialize connection to PostgreSQL with proper authentication"""
        try:
            self.client = PostgreSQLClient(config)
            await self.client.connect()
            
            # Initialize database schema
            await self.schema_manager.ensure_schema_exists(self.client)
            
            self.logger.info(f"Connected to PostgreSQL: {config.host}:{config.port}/{config.database}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise DatabaseConnectionError(f"PostgreSQL connection failed: {e}")
    
    async def write_batch(self, data: List[TransformedASTData]) -> WriteResult:
        """Write batch of AST data to PostgreSQL with transaction safety"""
        if not self.client:
            raise DatabaseConnectionError("PostgreSQL client not connected")
        
        try:
            async with self.client.transaction() as tx:
                # Convert AST data to relational records
                relational_records = await self._transform_to_relational(data)
                
                # Bulk insert with transaction safety
                write_result = await self._bulk_insert_records(relational_records, tx)
                
                # Validate referential integrity
                await self._validate_referential_integrity(relational_records, tx)
                
                # Update statistics
                self.stats.total_writes += len(data)
                self.stats.records_inserted += write_result.records_written
                
                return WriteResult(
                    success=True,
                    items_written=len(data),
                    operation_time=write_result.operation_time,
                    backend_specific_metrics=write_result.metrics
                )
                
        except Exception as e:
            self.logger.error(f"Failed to write batch to PostgreSQL: {e}")
            self.stats.write_errors += 1
            raise DatabaseWriteError(f"PostgreSQL write failed: {e}")
    
    async def _transform_to_relational(self, data: List[TransformedASTData]) -> List[RelationalRecord]:
        """Transform AST data to relational records for PostgreSQL storage"""
        relational_records = []
        
        for ast_data in data:
            # Create file record
            file_record = RelationalRecord(
                table="files",
                data={
                    "file_id": ast_data.file_id,
                    "file_path": ast_data.file_path,
                    "language": ast_data.language,
                    "size_bytes": ast_data.size_bytes,
                    "ast_content": json.dumps(ast_data.ast_content),
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            )
            relational_records.append(file_record)
            
            # Create function records
            for function in ast_data.functions:
                function_record = RelationalRecord(
                    table="functions",
                    data={
                        "function_id": function.function_id,
                        "file_id": ast_data.file_id,
                        "name": function.name,
                        "signature": function.signature,
                        "line_start": function.line_start,
                        "line_end": function.line_end,
                        "complexity": function.complexity,
                        "parameter_count": len(function.parameters),
                        "created_at": datetime.utcnow()
                    }
                )
                relational_records.append(function_record)
            
            # Create class records
            for class_info in ast_data.classes:
                class_record = RelationalRecord(
                    table="classes",
                    data={
                        "class_id": class_info.class_id,
                        "file_id": ast_data.file_id,
                        "name": class_info.name,
                        "namespace": class_info.namespace,
                        "line_start": class_info.line_start,
                        "line_end": class_info.line_end,
                        "is_abstract": class_info.is_abstract,
                        "method_count": len(class_info.methods),
                        "created_at": datetime.utcnow()
                    }
                )
                relational_records.append(class_record)
            
            # Create dependency records
            for dependency in ast_data.dependencies:
                dependency_record = RelationalRecord(
                    table="dependencies",
                    data={
                        "source_file_id": ast_data.file_id,
                        "target_file_id": dependency.target_file_id,
                        "dependency_type": dependency.dependency_type.value,
                        "strength": dependency.strength,
                        "line_number": dependency.line_number,
                        "created_at": datetime.utcnow()
                    }
                )
                relational_records.append(dependency_record)
        
        return relational_records
    
    async def _bulk_insert_records(
        self, 
        records: List[RelationalRecord], 
        transaction: Transaction
    ) -> BulkInsertResult:
        """Efficiently bulk insert records using COPY protocol"""
        start_time = time.time()
        records_by_table = self._group_records_by_table(records)
        total_inserted = 0
        
        for table_name, table_records in records_by_table.items():
            # Use COPY for maximum performance
            inserted_count = await self._copy_records_to_table(
                table_name, table_records, transaction
            )
            total_inserted += inserted_count
        
        operation_time = time.time() - start_time
        
        return BulkInsertResult(
            records_written=total_inserted,
            tables_affected=list(records_by_table.keys()),
            operation_time=operation_time,
            throughput=total_inserted / operation_time if operation_time > 0 else 0
        )

# Schema Management Pattern
class PostgreSQLSchemaManager:
    """Manages PostgreSQL schema for AST data storage"""
    
    def __init__(self, config: PostgreSQLConfig):
        self.config = config
        self.schema_version = "1.0"
    
    async def ensure_schema_exists(self, client: PostgreSQLClient) -> None:
        """Ensure database schema exists with proper tables and indexes"""
        # Create tables if not exist
        await self._create_tables(client)
        
        # Create indexes for performance
        await self._create_indexes(client)
        
        # Create foreign key constraints
        await self._create_foreign_keys(client)
        
        # Create views for common queries
        await self._create_views(client)
    
    async def _create_tables(self, client: PostgreSQLClient) -> None:
        """Create all required tables for AST data storage"""
        table_definitions = {
            "files": """
                CREATE TABLE IF NOT EXISTS files (
                    file_id VARCHAR(255) PRIMARY KEY,
                    file_path TEXT NOT NULL UNIQUE,
                    language VARCHAR(50) NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    ast_content JSONB,
                    hash_md5 VARCHAR(32),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """,
            "functions": """
                CREATE TABLE IF NOT EXISTS functions (
                    function_id VARCHAR(255) PRIMARY KEY,
                    file_id VARCHAR(255) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    signature TEXT,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    complexity INTEGER DEFAULT 0,
                    parameter_count INTEGER DEFAULT 0,
                    return_type VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
                )
            """,
            "classes": """
                CREATE TABLE IF NOT EXISTS classes (
                    class_id VARCHAR(255) PRIMARY KEY,
                    file_id VARCHAR(255) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    namespace VARCHAR(255),
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    is_abstract BOOLEAN DEFAULT FALSE,
                    method_count INTEGER DEFAULT 0,
                    parent_class_id VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_class_id) REFERENCES classes(class_id)
                )
            """,
            "dependencies": """
                CREATE TABLE IF NOT EXISTS dependencies (
                    dependency_id SERIAL PRIMARY KEY,
                    source_file_id VARCHAR(255) NOT NULL,
                    target_file_id VARCHAR(255) NOT NULL,
                    dependency_type VARCHAR(50) NOT NULL,
                    strength DECIMAL(3,2) DEFAULT 1.0,
                    line_number INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    FOREIGN KEY (source_file_id) REFERENCES files(file_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_file_id) REFERENCES files(file_id) ON DELETE CASCADE,
                    UNIQUE(source_file_id, target_file_id, dependency_type)
                )
            """,
            "metrics": """
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id SERIAL PRIMARY KEY,
                    file_id VARCHAR(255) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DECIMAL(10,2) NOT NULL,
                    metric_type VARCHAR(50) NOT NULL,
                    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    FOREIGN KEY (file_id) REFERENCES files(file_id) ON DELETE CASCADE
                )
            """
        }
        
        for table_name, definition in table_definitions.items():
            await client.execute(definition)
    
    async def _create_indexes(self, client: PostgreSQLClient) -> None:
        """Create indexes for optimal query performance"""
        index_definitions = [
            "CREATE INDEX IF NOT EXISTS idx_files_language ON files(language)",
            "CREATE INDEX IF NOT EXISTS idx_files_path ON files USING gin(to_tsvector('english', file_path))",
            "CREATE INDEX IF NOT EXISTS idx_files_ast_content ON files USING gin(ast_content)",
            "CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name)",
            "CREATE INDEX IF NOT EXISTS idx_functions_file_id ON functions(file_id)",
            "CREATE INDEX IF NOT EXISTS idx_classes_name ON classes(name)",
            "CREATE INDEX IF NOT EXISTS idx_classes_file_id ON classes(file_id)",
            "CREATE INDEX IF NOT EXISTS idx_dependencies_source ON dependencies(source_file_id)",
            "CREATE INDEX IF NOT EXISTS idx_dependencies_target ON dependencies(target_file_id)",
            "CREATE INDEX IF NOT EXISTS idx_dependencies_type ON dependencies(dependency_type)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_file_name ON metrics(file_id, metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_calculated ON metrics(calculated_at)"
        ]
        
        for index_definition in index_definitions:
            await client.execute(index_definition)

# Relational Mapping Pattern
class RelationalMapper:
    """Maps AST data to relational database structures"""
    
    def __init__(self):
        self.mapping_cache: Dict[str, TableMapping] = {}
    
    async def map_ast_to_tables(self, ast_data: TransformedASTData) -> List[TableRecord]:
        """Map AST data to appropriate database tables"""
        table_records = []
        
        # Map file information
        file_record = self._map_file_data(ast_data)
        table_records.append(file_record)
        
        # Map functions
        for function in ast_data.functions:
            function_record = self._map_function_data(function, ast_data.file_id)
            table_records.append(function_record)
        
        # Map classes
        for class_info in ast_data.classes:
            class_record = self._map_class_data(class_info, ast_data.file_id)
            table_records.append(class_record)
        
        # Map dependencies
        for dependency in ast_data.dependencies:
            dependency_record = self._map_dependency_data(dependency, ast_data.file_id)
            table_records.append(dependency_record)
        
        # Map metrics
        for metric in ast_data.metrics:
            metric_record = self._map_metric_data(metric, ast_data.file_id)
            table_records.append(metric_record)
        
        return table_records
    
    def _map_file_data(self, ast_data: TransformedASTData) -> TableRecord:
        """Map AST file data to files table record"""
        return TableRecord(
            table="files",
            operation="INSERT",
            data={
                "file_id": ast_data.file_id,
                "file_path": ast_data.file_path,
                "language": ast_data.language,
                "size_bytes": ast_data.size_bytes,
                "ast_content": ast_data.ast_content,
                "hash_md5": ast_data.hash_md5,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            primary_key="file_id"
        )
```

## Acceptance Criteria

### Functional Requirements
- [ ] **Relational Schema**: Implement comprehensive relational schema for AST data with proper normalization
- [ ] **ACID Transactions**: Support full ACID transaction guarantees for all write operations
- [ ] **Bulk Operations**: Efficient bulk insert using COPY protocol for high-throughput scenarios
- [ ] **Query Support**: Optimized SQL queries for analytical workloads and reporting
- [ ] **Referential Integrity**: Maintain foreign key constraints and data consistency
- [ ] **Schema Migrations**: Support database schema evolution and version management
- [ ] **Integration**: Full implementation of DatabaseBackend interface for plugin compatibility
- [ ] **Connection Pooling**: Efficient connection pool management with health monitoring

### Performance Requirements
- [ ] **Write Throughput**: Process >5000 AST records per minute with full ACID guarantees
- [ ] **Bulk Performance**: Insert 50,000+ records per minute using COPY protocol
- [ ] **Query Performance**: Execute analytical queries in <100ms on indexed columns
- [ ] **Connection Efficiency**: <5ms connection acquisition from connection pool
- [ ] **Memory Usage**: <1GB RAM for typical bulk insert and query operations

### Quality Requirements
- [ ] **Test Coverage**: Achieve 90%+ test coverage with unit, integration, and performance tests
- [ ] **Code Quality**: Pass all linting and type checking requirements
- [ ] **Documentation**: Complete schema documentation and query optimization guides
- [ ] **Logging**: Comprehensive logging with SQL operation metrics and error details
- [ ] **Monitoring**: Database health monitoring and performance metrics

### Integration Requirements
- [ ] **Backend Plugin Integration**: Full compliance with DatabaseBackend interface
- [ ] **Configuration System**: Integration with project-wide configuration management
- [ ] **Pipeline Integration**: Seamless integration with transform phase and multi-backend coordinator
- [ ] **Monitoring Integration**: Health checks and metrics for operational monitoring
- [ ] **Migration Integration**: Schema migration tools and version management

## Priority Guidelines

**Critical**: Relational schema design, ACID transaction support, bulk insert operations, backend interface compliance
**High**: Query optimization, connection pooling, referential integrity, performance tuning
**Medium**: Advanced SQL features, materialized views, read replicas, advanced analytics
**Low**: Custom functions, advanced indexing strategies, PostgreSQL extensions, optimization edge cases

**Focus**: Create a robust, high-performance PostgreSQL backend that provides ACID guarantees and powerful analytical capabilities while maintaining the performance and reliability standards required for production ETL pipelines.
