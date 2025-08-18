---
applyTo: "**"
---

# Python Code Guidelines and Review Standards - Snake-Pipe AST Processing ETL Pipeline

## 🚨 MANDATORY TESTING REQUIREMENT
**ALL Python code submissions MUST include comprehensive test cases. This is a non-negotiable requirement.**

> **📋 For complete testing guidelines and structure, see [test-case.instructions.md](./test-case.instructions.md)**

- ✅ **Required:** Unit tests for all ETL pipeline functions, methods, and classes
- ✅ **Required:** Integration tests for database operations and multi-backend coordination
- ✅ **Required:** End-to-end tests for complete AST processing workflows
- ✅ **Required:** Performance tests for throughput and scalability validation
- ✅ **Required:** Edge case and error condition testing (malformed JSON, database failures, file system issues)
- ✅ **Required:** Minimum 90% test coverage for new code (ETL pipelines require extensive testing)
- ❌ **Not Acceptable:** Code submissions without comprehensive tests

**Code reviews will be rejected if comprehensive tests are not included.**

## 🎯 Snake-Pipe ETL Pipeline Project-Specific Requirements

### Core Architecture Patterns (MANDATORY)
- **Clean Architecture:** Implement clear separation between domain, application, and infrastructure layers
- **Plugin Architecture:** Use ABC for defining extensible database backend interfaces
- **Factory Pattern:** Implement factories for creating database backends and processing strategies
- **Strategy Pattern:** Use for different ETL processing strategies and validation approaches
- **Dependency Injection:** All services must use dependency injection for testability and modularity
- **Async/Await Patterns:** Use async programming for I/O-bound operations and concurrent processing
- **Enums for Constants:** All constants MUST be defined as Enums (processing modes, error codes, backend types)

## 🚨 MANDATORY DOCUMENTATION AND PROJECT TRACKING REQUIREMENTS
**ALL code changes MUST include proper documentation updates and project tracking. This is a non-negotiable requirement.**

### Essential Documentation Files (MANDATORY)
- ✅ **Required:** Update ProjectDetails.md with any structural, dependency, or architectural changes
- ✅ **Required:** Keep requirements.txt and pyproject.toml up-to-date with all dependencies
- ✅ **Required:** Maintain current README.md with setup and usage instructions
- ✅ **Required:** Document all changes in CHANGELOG.md
- ✅ **Required:** Follow CONTRIBUTING.md guidelines for all contributions
- ✅ **Required:** All code changes must undergo peer review process
- ✅ **Required:** Update API documentation for new endpoints and interfaces
- ✅ **Required:** Document database schema changes and migration scripts
- ❌ **Not Acceptable:** Code changes without corresponding documentation updates
- ❌ **Not Acceptable:** Outdated dependency files or project documentation
- ❌ **Not Acceptable:** Missing changelog entries for significant changes
- ❌ **Not Acceptable:** Undocumented database backend additions or configuration changes

**Code reviews will be rejected if documentation requirements are not met.**

## 1. Code Style and Formatting (PEP 8)

### Essential Style Rules
- **Indentation:** Always use 4 spaces per level (never tabs)
- **Line Length:** Maximum 120 characters per line (appropriate for data processing code)
- **Naming Conventions:**
  - Variables/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants/Enums: `UPPER_SNAKE_CASE`
  - Database tables/columns: `snake_case` in schema definitions
  - Configuration keys: `snake_case` in config files
- **Blank Lines:** 
  - Two blank lines between top-level functions and classes
  - One blank line between methods within a class
- **Import Organization:**
  1. Standard library imports
  2. Third-party library imports (pydantic, asyncio, pandas, etc.)
  3. Database-specific imports (NebulaGraph, PostgreSQL, Elasticsearch clients)
  4. Local application imports
  - Separate each group with blank lines

### ETL Pipeline Specific Constants (MANDATORY)
All constants MUST be defined as Enums:

```python
from enum import Enum, auto

class ProcessingMode(Enum):
    """ETL processing modes."""
    BATCH = auto()
    STREAMING = auto()
    REAL_TIME = auto()

class BackendType(Enum):
    """Database backend types."""
    NEBULA_GRAPH = "nebula_graph"
    POSTGRESQL = "postgresql"
    VECTOR_DB = "vector_db"
    ELASTICSEARCH = "elasticsearch"

class ValidationLevel(Enum):
    """Data validation levels."""
    STRICT = "strict"
    LENIENT = "lenient"
    SKIP = "skip"

class ErrorCode(Enum):
    """ETL processing error codes."""
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    MALFORMED_JSON = "MALFORMED_JSON"
    ENRICHMENT_FAILED = "ENRICHMENT_FAILED"
    BACKEND_WRITE_ERROR = "BACKEND_WRITE_ERROR"

class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    QUARANTINE = "quarantine"
```

### Documentation Standards
- **Docstrings:** Use Google, NumPy, or Sphinx style for all functions, classes, and modules
- **Comments:** Explain WHY, not WHAT. Focus on business logic and non-obvious decisions
- **Comment When:**
  - Complex ETL processing logic and data transformation algorithms
  - Performance optimizations for high-throughput operations
  - Error handling strategies for database failures and data corruption
  - Cross-database coordination and transaction management
  - AST data enrichment and relationship mapping logic
  - Plugin configuration and backend selection decisions

### ETL Pipeline Code Organization
- **Single Responsibility:** Functions and classes should have one clear purpose
  - Separate extraction logic from transformation and loading
  - Isolate database backend implementations from coordination logic
  - Keep configuration management separate from business logic
- **Module Structure:** Group related functionality logically
  - `extract/` - File monitoring and data discovery implementations
  - `transform/` - Validation, normalization, and enrichment services
  - `load/` - Database coordination and backend write management
  - `config/` - Configuration management and settings
  - `utils/` - Shared utilities, logging, and helper functions
- **Package Structure:** Use `__init__.py` files properly for clean imports
- **Avoid Deep Nesting:** Keep indentation levels minimal in async operations and data processing loops

## 2. Correctness and Logic

### Functional Requirements
- **Requirements Compliance:** Code must correctly implement XML parsing specifications
- **Edge Case Handling:** Handle malformed XML, empty files, large files, and encoding issues
- **Data Validation:** Validate XML input before processing with lxml

### Error Handling Best Practices for ETL Processing
- **Specific Exceptions:** Catch specific exceptions rather than broad `except Exception:` blocks
  - `JSONDecodeError` for malformed AST JSON files
  - `ValidationError` for schema validation failures
  - `ConnectionError` for database connection issues
  - `FileNotFoundError` for missing input files
  - `PermissionError` for file access issues
  - `TimeoutError` for database operation timeouts
- **Graceful Degradation:** Provide informative error messages and handle expected failures gracefully
- **Resource Management:** Use `with` statements for files, database connections, and async context managers

### Algorithm Efficiency for ETL Processing
- **Async Processing:** Always use `async/await` for I/O-bound operations (file reading, database writes)
- **Memory Management:** Use streaming for large datasets and batch processing for efficiency
- **Data Structures:** Use appropriate data structures:
  - `dict` for AST nodes and configuration mappings
  - `list` for batch processing collections
  - `set` for deduplication and fast lookups
  - `collections.deque` for efficient queue operations
- **Performance:** Avoid redundant database calls and unnecessary data transformations

## 3. Design Principles and Architecture

### SOLID Principles (With Practical Examples)

#### Single Responsibility Principle (SRP) - "You had one job"
**A class should have only one reason to change and one responsibility.**

```python
# ❌ BAD: Violates SRP - Animal class has multiple responsibilities
class Animal:
    def __init__(self, name: str):
        self.name = name
    
    def get_name(self) -> str:
        return self.name
    
    def save(self, animal) -> None:
        # Database management - different responsibility
        # Save animal to database
        pass

# ✅ GOOD: Separates concerns
class Animal:
    def __init__(self, name: str):
        self.name = name
    
    def get_name(self) -> str:
        return self.name

class AnimalRepository:
    def save(self, animal: Animal) -> None:
        # Handle database operations
        pass
    
    def get_animal(self, name: str) -> Animal:
        # Retrieve animal from database
        pass
```

#### Open/Closed Principle (OCP) - Open for extension, closed for modification
**Software entities should be open for extension but closed for modification.**

```python
# ❌ BAD: Must modify function for each new animal
def animal_sound(animals: list):
    for animal in animals:
        if animal.name == 'lion':
            print('roar')
        elif animal.name == 'mouse':
            print('squeak')
        elif animal.name == 'snake':  # Added new case - modifying existing code
            print('hiss')

# ✅ GOOD: Extensible without modification
class Animal:
    def make_sound(self):
        raise NotImplementedError

class Lion(Animal):
    def make_sound(self):
        return 'roar'

class Mouse(Animal):
    def make_sound(self):
        return 'squeak'

class Snake(Animal):  # New animal - no modification of existing code
    def make_sound(self):
        return 'hiss'

def animal_sound(animals: list):
    for animal in animals:
        print(animal.make_sound())

# Discount system example
class Discount:
    def __init__(self, customer, price):
        self.customer = customer
        self.price = price
    
    def get_discount(self):
        return self.price * 0.2

class VIPDiscount(Discount):
    def get_discount(self):
        return self.price * 0.4

class SuperVIPDiscount(Discount):
    def get_discount(self):
        return self.price * 0.8
```

#### Liskov Substitution Principle (LSP) - Substitutability without breaking functionality
**A sub-class must be substitutable for its super-class without errors.**

```python
# ❌ BAD: Function must check types - violates LSP
def animal_leg_count(animals: list):
    for animal in animals:
        if isinstance(animal, Lion):
            print(lion_leg_count(animal))
        elif isinstance(animal, Mouse):
            print(mouse_leg_count(animal))
        elif isinstance(animal, Pigeon):
            print(pigeon_leg_count(animal))

# ✅ GOOD: Subtypes are substitutable for their base type
class Animal:
    def leg_count(self):
        raise NotImplementedError

class Lion(Animal):
    def leg_count(self):
        return 4

class Mouse(Animal):
    def leg_count(self):
        return 4

class Pigeon(Animal):
    def leg_count(self):
        return 2

def animal_leg_count(animals: list):
    for animal in animals:
        print(animal.leg_count())  # Works with any Animal subtype
```

#### Interface Segregation Principle (ISP) - Fine-grained interfaces
**Make fine-grained interfaces that are client-specific. Clients shouldn't depend on unused interfaces.**

```python
# ❌ BAD: Fat interface - clients forced to implement unused methods
class IShape:
    def draw_square(self):
        raise NotImplementedError
    
    def draw_rectangle(self):
        raise NotImplementedError
    
    def draw_circle(self):
        raise NotImplementedError

class Circle(IShape):
    def draw_square(self):
        pass  # Unused - forced to implement
    
    def draw_rectangle(self):
        pass  # Unused - forced to implement
    
    def draw_circle(self):
        # Only this method is actually needed
        pass

# ✅ GOOD: Segregated interfaces
class IShape:
    def draw(self):
        raise NotImplementedError

class Circle(IShape):
    def draw(self):
        # Draw circle implementation
        pass

class Square(IShape):
    def draw(self):
        # Draw square implementation
        pass
```

#### Dependency Inversion Principle (DIP) - Depend on abstractions
**High-level modules shouldn't depend on low-level modules. Both should depend on abstractions.**

```python
# ❌ BAD: High-level Http class depends on concrete XMLHttpService
class XMLHttpService:
    def request(self, url: str, method: str):
        # XML HTTP implementation
        pass

class Http:
    def __init__(self):
        self.xml_http_service = XMLHttpService()  # Concrete dependency
    
    def get(self, url: str):
        self.xml_http_service.request(url, 'GET')

# ✅ GOOD: Both depend on abstraction
from abc import ABC, abstractmethod

class Connection(ABC):
    @abstractmethod
    def request(self, url: str, method: str):
        pass

class XMLHttpService(Connection):
    def request(self, url: str, method: str):
        # XML HTTP implementation
        pass

class NodeHttpService(Connection):
    def request(self, url: str, method: str):
        # Node HTTP implementation
        pass

class Http:
    def __init__(self, connection: Connection):
        self.connection = connection  # Depends on abstraction
    
    def get(self, url: str):
        self.connection.request(url, 'GET')
```

### ETL Pipeline Specific Design Patterns (MANDATORY)

#### Abstract Base Classes for ETL Components
**Define clear interfaces for extensible ETL processing components.**

```python
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, List, AsyncIterator
import asyncio

class ProcessingMode(Enum):
    """ETL processing modes for different throughput requirements."""
    BATCH = auto()
    STREAMING = auto()
    REAL_TIME = auto()

class BackendType(Enum):
    """Available database backend types."""
    NEBULA_GRAPH = "nebula_graph"
    POSTGRESQL = "postgresql"
    VECTOR_DB = "vector_db"
    ELASTICSEARCH = "elasticsearch"

class ASTExtractor(ABC):
    """Abstract base class for AST file extractors."""
    
    @abstractmethod
    async def extract_files(self, source_path: str, filter_config: Dict[str, Any]) -> List[str]:
        """Extract AST files from source directory."""
        pass
    
    @abstractmethod
    async def watch_directory(self, source_path: str) -> AsyncIterator[str]:
        """Watch directory for new AST files."""
        pass

class ASTTransformer(ABC):
    """Abstract base class for AST data transformation."""
    
    @abstractmethod
    async def validate_schema(self, ast_data: Dict[str, Any]) -> bool:
        """Validate AST data against schema."""
        pass
    
    @abstractmethod
    async def normalize_data(self, ast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize AST data for consistency."""
        pass
    
    @abstractmethod
    async def enrich_relationships(self, ast_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrich with cross-file relationships."""
        pass

class DatabaseBackend(ABC):
    """Abstract base class for database backends."""
    
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> None:
        """Connect to database backend."""
        pass
    
    @abstractmethod
    async def write_batch(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Write batch of data to backend."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check backend health status."""
        pass
```

#### Factory Pattern for ETL Processing Components
**Use factories to create appropriate processors and backends based on configuration.**

```python
class DatabaseBackendFactory:
    """Factory for creating database backends based on configuration."""
    
    @staticmethod
    def create_backend(backend_type: BackendType, config: Dict[str, Any]) -> DatabaseBackend:
        """Create appropriate database backend based on type and configuration."""
        if backend_type == BackendType.NEBULA_GRAPH:
            return NebulaGraphBackend(config)
        elif backend_type == BackendType.POSTGRESQL:
            return PostgreSQLBackend(config)
        elif backend_type == BackendType.VECTOR_DB:
            return VectorDatabaseBackend(config)
        elif backend_type == BackendType.ELASTICSEARCH:
            return ElasticsearchBackend(config)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

class ProcessorFactory:
    """Factory for creating ETL processors."""
    
    @staticmethod
    def create_transformer(language: str, config: Dict[str, Any]) -> ASTTransformer:
        """Create appropriate transformer based on language and configuration."""
        if language == "python":
            return PythonASTTransformer(config)
        elif language == "java":
            return JavaASTTransformer(config)
        elif language == "javascript":
            return JavaScriptASTTransformer(config)
        else:
            return GenericASTTransformer(config)

class ExtractorFactory:
    """Factory for creating data extractors with different strategies."""
    
    @staticmethod
    def create_extractor(mode: ProcessingMode, config: Dict[str, Any]) -> ASTExtractor:
        """Create extractor based on processing mode."""
        if mode == ProcessingMode.REAL_TIME:
            return RealtimeFileWatcher(config)
        elif mode == ProcessingMode.STREAMING:
            return StreamingExtractor(config)
        else:
            return BatchExtractor(config)
```

#### Strategy Pattern for ETL Processing
**Use strategy pattern for different ETL processing approaches.**

```python
class ETLProcessingStrategy(ABC):
    """Strategy for ETL processing workflows."""
    
    @abstractmethod
    async def process(self, input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ETL processing strategy."""
        pass

class HighThroughputStrategy(ETLProcessingStrategy):
    """High-throughput ETL processing with minimal validation and batch optimization."""
    
    def __init__(self):
        self.extractor = ExtractorFactory.create_extractor(ProcessingMode.BATCH, {})
        self.transformer = ProcessorFactory.create_transformer("generic", {"validation": "lenient"})
        self.loader = DatabaseBackendFactory.create_backend(BackendType.POSTGRESQL, {})
    
    async def process(self, input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for high-throughput processing
        pass

class RealTimeStrategy(ETLProcessingStrategy):
    """Real-time ETL processing with immediate validation and multi-backend writes."""
    
    def __init__(self):
        self.extractor = ExtractorFactory.create_extractor(ProcessingMode.REAL_TIME, {})
        self.transformer = ProcessorFactory.create_transformer("python", {"validation": "strict"})
        self.backends = [
            DatabaseBackendFactory.create_backend(BackendType.NEBULA_GRAPH, {}),
            DatabaseBackendFactory.create_backend(BackendType.VECTOR_DB, {})
        ]
    
    async def process(self, input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for real-time processing
        pass

class QualityFocusedStrategy(ETLProcessingStrategy):
    """Quality-focused ETL processing with comprehensive validation and enrichment."""
    
    def __init__(self):
        self.extractor = ExtractorFactory.create_extractor(ProcessingMode.STREAMING, {})
        self.transformer = ProcessorFactory.create_transformer("java", {"validation": "strict", "enrichment": True})
        self.loader = DatabaseBackendFactory.create_backend(BackendType.ELASTICSEARCH, {})
    
    async def process(self, input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for quality-focused processing
        pass
```

### Error Handling with Enums
**Use enums for consistent error codes and messages.**

```python
class ETLErrorCode(Enum):
    """Error codes for ETL processing operations."""
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    MALFORMED_JSON = "MALFORMED_JSON"
    ENRICHMENT_FAILED = "ENRICHMENT_FAILED"
    BACKEND_WRITE_ERROR = "BACKEND_WRITE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"

class ETLProcessingError(Exception):
    """Custom exception for ETL processing errors."""
    
    def __init__(self, error_code: ETLErrorCode, message: str, source_info: str = None):
        self.error_code = error_code
        self.source_info = source_info
        super().__init__(f"{error_code.value}: {message}")

# Usage in ETL pipeline
async def process_ast_file(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return await process_ast_content(content)
    except FileNotFoundError:
        raise ETLProcessingError(
            ETLErrorCode.FILE_NOT_FOUND,
            f"AST file not found: {file_path}",
            file_path
        )
    except PermissionError:
        raise ETLProcessingError(
            ETLErrorCode.PERMISSION_DENIED,
            f"Permission denied accessing: {file_path}",
            file_path
        )
    except json.JSONDecodeError as e:
        raise ETLProcessingError(
            ETLErrorCode.MALFORMED_JSON,
            f"JSON decode error in AST file: {str(e)}",
            file_path
        )
```

### Comprehensive Design Patterns Library

#### Creational Patterns - Object Creation Mechanisms

##### Abstract Factory Pattern
**Provide an interface for creating families of related objects without specifying their concrete classes.**

```python
from abc import ABC, abstractmethod

class AbstractFactory(ABC):
    @abstractmethod
    def create_product_a(self):
        pass
    
    @abstractmethod
    def create_product_b(self):
        pass

class ConcreteFactory1(AbstractFactory):
    def create_product_a(self):
        return ConcreteProductA1()
    
    def create_product_b(self):
        return ConcreteProductB1()

class ConcreteFactory2(AbstractFactory):
    def create_product_a(self):
        return ConcreteProductA2()
    
    def create_product_b(self):
        return ConcreteProductB2()
```

##### Builder Pattern
**Construct complex objects step by step instead of using multiple constructors.**

```python
class Product:
    def __init__(self):
        self.parts = []
    
    def add_part(self, part):
        self.parts.append(part)

class ProductBuilder:
    def __init__(self):
        self.product = Product()
    
    def add_engine(self, engine):
        self.product.add_part(f"Engine: {engine}")
        return self
    
    def add_wheels(self, wheels):
        self.product.add_part(f"Wheels: {wheels}")
        return self
    
    def build(self):
        return self.product

# Usage
car = (ProductBuilder()
       .add_engine("V8")
       .add_wheels("Racing")
       .build())
```

##### Factory Method Pattern
**Create objects without specifying exact classes using a factory method.**

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        if animal_type.lower() == "dog":
            return Dog()
        elif animal_type.lower() == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")
```

##### Singleton Pattern
**Ensure only one instance exists (use with caution in Python).**

```python
class Singleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Initialize only once
            self._initialized = True

# Python-specific: Use module-level variables instead
class DatabaseConnection:
    def __init__(self):
        self.connection = "Connected"

# Create single instance at module level
database = DatabaseConnection()
```

##### Prototype Pattern
**Create new instances by cloning existing instances (useful for expensive instantiation).**

```python
import copy
from abc import ABC, abstractmethod

class Prototype(ABC):
    @abstractmethod
    def clone(self):
        pass

class ConcretePrototype(Prototype):
    def __init__(self, data):
        self.data = data
    
    def clone(self):
        return copy.deepcopy(self)
```

##### Object Pool Pattern
**Preinstantiate and maintain a group of reusable objects.**

```python
class ObjectPool:
    def __init__(self, create_func, reset_func, initial_size=5):
        self._create_func = create_func
        self._reset_func = reset_func
        self._pool = [create_func() for _ in range(initial_size)]
    
    def acquire(self):
        if self._pool:
            return self._pool.pop()
        return self._create_func()
    
    def release(self, obj):
        self._reset_func(obj)
        self._pool.append(obj)
```

#### Structural Patterns - Object Composition

##### Adapter Pattern
**Allow incompatible interfaces to work together using a white-list approach.**

```python
class OldInterface:
    def old_method(self):
        return "Old interface method"

class NewInterface:
    def new_method(self):
        return "New interface method"

class Adapter:
    def __init__(self, old_object):
        self.old_object = old_object
    
    def new_method(self):
        return self.old_object.old_method()
```

##### Bridge Pattern
**Separate abstraction from implementation to avoid permanent binding.**

```python
from abc import ABC, abstractmethod

class DrawAPI(ABC):
    @abstractmethod
    def draw_circle(self, x, y, radius):
        pass

class RedCircle(DrawAPI):
    def draw_circle(self, x, y, radius):
        print(f"Drawing red circle at ({x}, {y}) with radius {radius}")

class GreenCircle(DrawAPI):
    def draw_circle(self, x, y, radius):
        print(f"Drawing green circle at ({x}, {y}) with radius {radius}")

class Shape(ABC):
    def __init__(self, draw_api: DrawAPI):
        self.draw_api = draw_api
    
    @abstractmethod
    def draw(self):
        pass

class Circle(Shape):
    def __init__(self, x, y, radius, draw_api: DrawAPI):
        super().__init__(draw_api)
        self.x = x
        self.y = y
        self.radius = radius
    
    def draw(self):
        self.draw_api.draw_circle(self.x, self.y, self.radius)
```

##### Composite Pattern
**Treat individual objects and compositions uniformly.**

```python
from abc import ABC, abstractmethod

class Component(ABC):
    @abstractmethod
    def operation(self):
        pass

class Leaf(Component):
    def operation(self):
        return "Leaf operation"

class Composite(Component):
    def __init__(self):
        self.children = []
    
    def add(self, component: Component):
        self.children.append(component)
    
    def remove(self, component: Component):
        self.children.remove(component)
    
    def operation(self):
        results = []
        for child in self.children:
            results.append(child.operation())
        return f"Composite[{', '.join(results)}]"
```

##### Decorator Pattern
**Add new behaviors to objects without modifying their structure.**

```python
from abc import ABC, abstractmethod

class Component(ABC):
    @abstractmethod
    def operation(self):
        pass

class ConcreteComponent(Component):
    def operation(self):
        return "ConcreteComponent"

class Decorator(Component):
    def __init__(self, component: Component):
        self._component = component
    
    def operation(self):
        return self._component.operation()

class ConcreteDecorator(Decorator):
    def operation(self):
        return f"ConcreteDecorator({self._component.operation()})"

# Python decorator syntax
def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    import time
    time.sleep(1)
    return "Done"
```

##### Facade Pattern
**Provide a simplified interface to a complex subsystem.**

```python
class SubsystemA:
    def operation_a(self):
        return "SubsystemA operation"

class SubsystemB:
    def operation_b(self):
        return "SubsystemB operation"

class SubsystemC:
    def operation_c(self):
        return "SubsystemC operation"

class Facade:
    def __init__(self):
        self.subsystem_a = SubsystemA()
        self.subsystem_b = SubsystemB()
        self.subsystem_c = SubsystemC()
    
    def simple_operation(self):
        result = []
        result.append(self.subsystem_a.operation_a())
        result.append(self.subsystem_b.operation_b())
        result.append(self.subsystem_c.operation_c())
        return " | ".join(result)
```

##### Flyweight Pattern
**Share common data efficiently to minimize memory usage.**

```python
class Flyweight:
    def __init__(self, shared_state):
        self.shared_state = shared_state
    
    def operation(self, unique_state):
        return f"Shared: {self.shared_state}, Unique: {unique_state}"

class FlyweightFactory:
    _flyweights = {}
    
    @classmethod
    def get_flyweight(cls, shared_state):
        if shared_state not in cls._flyweights:
            cls._flyweights[shared_state] = Flyweight(shared_state)
        return cls._flyweights[shared_state]
```

##### Proxy Pattern
**Provide a placeholder or surrogate for another object to control access.**

```python
from abc import ABC, abstractmethod

class Subject(ABC):
    @abstractmethod
    def request(self):
        pass

class RealSubject(Subject):
    def request(self):
        return "RealSubject handling request"

class Proxy(Subject):
    def __init__(self):
        self._real_subject = None
    
    def request(self):
        if self._real_subject is None:
            self._real_subject = RealSubject()
        return f"Proxy: {self._real_subject.request()}"
```

### Procedural Programming Design Principles

Python is a multi-paradigm language that excels at procedural programming. Many design principles can be effectively applied without explicit class definitions, using functions, modules, and data structures.

#### Single Responsibility Principle (SRP) for Functions

Each function should have one well-defined purpose and one reason to change. Avoid creating functions that handle multiple, unrelated tasks.

```python
# ❌ BAD: Function handles both calculation and formatting
def calculate_and_display_sum(a, b):
    result = a + b
    print(f"The sum is: {result}")
    return result

# ✅ GOOD: Separate functions for calculation and display
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

def display_result(operation, result):
    """Display the result of an operation."""
    print(f"The {operation} result is: {result}")

def format_currency(amount):
    """Format a number as currency."""
    return f"${amount:.2f}"

# Usage - composable functions
sum_result = calculate_sum(10, 20)
display_result("addition", sum_result)
formatted = format_currency(sum_result)
```

#### Open/Closed Principle (OCP) for Functions

Functions and modules should be open for extension but closed for modification. Use higher-order functions, function composition, and decorators.

```python
# ✅ GOOD: Extensible operation system using higher-order functions
def apply_operation(operation_func, *args):
    """Apply any operation function to arguments."""
    return operation_func(*args)

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def power(a, b):
    return a ** b

# New operations can be added without modifying apply_operation
operations = {
    'add': add,
    'subtract': subtract,
    'multiply': multiply,
    'power': power
}

def calculate(operation_name, a, b):
    if operation_name in operations:
        return apply_operation(operations[operation_name], a, b)
    raise ValueError(f"Unknown operation: {operation_name}")

# Usage
result = calculate('add', 5, 3)  # 8
result = calculate('power', 2, 3)  # 8

# ✅ GOOD: Function composition for extensibility
def compose(*functions):
    """Compose multiple functions into a single function."""
    def composed_function(value):
        for func in reversed(functions):
            value = func(value)
        return value
    return composed_function

def add_ten(x):
    return x + 10

def multiply_by_two(x):
    return x * 2

def square(x):
    return x ** 2

# Create new operations by composing existing ones
complex_operation = compose(square, multiply_by_two, add_ten)
result = complex_operation(5)  # ((5 + 10) * 2) ** 2 = 900
```

#### Don't Repeat Yourself (DRY) in Procedural Code

Extract common logic into reusable functions to avoid code duplication.

```python
# ❌ BAD: Repeated validation and processing logic
def process_user_data(data):
    if not data or 'name' not in data:
        raise ValueError("Invalid user data")
    
    name = data['name'].strip().title()
    email = data.get('email', '').lower()
    
    # Process user-specific logic
    print(f"Processing user: {name}")
    return {'name': name, 'email': email, 'type': 'user'}

def process_admin_data(data):
    if not data or 'name' not in data:
        raise ValueError("Invalid admin data")
    
    name = data['name'].strip().title()
    email = data.get('email', '').lower()
    
    # Process admin-specific logic
    print(f"Processing admin: {name}")
    return {'name': name, 'email': email, 'type': 'admin'}

# ✅ GOOD: Extract common logic into reusable functions
def validate_data(data, data_type="data"):
    """Validate that data contains required fields."""
    if not data or 'name' not in data:
        raise ValueError(f"Invalid {data_type}")

def normalize_person_data(data):
    """Normalize person data fields."""
    return {
        'name': data['name'].strip().title(),
        'email': data.get('email', '').lower()
    }

def process_person_data(data, person_type):
    """Process person data with common validation and normalization."""
    validate_data(data, f"{person_type} data")
    normalized = normalize_person_data(data)
    normalized['type'] = person_type
    
    print(f"Processing {person_type}: {normalized['name']}")
    return normalized

def process_user_data(data):
    return process_person_data(data, 'user')

def process_admin_data(data):
    return process_person_data(data, 'admin')
```

#### Keep It Simple, Stupid (KISS) - Functional Approach

Strive for simplicity in your functions. Avoid unnecessary complexity.

```python
# ❌ BAD: Overly complex function trying to do everything
def complex_data_processor(data, mode='default', options=None, validators=None):
    options = options or {}
    validators = validators or []
    
    if mode == 'advanced':
        # Complex branching logic
        for validator in validators:
            if not validator(data):
                if options.get('strict', False):
                    raise ValueError("Validation failed")
                else:
                    data = options.get('fallback', {})
        
        # More complex processing...
        result = {}
        for key, value in data.items():
            if options.get('transform', False):
                result[key.upper()] = str(value).strip()
            else:
                result[key] = value
        return result
    else:
        return data

# ✅ GOOD: Simple, focused functions
def validate_data_with_validators(data, validators):
    """Validate data using provided validators."""
    for validator in validators:
        if not validator(data):
            return False
    return True

def transform_keys_to_uppercase(data):
    """Transform dictionary keys to uppercase."""
    return {key.upper(): value for key, value in data.items()}

def normalize_values(data):
    """Normalize string values by stripping whitespace."""
    return {key: str(value).strip() if isinstance(value, str) else value 
            for key, value in data.items()}

def process_data_simple(data):
    """Simple data processing with validation."""
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    return normalize_values(data)

def process_data_advanced(data, validators=None):
    """Advanced data processing with validation and transformation."""
    validators = validators or []
    
    if validators and not validate_data_with_validators(data, validators):
        raise ValueError("Data validation failed")
    
    processed = normalize_values(data)
    return transform_keys_to_uppercase(processed)
```

#### Modularity and Separation of Concerns

Divide your program into distinct modules, each responsible for a specific aspect.

```python
# data_validation.py - Handle data validation concerns
def is_valid_email(email):
    """Check if email format is valid."""
    return '@' in email and '.' in email.split('@')[1]

def is_valid_phone(phone):
    """Check if phone format is valid."""
    import re
    pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
    return bool(re.match(pattern, phone))

def validate_required_fields(data, required_fields):
    """Validate that all required fields are present."""
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

# data_processing.py - Handle data transformation concerns
def normalize_name(name):
    """Normalize a person's name."""
    return name.strip().title() if name else ""

def normalize_email(email):
    """Normalize email address."""
    return email.strip().lower() if email else ""

def extract_domain(email):
    """Extract domain from email address."""
    return email.split('@')[1] if '@' in email else ""

# business_logic.py - Handle business-specific concerns
def calculate_user_score(user_data):
    """Calculate user score based on profile completeness."""
    score = 0
    if user_data.get('name'):
        score += 20
    if user_data.get('email'):
        score += 30
    if user_data.get('phone'):
        score += 25
    if user_data.get('bio'):
        score += 25
    return score

def determine_user_tier(score):
    """Determine user tier based on score."""
    if score >= 80:
        return 'premium'
    elif score >= 50:
        return 'standard'
    else:
        return 'basic'

# main_processor.py - Orchestrate the workflow
from . import data_validation, data_processing, business_logic

def process_user_registration(raw_data):
    """Complete user registration processing workflow."""
    # Validation
    data_validation.validate_required_fields(raw_data, ['name', 'email'])
    
    if not data_validation.is_valid_email(raw_data['email']):
        raise ValueError("Invalid email format")
    
    # Processing
    processed_data = {
        'name': data_processing.normalize_name(raw_data['name']),
        'email': data_processing.normalize_email(raw_data['email']),
        'domain': data_processing.extract_domain(raw_data['email'])
    }
    
    # Business logic
    score = business_logic.calculate_user_score(processed_data)
    processed_data['tier'] = business_logic.determine_user_tier(score)
    
    return processed_data
```

#### Functional Programming Principles

```python
# ✅ GOOD: Pure functions (no side effects)
def calculate_tax(income, tax_rate):
    """Pure function - same input always produces same output."""
    return income * tax_rate

def format_currency(amount, currency='USD'):
    """Pure function for formatting currency."""
    return f"{amount:.2f} {currency}"

# ✅ GOOD: Higher-order functions for abstraction
def create_validator(min_length=0, max_length=float('inf')):
    """Higher-order function that returns a validator function."""
    def validator(value):
        return min_length <= len(str(value)) <= max_length
    return validator

def create_formatter(prefix='', suffix=''):
    """Higher-order function that returns a formatter function."""
    def formatter(value):
        return f"{prefix}{value}{suffix}"
    return formatter

# Usage
name_validator = create_validator(min_length=2, max_length=50)
currency_formatter = create_formatter(prefix='$', suffix=' USD')

print(name_validator("John"))  # True
print(currency_formatter(100))  # $100 USD

# ✅ GOOD: Function composition and chaining
def pipe(value, *functions):
    """Apply functions in sequence to a value."""
    for func in functions:
        value = func(value)
    return value

def clean_string(text):
    return text.strip()

def uppercase_string(text):
    return text.upper()

def add_prefix(text):
    return f"PROCESSED: {text}"

# Chain operations
result = pipe("  hello world  ", clean_string, uppercase_string, add_prefix)
print(result)  # "PROCESSED: HELLO WORLD"

# ✅ GOOD: Immutable data processing
def update_user_data(user_data, **updates):
    """Return new dict with updates instead of modifying original."""
    return {**user_data, **updates}

def add_computed_fields(user_data):
    """Add computed fields without modifying original data."""
    return update_user_data(
        user_data,
        full_name=f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
        initials=''.join(word[0] for word in user_data.get('full_name', '').split() if word)
    )

# Usage preserves original data
original_user = {'first_name': 'John', 'last_name': 'Doe', 'email': 'john@example.com'}
enhanced_user = add_computed_fields(original_user)
# original_user remains unchanged
```

#### Error Handling in Procedural Code

```python
# ✅ GOOD: Explicit error handling with meaningful messages
def divide_safely(a, b):
    """Divide two numbers with proper error handling."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers")
    
    if b == 0:
        raise ValueError("Cannot divide by zero")
    
    return a / b

def process_file_safely(file_path, processor_func):
    """Process a file with comprehensive error handling."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return processor_func(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except PermissionError:
        raise PermissionError(f"Permission denied accessing: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing file {file_path}: {e}")

# ✅ GOOD: Early returns for validation
def calculate_discount(price, customer_type, years_customer):
    """Calculate discount with early validation returns."""
    if price <= 0:
        return 0
    
    if customer_type not in ['regular', 'premium', 'vip']:
        return 0
    
    if years_customer < 0:
        return 0
    
    # Main logic only runs with valid inputs
    base_discount = {
        'regular': 0.05,
        'premium': 0.10,
        'vip': 0.15
    }[customer_type]
    
    loyalty_bonus = min(years_customer * 0.01, 0.10)  # Max 10% loyalty bonus
    total_discount = base_discount + loyalty_bonus
    
    return min(total_discount * price, price * 0.50)  # Max 50% discount
```

#### Testing Procedural Code

```python
# ✅ GOOD: Testable pure functions
def calculate_shipping_cost(weight, distance, shipping_type='standard'):
    """Calculate shipping cost - pure function, easy to test."""
    base_rates = {
        'standard': 0.50,
        'express': 1.00,
        'overnight': 2.00
    }
    
    if shipping_type not in base_rates:
        raise ValueError(f"Invalid shipping type: {shipping_type}")
    
    base_cost = weight * base_rates[shipping_type]
    distance_multiplier = 1 + (distance / 1000) * 0.1
    
    return round(base_cost * distance_multiplier, 2)

# Easy to test
def test_shipping_cost():
    assert calculate_shipping_cost(10, 100, 'standard') == 5.05
    assert calculate_shipping_cost(5, 500, 'express') == 5.25
    
    try:
        calculate_shipping_cost(10, 100, 'invalid')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
```

These procedural programming principles promote maintainable, flexible, and robust code without requiring object-oriented design. They emphasize simplicity, modularity, and functional composition while maintaining clean separation of concerns.

#### Behavioral Patterns - Object Interaction and Responsibilities

##### Chain of Responsibility Pattern
**Pass requests along a chain of handlers until one handles it.**

```python
from abc import ABC, abstractmethod

class Handler(ABC):
    def __init__(self):
        self._next_handler = None
    
    def set_next(self, handler):
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, request):
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

class ConcreteHandlerA(Handler):
    def handle(self, request):
        if request == "A":
            return f"HandlerA handled {request}"
        return super().handle(request)

class ConcreteHandlerB(Handler):
    def handle(self, request):
        if request == "B":
            return f"HandlerB handled {request}"
        return super().handle(request)
```

##### Command Pattern
**Encapsulate requests as objects to enable queuing, logging, and undo operations.**

```python
from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class Light:
    def turn_on(self):
        print("Light is ON")
    
    def turn_off(self):
        print("Light is OFF")

class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.turn_on()
    
    def undo(self):
        self.light.turn_off()

class RemoteControl:
    def __init__(self):
        self.commands = {}
        self.last_command = None
    
    def set_command(self, slot, command):
        self.commands[slot] = command
    
    def press_button(self, slot):
        if slot in self.commands:
            self.commands[slot].execute()
            self.last_command = self.commands[slot]
    
    def press_undo(self):
        if self.last_command:
            self.last_command.undo()
```

##### Iterator Pattern
**Provide a way to access elements of a collection sequentially.**

```python
class Iterator:
    def __init__(self, collection):
        self._collection = collection
        self._index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index < len(self._collection):
            result = self._collection[self._index]
            self._index += 1
            return result
        raise StopIteration

class NumberCollection:
    def __init__(self):
        self._items = []
    
    def add_item(self, item):
        self._items.append(item)
    
    def __iter__(self):
        return Iterator(self._items)

# Python built-in iterator support
class Fibonacci:
    def __init__(self, max_count):
        self.max_count = max_count
    
    def __iter__(self):
        self.count = 0
        self.current = 0
        self.next_val = 1
        return self
    
    def __next__(self):
        if self.count < self.max_count:
            result = self.current
            self.current, self.next_val = self.next_val, self.current + self.next_val
            self.count += 1
            return result
        raise StopIteration
```

##### Mediator Pattern
**Define how objects interact with each other through a mediator.**

```python
from abc import ABC, abstractmethod

class Mediator(ABC):
    @abstractmethod
    def notify(self, sender, event):
        pass

class ConcreteMediator(Mediator):
    def __init__(self, component1, component2):
        self._component1 = component1
        self._component1.mediator = self
        self._component2 = component2
        self._component2.mediator = self
    
    def notify(self, sender, event):
        if event == "A":
            print("Mediator reacts on A and triggers:")
            self._component2.do_c()
        elif event == "D":
            print("Mediator reacts on D and triggers:")
            self._component1.do_b()

class BaseComponent:
    def __init__(self, mediator=None):
        self._mediator = mediator

class Component1(BaseComponent):
    def do_a(self):
        print("Component 1 does A")
        self._mediator.notify(self, "A")
    
    def do_b(self):
        print("Component 1 does B")

class Component2(BaseComponent):
    def do_c(self):
        print("Component 2 does C")
    
    def do_d(self):
        print("Component 2 does D")
        self._mediator.notify(self, "D")
```

##### Observer Pattern
**Define a one-to-many dependency between objects.**

```python
from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer: Observer):
        self._observers.append(observer)
    
    def detach(self, observer: Observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)
    
    def set_state(self, state):
        self._state = state
        self.notify()
    
    def get_state(self):
        return self._state

class ConcreteObserver(Observer):
    def update(self, subject):
        print(f"Observer: Reacted to the event. Subject state: {subject.get_state()}")
```

##### State Pattern
**Allow an object to alter its behavior when its internal state changes.**

```python
from abc import ABC, abstractmethod

class State(ABC):
    @abstractmethod
    def handle(self, context):
        pass

class ConcreteStateA(State):
    def handle(self, context):
        print("ConcreteStateA handling request")
        context.set_state(ConcreteStateB())

class ConcreteStateB(State):
    def handle(self, context):
        print("ConcreteStateB handling request")
        context.set_state(ConcreteStateA())

class Context:
    def __init__(self, state: State):
        self._state = state
    
    def set_state(self, state: State):
        self._state = state
    
    def request(self):
        self._state.handle(self)
```

##### Strategy Pattern
**Define a family of algorithms and make them interchangeable.**

```python
from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def do_algorithm(self, data):
        pass

class ConcreteStrategyA(Strategy):
    def do_algorithm(self, data):
        return sorted(data)

class ConcreteStrategyB(Strategy):
    def do_algorithm(self, data):
        return sorted(data, reverse=True)

class Context:
    def __init__(self, strategy: Strategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: Strategy):
        self._strategy = strategy
    
    def do_some_business_logic(self, data):
        result = self._strategy.do_algorithm(data)
        print(f"Context: Sorted data using strategy: {result}")
```

##### Template Method Pattern
**Define the skeleton of an algorithm, letting subclasses override specific steps.**

```python
from abc import ABC, abstractmethod

class AbstractClass(ABC):
    def template_method(self):
        self.base_operation1()
        self.required_operation1()
        self.base_operation2()
        self.hook1()
        self.required_operation2()
        self.base_operation3()
        self.hook2()
    
    def base_operation1(self):
        print("AbstractClass: Base operation 1")
    
    def base_operation2(self):
        print("AbstractClass: Base operation 2")
    
    def base_operation3(self):
        print("AbstractClass: Base operation 3")
    
    @abstractmethod
    def required_operation1(self):
        pass
    
    @abstractmethod
    def required_operation2(self):
        pass
    
    def hook1(self):
        pass
    
    def hook2(self):
        pass

class ConcreteClass1(AbstractClass):
    def required_operation1(self):
        print("ConcreteClass1: Required operation 1")
    
    def required_operation2(self):
        print("ConcreteClass1: Required operation 2")

class ConcreteClass2(AbstractClass):
    def required_operation1(self):
        print("ConcreteClass2: Required operation 1")
    
    def required_operation2(self):
        print("ConcreteClass2: Required operation 2")
    
    def hook1(self):
        print("ConcreteClass2: Hook 1")
```

##### Visitor Pattern
**Represent operations to be performed on elements of an object structure.**

```python
from abc import ABC, abstractmethod

class Visitor(ABC):
    @abstractmethod
    def visit_concrete_component_a(self, element):
        pass
    
    @abstractmethod
    def visit_concrete_component_b(self, element):
        pass

class Component(ABC):
    @abstractmethod
    def accept(self, visitor: Visitor):
        pass

class ConcreteComponentA(Component):
    def accept(self, visitor: Visitor):
        visitor.visit_concrete_component_a(self)
    
    def exclusive_method_of_concrete_component_a(self):
        return "A"

class ConcreteComponentB(Component):
    def accept(self, visitor: Visitor):
        visitor.visit_concrete_component_b(self)
    
    def special_method_of_concrete_component_b(self):
        return "B"

class ConcreteVisitor1(Visitor):
    def visit_concrete_component_a(self, element):
        print(f"ConcreteVisitor1: {element.exclusive_method_of_concrete_component_a()}")
    
    def visit_concrete_component_b(self, element):
        print(f"ConcreteVisitor1: {element.special_method_of_concrete_component_b()}")
```

##### Interpreter Pattern
**Define a grammar for a language and interpret sentences in that language.**

```python
from abc import ABC, abstractmethod

class Expression(ABC):
    @abstractmethod
    def interpret(self, context):
        pass

class TerminalExpression(Expression):
    def __init__(self, data):
        self.data = data
    
    def interpret(self, context):
        return self.data in context

class OrExpression(Expression):
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2
    
    def interpret(self, context):
        return self.expr1.interpret(context) or self.expr2.interpret(context)

class AndExpression(Expression):
    def __init__(self, expr1, expr2):
        self.expr1 = expr1
        self.expr2 = expr2
    
    def interpret(self, context):
        return self.expr1.interpret(context) and self.expr2.interpret(context)

# Example: Simple boolean expression interpreter
def get_male_expression():
    robert = TerminalExpression("Robert")
    john = TerminalExpression("John")
    return OrExpression(robert, john)

def get_married_woman_expression():
    julie = TerminalExpression("Julie")
    married = TerminalExpression("Married")
    return AndExpression(julie, married)

# Usage
male_expr = get_male_expression()
married_woman_expr = get_married_woman_expression()

print(male_expr.interpret("Robert"))  # True
print(married_woman_expr.interpret("Julie Married"))  # True
```

#### Additional Python-Specific Patterns

##### Borg Pattern (Shared State Singleton)
**Share state among instances instead of enforcing single instance.**

```python
class Borg:
    _shared_state = {}
    
    def __init__(self):
        self.__dict__ = self._shared_state

class BorgSingleton(Borg):
    def __init__(self, value=None):
        super().__init__()
        if value is not None:
            self.value = value

# Usage
b1 = BorgSingleton("first")
b2 = BorgSingleton("second")
print(b1.value)  # "second" - shared state
print(b2.value)  # "second" - shared state
```

##### Monostate Pattern (Alternative Singleton)
**All instances share the same state, but can have different identities.**

```python
class Monostate:
    _shared_attrs = {}
    
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.__dict__ = cls._shared_attrs
        return obj

class DatabaseConnection(Monostate):
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.connection_string = "default_connection"
            self.is_connected = False
            self.initialized = True
    
    def connect(self):
        self.is_connected = True
        print(f"Connected to {self.connection_string}")

# Usage - different instances, shared state
db1 = DatabaseConnection()
db2 = DatabaseConnection()
db1.connect()
print(db2.is_connected)  # True - shared state
```

##### Lazy Evaluation Pattern
**Defer computation until the result is actually needed.**

```python
class LazyProperty:
    def __init__(self, func):
        self.func = func
    
    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.func.__name__, value)
        return value

class DataProcessor:
    @LazyProperty
    def expensive_computation(self):
        print("Performing expensive computation...")
        return sum(range(1000000))
```

##### Null Object Pattern
**Provide a neutral object that exhibits neutral behavior to eliminate conditional checks for null.**

```python
from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log(self, message: str) -> None:
        pass

class FileLogger(Logger):
    def __init__(self, filename: str):
        self.filename = filename
    
    def log(self, message: str) -> None:
        with open(self.filename, 'a') as f:
            f.write(f"{message}\n")

class ConsoleLogger(Logger):
    def log(self, message: str) -> None:
        print(f"LOG: {message}")

class NullLogger(Logger):
    """Null Object - does nothing, eliminating need for null checks."""
    def log(self, message: str) -> None:
        pass  # Do nothing

class Service:
    def __init__(self, logger: Logger = None):
        # Use Null Object instead of None checks
        self.logger = logger or NullLogger()
    
    def do_work(self):
        # No need to check if logger is None
        self.logger.log("Starting work")
        # ... do actual work ...
        self.logger.log("Work completed")

# Usage - no conditional logic needed
service1 = Service(FileLogger("app.log"))
service2 = Service(ConsoleLogger())
service3 = Service()  # Uses NullLogger automatically

service1.do_work()  # Logs to file
service2.do_work()  # Logs to console
service3.do_work()  # Logs nowhere, but no errors
```

##### Dependency Injection Pattern
**Provide dependencies from external sources rather than creating them internally.**

```python
from abc import ABC, abstractmethod

class DatabaseInterface(ABC):
    @abstractmethod
    def save(self, data):
        pass

class MySQLDatabase(DatabaseInterface):
    def save(self, data):
        print(f"Saving to MySQL: {data}")

class PostgreSQLDatabase(DatabaseInterface):
    def save(self, data):
        print(f"Saving to PostgreSQL: {data}")

class UserService:
    def __init__(self, database: DatabaseInterface):
        self.database = database
    
    def create_user(self, user_data):
        # Process user data
        self.database.save(user_data)

# Dependency injection container
class DIContainer:
    def __init__(self):
        self._services = {}
    
    def register(self, service_type, implementation):
        self._services[service_type] = implementation
    
    def get(self, service_type):
        return self._services.get(service_type)

# Usage
container = DIContainer()
container.register(DatabaseInterface, MySQLDatabase())
user_service = UserService(container.get(DatabaseInterface))
```

### Core Design Principles
- **Don't Repeat Yourself (DRY):** Extract common logic into reusable functions, classes, or modules. If you find yourself writing the same logic more than once, abstract it to reduce maintenance overhead and likelihood of bugs
- **Keep It Simple, Stupid (KISS):** Always strive for the simplest possible solution that meets requirements. Avoid unnecessary complexity, fancy patterns, or over-engineering. Simplicity is the best path to reliability and maintainability
- **You Aren't Gonna Need It (YAGNI):** Don't implement functionality that is not currently required, no matter how certain you are it will be needed in the future. Focus on immediate needs and build incrementally
- **Composition Over Inheritance:** Prefer composing objects with desired behaviors rather than relying on deep inheritance hierarchies. This leads to more flexible and robust designs, avoiding the "fragile base class" problem

### Additional Python Design Patterns and Best Practices

#### Functional Programming Patterns

##### Registry Pattern
**Keep track of all subclasses of a given class.**

```python
class RegisteredMeta(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        cls.registry[name] = cls

class BaseHandler(metaclass=RegisteredMeta):
    pass

class JSONHandler(BaseHandler):
    def handle(self, data):
        return f"Handling JSON: {data}"

class XMLHandler(BaseHandler):
    def handle(self, data):
        return f"Handling XML: {data}"

# Usage
handler_class = BaseHandler.registry['JSONHandler']
handler = handler_class()
```

##### Specification Pattern
**Business rules can be recombined by chaining using boolean logic.**

```python
from abc import ABC, abstractmethod

class Specification(ABC):
    @abstractmethod
    def is_satisfied_by(self, candidate):
        pass
    
    def and_(self, other):
        return AndSpecification(self, other)
    
    def or_(self, other):
        return OrSpecification(self, other)
    
    def not_(self):
        return NotSpecification(self)

class AndSpecification(Specification):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, candidate):
        return self.left.is_satisfied_by(candidate) and self.right.is_satisfied_by(candidate)

class OrSpecification(Specification):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, candidate):
        return self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(candidate)

class NotSpecification(Specification):
    def __init__(self, specification):
        self.specification = specification
    
    def is_satisfied_by(self, candidate):
        return not self.specification.is_satisfied_by(candidate)

# Example specifications
class AgeSpecification(Specification):
    def __init__(self, min_age):
        self.min_age = min_age
    
    def is_satisfied_by(self, user):
        return user.age >= self.min_age

class EmailSpecification(Specification):
    def is_satisfied_by(self, user):
        return '@' in user.email

# Usage
adult_spec = AgeSpecification(18)
email_spec = EmailSpecification()
valid_user_spec = adult_spec.and_(email_spec)
```

##### Memento Pattern
**Generate an opaque token to go back to a previous state.**

```python
class Memento:
    def __init__(self, state):
        self._state = state
    
    def get_state(self):
        return self._state

class Originator:
    def __init__(self):
        self._state = None
    
    def set_state(self, state):
        self._state = state
    
    def get_state(self):
        return self._state
    
    def create_memento(self):
        return Memento(self._state)
    
    def restore_from_memento(self, memento):
        self._state = memento.get_state()

class Caretaker:
    def __init__(self):
        self._mementos = []
    
    def add_memento(self, memento):
        self._mementos.append(memento)
    
    def get_memento(self, index):
        return self._mementos[index]
```

##### Catalog Pattern
**General methods call different specialized methods based on construction parameter.**

```python
class Catalog:
    def __init__(self):
        self._catalog = {}
    
    def register(self, name, func):
        self._catalog[name] = func
    
    def call(self, name, *args, **kwargs):
        if name in self._catalog:
            return self._catalog[name](*args, **kwargs)
        raise ValueError(f"Function {name} not found in catalog")

# Usage
catalog = Catalog()
catalog.register('add', lambda x, y: x + y)
catalog.register('multiply', lambda x, y: x * y)

result = catalog.call('add', 5, 3)  # Returns 8
```

#### Architectural Patterns

##### Model-View-Controller (MVC)
**Separate data (model), presentation (view), and logic (controller).**

```python
# Model
class UserModel:
    def __init__(self):
        self.users = []
    
    def add_user(self, user):
        self.users.append(user)
    
    def get_users(self):
        return self.users

# View
class UserView:
    def display_users(self, users):
        for user in users:
            print(f"User: {user['name']}, Email: {user['email']}")
    
    def get_user_input(self):
        name = input("Enter name: ")
        email = input("Enter email: ")
        return {'name': name, 'email': email}

# Controller
class UserController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
    
    def add_user(self):
        user_data = self.view.get_user_input()
        self.model.add_user(user_data)
    
    def show_users(self):
        users = self.model.get_users()
        self.view.display_users(users)
```

##### Repository Pattern
**Separate data access logic from business logic.**

```python
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def save(self, entity):
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id):
        pass
    
    @abstractmethod
    def find_all(self):
        pass
    
    @abstractmethod
    def delete(self, entity_id):
        pass

class InMemoryUserRepository(Repository):
    def __init__(self):
        self._users = {}
        self._next_id = 1
    
    def save(self, user):
        if not hasattr(user, 'id') or user.id is None:
            user.id = self._next_id
            self._next_id += 1
        self._users[user.id] = user
        return user
    
    def find_by_id(self, user_id):
        return self._users.get(user_id)
    
    def find_all(self):
        return list(self._users.values())
    
    def delete(self, user_id):
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False
```

##### Unit of Work Pattern
**Maintain a list of objects affected by a business transaction.**

```python
class UnitOfWork:
    def __init__(self):
        self._new_objects = []
        self._dirty_objects = []
        self._removed_objects = []
    
    def register_new(self, obj):
        if obj not in self._new_objects:
            self._new_objects.append(obj)
    
    def register_dirty(self, obj):
        if obj not in self._dirty_objects:
            self._dirty_objects.append(obj)
    
    def register_removed(self, obj):
        if obj not in self._removed_objects:
            self._removed_objects.append(obj)
    
    def commit(self):
        # Insert new objects
        for obj in self._new_objects:
            self._insert(obj)
        
        # Update dirty objects
        for obj in self._dirty_objects:
            self._update(obj)
        
        # Delete removed objects
        for obj in self._removed_objects:
            self._delete(obj)
        
        # Clear all lists
        self._new_objects.clear()
        self._dirty_objects.clear()
        self._removed_objects.clear()
    
    def _insert(self, obj):
        # Implementation for inserting object
        pass
    
    def _update(self, obj):
        # Implementation for updating object
        pass
    
    def _delete(self, obj):
        # Implementation for deleting object
        pass
```

### Common Anti-Patterns to Avoid

#### God Object Anti-Pattern
**Avoid creating classes that know too much or do too much.**

```python
# ❌ BAD: God Object - does everything
class UserManager:
    def __init__(self):
        self.database = Database()
        self.email_service = EmailService()
        self.logger = Logger()
    
    def create_user(self, user_data):
        # Validates user data
        if not self._validate_email(user_data['email']):
            raise ValueError("Invalid email")
        
        # Saves to database
        user_id = self.database.save_user(user_data)
        
        # Sends welcome email
        self.email_service.send_welcome_email(user_data['email'])
        
        # Logs the action
        self.logger.log(f"User {user_id} created")
        
        # Generates report
        self._generate_user_report(user_id)
        
        return user_id

# ✅ GOOD: Separate responsibilities
class UserValidator:
    def validate_email(self, email):
        return '@' in email

class UserRepository:
    def save(self, user_data):
        # Save to database
        pass

class UserService:
    def __init__(self, validator, repository, email_service, logger):
        self.validator = validator
        self.repository = repository
        self.email_service = email_service
        self.logger = logger
    
    def create_user(self, user_data):
        if not self.validator.validate_email(user_data['email']):
            raise ValueError("Invalid email")
        
        user_id = self.repository.save(user_data)
        self.email_service.send_welcome_email(user_data['email'])
        self.logger.log(f"User {user_id} created")
        
        return user_id
```

#### Spaghetti Code Anti-Pattern
**Avoid unstructured, tangled control flow.**

```python
# ❌ BAD: Spaghetti code with complex nested conditions
def process_order(order):
    if order.status == 'pending':
        if order.payment_method == 'credit_card':
            if order.amount > 1000:
                if order.customer.vip_status:
                    discount = 0.15
                else:
                    discount = 0.05
            else:
                discount = 0.02
            # Process payment...
        elif order.payment_method == 'paypal':
            # Different logic...
            pass
        # Update inventory...
        # Send confirmation email...
    elif order.status == 'shipped':
        # Different processing...
        pass

# ✅ GOOD: Clean, structured approach
class OrderProcessor:
    def process(self, order):
        if order.status == 'pending':
            self._process_pending_order(order)
        elif order.status == 'shipped':
            self._process_shipped_order(order)
    
    def _process_pending_order(self, order):
        discount = self._calculate_discount(order)
        self._process_payment(order, discount)
        self._update_inventory(order)
        self._send_confirmation(order)
    
    def _calculate_discount(self, order):
        if order.amount > 1000:
            return 0.15 if order.customer.vip_status else 0.05
        return 0.02
```

#### Magic Numbers/Strings Anti-Pattern
**Use named constants instead of magic values.**

```python
# ❌ BAD: Magic numbers and strings
def calculate_tax(amount):
    if amount > 10000:
        return amount * 0.25  # What is 0.25?
    elif amount > 5000:
        return amount * 0.15  # What is 0.15?
    else:
        return amount * 0.10  # What is 0.10?

def get_user_status(user):
    if user.login_attempts > 3:  # What is 3?
        return "locked"
    return "active"

# ✅ GOOD: Named constants
class TaxRates:
    HIGH_INCOME_TAX = 0.25
    MEDIUM_INCOME_TAX = 0.15
    LOW_INCOME_TAX = 0.10

class UserSettings:
    MAX_LOGIN_ATTEMPTS = 3
    HIGH_INCOME_THRESHOLD = 10000
    MEDIUM_INCOME_THRESHOLD = 5000

def calculate_tax(amount):
    if amount > UserSettings.HIGH_INCOME_THRESHOLD:
        return amount * TaxRates.HIGH_INCOME_TAX
    elif amount > UserSettings.MEDIUM_INCOME_THRESHOLD:
        return amount * TaxRates.MEDIUM_INCOME_TAX
    else:
        return amount * TaxRates.LOW_INCOME_TAX

def get_user_status(user):
    if user.login_attempts > UserSettings.MAX_LOGIN_ATTEMPTS:
        return "locked"
    return "active"
```

#### Copy-Paste Programming Anti-Pattern
**Avoid duplicating code blocks.**

```python
# ❌ BAD: Code duplication
def send_email_to_user(user):
    smtp_server = "smtp.example.com"
    smtp_port = 587
    username = "noreply@example.com"
    password = "password123"
    
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(username, password)
    # Send email logic...
    server.quit()

def send_email_to_admin(admin):
    smtp_server = "smtp.example.com"
    smtp_port = 587
    username = "noreply@example.com"
    password = "password123"
    
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(username, password)
    # Send email logic...
    server.quit()

# ✅ GOOD: Extract common functionality
class EmailService:
    def __init__(self):
        self.smtp_server = "smtp.example.com"
        self.smtp_port = 587
        self.username = "noreply@example.com"
        self.password = "password123"
    
    def _create_connection(self):
        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        server.starttls()
        server.login(self.username, self.password)
        return server
    
    def send_email(self, recipient, subject, body):
        server = self._create_connection()
        try:
            # Send email logic...
            pass
        finally:
            server.quit()
    
    def send_email_to_user(self, user):
        self.send_email(user.email, "User Notification", "...")
    
    def send_email_to_admin(self, admin):
        self.send_email(admin.email, "Admin Notification", "...")
```

### Architecture Guidelines
- **Separation of Concerns:** High cohesion within modules, low coupling between modules
- **Dependency Injection:** Pass dependencies rather than creating them internally

## 4. Pythonic Practices and Idioms

### Core Python Features
- **List Comprehensions/Generator Expressions:** Use for transformations and filtering where they improve readability
- **Context Managers (`with` statements):** Use for managing resources (files, locks, database connections)
- **enumerate() for Loops:** Use when both index and value are needed in a loop
- **zip() for Parallel Iteration:** Use when iterating over multiple iterables in parallel
- **dict.get() for Dictionary Access:** Use when a default value is preferable to raising a KeyError
- **f-strings for String Formatting:** Preferred for clear and concise string formatting

### Advanced Python Features
- **collections Module:** Use appropriate data structures (defaultdict, Counter, deque) where they provide advantages
- **Custom Context Managers:** Use for managing application-specific resources
- **Generators/Iterators:** Use for producing large sequences efficiently when memory is a concern
- **Decorators:** Use for adding cross-cutting concerns (logging, authentication, timing) without modifying original function code

### Pythonic Code Examples and Best Practices

#### Idiomatic Python Constructs

```python
# ✅ GOOD: List comprehensions for simple transformations
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
even_squares = [x**2 for x in numbers if x % 2 == 0]

# ✅ GOOD: Dictionary comprehensions
words = ['hello', 'world', 'python']
word_lengths = {word: len(word) for word in words}

# ✅ GOOD: Using enumerate when you need both index and value
for i, item in enumerate(['a', 'b', 'c']):
    print(f"{i}: {item}")

# ✅ GOOD: Using zip for parallel iteration
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# ✅ GOOD: Using dict.get() with defaults
config = {'debug': True, 'port': 8000}
debug_mode = config.get('debug', False)
timeout = config.get('timeout', 30)

# ✅ GOOD: f-string formatting (Python 3.6+)
name = "World"
greeting = f"Hello, {name}!"
number = 3.14159
formatted = f"Pi is approximately {number:.2f}"
```

#### Context Managers and Resource Management

```python
# ✅ GOOD: Using context managers for file operations
with open('data.txt', 'r') as file:
    content = file.read()
    # File is automatically closed

# ✅ GOOD: Custom context manager
class DatabaseConnection:
    def __enter__(self):
        self.connection = connect_to_database()
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

with DatabaseConnection() as db:
    db.execute("SELECT * FROM users")

# ✅ GOOD: Context manager using contextlib
from contextlib import contextmanager

@contextmanager
def temporary_setting(setting_name, new_value):
    old_value = get_setting(setting_name)
    set_setting(setting_name, new_value)
    try:
        yield
    finally:
        set_setting(setting_name, old_value)

with temporary_setting('debug', True):
    # Code runs with debug=True
    pass
# debug setting restored to original value
```

#### Generators and Memory Efficiency

```python
# ✅ GOOD: Generator for memory-efficient processing
def read_large_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()

# ✅ GOOD: Generator expressions for lazy evaluation
def process_numbers(numbers):
    return (x * 2 for x in numbers if x > 0)

# Usage - memory efficient for large datasets
large_numbers = range(1000000)
doubled_positives = process_numbers(large_numbers)
first_ten = list(itertools.islice(doubled_positives, 10))

# ✅ GOOD: Using itertools for efficient iteration
import itertools

def batch_processor(iterable, batch_size):
    """Process items in batches."""
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch

for batch in batch_processor(range(100), 10):
    process_batch(batch)
```

#### Decorators for Cross-Cutting Concerns

```python
# ✅ GOOD: Timing decorator
import functools
import time

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# ✅ GOOD: Retry decorator
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@timing_decorator
@retry(max_attempts=3, delay=2)
def unreliable_api_call():
    # Simulated API call that might fail
    import random
    if random.random() < 0.7:
        raise ConnectionError("API temporarily unavailable")
    return {"status": "success", "data": "some data"}

# ✅ GOOD: Caching decorator
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n):
    # Expensive recursive computation with memoization
    if n <= 1:
        return n
    return expensive_computation(n-1) + expensive_computation(n-2)
```

#### Collections Module Usage

```python
from collections import defaultdict, Counter, deque, namedtuple

# ✅ GOOD: defaultdict for grouping
students_by_grade = defaultdict(list)
for student in students:
    students_by_grade[student.grade].append(student.name)

# ✅ GOOD: Counter for counting
from collections import Counter
text = "hello world"
letter_counts = Counter(text)
most_common = letter_counts.most_common(3)

# ✅ GOOD: deque for efficient append/pop operations
from collections import deque
queue = deque()
queue.append('first')
queue.append('second')
first_item = queue.popleft()  # O(1) operation

# ✅ GOOD: namedtuple for lightweight data containers
Point = namedtuple('Point', ['x', 'y'])
origin = Point(0, 0)
print(f"Origin: ({origin.x}, {origin.y})")

# ✅ GOOD: Modern dataclasses (Python 3.7+)
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str = ""  # Default value
    
    def is_adult(self) -> bool:
        return self.age >= 18

person = Person("Alice", 30, "alice@example.com")
```

#### Error Handling and Exception Management

```python
# ✅ GOOD: Specific exception handling
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Config file not found, using defaults")
    config = DEFAULT_CONFIG
except json.JSONDecodeError as e:
    print(f"Invalid JSON in config file: {e}")
    config = DEFAULT_CONFIG
except PermissionError:
    print("Permission denied accessing config file")
    raise

# ✅ GOOD: Custom exceptions with meaningful names
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass

def validate_email(email):
    if '@' not in email:
        raise ValidationError(f"Invalid email format: {email}")
    return email

# ✅ GOOD: Exception chaining for debugging
try:
    result = risky_operation()
except SomeException as e:
    raise ProcessingError("Failed to process data") from e
```

#### Type Hints and Modern Python Features

```python
from typing import List, Dict, Optional, Union, Callable, TypeVar, Generic

# ✅ GOOD: Type hints for better code documentation
def process_user_data(
    users: List[Dict[str, Union[str, int]]], 
    filter_func: Optional[Callable[[Dict], bool]] = None
) -> List[Dict[str, Union[str, int]]]:
    """Process user data with optional filtering."""
    if filter_func:
        users = [user for user in users if filter_func(user)]
    return users

# ✅ GOOD: Generic types
T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()

# ✅ GOOD: Protocol for structural typing (Python 3.8+)
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

def render_shape(shape: Drawable) -> None:
    shape.draw()  # Any object with a draw() method works
```

#### String Operations and Processing

```python
# ✅ GOOD: String operations and formatting
def format_user_info(name: str, age: int, city: str) -> str:
    # f-strings for readability
    return f"User: {name.title()}, Age: {age}, City: {city.upper()}"

# ✅ GOOD: String methods for common operations
text = "  Hello, World!  "
cleaned = text.strip().lower()
words = cleaned.split(', ')

# ✅ GOOD: Regular expressions for complex patterns
import re

def extract_emails(text: str) -> List[str]:
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)

def validate_phone_number(phone: str) -> bool:
    pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
    return bool(re.match(pattern, phone))
```

#### Functional Programming Patterns

```python
from functools import reduce, partial
import operator

# ✅ GOOD: Using map, filter, reduce appropriately
numbers = [1, 2, 3, 4, 5]

# Simple transformations
doubled = list(map(lambda x: x * 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
total = reduce(operator.add, numbers, 0)

# ✅ GOOD: Partial application for reusable functions
def multiply(x, y):
    return x * y

double = partial(multiply, 2)
triple = partial(multiply, 3)

doubled_numbers = list(map(double, numbers))

# ✅ GOOD: Function composition
def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def add_one(x):
    return x + 1

def multiply_by_two(x):
    return x * 2

def square(x):
    return x ** 2

# Compose functions: square(multiply_by_two(add_one(x)))
composed_func = compose(square, multiply_by_two, add_one)
result = composed_func(3)  # ((3 + 1) * 2) ** 2 = 64
```

## 5. Code Quality and Maintenance

### Code Smells to Avoid
- **Magic Numbers/Strings:** Replace hardcoded literal values with named constants
- **Long Parameter Lists:** Group related parameters into objects
- **Feature Envy:** Methods should focus on their own class's data
- **Shotgun Surgery:** Single changes shouldn't require modifications in many places
- **Duplicated Code:** Extract common logic into reusable functions
- **Dead Code:** Remove unreachable or unused code
- **Over/Under-engineering:** Balance complexity with current and future needs

### Testing Requirements (MANDATORY)
**⚠️ CRITICAL REQUIREMENT: All code changes MUST include comprehensive test cases. No exceptions.**

#### Test Coverage Standards
- **Minimum Coverage:** 80% line coverage for all new code
- **Critical Path Coverage:** 100% coverage for core business logic, API endpoints, and data processing functions
- **Edge Case Testing:** All edge cases, error conditions, and boundary values must be tested
- **Regression Testing:** All bug fixes must include tests that prevent regression

#### Test Types and Organization
- **Unit Tests (`tests/unit/`):** Test individual functions, methods, and classes in isolation
  - Test all public methods and functions
  - Test private methods when they contain complex logic
  - Mock external dependencies (databases, APIs, file systems)
  - Test both happy path and error conditions
  
- **Integration Tests (`tests/integration/`):** Test interactions between components
  - Test API endpoints end-to-end
  - Test database operations with real database connections
  - Test file I/O operations with temporary files
  - Test external service integrations
  
- **End-to-End Tests (`tests/e2e/`):** Test complete user workflows
  - Test critical user journeys
  - Test system behavior under realistic conditions
  - Use realistic test data and scenarios

#### Test Structure and Best Practices
- **Test Organization:** Follow the standard `tests/` directory structure:
  ```
  tests/
  ├── unit/                    # Unit tests
  ├── integration/             # Integration tests
  ├── e2e/                     # End-to-end tests
  ├── fixtures/                # Test data and factories
  ├── conftest.py             # Pytest configuration and shared fixtures
  └── __init__.py
  ```

- **Test Naming Conventions:**
  - Test files: `test_<module_name>.py`
  - Test classes: `Test<ClassName>` or `Test<Functionality>`
  - Test methods: `test_<method_name>_<scenario>` (e.g., `test_create_user_with_valid_data`)

- **Test Documentation:**
  - Each test function must have a clear docstring explaining what is being tested
  - Use descriptive test names that explain the scenario being tested
  - Include comments for complex test setup or assertions

#### Mandatory Test Scenarios
**Every function/method MUST have tests for:**
1. **Happy Path:** Normal operation with valid inputs
2. **Edge Cases:** Boundary values, empty inputs, maximum/minimum values
3. **Error Conditions:** Invalid inputs, exceptions, system failures
4. **State Changes:** Verify object state changes correctly
5. **Side Effects:** File creation, database changes, API calls

#### Test Code Quality Standards
- **Clear Assertions:** Use specific assertions that explain what is being verified
  ```python
  # Good
  assert user.email == "test@example.com", "User email should be set correctly"
  assert len(users) == 2, "Should return exactly 2 users"
  
  # Avoid
  assert user
  assert users
  ```

- **Test Isolation:** Each test should be independent and not rely on other tests
- **Test Data:** Use factories, fixtures, or builders for consistent test data
- **Mocking Strategy:** Mock external dependencies but not the code under test
  ```python
  # Good - Mock external service
  @patch('app.services.external_api.make_request')
  def test_process_data_with_api_call(self, mock_api):
      mock_api.return_value = {'status': 'success'}
      result = process_data(test_data)
      assert result['processed'] == True
  ```

- **Exception Testing:** Test that appropriate exceptions are raised
  ```python
  def test_divide_by_zero_raises_error():
      with pytest.raises(ValueError, match="Cannot divide by zero"):
          divide(10, 0)
  ```

#### Test Configuration and Tools
- **Pytest Configuration:** Use `pytest.ini` or `pyproject.toml` for test configuration
- **Coverage Tools:** Use `pytest-cov` for coverage reporting
- **Test Fixtures:** Create reusable fixtures in `conftest.py`
- **Parameterized Tests:** Use `@pytest.mark.parametrize` for testing multiple scenarios
  ```python
  @pytest.mark.parametrize("input_val,expected", [
      (1, 2),
      (2, 4),
      (3, 6),
  ])
  def test_double_value(input_val, expected):
      assert double(input_val) == expected
  ```

#### Test Data Management
- **Test Factories:** Use factory_boy or similar for creating test objects
- **Fixtures:** Create reusable test data in `tests/fixtures/`
- **Database Testing:** Use separate test database or in-memory database
- **File Testing:** Use `tempfile` module for file operations testing

#### Continuous Integration Requirements
- **Pre-commit Hooks:** Tests must pass before commits
- **CI Pipeline:** All tests must pass in CI/CD pipeline
- **Coverage Gates:** CI must fail if coverage drops below minimum threshold
- **Test Performance:** Tests should run in reasonable time (unit tests < 1s each)

#### Code Review Requirements
- **Test Review:** All test code must be reviewed as thoroughly as production code
- **Test Documentation:** Tests must be self-documenting and explain business requirements
- **Test Maintenance:** Update tests when requirements change

#### Examples of Comprehensive Test Coverage
```python
# Example: Testing a user service
class TestUserService:
    """Comprehensive test suite for UserService."""
    
    def test_create_user_with_valid_data(self):
        """Test creating user with all required fields."""
        user_data = {"email": "test@example.com", "name": "Test User"}
        user = UserService.create_user(user_data)
        
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.id is not None
        assert user.created_at is not None
    
    def test_create_user_with_duplicate_email_raises_error(self):
        """Test that creating user with existing email raises appropriate error."""
        existing_user = UserFactory(email="test@example.com")
        
        with pytest.raises(DuplicateEmailError, match="Email already exists"):
            UserService.create_user({"email": "test@example.com", "name": "New User"})
    
    def test_create_user_with_invalid_email_raises_error(self):
        """Test that invalid email format raises validation error."""
        invalid_emails = ["", "invalid", "@example.com", "test@"]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                UserService.create_user({"email": email, "name": "Test"})
    
    @patch('app.services.email_service.send_welcome_email')
    def test_create_user_sends_welcome_email(self, mock_send_email):
        """Test that welcome email is sent after user creation."""
        user_data = {"email": "test@example.com", "name": "Test User"}
        user = UserService.create_user(user_data)
        
        mock_send_email.assert_called_once_with(user.email, user.name)
```

#### Common Testing Patterns and Anti-Patterns

**✅ Good Testing Practices:**
```python
# Clear test names that describe scenario
def test_user_login_with_invalid_password_returns_error():
    """Test that login with wrong password returns appropriate error."""
    user = UserFactory(password="correct_password")
    
    with pytest.raises(AuthenticationError, match="Invalid credentials"):
        authenticate_user(user.email, "wrong_password")

# Use fixtures for consistent test data
@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return UserFactory(
        email="test@example.com",
        name="Test User",
        is_active=True
    )

# Parameterized tests for multiple scenarios
@pytest.mark.parametrize("email,expected_valid", [
    ("test@example.com", True),
    ("invalid.email", False),
    ("", False),
    ("test@", False),
])
def test_email_validation(email, expected_valid):
    """Test email validation with various inputs."""
    assert is_valid_email(email) == expected_valid
```

**❌ Testing Anti-Patterns to Avoid:**
```python
# Don't write tests that don't test anything meaningful
def test_user_exists():
    user = User()
    assert user  # This doesn't test anything useful

# Don't write tests that are too broad
def test_entire_user_workflow():
    # This test is doing too much - break it down
    user = create_user()
    login_user(user)
    update_user_profile(user)
    delete_user(user)

# Don't use production data in tests
def test_with_production_data():
    users = User.objects.all()  # Never use production data
    assert len(users) > 0

# Don't write tests that depend on external services
def test_api_integration():
    response = requests.get("https://api.external-service.com")  # Unreliable
    assert response.status_code == 200
```

#### Test-Driven Development (TDD) - Recommended Approach
- **Red-Green-Refactor Cycle:**
  1. **Red:** Write a failing test that defines the desired functionality
  2. **Green:** Write the minimum code necessary to make the test pass
  3. **Refactor:** Improve the code while keeping tests passing

- **Benefits of TDD:**
  - Ensures all code is testable by design
  - Provides immediate feedback on code changes
  - Results in better code architecture and design
  - Creates comprehensive test documentation of requirements

- **When to Use TDD:**
  - New feature development
  - Bug fixes (write test to reproduce bug first)
  - API endpoint development
  - Complex business logic implementation

### Security Considerations
- **Input Validation:** Validate and sanitize all user input and external data
- **Sensitive Data Handling:** Never hardcode secrets; use environment variables or secret management
- **Least Privilege:** Operate with minimum necessary privileges

## 6. Documentation Standards and Project Tracking (MANDATORY)

### Core Documentation Requirements
All code changes must include corresponding updates to project documentation:

#### PROJECT.md (MANDATORY)
- **Purpose:** Central reference for project structure, dependencies, and architecture
- **Update Requirements:**
  - Any changes to project structure or directory layout
  - Addition, removal, or modification of dependencies
  - Architectural decisions or pattern changes
  - Configuration changes or new environment variables
  - API endpoint modifications or new services
- **Review Process:** PROJECT.md updates must be reviewed alongside code changes

#### requirements.txt and dependency management (MANDATORY)
- **Production Dependencies:** Keep requirements.txt current with all production dependencies
- **Development Dependencies:** Maintain requirements-dev.txt for development tools
- **Version Pinning:** Pin all dependencies to specific versions for reproducibility
- **Security Updates:** Regularly update dependencies for security patches
- **Documentation:** Include comments in requirements files explaining purpose of dependencies

```txt
# requirements.txt example
# Core XML processing dependencies
lxml==4.9.3                   # High-performance XML parser with C bindings
defusedxml==0.7.1            # XML bomb protection for secure parsing

# CLI and argument parsing
argparse                     # Standard library - command-line interface

# Data validation and serialization
jsonschema==4.19.0           # JSON schema validation for output
pydantic==2.5.0              # Data validation for configuration

# Development and testing
pytest==7.4.0               # Testing framework
pytest-cov==4.1.0           # Coverage reporting
black==23.7.0                # Code formatting
flake8==6.0.0                # Linting
mypy==1.5.0                  # Type checking
```

#### README.md (MANDATORY)
- **Project Overview:** Clear description of the XML parser plugin and its capabilities
- **Setup Instructions:** Step-by-step installation and configuration guide
- **Usage Examples:** Basic usage patterns and CLI commands
- **Architecture Overview:** High-level system architecture and component interactions
- **Contributing Guidelines:** Link to CONTRIBUTING.md and development setup
- **Update Requirements:** Must be updated when:
  - Installation steps change
  - New features are added
  - Configuration requirements change
  - Usage patterns are modified

#### CHANGELOG.md (MANDATORY)
- **Format:** Follow Keep a Changelog format (https://keepachangelog.com/)
- **Required Sections:** Added, Changed, Deprecated, Removed, Fixed, Security
- **Update Requirements:** Every code change must include changelog entry
- **Version Management:** Align with semantic versioning (major.minor.patch)

```markdown
# Changelog

## [Unreleased]

### Added
- New recursive AST builder with tail text support
- CLI option for custom output file paths
- Support for XML namespace processing

### Changed
- Improved lxml error handling and recovery modes
- Enhanced JSON output formatting with configurable indentation
- Updated XML validation to handle edge cases

### Fixed
- Fixed memory usage for large XML files
- Resolved encoding issues with non-UTF8 files
- Fixed AST generation for self-closing XML tags

## [0.1.0] - 2024-08-12

### Added
- Initial XML parser implementation using lxml
- Command-line interface with argparse
- JSON AST output with pretty printing

...existing entries...
```

#### CONTRIBUTING.md (MANDATORY)
- **Development Setup:** Detailed environment setup instructions
- **Code Standards:** Reference to this code-guidelines.instructions.md
- **Testing Requirements:** Testing standards and coverage requirements
- **Pull Request Process:** Step-by-step contribution workflow
- **Code Review Guidelines:** What reviewers should check

### Documentation Quality Standards
- **Clarity:** Use clear, concise language accessible to new team members
- **Completeness:** Cover all aspects necessary for understanding and using the code
- **Accuracy:** Ensure documentation matches current code implementation
- **Examples:** Include practical examples and code snippets
- **Maintenance:** Keep documentation current with code changes

### Code Review Process (MANDATORY)

#### Review Requirements
**All code changes must be reviewed by at least one other developer before merging.**

#### Pre-Review Checklist (Developer)
- [ ] All tests pass locally with minimum 80% coverage
- [ ] Code follows style guidelines (black, flake8, mypy)
- [ ] Documentation is updated (PROJECT.md, README.md, CHANGELOG.md)
- [ ] Dependencies are updated in requirements files
- [ ] Commit messages are clear and descriptive
- [ ] No sensitive data or secrets in code

#### Review Checklist (Reviewer)
- [ ] **Code Quality:** Adherence to coding standards and best practices
- [ ] **Testing:** Comprehensive test coverage including edge cases
- [ ] **Documentation:** All required documentation files are updated
- [ ] **Architecture:** Changes align with project architecture principles
- [ ] **Security:** No security vulnerabilities introduced
- [ ] **Performance:** No obvious performance regressions
- [ ] **Dependencies:** New dependencies are justified and secure
- [ ] **Breaking Changes:** Breaking changes are documented and versioned

#### Review Process Steps
1. **Initial Review:** Check code quality, tests, and documentation
2. **Feedback:** Provide constructive feedback and improvement suggestions
3. **Iteration:** Developer addresses feedback and updates code
4. **Final Approval:** Reviewer approves changes after requirements are met
5. **Merge:** Code is merged with proper commit message and tags

#### Review Standards
- **Constructive Feedback:** Focus on improvement rather than criticism
- **Knowledge Sharing:** Use reviews as learning opportunities
- **Consistency:** Apply standards consistently across all code changes
- **Timeliness:** Provide reviews within 24-48 hours during business days

## 7. Tools and Automation

### Testing Tools (MANDATORY)
- **pytest:** Primary testing framework - supports fixtures, parameterization, and plugins
- **pytest-cov:** Code coverage reporting and enforcement
- **pytest-mock:** Enhanced mocking capabilities
- **factory_boy:** Test data factories for consistent test objects
- **freezegun:** Time-travel testing for date/time dependent code
- **responses:** Mock HTTP requests for external API testing
- **pytest-asyncio:** Testing async/await code
- **tox:** Testing across multiple Python versions and environments

### Documentation Tools
- **Markdown Linting:** Use markdownlint for consistent documentation formatting
- **Link Checking:** Validate all documentation links are functional
- **Spell Checking:** Use automated spell checking for documentation
- **Documentation Generation:** Consider automated API documentation generation

### Automation Requirements
- **Pre-commit Hooks:** Include documentation validation in pre-commit checks
- **CI/CD Pipeline:** Validate documentation completeness in automated builds
- **Dependency Scanning:** Automated security scanning of dependencies
- **Documentation Builds:** Automated building and deployment of documentation

### Pre-Commit Requirements
All commits must pass these checks:
```bash
# Run before every commit
pytest --cov=app --cov-fail-under=80    # Test coverage
black --check .                         # Code formatting
flake8 .                                # Linting
mypy app/                               # Type checking
markdownlint docs/ *.md                 # Documentation formatting
pip-audit                               # Security vulnerability scanning
```

## 8. Architectural Guidelines

### Architectural Patterns
- **Choose Appropriate Style:**
  - **Monolith:** Simple, single deployment unit (good for small projects)
  - **Layered Architecture:** Separate presentation, application, business logic, and data access layers
  - **Hexagonal Architecture:** Separate business logic from external concerns (UI, databases, APIs)
  - **Event-Driven Architecture:** Components communicate via events for decoupling

### Design Considerations
- **Separation of Concerns:** High cohesion within modules, low coupling between modules
- **Scalability Planning:** Identify potential bottlenecks and consider asynchronous programming for I/O-bound tasks
- **Security by Design:** Implement input validation, authentication, authorization, and secret management
- **Observability:** Use structured logging, metrics, and tracing for monitoring and debugging

### Core Design Principles
- **Single Responsibility Principle:** A class or module should have only one reason to change
- **Open/Closed Principle:** Open for extension, closed for modification
- **Liskov Substitution Principle:** Subtypes must be substitutable for their base types
- **Interface Segregation Principle:** Create fine-grained, specific interfaces
- **Dependency Inversion Principle:** Depend on abstractions, not concretions
- **Composition Over Inheritance:** Prefer composing objects over deep inheritance hierarchies

## 6. Documentation Requirements

**Remember: Documentation is as important as code. Treat documentation updates with the same rigor as code changes. Well-maintained documentation ensures project sustainability and team productivity.**

### Code Review Checklist for Testing
**Reviewers must verify:**
- [ ] All new functions/methods have corresponding unit tests
- [ ] All XML parsing logic has integration tests
- [ ] All error conditions are tested with appropriate assertions
- [ ] Test coverage meets minimum 80% threshold
- [ ] Tests are properly organized in `tests/` directory structure
- [ ] Test names clearly describe what is being tested
- [ ] Tests use appropriate fixtures and factories
- [ ] External dependencies are properly mocked
- [ ] Tests include both positive and negative scenarios
- [ ] Edge cases and boundary conditions are covered

### Documentation Review Checklist
**Reviewers must verify:**
- [ ] PROJECT.md reflects any structural or architectural changes
- [ ] requirements.txt includes all new dependencies with pinned versions
- [ ] README.md is updated for any usage or setup changes
- [ ] CHANGELOG.md includes appropriate entries for all changes
- [ ] Code comments explain complex XML parsing logic
- [ ] XML parser documentation is updated for new features
- [ ] Breaking changes are clearly documented

### Project Tracking and Maintenance

#### Regular Maintenance Tasks
- **Weekly:** Review and update dependencies for security patches
- **Monthly:** Review documentation for accuracy and completeness
- **Quarterly:** Audit and cleanup unused dependencies
- **Release Cycle:** Update version numbers across all documentation

#### Quality Metrics Tracking
- **Test Coverage:** Monitor coverage trends over time
- **Documentation Coverage:** Track completeness of API and code documentation
- **Code Quality:** Monitor technical debt and code complexity metrics
- **Dependencies:** Track dependency freshness and security status

**Remember: Documentation is as important as code. Treat documentation updates with the same rigor as code changes. Well-maintained documentation ensures project sustainability and team productivity.**