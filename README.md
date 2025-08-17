# 🐍 Snake Pipe

A modern, flexible Python-based ETL (Extract, Transform, Load) pipeline framework designed for data engineers and analysts who need to build robust data processing workflows.

## 🌟 Features

- **Multiple Data Sources**: Extract from CSV files, databases, APIs, and more
- **Powerful Transformations**: Built-in data cleaning, validation, and enrichment tools
- **Flexible Loading**: Save to files, databases, or data warehouses (Snowflake, BigQuery, Redshift)
- **Method Chaining**: Intuitive fluent interface for building pipelines
- **Type Safety**: Built with modern Python typing and Pydantic validation
- **Comprehensive Logging**: Full observability of your data pipelines
- **Extensible**: Easy to add custom transformation functions

## 📦 Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/vivekbarsagadey/snake-pipe.git
cd snake-pipe

# Install with uv
uv sync

# Install with optional dependencies
uv sync --extra warehouse  # For data warehouse support
uv sync --extra dev        # For development tools
uv sync --extra all        # For everything
```

### Using pip

```bash
pip install -e .
# or with optional dependencies
pip install -e ".[warehouse,dev]"
```

## 🚀 Quick Start

### 1. Set up environment variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Basic Pipeline Example

```python
from snake_pipe.pipeline import Pipeline

# Create a simple ETL pipeline
pipeline = Pipeline("my_first_pipeline")

result = (pipeline
    .extract_from_csv("data/input.csv")
    .clean_data()
    .load_to_csv("data/output.csv")
    .run())

print(f"Processed {len(result)} rows successfully!")
```

### 3. Advanced Pipeline Example

```python
from snake_pipe.pipeline import Pipeline

# Create a more complex pipeline
pipeline = Pipeline("advanced_pipeline")

result = (pipeline
    # Extract data
    .extract_from_database(
        query="SELECT * FROM users WHERE created_date >= '2024-01-01'",
        connection_string="postgresql://user:pass@localhost:5432/mydb"
    )
    
    # Clean the data
    .clean_data(
        remove_duplicates={'subset': ['email']},
        handle_missing_values={'strategy': 'drop'},
        standardize_columns={'case': 'lower'}
    )
    
    # Validate data quality
    .validate_data(
        check_required_columns=['id', 'email', 'name'],
        check_data_types={'id': 'int', 'email': 'string'},
        check_unique_values=['id', 'email']
    )
    
    # Enrich the data
    .enrich_data(
        add_datetime_features={'datetime_column': 'created_date'},
        add_calculated_column={
            'column_name': 'full_name',
            'calculation': lambda df: df['first_name'] + ' ' + df['last_name']
        }
    )
    
    # Load to warehouse
    .load_to_database(
        table_name="processed_users",
        if_exists="replace"
    )
    
    .run())
```

## 📊 Data Sources & Destinations

### Extract From:
- **CSV Files**: `extract_from_csv(file_path)`
- **Databases**: `extract_from_database(query, connection_string)`
- **REST APIs**: `extract_from_api(endpoint, base_url, api_key)`

### Load To:
- **CSV Files**: `load_to_csv(filename)`
- **Excel Files**: `load_to_excel(filename)`
- **Databases**: `load_to_database(table_name)`
- **Data Warehouses**: Support for Snowflake, BigQuery, Redshift

## 🔧 Transformations

### Data Cleaning
```python
.clean_data(
    remove_duplicates={'subset': ['id']},
    handle_missing_values={'strategy': 'fill', 'fill_value': 0},
    standardize_columns={'case': 'lower'},
    remove_outliers={'columns': ['price'], 'method': 'iqr'}
)
```

### Data Validation
```python
.validate_data(
    check_required_columns=['id', 'name'],
    check_data_types={'id': 'int', 'price': 'float'},
    check_null_values={'max_null_percentage': 5.0},
    check_value_ranges={'age': {'min': 0, 'max': 120}}
)
```

### Data Enrichment
```python
.enrich_data(
    add_datetime_features={'date_column': 'created_at'},
    add_categorical_encoding={'categorical_columns': ['category']},
    add_text_features={'text_column': 'description'}
)
```

## 🖥️ Command Line Interface

```bash
# Run the default pipeline
uv run python scripts/run_pipeline.py

# Process a specific file
uv run python scripts/run_pipeline.py --input data/input.csv --output data/output.csv

# Run with verbose logging
uv run python scripts/run_pipeline.py --verbose

# Use as installed script
snake-pipe --help
```

## 🏗️ Project Structure

```
snake-pipe/
├── snake_pipe/              # Main package
│   ├── __init__.py
│   ├── pipeline.py           # Main pipeline orchestration
│   ├── config/
│   │   └── settings.py       # Configuration management
│   ├── extract/              # Data extraction modules
│   │   ├── csv_extractor.py
│   │   ├── db_extractor.py
│   │   └── api_extractor.py
│   ├── transform/            # Data transformation modules
│   │   ├── cleaners.py
│   │   ├── validators.py
│   │   └── enrichers.py
│   ├── load/                 # Data loading modules
│   │   ├── db_loader.py
│   │   ├── file_loader.py
│   │   └── warehouse_loader.py
│   └── utils/                # Utility functions
│       ├── logger.py
│       └── helpers.py
├── tests/                    # Test suite
├── scripts/                  # CLI scripts
│   ├── run_pipeline.py
│   └── sample_data.csv
├── pyproject.toml           # Project configuration
├── .env.example             # Environment variables template
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=snake_pipe --cov-report=html

# Run specific test file
uv run pytest tests/test_pipeline.py
```

## 🔧 Development

```bash
# Install development dependencies
uv sync --extra dev

# Format code
uv run black snake_pipe/
uv run isort snake_pipe/

# Type checking
uv run mypy snake_pipe/

# Linting
uv run flake8 snake_pipe/
```

## 📝 Configuration

Create a `.env` file in your project root:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# API
API_BASE_URL=https://api.example.com
API_KEY=your_api_key

# Paths
DATA_DIR=./data
LOG_DIR=./logs

# Pipeline Settings
BATCH_SIZE=1000
MAX_RETRIES=3
LOG_LEVEL=INFO
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`uv run pytest`)
6. Format your code (`uv run black .`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 📋 Requirements

- Python 3.10+
- pandas >= 2.0.0
- SQLAlchemy >= 2.0.0
- Pydantic >= 2.0.0
- Additional dependencies for specific features (see `pyproject.toml`)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- 📧 Email: vivek@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/vivekbarsagadey/snake-pipe/issues)
- 📖 Documentation: [GitHub Repository](https://github.com/vivekbarsagadey/snake-pipe)

## 🎯 Roadmap

- [ ] Web UI for pipeline monitoring
- [ ] Support for streaming data
- [ ] Integration with Apache Airflow
- [ ] More data warehouse connectors
- [ ] Docker containerization
- [ ] Kubernetes deployment templates

---

Made with ❤️ by [Vivek Barsagadey](https://github.com/vivekbarsagadey)
