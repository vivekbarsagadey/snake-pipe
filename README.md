## 📦 Installation

### Using uv (Recommended)

```bash

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

```text
snake-pipe/
├── snake_pipe/              # Main package
│   ├── __init__.py
│   ├── pipeline.py           # Main pipeline orchestration
│   ├── config/
│   │   └── settings.py       # Configuration management
│   ├── extract/              # Data extraction modules
│   ├── transform/            # Data transformation modules
│   ├── load/                 # Data loading modules
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

# Run tests with verbose output
uv run pytest -v
```



## 🔧 Development

This project uses comprehensive linting and formatting tools with a 200-character line length configuration.

### Quick Development Commands

```bash
# Install development dependencies
uv sync

# Auto-format code (200 char line length)
uv run black --line-length=200 snake_pipe/
uv run isort --profile=black --line-length=200 snake_pipe/

# Run linting
uv run flake8 snake_pipe/
uv run pylint snake_pipe/
uv run mypy snake_pipe/

# Run pre-commit hooks on all files
uv run pre-commit run --all-files

# Run tests with coverage
uv run pytest --cov=snake_pipe --cov-report=html
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development instructions.

### Building & Testing

```bash
# Build the package
uv build

# Install in development mode
uv pip install -e .

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
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
1. Make your changes
1. Add tests for new functionality
1. Run the test suite (`uv run pytest`)
1. Format your code (`uv run black .`)
1. Commit your changes (`git commit -m 'Add amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

## 📋 Requirements

- Python 3.11+
- pandas >= 2.0.0
- SQLAlchemy >= 2.0.0
- Pydantic >= 2.0.0
- Additional dependencies for specific features (see `pyproject.toml`)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

______________________________________________________________________