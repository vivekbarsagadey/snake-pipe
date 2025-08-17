# Development commands

## Linting and Formatting

### Python Linting

```bash
# Run all Python linters
uv run pylint snake_pipe/
uv run flake8 snake_pipe/
uv run mypy snake_pipe/
uv run bandit -r snake_pipe/

# Auto-format Python code (200 char line length)
uv run black --line-length=200 snake_pipe/
uv run isort --profile=black --line-length=200 snake_pipe/
uv run autopep8 --in-place --max-line-length=200 --aggressive --aggressive -r snake_pipe/
```

### Markdown Formatting

```bash
# Format markdown files (200 char line length)
uv run mdformat --wrap=200 README.md
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run black
uv run pre-commit run flake8
```

## Testing

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=snake_pipe --cov-report=html

# Run specific test
uv run pytest tests/test_pipeline.py::test_pipeline_creation
```

## Building and Installation

```bash
# Build the package
uv build

# Install in development mode
uv pip install -e .

# Run CLI
snake-pipe
```
