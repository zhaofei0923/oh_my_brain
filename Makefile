.PHONY: install dev test lint format typecheck docs clean all

# Default target
all: lint typecheck test

# Install production dependencies
install:
	uv sync

# Install all dependencies including dev
dev:
	uv sync --all-groups
	uv run pre-commit install

# Run tests
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ --cov=oh_my_brain --cov-report=html --cov-report=term

# Run linter
lint:
	uv run ruff check src/ tests/

# Run linter with auto-fix
lint-fix:
	uv run ruff check --fix src/ tests/

# Format code
format:
	uv run ruff format src/ tests/

# Run type checker
typecheck:
	uv run mypy src/

# Run all checks (lint + typecheck + test)
check: lint typecheck test

# Build documentation
docs:
	uv run mkdocs build

# Serve documentation locally
docs-serve:
	uv run mkdocs serve

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	uv build

# Start Brain server (development)
start-brain:
	uv run oh-my-brain start

# Start Worker (development)
start-worker:
	uv run oh-my-brain worker --count 1

# Start Redis (requires Docker)
redis:
	docker run -d --name oh-my-brain-redis -p 6379:6379 redis:alpine

# Stop Redis
redis-stop:
	docker stop oh-my-brain-redis && docker rm oh-my-brain-redis

# Show help
help:
	@echo "Available targets:"
	@echo "  install     - Install production dependencies"
	@echo "  dev         - Install all dependencies and setup pre-commit"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage report"
	@echo "  lint        - Run linter"
	@echo "  lint-fix    - Run linter with auto-fix"
	@echo "  format      - Format code"
	@echo "  typecheck   - Run type checker"
	@echo "  check       - Run all checks (lint + typecheck + test)"
	@echo "  docs        - Build documentation"
	@echo "  docs-serve  - Serve documentation locally"
	@echo "  clean       - Clean build artifacts"
	@echo "  build       - Build package"
	@echo "  start-brain - Start Brain server"
	@echo "  start-worker- Start Worker"
	@echo "  redis       - Start Redis in Docker"
	@echo "  redis-stop  - Stop Redis container"
