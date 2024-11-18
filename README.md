# Credit Card Fraud Detection System

## Overview
Production-ready machine learning system for detecting fraudulent credit card transactions. Built with clean architecture principles, monitoring, and CI/CD integration.

## Prerequisites
- Python 3.10+
- Poetry

## Quick Start
```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Train model
poetry run python -m fraud_detection.core.trainer

# Start API
poetry run uvicorn fraud_detection.api.app:app --reload
```

## Project Structure
- `fraud_detection/`: Main package directory
  - `core/`: Core ML pipeline components
  - `models/`: Model definition and artifacts
  - `api/`: FastAPI service
- `tests/`: Unit and integration tests
- `config/`: Configuration files

## Development
```bash
# Install dev dependencies
poetry install --with dev

# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy .

# Run tests with coverage
poetry run pytest --cov
```

other files[]: # Path: .gitignore