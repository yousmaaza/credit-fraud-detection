# Credit Card Fraud Detection System

A production-ready machine learning system for detecting fraudulent credit card transactions. This project implements a complete ML pipeline with real-time prediction capabilities, monitoring, and MLflow experiment tracking.

## 🌟 Features

- Real-time fraud detection via REST API
- MLflow experiment tracking and model versioning
- Comprehensive CLI for data and model management
- Configurable environments (development/production)
- Automated testing and CI/CD pipeline

## 🏗️ Project Structure

```
fraud_detection/
├── config/                 # Configuration files
│   ├── development.yaml   # Development environment config
│   └── production.yaml    # Production environment config
├── data/                  # Data directory (gitignored)
│   ├── raw/              # Raw dataset files
│   └── processed/        # Processed datasets
├── fraud_detection/       # Main package
│   ├── api/              # API implementation
│   │   ├── app.py        # FastAPI application
│   │   └── schemas.py    # features and response schemas
│   ├── cli/              # CLI commands
│   │   ├── data.py       # Data management commands
│   │   └── inference.py  # Inference commands
│   │   ├── main.py       # Fraud Application commands
│   │   └── promote.py    # Promoting the model from staging to production
│   │   └── train.py      # Training commands
│   ├── core/             # Core functionality
│   │   ├── config.py    # Configuration management
│   │   ├── data_loader.py
│   │   ├── mlflow_handler.py
│   │   └── trainer.py
│   └── api/              # API implementation
├── models/               # Model artifacts (gitignored)
├── tests/               # Test suite
├── .env.template        # Environment variables template
└── pyproject.toml       # Project dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Poetry for dependency management
- Kaggle account for dataset access

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fraud-detection
```

2. Install dependencies:
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

3. Set up environment:
```bash
# Copy environment template
cp .env.template .env.development

# Edit environment variables
vim .env.development

# Required variables:
FRAUD_DETECTION_ENV=development
KAGGLE_TOKEN=your_kaggle_token
```

### Using the CLI

The project provides a comprehensive CLI for managing data and training models:

1. Data Management:
```bash
# Show available data commands
poetry run fraud-detection data --help

# Download the dataset
poetry run fraud-detection data download

# Validate downloaded data
poetry run fraud-detection data validate

# Show data information
poetry run fraud-detection data info
```

2. Model Training:
```bash
# Show available training commands
poetry run fraud-detection train --help

# Train the model
poetry run fraud-detection train run

```
3. Promote the model from staging to production:
```bash
# Promote the model
poetry run fraud-detection promote
```

4. Batch inference:
```bash
# Show available inference commands
poetry run fraud-detection inference --help

# Run batch inference
poetry run fraud-detection inference INPUT_DATA OUTPUT_DATA

```
### Configuration

The project uses both YAML configuration files and environment variables:

1. Environment Variables (.env files):
```bash
# Development environment
FRAUD_DETECTION_ENV=development
DATASET_URL=https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud
DATASET_MIN_FILE_SIZE=100000000
KAGGLE_TOKEN=your_token_here
```

2. Configuration Files (config/):
```yaml
# development.yaml
data:
  train_path: "data/raw/creditcard.csv"
  processed_data_dir: "data/processed"
  # ...

model:
  target_column: "Class"
  features_to_exclude: ["Time", "Class"]
  # ...
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=fraud_detection
```

### Code Quality

```bash
# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy fraud_detection/

# Lint
poetry run flake8 fraud_detection/
```

### Pre-commit Hooks

```bash
# Install hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

## 📊 MLflow Tracking

View and manage experiments:

```bash
# Start MLflow UI
poetry run mlflow ui

# View at http://localhost:5000
```

## 🔍 API Usage

Start the API server:

```bash
# Development server
poetry run uvicorn fraud_detection.api.app:app --reload

# Production server
poetry run gunicorn fraud_detection.api.app:app
```

API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for CI/CD:

1. On Pull Request:
   - Code quality checks
   - Tests
   - Development training

2. On Main Branch:
   - Full training pipeline
   - Model evaluation
   - Optional deployment


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Credit Card Fraud Detection dataset from Kaggle
- MLflow for experiment tracking
- FastAPI for API implementation
