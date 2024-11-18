# Credit Card Fraud Detection System

A production-ready machine learning system for detecting fraudulent credit card transactions. This project implements a complete ML pipeline with real-time prediction capabilities, monitoring, and MLflow experiment tracking.

## 🌟 Features

- Real-time fraud detection via REST API
- MLflow experiment tracking and model versioning
- Configurable environments (development/production)
- Comprehensive testing and CI/CD pipeline
- Prometheus metrics for monitoring
- Docker support for deployment

## 🏗️ Project Structure

```
fraud_detection/
├── config/                 # Configuration files
│   ├── development.yaml
│   └── production.yaml
├── data/                  # Data directory (gitignored)
│   ├── raw/
│   └── processed/
├── fraud_detection/       # Main package
│   ├── core/             # Core functionality
│   │   ├── config.py
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
- Access to the Credit Card Fraud Detection dataset

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
```

4. Download the dataset:
```bash
# Using provided script
poetry run python scripts/download_data.py
```

### Running the System

1. Train the model:
```bash
# Run training pipeline
poetry run python -m fraud_detection.core.trainer
```

2. Start the API:
```bash
poetry run uvicorn fraud_detection.api.app:app --reload
```

3. View MLflow dashboard:
```bash
poetry run mlflow ui
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/core/test_simple.py

# Run with coverage
poetry run pytest --cov=fraud_detection
```

### Code Quality Checks

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
# Install pre-commit hooks
poetry run pre-commit install

# Run hooks manually
poetry run pre-commit run --all-files
```

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for CI/CD with the following stages:

1. **Lint**: Code quality checks
2. **Test**: Run test suite
3. **Train**: Train and validate model
4. **Deploy**: (Optional) Deploy model to production

To run the pipeline locally:
```bash
# Run pipeline script
./scripts/run_pipeline.sh
```

## 📊 MLflow Tracking

MLflow is used for experiment tracking and model versioning:

```bash
# View experiments
poetry run mlflow ui

# Train with tracking
MLFLOW_TRACKING_URI=mlruns poetry run python -m fraud_detection.core.trainer
```

## 🛠️ Configuration

The system uses a hierarchical configuration system:

1. Environment variables (highest priority)
2. Environment-specific config files
3. Default configuration (lowest priority)

Example configuration:
```bash
# Set environment
export FRAUD_DETECTION_ENV=development

# Use specific config
export FRAUD_DETECTION_CONFIG=config/development.yaml
```

## 🔍 API Documentation

Once running, API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📈 Monitoring

Prometheus metrics are exposed at `/metrics` endpoint:
- Model prediction latency
- Request counts
- Error rates

## 🔐 Security

- Sensitive data is handled via environment variables
- Model artifacts are versioned and tracked
- API endpoints include rate limiting

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests and quality checks
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Credit Card Fraud Detection dataset from Kaggle
- MLflow for experiment tracking
- FastAPI for API implementation

Would you like me to:
1. Add more specific examples?
2. Add troubleshooting section?
3. Add deployment instructions?
4. Add performance benchmarks?
