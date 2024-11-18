#!/bin/bash

# Help message
show_help() {
    echo "Usage: ./scripts/run_pipeline.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -s, --step        Specific step to run (lint, test, train)"
    echo "  -e, --env         Environment (development, production)"
    echo "  --dry-run         Show what would be executed without running"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_pipeline.sh                    # Run full pipeline"
    echo "  ./scripts/run_pipeline.sh -s lint           # Run only lint step"
    echo "  ./scripts/run_pipeline.sh -e production     # Run in production environment"
}

# Default values
STEP="all"
ENV="development"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--step)
            STEP="$2"
            shift
            shift
            ;;
        -e|--env)
            ENV="$2"
            shift
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Function to run a command or show it in dry-run mode
run_command() {
    if [ "$DRY_RUN" = true ]; then
        echo "Would run: $@"
    else
        echo "Running: $@"
        "$@"
    fi
}

# Function to handle errors
handle_error() {
    echo "Error: Step $1 failed"
    exit 1
}

# Run lint checks
run_lint() {
    echo "Running lint checks..."
    run_command poetry run black . --check || handle_error "black"
    run_command poetry run isort . --check-only || handle_error "isort"
    run_command poetry run flake8 fraud_detection/ tests/ || handle_error "flake8"
    run_command poetry run mypy fraud_detection/ || handle_error "mypy"
    echo "Lint checks passed! ✨"
}

# Run tests
run_tests() {
    echo "Running tests..."
    # Run unit tests
    poetry run pytest tests/core/test_simple.py -v || handle_error "pytest"
    echo "Tests passed! ✨"
}

# Run training
run_training() {
    echo "Running training in $ENV environment..."

    # Create environment file
    if [ "$ENV" = "production" ]; then
        cat > .env << EOL
FRAUD_DETECTION_ENV=production
FRAUD_DETECTION_CONFIG=config/production.yaml
MLFLOW_RUN_NAME=lightgbm_prod_local
EOL
    else
        cat > .env << EOL
FRAUD_DETECTION_ENV=development
FRAUD_DETECTION_CONFIG=config/development.yaml
MLFLOW_RUN_NAME=lightgbm_dev_local
EOL
    fi

    # Set up directories
    mkdir -p data/raw data/processed models/artifacts logs mlruns

    source .env.local

    # Download dataset if not exists
    if [ ! -f data/raw/creditcard.csv ]; then
        echo "Downloading dataset..."
        if [ -z "$KAGGLE_TOKEN" ]; then
            echo "Error: KAGGLE_TOKEN environment variable not set"
            exit 1
        fi
        run_command curl -L -o archive.zip \
            -H "Authorization: Basic $KAGGLE_TOKEN" \
            https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud

        run_command unzip archive.zip -d data/raw
        run_command rm archive.zip
    fi

    # Run training
    run_command poetry run python -m fraud_detection.core.trainer || handle_error "training"

    echo "Training completed! ✨"
}

# Main execution
echo "Starting pipeline in $ENV environment..."

case $STEP in
    "lint")
        run_lint
        ;;
    "test")
        run_tests
        ;;
    "train")
        run_training
        ;;
    "all")
        run_lint
#        run_tests
        run_training
        ;;
    *)
        echo "Invalid step: $STEP"
        show_help
        exit 1
        ;;
esac

echo "Pipeline completed successfully! ✨ 🚀 ✨"
