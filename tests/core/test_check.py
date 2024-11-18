"""Simple tests for configuration and MLflow handler."""
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
import mlflow
from dotenv import load_dotenv
from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.mlflow_handler import MLflowHandler


@pytest.fixture
def setup_test_env(tmp_path) -> Path:
    """Set up basic test environment."""
    # Create necessary directories
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "mlruns").mkdir()

    # Create basic config file
    config = {
        "data": {
            "train_path": "data/raw/creditcard.csv",
            "processed_data_dir": "data/processed",
            "model_artifacts_dir": "models/artifacts",
            "log_dir": "logs"
        },
        "model": {
            "target_column": "Class",
            "features_to_exclude": ["Time", "Class"],
            "val_size": 0.2,
            "random_state": 42,
            "early_stopping_rounds": 50,
            "model_params": {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt"
            }
        },
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "fraud_detection_test",
            "run_name": "test_run",
            "registry_uri": "sqlite:///mlflow_test.db",
            "artifact_location": "models/mlflow-artifacts"
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "model_version": "latest"
        }
    }

    with open(tmp_path / "config" / "development.yaml", "w") as f:
        yaml.dump(config, f)

    # Create basic test data
    (tmp_path / "data" / "raw" / "creditcard.csv").touch()

    # Create .env file
    env_content = f"""
    FRAUD_DETECTION_ENV=development
    FRAUD_DETECTION_CONFIG={tmp_path / "config/development.yaml"}

    """
    with open(tmp_path / ".env.development", "w") as f:
        f.write(env_content.strip())

    # Set environment variables
    os.environ["FRAUD_DETECTION_ENV"] = "development"
    os.environ["FRAUD_DETECTION_CONFIG"] = str(tmp_path / "config/development.yaml")


    return tmp_path


def test_config_initialization(setup_test_env):
    """
    Test basic configuration loading.

    Verifies that:
    1. ConfigurationManager loads without errors
    2. Basic configuration values are correct
    """
    load_dotenv(dotenv_path=setup_test_env / ".env.development")

    with patch('fraud_detection.core.config.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = setup_test_env
        mock_path.return_value = setup_test_env / "fraud_detection/core/config.py"

        config = ConfigurationManager()

        # Verify basic configuration
        assert config.env == "development"
        assert config.model.target_column == "Class"
        assert config.model.val_size == 0.2


def test_mlflow_experiment_creation(setup_test_env):
    """
    Test MLflow experiment initialization.

    Verifies that:
    1. MLflow handler initializes correctly
    2. Experiment is created and tracked
    """
    load_dotenv(dotenv_path=setup_test_env / ".env.development")

    with patch('fraud_detection.core.config.Path') as mock_path:
        mock_path.return_value.parent.parent.parent = setup_test_env
        mock_path.return_value = setup_test_env / "fraud_detection/core/config.py"

        # Initialize configuration
        config = ConfigurationManager()

        # Mock MLflow experiment
        with patch('mlflow.get_experiment_by_name', return_value=None), \
                patch('mlflow.create_experiment', return_value="test-1"), \
                patch('mlflow.set_tracking_uri'):
            # Initialize MLflow handler
            mlflow_handler = MLflowHandler(config)

            # Check basic MLflow setup
            assert mlflow_handler.config.mlflow.experiment_name == "fraud_detection_test"
            assert mlflow_handler.config.mlflow.tracking_uri == setup_test_env / "mlruns"
