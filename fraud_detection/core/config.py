import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from fraud_detection.core.logger import setup_logger


@dataclass
class DataConfig:
    """Data configuration parameters."""

    train_path: Path
    processed_data_dir: Path
    model_artifacts_dir: Path
    log_dir: Path


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    target_column: str
    features_to_exclude: list[str]
    val_size: float
    random_state: int
    early_stopping_rounds: int
    model_params: Dict[str, Any]


@dataclass
class MLflowConfig:
    """MLflow's configuration parameters."""

    tracking_uri: str
    experiment_name: str
    run_name: str
    registry_uri: str
    artifact_location: Path


@dataclass
class APIConfig:
    """API configuration parameters."""

    host: str
    port: int
    model_version: str


class ConfigurationManager:
    """Manages configuration loading and access."""

    def __init__(self) -> None:
        """Initialize configuration manager using environment variables."""
        # Set up project root (2 levels up from this file)
        self.project_root = Path(__file__).parent.parent.parent

        # Load environment variables
        self._load_env_vars()

        # Set up configuration
        self.config_path = self._get_config_path()
        self._load_config()

        # Set up directories and logger
        self._setup_directories()
        self.logger = setup_logger(self.data.log_dir)

        # Log initialization
        self.logger.info(f"Initialized configuration for environment: {self.env}")
        self.logger.info(f"Using config file: {self.config_path}")

    def _load_env_vars(self) -> None:
        """Load environment variables from .env file."""
        # First, check if FRAUD_DETECTION_ENV is already set
        self.env = os.getenv("FRAUD_DETECTION_ENV", "development")

        # Look for environment-specific .env file first
        env_paths = [
            self.project_root
            / f".env.{self.env}",  # .env.development or .env.production
            self.project_root / ".env",  # fallback to default .env
        ]

        env_file_loaded = False
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                env_file_loaded = True
                break

        if not env_file_loaded:
            raise FileNotFoundError(
                f"No environment file found. Looked for: {[str(p) for p in env_paths]}"
            )

        # Update env after loading .env file (in case it was changed)
        self.env = os.getenv("FRAUD_DETECTION_ENV", "development")

    def _get_config_path(self) -> Path:
        """Get configuration file path from environment variables."""
        config_path = os.getenv("FRAUD_DETECTION_CONFIG")
        if config_path:
            return self.project_root / config_path
        return self.project_root / "config" / f"{self.env}.yaml"

    def _setup_directories(self) -> None:
        """Set up project directories based on environment variables."""
        data_dir = os.getenv("FRAUD_DETECTION_DATA")
        if data_dir:
            data_path = self.project_root / data_dir
            self.data.train_path = data_path / "creditcard.csv"
            self.data.processed_data_dir = data_path / "processed"

        # Create necessary directories
        self.data.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.data.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.data.log_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow.artifact_location.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> None:
        """Load and validate configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Initialize data paths relative to project root
        data_config = config["data"]
        self.data = DataConfig(
            train_path=self.project_root
            / os.getenv("FRAUD_DETECTION_DATA", data_config["train_path"]),
            processed_data_dir=self.project_root / data_config["processed_data_dir"],
            model_artifacts_dir=self.project_root / data_config["model_artifacts_dir"],
            log_dir=self.project_root / data_config["log_dir"],
        )

        # Initialize Model configs
        self.model = ModelConfig(**config["model"])

        # Initialize MLflow configs with environment variables
        self.mlflow = MLflowConfig(
            tracking_uri=self.project_root / config["mlflow"]["tracking_uri"],
            experiment_name=config["mlflow"]["experiment_name"],
            run_name=config["mlflow"]["run_name"],
            registry_uri=config["mlflow"]["registry_uri"],
            artifact_location=self.project_root / config["mlflow"]["artifact_location"],
        )

        self.api = APIConfig(**config["api"])

    def validate(self) -> None:
        """Validate the configuration."""
        required_env_vars = [
            "FRAUD_DETECTION_ENV",
            "FRAUD_DETECTION_DATA",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_REGISTRY_URI",
        ]

        # Check required environment variables
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        # Validate data paths
        if not self.data.train_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.data.train_path}")

        # Validate model configuration
        if not 0 < self.model.val_size < 1:
            raise ValueError(f"Invalid validation size: {self.model.val_size}")

        # Environment-specific validations
        if self.env == "production":
            if not Path(self.mlflow.tracking_uri).exists():
                raise ValueError(
                    f"MLflow tracking URI not found: {self.mlflow.tracking_uri}"
                )

        self.logger.info("Configuration validated successfully")
