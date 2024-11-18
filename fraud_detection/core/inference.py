from typing import Any

import mlflow.pyfunc
import pandas as pd

from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.mlflow_handler import MLflowHandler


class ModelInference:
    """Handles model inference using the latest registered model from MLflow."""

    def __init__(self, config: ConfigurationManager) -> None:
        """
        Initialize the inference handler.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = config.logger
        self.mlflow = MLflowHandler(config)
        self.model = self._load_latest_model()

    def _load_latest_model(self) -> Any:
        """Load the latest registered model from MLflow."""
        self.logger.info("Loading the latest registered model from MLflow")

        latest_version = self.mlflow.client.get_latest_versions(
            self.config.mlflow.registered_model_name, stages=["Production"]
        )[0].version
        model_uri = (
            f"models:/{self.config.mlflow.registered_model_name}/{latest_version}"
        )
        model = mlflow.pyfunc.load_model(model_uri)
        self.logger.info("Latest registered model loaded successfully")
        return model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform inference on new data.

        Args:
            data: DataFrame containing the input data

        Returns:
            DataFrame containing the predictions
        """
        self.logger.info("Performing inference")
        predictions = self.model.predict(data)
        return pd.DataFrame(predictions, columns=["prediction"])
