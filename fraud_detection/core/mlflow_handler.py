# fraud_detection/core/mlflow_handler.py
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from fraud_detection.core.config import ConfigurationManager


class MLflowHandler:
    """Handles MLflow experiment tracking and model registry."""

    def __init__(self, config: ConfigurationManager) -> None:
        """
        Initialize MLflow handler.

        Args:
            config: Configuration manager instance containing MLflow settings
                   and project logger
        """
        self.config = config
        self.logger = config.logger

        # Set up MLflow paths
        tracking_uri = (
            self.config.project_root / self.config.mlflow.tracking_uri
        ).resolve()
        tracking_uri.mkdir(parents=True, exist_ok=True)

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(str(tracking_uri))
        self.client = MlflowClient()

        # Set up and get experiment ID
        self.experiment_id = self._setup_experiment()

        # Set active experiment
        mlflow.set_experiment(self.config.mlflow.experiment_name)

        self.logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        self.logger.info(f"Active experiment: {self.config.mlflow.experiment_name}")

    def _setup_experiment(self) -> str:
        """
        Create or get experiment.

        Returns:
            experiment_id: ID of the experiment
        """
        # Resolve artifact location path
        artifact_location = (
            self.config.project_root / self.config.mlflow.artifact_location
        ).resolve()
        artifact_location.mkdir(parents=True, exist_ok=True)

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(self.config.mlflow.experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=self.config.mlflow.experiment_name,
                artifact_location=str(artifact_location),
            )
            self.logger.info(
                f"Created new MLflow experiment with id: {experiment_id}\n"
                f"Artifact Location: {artifact_location}"
            )
        else:
            experiment_id = experiment.experiment_id
            self.logger.info(
                f"Using existing MLflow experiment: {experiment_id}\n"
                f"Artifact Location: {experiment.artifact_location}"
            )

        return str(experiment_id)

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional run name (defaults to config run_name)

        Returns:
            MLflow ActiveRun context
        """
        # Ensure we're using the correct experiment
        mlflow.set_experiment(self.config.mlflow.experiment_name)

        run_name = run_name or self.config.mlflow.run_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{run_name}_{timestamp}"

        self.logger.info(
            f"Starting MLflow run: {run_name} "
            f"in experiment: {self.config.mlflow.experiment_name}"
        )

        return mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to current run.

        Args:
            params: Dictionary of parameters to log
        """
        try:
            mlflow.log_params(params)
            self.logger.debug(
                f"Logged parameters in experiment "
                f"'{self.config.mlflow.experiment_name}': {params}"
            )
        except Exception as e:
            self.logger.error(f"Error logging parameters: {str(e)}")
            raise

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to current run.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            self.logger.debug(
                f"Logged metrics in experiment "
                f"'{self.config.mlflow.experiment_name}' "
                f"at step {step}: {metrics}"
            )
        except Exception as e:
            self.logger.error(f"Error logging metrics: {str(e)}")
            raise

    def log_artifact(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """
        Log an artifact to the current MLflow run.

        Args:
            local_path: Path to the file to log
            artifact_path: Optional path within the artifact directory to log to
        """
        try:
            # Ensure local_path is a Path object
            local_path = Path(local_path)

            if not local_path.exists():
                raise FileNotFoundError(f"Artifact not found: {local_path}")

            # Log the artifact
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

            artifact_info = f" in {artifact_path}" if artifact_path else ""
            self.logger.debug(
                f"Logged artifact '{local_path.name}'{artifact_info} "
                f"in experiment '{self.config.mlflow.experiment_name}'"
            )

        except Exception as e:
            self.logger.error(f"Error logging artifact {local_path}: {str(e)}")
            raise

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        input_example: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Log and optionally register a model.

        Args:
            model: Model object to log
            artifact_path: Path within the MLflow run to save the model
            registered_model_name: Optional name to register the model
            input_example: Optional input example for model signature
        """
        try:
            # Create model signature
            signature = mlflow.models.infer_signature(
                input_example, model.predict(input_example)
            )

            # Ensure we're in the correct experiment
            mlflow.set_experiment(self.config.mlflow.experiment_name)

            mlflow.lightgbm.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
            )

            message = (
                f"Logged model to {artifact_path} "
                f"in experiment '{self.config.mlflow.experiment_name}'"
            )

            if registered_model_name:
                try:
                    model_version = self.client.get_latest_versions(
                        registered_model_name, stages=["None"]
                    )[0].version
                    message += (
                        f" and updated registered model '{registered_model_name}' "
                        f"to version {model_version}"
                    )
                except Exception:
                    message += f" and registered as new model '{registered_model_name}'"

            self.logger.info(message)

        except Exception as e:
            self.logger.error(f"Error logging model: {str(e)}")
            raise

    # fraud_detection/core/mlflow_handler.py

    def promote_model(self) -> None:
        """Promote the latest staging model to production."""
        try:
            # Get the latest version of the model in the "Staging" stage
            staging_versions = self.client.get_latest_versions(
                self.config.mlflow.registered_model_name, stages=["Staging"]
            )
            if not staging_versions:
                raise ValueError("No model found in the 'Staging' stage")

            latest_staging_version = staging_versions[0].version

            # Transition the latest staging model to the "Production" stage
            self.client.transition_model_version_stage(
                name=self.config.mlflow.registered_model_name,
                version=latest_staging_version,
                stage="Production",
            )

            # Archive the previous production version if it exists
            production_versions = self.client.get_latest_versions(
                self.config.mlflow.registered_model_name, stages=["Production"]
            )
            for version in production_versions:
                if version.version != latest_staging_version:
                    self.client.transition_model_version_stage(
                        name=self.config.mlflow.registered_model_name,
                        version=version.version,
                        stage="Archived",
                    )

            self.config.logger.info(
                f"Model version {latest_staging_version} promoted to 'Production' stage"
            )

        except Exception as e:
            self.config.logger.error(f"Error promoting model: {str(e)}")
            raise
