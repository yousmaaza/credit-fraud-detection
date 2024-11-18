from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.mlflow_handler import MLflowHandler


class ModelTrainer:
    """Handles model training and evaluation with MLflow tracking."""

    def __init__(self, config: ConfigurationManager) -> None:
        """
        Initialize the trainer.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = config.logger
        self.mlflow = MLflowHandler(config)

        # Create model directory if it doesn't exist
        self.model_dir = Path("output")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, features: list[str]
    ) -> lgb.Booster:
        """
        Train the model and log metrics with MLflow.

        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            features: List of feature columns

        Returns:
            Trained model
        """
        self.logger.info("Starting model training")

        try:
            # Prepare datasets
            train_features, train_target = self._prepare_data(train_data, features)
            val_features, val_target = self._prepare_data(val_data, features)

            # Create LightGBM datasets
            train_set = lgb.Dataset(train_features, train_target)
            val_set = lgb.Dataset(val_features, val_target, reference=train_set)

            # Start MLflow run
            with self.mlflow.start_run():
                # Log parameters
                self._log_training_params(features)

                # Train model
                model = self._train_model(train_set, val_set)

                # Evaluate model
                self._evaluate_model(
                    model, train_features, train_target, val_features, val_target
                )

                # Save artifacts
                self._save_artifacts(model, features)

                self.logger.info("Training completed successfully")
                return model

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def _prepare_data(
        self, data: pd.DataFrame, features: list[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        return (data[features], data[self.config.model.target_column])

    def _train_model(self, train_set: lgb.Dataset, val_set: lgb.Dataset) -> lgb.Booster:
        """Train LightGBM model."""
        self.logger.info("Training LightGBM model")

        model = lgb.train(
            params=self.config.model.model_params,
            train_set=train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.config.model.early_stopping_rounds,
                    verbose=False,
                ),
                lgb.log_evaluation(period=100),
            ],
        )

        return model

    def _evaluate_model(
        self,
        model: lgb.Booster,
        train_features: pd.DataFrame,
        train_target: pd.Series,
        val_features: pd.DataFrame,
        val_target: pd.Series,
    ) -> None:
        """Evaluate model and log metrics."""
        # Get predictions
        train_preds = model.predict(train_features)
        val_preds = model.predict(val_features)

        # Calculate metrics
        metrics = self._calculate_metrics(
            train_target, train_preds, val_target, val_preds, thresholds=[0.5, 0.6, 0.7]
        )

        # Log metrics
        self.mlflow.log_metrics(metrics)
        self.logger.info(f"Model metrics: {metrics}")

    def _calculate_metrics(
        self,
        train_target: pd.Series,
        train_preds: np.ndarray,
        val_target: pd.Series,
        val_preds: np.ndarray,
        thresholds: list[float],
    ) -> Dict[str, float]:
        """Calculate model performance metrics."""
        metrics = {
            "train_auc": roc_auc_score(train_target, train_preds),
            "val_auc": roc_auc_score(val_target, val_preds),
            "train_avg_precision": average_precision_score(train_target, train_preds),
            "val_avg_precision": average_precision_score(val_target, val_preds),
        }

        # Calculate metrics at different thresholds
        for threshold in thresholds:
            train_pred_labels = (train_preds >= threshold).astype(int)
            val_pred_labels = (val_preds >= threshold).astype(int)

            # Training metrics
            metrics[f"train_precision_{threshold}"] = precision_score(
                train_target, train_pred_labels
            )
            metrics[f"train_recall_{threshold}"] = recall_score(
                train_target, train_pred_labels
            )
            metrics[f"train_f1_{threshold}"] = f1_score(train_target, train_pred_labels)

            # Validation metrics
            metrics[f"val_precision_{threshold}"] = precision_score(
                val_target, val_pred_labels
            )
            metrics[f"val_recall_{threshold}"] = recall_score(
                val_target, val_pred_labels
            )
            metrics[f"val_f1_{threshold}"] = f1_score(val_target, val_pred_labels)

        return metrics

    def _log_training_params(self, features: list[str]) -> None:
        """Log training parameters to MLflow."""
        params = {
            "features": features,
            "n_features": len(features),
            "early_stopping_rounds": self.config.model.early_stopping_rounds,
            "random_state": self.config.model.random_state,
            **self.config.model.model_params,
        }
        self.mlflow.log_params(params)

    def _save_artifacts(self, model: lgb.Booster, features: list[str]) -> None:
        """Save and log model artifacts."""
        # Create and save feature importance plot
        importance_plot_path = self._create_feature_importance_plot(model, features)
        self.mlflow.log_artifact(importance_plot_path)

        # Create model signature and input example
        input_example = self._create_input_example(features)

        # Log the model
        self.mlflow.log_model(
            model,
            artifact_path="model",
            registered_model_name="fraud_detection_model",
            input_example=input_example,
        )

    def _create_input_example(self, features: list[str]) -> pd.DataFrame:
        """
        Create an input example for model signature.

        Args:
            features: List of feature names

        Returns:
            DataFrame with one row of example data
        """
        # Get a sample from training data if available
        example_data = {feature: 0.0 for feature in features}
        return pd.DataFrame([example_data])

    def _create_feature_importance_plot(
        self, model: lgb.Booster, features: list[str]
    ) -> Path:
        """Create feature importance plot."""
        plt.figure(figsize=(10, 6))

        importance_df = pd.DataFrame(
            {
                "feature": features,
                "importance": model.feature_importance(importance_type="gain"),
            }
        )
        importance_df = importance_df.sort_values("importance", ascending=True)

        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.title("Feature Importance (Gain)")
        plt.xlabel("Importance")

        # Save plot
        plot_path = self.model_dir / "feature_importance.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        return plot_path
