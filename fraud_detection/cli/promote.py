# fraud_detection/cli/promote.py

import click

from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.mlflow_handler import MLflowHandler


@click.command(name="promote")
def promote_model():
    """Promote the latest champion model to production."""
    config = ConfigurationManager()
    mlflow_handler = MLflowHandler(config)
    mlflow_handler.promote_model()
