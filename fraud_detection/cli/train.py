"""Training commands."""
import sys


import click
from dotenv import load_dotenv

from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.data_loader import DataLoader
from fraud_detection.core.trainer import ModelTrainer


@click.group(name="train")
def train_cli():
    """Model training commands."""
    pass


@train_cli.command(name="run")
@click.option(
    "--env",
    type=click.Choice(["development", "production"]),
    default="development",
    help="Environment to use",
)
@click.option(
    "--force-download", is_flag=True, help="Force download of dataset even if exists"
)
def train_model(env: str, force_download: bool) -> None:
    """Train the fraud detection model."""
    try:
        # Load environment variables
        load_dotenv(f".env.{env}")

        # Initialize components
        config_manager = ConfigurationManager()
        data_loader = DataLoader(config_manager)
        trainer = ModelTrainer(config_manager)

        # Ensure data is available
        if force_download or not config_manager.data.train_path.exists():
            click.echo("Downloading dataset...")
            data_loader.download_data(force=force_download)

        # Load and split data
        click.echo("Loading data...")
        raw_data = data_loader.load_raw_data()
        train_df, val_df = data_loader.split_data(raw_data)

        # Get feature columns
        features = [
            col
            for col in train_df.columns
            if col not in config_manager.model.features_to_exclude
        ]

        # Train model
        click.echo("Training model...")
        trainer.train(train_df, val_df, features)

        click.echo("Training completed successfully!")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


# @train_cli.command(name="evaluate")
# @click.option(
#     "--env",
#     type=click.Choice(["development", "production"]),
#     default="development",
#     help="Environment to use",
# )
# @click.option("--run-id", help="MLflow run ID to evaluate")
# def evaluate_model(env: str, run_id: Optional[str]) -> None:
#     """Evaluate a trained model."""
#     try:
#         load_dotenv(f".env.{env}")
#         config = ConfigurationManager()
#         data_loader = DataLoader(config)
#         trainer = ModelTrainer(config)
#
#         click.echo("Loading validation data...")
#         val_data = data_loader.load_processed_data("validation")
#
#         click.echo("Evaluating model...")
#         metrics = trainer.evaluate(val_data, run_id)
#
#         click.echo("\nModel Metrics:")
#         for metric, value in metrics.items():
#             click.echo(f"{metric}: {value:.4f}")
#
#     except Exception as e:
#         click.echo(f"Error: {str(e)}", err=True)
#         sys.exit(1)
