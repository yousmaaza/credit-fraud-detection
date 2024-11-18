"""Main CLI entry point for fraud detection project."""
import click

from fraud_detection.cli.data import data_cli
from fraud_detection.cli.inference import inference
from fraud_detection.cli.promote import promote_model
from fraud_detection.cli.train import train_cli


@click.group()
def cli():
    """Fraud Detection CLI commands."""
    pass


# Add command groups
cli.add_command(data_cli)
cli.add_command(train_cli)
cli.add_command(promote_model)
cli.add_command(inference)

if __name__ == "__main__":
    cli()
