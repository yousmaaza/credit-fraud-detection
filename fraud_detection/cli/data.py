"""Data management commands."""
import sys


import click
from dotenv import load_dotenv
from tqdm import tqdm

from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.data_loader import DataLoader


@click.group(name="data")
def data_cli():
    """Manage data commands."""
    pass


@data_cli.command(name="download")
@click.option("--force", is_flag=True, help="Force download even if file exists")
@click.option(
    "--env",
    type=click.Choice(["development", "production"]),
    default="development",
    help="Environment to use",
)
def download_data(force: bool, env: str) -> None:
    """Download the fraud detection dataset."""
    try:
        # Load environment variables
        load_dotenv(f".env.{env}")

        # Initialize configuration
        config_manager = ConfigurationManager()
        data_loader = DataLoader(config_manager)

        # Download data
        click.echo("Starting dataset download...")
        with tqdm(total=100, desc="Downloading") as pbar:
            data_loader.download_data(force=force)
            pbar.update(100)

        click.echo("Download completed successfully!")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@data_cli.command(name="validate")
@click.option(
    "--env",
    type=click.Choice(["development", "production"]),
    default="development",
    help="Environment to use",
)
def validate_data(env: str) -> None:
    """Validate the downloaded dataset."""
    try:
        load_dotenv(f".env.{env}")
        config = ConfigurationManager()
        data_loader = DataLoader(config)

        click.echo("Validating dataset...")
        df = data_loader.load_raw_data()

        click.echo("\nDataset Information:")
        click.echo(f"Number of records: {len(df)}:,")
        click.echo(f"Number of features: {len(df.columns)}")
        click.echo("\nClass distribution:")
        click.echo(df["Class"].value_counts().to_string())
        click.echo("\nValidation completed successfully!")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@data_cli.command(name="info")
@click.option(
    "--env",
    type=click.Choice(["development", "production"]),
    default="development",
    help="Environment to use",
)
def data_info(env: str) -> None:
    """Show information about the dataset configuration."""
    try:
        load_dotenv(f".env.{env}")
        config = ConfigurationManager()
        data_loader = DataLoader(config)

        click.echo("\nDataset Configuration:")
        click.echo(f"Dataset URL: {data_loader.dataset_url}")
        click.echo(f"Minimum file size: {data_loader.min_file_size:,} bytes")
        click.echo(f"Data directory: {data_loader.data_dir}")
        click.echo(f"Processed directory: {data_loader.processed_dir}")

        data_path = config.data.train_path
        if data_path.exists():
            click.echo(
                f"\nDataset status: Downloaded ({data_path.stat().st_size:,} bytes)"
            )
        else:
            click.echo("\nDataset status: Not downloaded")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
