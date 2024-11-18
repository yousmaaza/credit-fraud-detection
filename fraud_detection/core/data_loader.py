import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.core.config import ConfigurationManager


class DataValidationError(Exception):
    """Raised when data validation fails."""

    pass


class DataDownloadError(Exception):
    """Raised when data download fails."""

    pass


class DataLoader:
    """Handles data loading, validation, and downloading operations."""

    REQUIRED_COLUMNS = {"Time", "Amount", "Class"}

    def __init__(self, config: "ConfigurationManager") -> None:
        """
        Initialize the data loader.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = config.logger
        self.data_dir = config.data.train_path.parent
        self.processed_dir = config.data.processed_data_dir

        # Get dataset configuration from environment with defaults
        self.dataset_url = os.getenv(
            "DATASET_URL",
            "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud",
        )

        try:
            self.min_file_size = int(os.getenv("DATASET_MIN_FILE_SIZE", "100000000"))
        except ValueError:
            self.logger.warning(
                "Invalid DATASET_MIN_FILE_SIZE in environment, using default"
            )
            self.min_file_size = 100_000_000

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Log configuration
        self.logger.debug(
            f"DataLoader initialized with:\n"
            f"  Dataset URL: {self.dataset_url}\n"
            f"  Min file size: {self.min_file_size:,} bytes\n"
            f"  Data directory: {self.data_dir}\n"
            f"  Processed directory: {self.processed_dir}"
        )

    def download_data(self, force: bool = False) -> None:
        """
        Download credit card fraud dataset from Kaggle.

        Args:
            force: Force download even if file exists

        Raises:
            DataDownloadError: If download or validation fails
        """
        archive_path = self.data_dir / "archive.zip"
        dataset_path = self.config.data.train_path

        # Check if data already exists
        if dataset_path.exists() and self._check_file_size(dataset_path) and not force:
            self.logger.info(f"Dataset already exists at {dataset_path}")
            return

        self.logger.info("Downloading credit card fraud dataset...")

        kaggle_token = self.config._get_kaggle_token()
        if not kaggle_token:
            raise DataDownloadError(
                "KAGGLE_TOKEN not set in environment. "
                "Please set KAGGLE_TOKEN in your .env file"
            )

        try:
            # Download dataset
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "-o",
                    str(archive_path),
                    "-H",
                    f"Authorization: Basic {kaggle_token}",
                    self.dataset_url,
                ],
                check=True,
            )

            # Unzip archive
            self.logger.info("Extracting dataset...")
            subprocess.run(
                ["unzip", "-o", str(archive_path), "-d", str(self.data_dir)], check=True
            )

            # Cleanup
            archive_path.unlink()

            # Validate download
            if not dataset_path.exists():
                raise DataDownloadError("Dataset not found after extraction")

            if not self._check_file_size(dataset_path):
                dataset_path.unlink()  # Remove potentially corrupted file
                raise DataDownloadError(
                    f"Downloaded file is too small "
                    f"(minimum size: {self.min_file_size:,} bytes), "
                    "might be corrupted"
                )

            self.logger.info(f"Successfully downloaded dataset to {dataset_path}")

        except subprocess.CalledProcessError as e:
            raise DataDownloadError(f"Error downloading dataset: {str(e)}")
        except Exception as e:
            raise DataDownloadError(f"Unexpected error during download: {str(e)}")

    def _check_file_size(self, file_path: Path) -> bool:
        """
        Check if file size meets minimum requirement.

        Args:
            file_path: Path to file to check

        Returns:
            True if file size is adequate, False otherwise
        """
        return file_path.stat().st_size > self.min_file_size

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load and validate the raw credit card fraud dataset.

        Returns:
            DataFrame containing the validated raw data

        Raises:
            FileNotFoundError: If the data file doesn't exist
            DataValidationError: If the data doesn't meet requirements
        """
        self.logger.info(f"Loading data from {self.config.data.train_path}")

        if not self.config.data.train_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {self.config.data.train_path}"
            )

        df = pd.read_csv(self.config.data.train_path)
        self._validate_data(df)

        self.logger.info(f"Successfully loaded {len(df)} records")
        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the loaded data meets requirements.

        Args:
            df: DataFrame to validate
        """
        # Check required columns
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # Validate target variable
        if not set(df[self.config.model.target_column].unique()).issubset({0, 1}):
            raise DataValidationError(
                f"Target column '{self.config.model.target_column}' "
                "must be binary (0, 1)"
            )
        # Log class distribution
        class_dist = df[self.config.model.target_column].value_counts()
        self.logger.info(f"Class distribution:\n{class_dist.to_dict()}")

    def save_processed_data(self, df: pd.DataFrame, name: str) -> None:
        """
        Save processed data to parquet format.

        Args:
            df: DataFrame to save
            name: Name of the processed dataset
        """
        output_path = self.processed_dir / f"{name}.parquet"
        df.to_parquet(output_path, index=False)
        self.logger.info(f"Saved processed data to {output_path}")

    def load_processed_data(self, name: str) -> pd.DataFrame:
        """
        Load processed data from parquet format.

        Args:
            name: Name of the processed dataset

        Returns:
            Loaded DataFrame
        """
        input_path = self.processed_dir / f"{name}.parquet"
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found at {input_path}")

        return pd.read_parquet(input_path)

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets.

        Args:
            df: DataFrame to split

        Returns:
            Tuple of (training_df, validation_df)
        """
        train_df, val_df = train_test_split(
            df,
            test_size=self.config.model.val_size,
            random_state=self.config.model.random_state,
            stratify=df[self.config.model.target_column],
        )

        # Save splits
        self.save_processed_data(train_df, "train")
        self.save_processed_data(val_df, "validation")

        self.logger.info(
            f"Split data into train ({len(train_df)} samples) "
            f"and validation ({len(val_df)} samples)"
        )

        return train_df, val_df

    def get_feature_columns(self) -> list[str]:
        """Get list of feature columns (excluding target and excluded features)."""
        df = pd.read_csv(self.config.data.train_path, nrows=1)
        all_columns = set(df.columns)
        excluded = set(self.config.model.features_to_exclude)
        return list(all_columns - excluded)

    def validate_environment(self) -> None:
        """Validate environment variables for data handling."""
        required_vars = {
            "KAGGLE_TOKEN": "Authentication token for Kaggle API",
        }

        optional_vars = {
            "DATASET_URL": "URL for downloading the dataset",
            "DATASET_MIN_FILE_SIZE": "Minimum expected file size in bytes",
        }

        # Check required variables
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            print("Missing required environment variables:")
            for var in missing_vars:
                print(f"  {var}: {required_vars[var]}")
            print("\nPlease set these variables in your .env file")
            sys.exit(1)

        # Check optional variables
        for var, description in optional_vars.items():
            value = os.getenv(var)
            if value:
                print(f"{var}: {value}")
            else:
                print(f"{var}: Using default value ({description})")
