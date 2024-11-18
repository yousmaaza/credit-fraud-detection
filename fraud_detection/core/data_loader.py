import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.core.config import ConfigurationManager

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""

    pass


class DataLoader:
    """Handles data loading, validation, and splitting operations."""

    REQUIRED_COLUMNS = {"Time", "Amount", "Class"}

    def __init__(self, config: ConfigurationManager) -> None:
        """
        Initialize the data loader.

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.processed_dir = config.data.processed_data_dir
        self.logger = config.logger
        self.processed_dir.mkdir(parents=True, exist_ok=True)

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
