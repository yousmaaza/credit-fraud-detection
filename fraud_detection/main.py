from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.data_loader import DataLoader
from fraud_detection.core.trainer import ModelTrainer


def main() -> None:
    # Initialize
    config = ConfigurationManager()
    data_loader = DataLoader(config)
    trainer = ModelTrainer(config)

    try:
        # Load and prepare data
        data_loader.download_data()  # Will only download if needed
        raw_data = data_loader.load_raw_data()
        train_df, val_df = data_loader.split_data(raw_data)

        # Get features
        features = [
            col
            for col in train_df.columns
            if col not in config.model.features_to_exclude
        ]

        # Train model
        model = trainer.train(train_df, val_df, features)

        config.logger.info("Training pipeline completed successfully")

    except Exception as e:
        config.logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    import os

    os.environ["FRAUD_DETECTION_ENV"] = "development"
    main()
