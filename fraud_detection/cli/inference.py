import click
import pandas as pd

from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.inference import ModelInference


@click.command(name="inference")
@click.argument("input_data", type=click.Path(exists=True))
@click.argument("output_data", type=click.Path())
def inference(input_data: str, output_data: str):
    """Perform inference on input data and save predictions to output data."""
    config = ConfigurationManager()
    model_inference = ModelInference(config)

    # Load input data
    data = pd.read_csv(input_data)

    # Perform inference
    predictions = model_inference.predict(data)

    # Save predictions to output file
    predictions.to_csv(output_data, index=False)
    config.logger.info(f"Predictions saved to {output_data}")
