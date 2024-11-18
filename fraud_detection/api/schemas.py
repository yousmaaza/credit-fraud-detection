from typing import Dict, Type

from pydantic import BaseModel, create_model

from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.inference import ModelInference


# Dynamically generate and cache the PredictionRequest schema
def generate_prediction_request_schema() -> Type[BaseModel]:
    """Dynamically generate a schema for PredictionRequest."""
    # Load model configuration and schema
    config = ConfigurationManager()
    model_inference = ModelInference(config)
    model = model_inference.model
    input_schema = model._model_meta.get_input_schema()
    input_schema_dict = input_schema.input_types_dict()

    # Create a dictionary of fields for Pydantic's create_model
    fields: Dict[str, tuple] = {key: (float, ...) for key in input_schema_dict.keys()}

    # Dynamically create the PredictionRequest model
    DynamicPredictionRequest = create_model("PredictionRequest", **fields)

    return DynamicPredictionRequest


# Explicitly assign the dynamically generated schema
PredictionRequest = generate_prediction_request_schema()  # type: ignore


# Define a concrete response schema
class PredictionResponse(BaseModel):
    """Dynamically generate a schema for PredictionRequest."""

    prediction: float
