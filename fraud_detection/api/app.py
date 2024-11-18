from fastapi import FastAPI, HTTPException
import pandas as pd
from fraud_detection.api.schemas import PredictionRequest, PredictionResponse
from fraud_detection.core.config import ConfigurationManager
from fraud_detection.core.inference import ModelInference

app = FastAPI()

# Initialize the model inference pipeline
config = ConfigurationManager()
model_inference = ModelInference(config)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):   # type: ignore
    try:
        # Convert the request into a DataFrame
        input_data = pd.DataFrame([request.dict()])  # PredictionRequest provides .dict()

        # Perform prediction
        prediction = model_inference.predict(input_data)

        # Validate and return the prediction
        prediction_value = prediction.iloc[0, 0]
        if not isinstance(prediction_value, (float, int)):
            raise ValueError(f"Prediction result is not a valid float: {prediction_value}")

        return PredictionResponse(prediction=float(prediction_value))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
