import subprocess

from fraud_detection.core.config import ConfigurationManager

# Load configuration
config = ConfigurationManager()

# Extract parameters
tracking_uri = config.mlflow.tracking_uri
artifact_location = config.mlflow.artifact_location

# Construct the command
command = [
    "mlflow",
    "server",
    "--backend-store-uri",
    tracking_uri,
    "--default-artifact-root",
    str(artifact_location),
    "--host",
    "0.0.0.0",
    "--port",
    "5050",
]

# Run the command
subprocess.run(command)
