import mlflow
from mlflow import MlflowClient

# Set up MLflow for Unity Catalog
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Fully qualified model name in Unity Catalog
uc_model_name = "poc_catalog.model.imagepromptmodel"
version = 1  # Use actual version here

# Optional: Create or update alias 'latest'
client = MlflowClient()
client.set_registered_model_alias(name=uc_model_name, alias="latest", version=version)

# Load the model by alias
model_uri = f"models:/{uc_model_name}@latest"
model = mlflow.pyfunc.load_model(model_uri)

# Save model to local directory (e.g., for Docker image build)
model.save("model")

print("✅ Model has been downloaded and saved to 'model/' folder.")
