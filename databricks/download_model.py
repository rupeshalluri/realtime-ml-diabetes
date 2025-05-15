import os
import mlflow
from mlflow import MlflowClient

# Set environment variables
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

# Unity Catalog-specific setup
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Define the full Unity Catalog model name (catalog.schema.model_name)
uc_model_name = "poc_catalog.model.imagepromptmodel"

# Initialize MLflow client
client = MlflowClient()

# Get all versions of the model and find the highest version
versions = client.search_model_versions(f"name='{uc_model_name}'")
latest_version = max([int(m.version) for m in versions])

# Print the version for logging
print(f"Latest version of model '{uc_model_name}' is: {latest_version}")

# Construct the model URI
model_uri = f"models:/{uc_model_name}/{latest_version}"

# Load the model
model = mlflow.pyfunc.load_model(model_uri)

# Save the model locally in a folder named "model"
save_path = "model"
model.save(save_path)

print(f"Model v{latest_version} has been downloaded and saved to: {save_path}")
