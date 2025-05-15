import os
import mlflow
from mlflow.artifacts import download_artifacts
from mlflow import MlflowClient

# Validate env vars
databricks_host = os.getenv("DATABRICKS_HOST")
databricks_token = os.getenv("DATABRICKS_TOKEN")

if not databricks_host or not databricks_token:
    raise EnvironmentError("DATABRICKS_HOST and/or DATABRICKS_TOKEN must be set.")

# Set Databricks URIs
os.environ["DATABRICKS_HOST"] = databricks_host
os.environ["DATABRICKS_TOKEN"] = databricks_token
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Set model name in Unity Catalog
uc_model_name = "poc_catalog.model.imagepromptmodel"

# Get latest model version
client = MlflowClient()
versions = client.search_model_versions(f"name='{uc_model_name}'")
latest_version = max([int(m.version) for m in versions])

# Model URI for Unity Catalog
model_uri = f"models:/{uc_model_name}/{latest_version}"

# ✅ Save to custom path
custom_path = "/home/runner/work/realtime-ml-diabetes/realtime-ml-diabetes/saved_model"
local_path = download_artifacts(artifact_uri=model_uri, dst_path=custom_path)

print(f"Model saved locally at: {local_path}")
