import os
import mlflow
from mlflow import MlflowClient

# Validate environment variables are set
databricks_host = os.getenv("DATABRICKS_HOST")
databricks_token = os.getenv("DATABRICKS_TOKEN")

if not databricks_host or not databricks_token:
    raise EnvironmentError("DATABRICKS_HOST and/or DATABRICKS_TOKEN are not set in the environment.")

# Set them in os.environ if needed
os.environ["DATABRICKS_HOST"] = databricks_host
os.environ["DATABRICKS_TOKEN"] = databricks_token

# Set URIs for Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Define model name
uc_model_name = "poc_catalog.model.imagepromptmodel"

# Init client
client = MlflowClient()

# Get latest version
versions = client.search_model_versions(f"name='{uc_model_name}'")
latest_version = max([int(m.version) for m in versions])

# Load & save model
model_uri = f"models:/{uc_model_name}/{latest_version}"
model = mlflow.pyfunc.load_model(model_uri)
model.save("model")

print(f"Model v{latest_version} downloaded and saved.")
