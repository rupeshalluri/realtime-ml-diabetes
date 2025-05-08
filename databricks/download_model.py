import mlflow

# Make sure to set the appropriate registry URI and tracking URI for Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Create an alias for the model version
model_name = "poc_catalog.model.imagepromptmodel"
version = 1  # Example: version number of your model you want to use as the latest

# Create alias 'latest' for version 1 (replace version with actual version number)
mlflow.register_model(model_uri=f"models:/{model_name}/{version}", name=f"{model_name}_latest")
