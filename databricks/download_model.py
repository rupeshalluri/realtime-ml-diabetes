import mlflow

# Set the appropriate registry URI and tracking URI for Unity Catalog
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Define the model name and version
model_name = "poc_catalog.model.imagepromptmodel"
version = 1  # Replace this with the actual version number you want to use

# Register the model and create an alias (if not already done)
mlflow.register_model(model_uri=f"models:/{model_name}/{version}", name=f"{model_name}_latest")

# Now load the model from the registry by the alias
model_uri = f"models:/{model_name}_latest"
model = mlflow.pyfunc.load_model(model_uri)

# Save the model to a local directory
model.save("docker")  # Save to the 'model' folder in the current working directory

print("Model has been downloaded and saved locally.")
