import subprocess
import sys
import os

# Get the current directory of the script
current_dir = "/Workspace/Users/alluri.rupesh@akira.co.in/realtime-ml-diabetes/databricks/"

# Construct full path to requirements.txt
requirements_path = os.path.join(current_dir, "requirements.txt")

# Install packages using pip
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
except subprocess.CalledProcessError as e:
    print("Failed to install requirements:", e)
    sys.exit(1)

## IMPORTING LIBRARIES
import cv2
from clip_interrogator import Config, Interrogator
from PIL import Image, ImageEnhance
import mlflow
import torch
from mlflow.models.signature import infer_signature, ModelSignature
import numpy as np
import pandas as pd
import numpy as np
import base64
import io

## MODEL CODE

class ImagePromptModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, input_img_dict):
        """
        Model input: a numpy array or PIL Image
        Output: the generated image prompt
        """ 
        input_image = input_img_dict["input_image"]

        # Convert input to a PIL Image if it's not already
        image = Image.fromarray(input_image).convert("RGB")
        # Load the interrogator and process the image
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        img_prompt = ci.interrogate(image)
        del ci
        torch.cuda.empty_cache()

        return img_prompt

# INVOKING THE MODEL
model = ImagePromptModel()

# INPUT AND OUTPUT
img = Image.open('./test_image.jpg').resize((512,512))
model_input = {'input_image' : np.array(img)}
context = None
prompt = model.predict(context, model_input)
print("Generated Prompt:", prompt)

# MODEL SIGNATURE
signature = infer_signature(model_input, prompt)
signature

# SAVING THE MODEL
# Replace Unity Catalog URI with the default model registry URI
mlflow.set_registry_uri("databricks")

# Define your catalog and schema (you can keep these for reference)
CATALOG_NAME = "gdna_az_poc_catalog"    
SCHEMA_NAME = "ml_model"

# Define pip requirements for the model environment
pip_requirements = ["scikit-build", "opencv-python", "clip-interrogator", "pillow", "torch"]

# Log the model to the Databricks model registry
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="image_prompt_model",  # The path to store the model artifacts
        python_model=model,
        registered_model_name="imagepromptmodel",  # The model name in the registry
        signature=signature,
        pip_requirements=pip_requirements  # Specify any additional requirements if needed
    )
