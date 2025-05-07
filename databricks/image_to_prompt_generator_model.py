## IMPORTING LIBRARIES
import cv2
import os
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

##MODEL CODE

class ImagePromptModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, input_img_dict):
        """
        Model input: a numpy array or PIL Image
        Output: the generated image prompt
        """ 
        input_image = input_img_dict["input_image"]

        # decoded = base64.b64decode(input_image) 
        # restored_image = Image.frombytes("RGB", (512,512), decoded)
        # restored_image = Image.open(io.BytesIO(decoded)).convert("RGB")
        
        # Convert input to a PIL Image if it's not already
        image = Image.fromarray(input_image).convert("RGB")
        # Load the interrogator and process the image
        os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache/transformers"
        os.environ["HF_HOME"] = "/tmp/hf_cache"
        os.environ["TORCH_HOME"] = "/tmp/torch_cache"
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        img_prompt = ci.interrogate(image)
        del ci
        torch.cuda.empty_cache()

        return img_prompt

#INVOKING THE MODEL
model = ImagePromptModel()


#INPUT AND OUTPUT
img = Image.open('./test_image.jpg').resize((512,512))
model_input = {'input_image' : np.array(img)}
context = None
prompt = model.predict(context, model_input)
print("Generated Prompt:", prompt)




# MODEL SIGNATURE
signature = infer_signature(model_input, prompt)
signature



# SAVING THE MODEL
mlflow.set_registry_uri("databricks-uc")
CATALOG_NAME = "poc_catalog"    
SCHEMA_NAME = "model"
mlflow.set_experiment("/Users/alluri.rupesh@akira.co.in/realtime-ml-diabetes/databricks/image_to_prompt_generator_model.py")



pip_requirements = ["scikit-build", "opencv-python", "clip-interrogator", "pillow", "torch"]

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="image_prompt_model",
        python_model=model,
        registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.imagepromptmodel",
        signature= signature,
        pip_requirements=pip_requirements
    )

