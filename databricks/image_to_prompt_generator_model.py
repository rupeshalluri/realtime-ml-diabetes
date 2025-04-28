## IMPORTING LIBRARIES
import cv2
from clip_interrogator import Config, Interrogator
from PIL import Image
import mlflow
import torch
from mlflow.models.signature import infer_signature
import numpy as np

## MODEL CODE

class ImagePromptModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, input_img_dict):
        input_image = input_img_dict["input_image"]
        image = Image.fromarray(input_image).convert("RGB")
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

# SAVING THE MODEL
mlflow.set_registry_uri("databricks")
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="image_prompt_model",
        python_model=model,
        registered_model_name="imagepromptmodel",
        signature=signature,
        pip_requirements=["scikit-build", "opencv-python", "clip-interrogator", "pillow", "torch"]
    )
