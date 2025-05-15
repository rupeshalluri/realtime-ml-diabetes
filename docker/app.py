import mlflow.pyfunc
from fastapi import FastAPI, Request
import pandas as pd

app = FastAPI()
model = mlflow.pyfunc.load_model("/home/runner/work/realtime-ml-diabetes/realtime-ml-diabetes/saved_model")
@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    inputs = pd.DataFrame(payload["inputs"])
    predictions = model.predict(inputs)
    return {"predictions": predictions.tolist()}
