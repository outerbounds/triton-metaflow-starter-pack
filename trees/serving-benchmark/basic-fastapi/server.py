from fastapi import FastAPI
import pickle
import os
from pydantic import BaseModel
import numpy as np
import json

app = FastAPI(title="My sklearn API")

@app.on_event("startup")
def load_model():
    model_path =  os.path.join(
        os.path.join("/", *__file__.split("/")[:-1]), 
        "model-repo", 
        "model.pkl"
    )
    app.model = pickle.load(open(model_path, 'rb'))

@app.get("/predict")
async def predict(data):
    data = np.asarray(json.loads(data)).reshape(1, -1)
    pred = app.model.predict(data)[0]
    if isinstance(pred, np.int64):
        pred = pred.item()
    return {"prediction": pred}