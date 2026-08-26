import pickle 
from pathlib import Path

import numpy as np
from fastapi import FastAPI,HTTPException

from app.schemas import WaterSample

api=FastAPI()

MODEL_PATH=Path(__file__).resolve().parent.parent/"models"/"model.pkl"

model=None
if MODEL_PATH.exists():
    with open(MODEL_PATH ,"rb") as f:
        model=pickle.load(f)
@api.get("/ping")
def ping():
    return{"status":"ok"}

@api.post("/predict")
def predict(sample:WaterSample):
    if model is None:
        raise HTTPException(status_code=503)
    data=np.array([[sample.ph,sample.Hardness,sample.Solids,sample.Chloramines,sample.Sulfate,sample.Conductivity,sample.Organic_carbon,sample.Trihalomethanes,sample.Turbidity]])
    prediction=int(model.predict(data[0]))
    return {"potable": bool(prediction), "prediction": prediction}