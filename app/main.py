import pickle
from pathlib import Path

import pandas as pd
import xgboost
from fastapi import FastAPI, HTTPException
from xgboost import XGBClassifier

from .schemas import WaterInput

api = FastAPI(title="Water Potability API")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.json"
MEDIAN_PATH = MODEL_DIR / "medians.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"

NON_NEGATIVE = [
    "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]

model = None
medians = None
FEATURES = []

try:
    missing = [p.name for p in (MODEL_PATH, MEDIAN_PATH, FEATURES_PATH) if not p.exists()]
    if missing:
        print(f"Artifacts missing: {missing} — run python -m src.train")
    else:
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        with open(MEDIAN_PATH, "rb") as f:
            medians = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            FEATURES = pickle.load(f)
        print(f"Loaded artifacts (xgboost {xgboost.__version__}): {FEATURES}")
except Exception as e:
    print(f"artifact load failed: {e}")
    model = None
    medians = None
    FEATURES = []


@api.get("/ping")
def ping():
    return {
        "status": "alive",
        "model_loaded": model is not None,
        "xgboost_version": xgboost.__version__,
    }


@api.get("/")
def root():
    return {
        "title": api.title,
        "docs": "/docs",
        "ping": "/ping",
        "model_loaded": model is not None,
    }


@api.post("/predict")
def predict(sample: WaterInput):
    if model is None or medians is None or not FEATURES:
        raise HTTPException(status_code=503, detail="Model not loaded - run train.py")

    data = sample.model_dump()

    if all(v is None for v in data.values()):
        raise HTTPException(status_code=422, detail="All fields None - provide at least one value")

    if data.get("ph") is not None and not (0 <= data["ph"] <= 14):
        raise HTTPException(status_code=422, detail="ph must be 0-14")

    for k in NON_NEGATIVE:
        if data.get(k) is not None and data[k] < 0:
            raise HTTPException(status_code=422, detail=f"{k} must be >=0")

    imputed_fields = [k for k, v in data.items() if v is None]
  
    df = pd.DataFrame([data], columns=FEATURES).fillna(medians).astype(float)

    pred = int(model.predict(df)[0])
    proba_potable = float(model.predict_proba(df)[0][1])

    return {
        "potability": pred,
        "probability_potable": proba_potable,
        "imputed_fields": imputed_fields,
    }