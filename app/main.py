from fastapi import FastAPI, HTTPException
import pickle
from pathlib import Path
import pandas as pd
from .schemas import WaterInput
api = FastAPI(title="Water Potability API")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
MEDIAN_PATH = Path(__file__).resolve().parent.parent / "models" / "medians.pkl"

model = None
medians = None
FEATURES = []

try:
    if MODEL_PATH.exists() and MEDIAN_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(MEDIAN_PATH, "rb") as f:
            medians = pickle.load(f)
        FEATURES = list(medians.index)
        print(f"Loaded artifacts: {FEATURES}")
    else:
        print(f"Artifacts missing: model={MODEL_PATH.exists()}, medians={MEDIAN_PATH.exists()}")
except Exception as e:
    print(f"artifact load failed: {e}")
    model = None
    medians = None
    FEATURES = []



@api.get("/ping")
def ping():
    return {"status": "alive", "model_loaded": model is not None}

@api.get("/")
def root():
    return {"title": api.title, "docs": "/docs", "ping": "/ping", "model_loaded": model is not None}

@api.post("/predict")
def predict(sample: WaterInput):
    if model is None or medians is None or not FEATURES:
        raise HTTPException(status_code=503, detail="Model not loaded - run train.py")

    data = sample.model_dump()

    if all(v is None for v in data.values()):
        raise HTTPException(status_code=422, detail="All fields None - provide at least one value")

    if data.get("ph") is not None and not (0 <= data["ph"] <= 14):
        raise HTTPException(status_code=422, detail="ph must be 0-14")

    for k in ["Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]:
        if data.get(k) is not None and data[k] < 0:
            raise HTTPException(status_code=422, detail=f"{k} must be >=0")

    imputed_fields = [k for k, v in data.items() if v is None]
    df = pd.DataFrame([data], columns=FEATURES).fillna(medians)

    pred = int(model.predict(df)[0])
    proba_potable = float(model.predict_proba(df)[0][1])

    return {
        "potability": pred,
        "probability_potable": proba_potable,
        "imputed_fields": imputed_fields
    }

