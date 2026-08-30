import math
import pickle
from pathlib import Path

import pandas as pd
import xgboost
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from xgboost import XGBClassifier

from .schemas import WaterInput

api = FastAPI(title="Water Potability API")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.json"
MEDIAN_PATH = MODEL_DIR / "medians.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"

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


def _json_safe(o):
    """Replace non-finite floats with their repr, recursively.

    Pydantic echoes the offending value back as `input`, and Starlette renders
    responses with json.dumps(allow_nan=False). A rejected inf/NaN would then
    raise while serialising the 422 and surface as a 500 instead.
    """
    if isinstance(o, float) and not math.isfinite(o):
        return repr(o)
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


@api.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))}
    )


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

    imputed_fields = [k for k, v in data.items() if v is None]
  
    df = pd.DataFrame([data], columns=FEATURES).fillna(medians).astype(float)

    proba_potable = float(model.predict_proba(df)[0][1])
    pred=int(proba_potable>0.5)

    return {
        "potability": pred,
        "probability_potable": proba_potable,
        "imputed_fields": imputed_fields,
    }