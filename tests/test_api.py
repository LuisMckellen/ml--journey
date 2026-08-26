
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import api

client = TestClient(api)

def test_empty_json_returns_422():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
    assert "All fields None" in resp.text

def test_invalid_ph_returns_422():
    payload = {
        "ph": -5,
        "Hardness": 200,
        "Solids": 20000,
        "Chloramines": 7,
        "Sulfate": 300,
        "Conductivity": 400,
        "Organic_carbon": 15,
        "Trihalomethanes": 60,
        "Turbidity": 4
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert "ph must be 0-14" in resp.text

def test_ping():
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] == True