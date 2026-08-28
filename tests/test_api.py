import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from app.main import api
client = TestClient(api)

VALID = {
    "ph": 7.0,
    "Hardness": 200,
    "Solids": 20000,
    "Chloramines": 7,
    "Sulfate": 300,
    "Conductivity": 400,
    "Organic_carbon": 15,
    "Trihalomethanes": 60,
    "Turbidity": 4,
}


def test_ping():
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_empty_json_returns_422():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
    assert "All fields None" in resp.text


def test_invalid_ph_returns_422():
    resp = client.post("/predict", json={**VALID, "ph": -5})
    assert resp.status_code == 422
    assert "ph must be 0-14" in resp.text


def test_valid_payload_returns_200():
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 200

    body = resp.json()
    assert body["potability"] in (0, 1)
    assert 0.0 <= body["probability_potable"] <= 1.0
    assert body["imputed_fields"] == []


def test_missing_field_is_imputed():
    payload = {k: v for k, v in VALID.items() if k != "Sulfate"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["imputed_fields"] == ["Sulfate"]