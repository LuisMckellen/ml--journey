import json
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


def post_raw(body):
    """POST a raw JSON string.

    httpx serialises with allow_nan=False, so the bare NaN/Infinity literals
    that json.loads accepts on the server side cannot be sent via json=.
    """
    return client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )


def body_with(field, literal):
    """VALID as raw JSON, with one field replaced by a bare JSON literal."""
    rest = {k: v for k, v in VALID.items() if k != field}
    return json.dumps(rest).rstrip("}") + f', "{field}": {literal}}}'


def test_ping():
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_empty_json_returns_422():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
    assert "All fields None" in resp.text


def errors(resp):
    """(field, error type) pairs from a pydantic 422 body."""
    return {(e["loc"][-1], e["type"]) for e in resp.json()["detail"]}


def test_invalid_ph_returns_422():
    resp = client.post("/predict", json={**VALID, "ph": -5})
    assert resp.status_code == 422
    assert ("ph", "greater_than_equal") in errors(resp)


def test_ph_above_14_returns_422():
    resp = client.post("/predict", json={**VALID, "ph": 15})
    assert resp.status_code == 422
    assert ("ph", "less_than_equal") in errors(resp)


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


def test_infinity_is_rejected():
    resp = post_raw(body_with("Hardness", "Infinity"))
    assert resp.status_code == 422
    assert ("Hardness", "finite_number") in errors(resp)


def test_negative_infinity_is_rejected():
    resp = post_raw(body_with("Hardness", "-Infinity"))
    assert resp.status_code == 422
    assert ("Hardness", "finite_number") in errors(resp)


def test_float_overflow_is_rejected():
    # 1e400 has no float representation and parses to inf
    resp = post_raw(body_with("Solids", "1e400"))
    assert resp.status_code == 422
    assert ("Solids", "finite_number") in errors(resp)


def test_nan_is_rejected_not_silently_imputed():
    # NaN used to pass validation, get median-imputed by fillna, and then be
    # omitted from imputed_fields - the caller saw a population value reported
    # as their own
    resp = post_raw(body_with("Sulfate", "NaN"))
    assert resp.status_code == 422
    assert ("Sulfate", "finite_number") in errors(resp)


def test_rejecting_infinity_does_not_crash_the_error_renderer():
    # pydantic echoes the bad value back as `input`; Starlette serialises with
    # allow_nan=False, so an unsanitised inf turns this 422 into a 500
    resp = post_raw(body_with("Hardness", "Infinity"))
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["input"] == "inf"


def test_absurdly_large_value_is_rejected():
    # 50 nines parses to a finite 1e50, so the finite check passes it through;
    # only an upper bound catches it
    resp = post_raw(body_with("Hardness", "9" * 50))
    assert resp.status_code == 422
    assert ("Hardness", "less_than_equal") in errors(resp)


def test_negative_concentration_is_rejected():
    resp = client.post("/predict", json={**VALID, "Hardness": -1})
    assert resp.status_code == 422
    assert ("Hardness", "greater_than_equal") in errors(resp)


def test_value_just_inside_the_bound_is_accepted():
    # 323.124 is the training max for Hardness; the 20% buffer must admit it
    resp = client.post("/predict", json={**VALID, "Hardness": 323.124})
    assert resp.status_code == 200

