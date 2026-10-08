import pytest
from fastapi.testclient import TestClient
from src.api.main import create_app


@pytest.mark.parametrize("horizon", [7, 14])
def test_api_prediction(trained, horizon):
    with TestClient(create_app(trained[1])) as client:
        assert client.get("/health").json() == {"status": "ok"}
        info = client.get("/model/info")
        assert info.status_code == 200
        assert info.json()["dataset_origin"] == "synthetic"
        result = client.post("/predict", json=dict(hospital_id="HOSP_A", medicine_id="PGMED001", horizon_days=horizon))
        assert result.status_code == 200
        assert len(result.json()["predictions"]) == horizon
        assert all(row["predicted_demand"] >= 0 for row in result.json()["predictions"])
        assert result.json()["risk_level"] in {"normal", "elevated", "high_surge", "critical_surge"}
        assert result.json()["average_predicted_demand"] >= 0
        assert "surge_percent" in result.json()
        assert client.post("/predict", json=dict(hospital_id="UNKNOWN", medicine_id="PGMED001")).status_code == 400
        assert client.post("/predict", json=dict(hospital_id="HOSP_A", medicine_id="PGMED001", horizon_days=8)).status_code == 422


def test_missing_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model/info").status_code == 503
        result = client.post("/predict", json=dict(hospital_id="HOSP_A", medicine_id="PGMED001"))
        assert result.status_code == 503
        assert "not trained" in result.json()["detail"]
