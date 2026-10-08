import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from src.ml.predict import predict_demand, predict_from_real_history


def real_history(days=28):
    start = date(2026, 1, 1)
    return pd.DataFrame([
        {
            "date": start + timedelta(days=index),
            "hospital_id": "real-hospital",
            "medicine_id": "MED-REAL",
            "quantity_requested": 10 + index % 4,
            "outbreak_flag": int(index >= 21),
        }
        for index in range(days)
    ])


@pytest.mark.parametrize("horizon", [7, 14])
def test_forecast_shape_dates_values(trained, horizon):
    artifact, path, _ = trained
    result = predict_demand("HOSP_A", "PGMED001", horizon, model_path=path)
    assert len(result["predictions"]) == horizon
    assert result["model_version"] == "lightgbm-outbreak-v2"
    dates = pd.to_datetime([row["date"] for row in result["predictions"]])
    assert dates[0] == artifact["history"].date.max() + pd.Timedelta(days=1)
    assert (dates[1:] - dates[:-1] == pd.Timedelta(days=1)).all()
    values = [row["predicted_demand"] for row in result["predictions"]]
    assert np.isfinite(values).all() and min(values) >= 0
    assert result == predict_demand("HOSP_A", "PGMED001", horizon, model_path=path)


@pytest.mark.parametrize("hospital,medicine", [("UNKNOWN", "PGMED001"), ("HOSP_A", "UNKNOWN")])
def test_unknown_ids(trained, hospital, medicine):
    with pytest.raises(ValueError, match="Unknown hospital-medicine"):
        predict_demand(hospital, medicine, model_path=trained[1])


def test_known_ids_unknown_combination(trained):
    artifact = dict(trained[0])
    history = artifact["history"]
    artifact["history"] = history[~((history.hospital_id == "HOSP_A") & (history.medicine_id == "PGMED001"))]
    with pytest.raises(ValueError, match="Unknown hospital-medicine"):
        predict_demand("HOSP_A", "PGMED001", artifact=artifact)


@pytest.mark.parametrize("horizon", [0, 1, 8, 15])
def test_invalid_horizon(trained, horizon):
    with pytest.raises(ValueError, match="7 or 14"):
        predict_demand("HOSP_A", "PGMED001", horizon, model_path=trained[1])


def test_real_history_uses_external_ids():
    result = predict_from_real_history(real_history(), 7, future_outbreak_flag=1)
    assert result["model_version"] == "lightgbm-real-history-v1"
    assert result["training_observations"] == 28
    assert len(result["predictions"]) == 7


def test_real_history_requires_consecutive_records():
    history = real_history()
    history.loc[14, "date"] = history.loc[13, "date"]
    with pytest.raises(ValueError, match="consecutive"):
        predict_from_real_history(history)
