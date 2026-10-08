"""Load persisted artifacts and predict recursively, without retraining."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from .data_loader import ROOT, KEYS
from .features import FEATURE_NAMES, build_features, category_mappings

DEFAULT_MODEL = ROOT / "intelligence/models/lightgbm_outbreak_v2.joblib"


def load_artifact(model_path: str | Path = DEFAULT_MODEL) -> dict:
    """Load a trusted local joblib artifact. Never load untrusted pickle files."""
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not trained: {model_path}. Run src.ml.train first.")
    artifact = joblib.load(model_path)
    if artifact["feature_names"] != FEATURE_NAMES:
        raise ValueError("Saved model feature definitions do not match this module")
    return artifact


def recursive_forecast(model, history: pd.DataFrame, categories: dict,
                       horizon_days: int, future_outbreak_flag: int = 0) -> pd.DataFrame:
    """Append only model predictions; never read future actual demand."""
    if horizon_days not in (7, 14):
        raise ValueError("horizon_days must be 7 or 14")
    if len(history) < 14 or len(history[KEYS].drop_duplicates()) != 1:
        raise ValueError("Recursive prediction needs one series with at least 14 days")
    working = history[["date", *KEYS, "quantity_requested", "outbreak_flag"]].sort_values("date").tail(14).copy()
    if not working.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Recursive prediction requires consecutive daily history")
    rows = []
    for _ in range(horizon_days):
        last = working.iloc[-1]
        date = last.date + pd.Timedelta(days=1)
        candidate = pd.DataFrame([dict(date=date, hospital_id=str(last.hospital_id),
            medicine_id=str(last.medicine_id), quantity_requested=np.nan,
            outbreak_flag=future_outbreak_flag)])
        expanded = pd.concat([working, candidate], ignore_index=True)
        features = build_features(expanded, categories).iloc[[-1]][FEATURE_NAMES]
        value = float(model.predict(features)[0])
        if not np.isfinite(value):
            raise ValueError("Model produced nonfinite demand")
        value = max(0.0, value)
        candidate["quantity_requested"] = value
        working = pd.concat([working, candidate], ignore_index=True).tail(14)
        rows.append({"date": date, "predicted_demand": value})
    return pd.DataFrame(rows)


def predict_demand(hospital_id: str, medicine_id: str, horizon_days: int = 7,
                   future_outbreak_flag: int = 0,
                   *, model_path: str | Path = DEFAULT_MODEL,
                   artifact: dict | None = None) -> dict:
    """Return exactly 7 or 14 daily forecasts after this pair's latest history."""
    if horizon_days not in (7, 14):
        raise ValueError("horizon_days must be 7 or 14")
    saved = load_artifact(model_path) if artifact is None else artifact
    history = saved["history"]
    history = history[(history.hospital_id == hospital_id) & (history.medicine_id == medicine_id)]
    if history.empty:
        raise ValueError(f"Unknown hospital-medicine combination: {hospital_id}/{medicine_id}")
    forecast = recursive_forecast(
        saved["model"], history, saved["categories"], horizon_days, future_outbreak_flag
    )
    baseline_daily_demand = float(history["quantity_requested"].tail(7).mean())
    average_predicted_demand = float(forecast["predicted_demand"].mean())
    surge_percent = (
        ((average_predicted_demand - baseline_daily_demand) / baseline_daily_demand) * 100
        if baseline_daily_demand > 0
        else 0.0
    )
    if surge_percent >= 25:
        risk_level = "critical_surge"
    elif surge_percent >= 10:
        risk_level = "high_surge"
    elif surge_percent > 0:
        risk_level = "elevated"
    else:
        risk_level = "normal"
    return dict(hospital_id=hospital_id, medicine_id=medicine_id,
        model_version=saved["model_version"], horizon_days=horizon_days,
        outbreak_flag=future_outbreak_flag,
        baseline_daily_demand=round(baseline_daily_demand, 2),
        average_predicted_demand=round(average_predicted_demand, 2),
        surge_percent=round(surge_percent, 2),
        risk_level=risk_level,
        predictions=[dict(date=row.date.strftime("%Y-%m-%d"),
                          predicted_demand=float(row.predicted_demand))
                     for row in forecast.itertuples()])


def predict_from_real_history(history: pd.DataFrame, horizon_days: int = 7,
                              future_outbreak_flag: int = 0) -> dict:
    """Train a small per-series model from validated hospital history.

    This path is deliberately separate from the persisted synthetic artifact:
    real UUID/SKU categories are not silently passed to a model trained on
    unrelated demo IDs.
    """
    if horizon_days not in (7, 14):
        raise ValueError("horizon_days must be 7 or 14")
    required = {"date", "hospital_id", "medicine_id", "quantity_requested", "outbreak_flag"}
    if not required.issubset(history.columns):
        raise ValueError(f"Real history is missing columns: {sorted(required - set(history.columns))}")
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], format="ISO8601", errors="raise")
    history["quantity_requested"] = pd.to_numeric(history["quantity_requested"], errors="raise")
    history["outbreak_flag"] = pd.to_numeric(history["outbreak_flag"], errors="raise")
    if len(history) < 28 or history[["hospital_id", "medicine_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Real ML forecasting needs one medicine series with at least 28 daily records")
    history = history.sort_values("date").reset_index(drop=True)
    expected = pd.date_range(history.date.min(), history.date.max(), freq="D")
    if history["date"].tolist() != list(expected):
        raise ValueError("Real ML forecasting needs consecutive daily records without gaps")
    categories = category_mappings(history)
    features = build_features(history, categories).dropna(subset=FEATURE_NAMES)
    if len(features) < 14:
        raise ValueError("Real ML forecasting needs at least 14 usable feature rows")
    model = lgb.LGBMRegressor(
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=7,
        max_depth=4,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(features[FEATURE_NAMES], features.quantity_requested)
    forecast = recursive_forecast(model, history, categories, horizon_days, future_outbreak_flag)
    baseline = float(history.quantity_requested.tail(7).mean())
    average = float(forecast.predicted_demand.mean())
    surge = ((average - baseline) / baseline) * 100 if baseline > 0 else 0.0
    risk = "critical_surge" if surge >= 25 else "high_surge" if surge >= 10 else "elevated" if surge > 0 else "normal"
    return {
        "model_version": "lightgbm-real-history-v1",
        "training_observations": len(history),
        "training_date_range": {
            "start": history.date.min().strftime("%Y-%m-%d"),
            "end": history.date.max().strftime("%Y-%m-%d"),
        },
        "horizon_days": horizon_days,
        "outbreak_flag": future_outbreak_flag,
        "baseline_daily_demand": round(baseline, 2),
        "average_predicted_demand": round(average, 2),
        "surge_percent": round(surge, 2),
        "risk_level": risk,
        "predictions": [
            {"date": row.date.strftime("%Y-%m-%d"), "predicted_demand": float(row.predicted_demand)}
            for row in forecast.itertuples()
        ],
    }
