"""One feature definition for both training and recursive inference."""
import pandas as pd
from .data_loader import KEYS

FEATURE_NAMES = ["hospital_id", "medicine_id", "day_of_week", "month",
                 "outbreak_flag", "outbreak_rate_7",
                 "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14"]


def category_mappings(data: pd.DataFrame) -> dict[str, list[str]]:
    """Stable category order is persisted with the model."""
    return {key: sorted(data[key].astype(str).unique().tolist()) for key in KEYS}


def build_features(data: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Build causal features; target at date t cannot affect its own features."""
    frame = data.sort_values([*KEYS, "date"]).copy()
    groups = frame.groupby(KEYS, observed=True)["quantity_requested"]
    frame["lag_1"] = groups.shift(1)
    frame["lag_7"] = groups.shift(7)
    for window in (7, 14):
        frame[f"rolling_mean_{window}"] = groups.transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean())
    frame["day_of_week"] = frame.date.dt.dayofweek
    frame["month"] = frame.date.dt.month
    frame["outbreak_flag"] = pd.to_numeric(frame["outbreak_flag"], errors="raise").astype(float)
    frame["outbreak_rate_7"] = frame.groupby(KEYS, observed=True)["outbreak_flag"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).mean()
    )
    for key in KEYS:
        if not frame[key].astype(str).isin(categories[key]).all():
            raise ValueError(f"Unknown categories in {key}")
        frame[key] = pd.Categorical(frame[key], categories=categories[key])
    return frame.sort_values(["date", *KEYS]).reset_index(drop=True)
