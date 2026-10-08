import numpy as np
import pandas as pd
import pytest
from src.ml.data_loader import validate_data
from src.ml.features import build_features, category_mappings, FEATURE_NAMES
from src.ml.synthetic_data import generate_synthetic


@pytest.mark.parametrize("column", ["date", "hospital_id", "medicine_id", "quantity_requested", "source_medicine_id"])
def test_missing_columns(observed, column):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(observed.drop(columns=column))


@pytest.mark.parametrize("column,value", [
    ("date", "not-a-date"), ("quantity_requested", "invalid"),
    ("quantity_requested", -1), ("quantity_requested", np.nan),
    ("quantity_requested", np.inf), ("hospital_id", ""), ("medicine_id", None)])
def test_invalid_values(observed, column, value):
    broken = observed.copy()
    broken[column] = broken[column].astype(object)
    broken.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_data(broken)


def test_duplicates(observed):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_data(pd.concat([observed, observed.iloc[[0]]]))


def test_missing_daily_date(observed):
    with pytest.raises(ValueError, match="Missing dates"):
        validate_data(observed.drop(index=20))


def test_sorted_and_canonical_ids(observed):
    checked = validate_data(observed.sample(frac=1, random_state=9))
    assert checked.date.is_monotonic_increasing
    assert set(checked.medicine_id) == {f"PGMED00{i}" for i in range(1, 6)}


def test_causal_features(synthetic):
    categories = category_mappings(synthetic)
    cutoff = synthetic.date.min() + pd.Timedelta(days=30)
    original = build_features(synthetic, categories)
    changed = synthetic.copy()
    changed.loc[changed.date >= cutoff, "quantity_requested"] = 1000000
    changed_features = build_features(changed, categories)
    pd.testing.assert_frame_equal(original.loc[original.date <= cutoff, FEATURE_NAMES],
                                  changed_features.loc[changed_features.date <= cutoff, FEATURE_NAMES])
    series = synthetic[(synthetic.hospital_id == "HOSP_A") & (synthetic.medicine_id == "PGMED001")]
    rows = original[(original.hospital_id == "HOSP_A") & (original.medicine_id == "PGMED001")]
    row = rows.iloc[20]
    assert row.lag_1 == series.iloc[19].quantity_requested
    assert row.lag_7 == series.iloc[13].quantity_requested
    assert row.rolling_mean_7 == pytest.approx(series.iloc[13:20].quantity_requested.mean())
    assert row.rolling_mean_14 == pytest.approx(series.iloc[6:20].quantity_requested.mean())
    assert "outbreak_flag" in FEATURE_NAMES
    assert "outbreak_rate_7" in FEATURE_NAMES
    assert "quantity_dispensed" not in FEATURE_NAMES


def test_synthetic_reproducible(observed):
    one = generate_synthetic(days=30, seed=42, catalog=observed)
    two = generate_synthetic(days=30, seed=42, catalog=observed)
    pd.testing.assert_frame_equal(one, two)
    assert set(one.data_origin) == {"synthetic"}
