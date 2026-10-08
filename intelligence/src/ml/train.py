"""Chronological pooled training, recursive holdout testing, and persistence."""
import argparse
from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd
from .baseline import seasonal_naive
from .data_loader import KEYS, load_data
from .evaluate import DEFAULT_REPORTS, save_evaluation
from .features import FEATURE_NAMES, build_features, category_mappings
from .predict import DEFAULT_MODEL, recursive_forecast
from .synthetic_data import DEFAULT_SYNTHETIC, SEED

MODEL_VERSION = "lightgbm-outbreak-v2"
PARAMETERS = dict(n_estimators=250, learning_rate=0.04, num_leaves=15,
    max_depth=5, min_child_samples=30, reg_lambda=2.0,
    random_state=SEED, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)


def date_range(data: pd.DataFrame) -> dict[str, str]:
    """JSON-safe chronological range."""
    return {"start": data.date.min().strftime("%Y-%m-%d"),
            "end": data.date.max().strftime("%Y-%m-%d")}


def train_model(data_path: str | Path = DEFAULT_SYNTHETIC,
                model_path: str | Path = DEFAULT_MODEL,
                report_dir: str | Path = DEFAULT_REPORTS,
                test_days: int = 14) -> dict:
    """Select iterations on later dates; test recursively then refit serving model."""
    if test_days not in (7, 14):
        raise ValueError("test_days must be 7 or 14")
    data = load_data(data_path)
    dates = sorted(data.date.unique())
    validation_days = 14
    # 14 warm-up days plus at least 28 training dates, validation and holdout.
    if len(dates) < 14 + 28 + validation_days + test_days:
        raise ValueError("Insufficient history for defensible chronological training/validation/test; "
                         "generate synthetic data (observed 15-day history is not enough)")
    for ids, series in data.groupby(KEYS, observed=True):
        if len(series) != len(dates) or series.date.min() != dates[0] or series.date.max() != dates[-1]:
            raise ValueError(f"All series must share common daily date coverage for splitting: {ids}")
    if "data_origin" in data:
        origins = data.data_origin
        if origins.isna().any() or origins.astype(str).str.strip().eq("").any():
            raise ValueError("data_origin must be nonmissing when provided")
        origin = ",".join(sorted(origins.astype(str).unique()))
    else:
        origin = "observed"
    categories = category_mappings(data)
    test_start = pd.Timestamp(dates[-test_days])
    validation_start = pd.Timestamp(dates[-test_days - validation_days])
    train_data = data[data.date < validation_start]
    val_data = data[(data.date >= validation_start) & (data.date < test_start)]
    test_data = data[data.date >= test_start]
    pretest = data[data.date < test_start]
    # Features for fitting/early stopping exclude the final test entirely.
    featured = build_features(pretest, categories).dropna(subset=FEATURE_NAMES)
    train_rows = featured[featured.date < validation_start]
    val_rows = featured[featured.date >= validation_start]
    selection_model = lgb.LGBMRegressor(**PARAMETERS)
    selection_model.fit(train_rows[FEATURE_NAMES], train_rows.quantity_requested,
        eval_X=val_rows[FEATURE_NAMES], eval_y=val_rows.quantity_requested,
        eval_metric="l1", callbacks=[lgb.early_stopping(25, verbose=False)])
    iterations = selection_model.best_iteration_ or PARAMETERS["n_estimators"]
    final_parameters = {**PARAMETERS, "n_estimators": iterations}
    evaluation_model = lgb.LGBMRegressor(**final_parameters)
    evaluation_model.fit(featured[FEATURE_NAMES], featured.quantity_requested)
    rows = []
    for (hospital, medicine), history in pretest.groupby(KEYS, observed=True):
        forecast = recursive_forecast(evaluation_model, history, categories, test_days)
        baseline = seasonal_naive(history, test_days)
        actual = test_data[(test_data.hospital_id == hospital) & (test_data.medicine_id == medicine)]
        joined = forecast.merge(actual[["date", "quantity_requested"]], on="date", validate="one_to_one")
        joined = joined.merge(baseline.rename(columns={"predicted_demand": "baseline_prediction"}),
                              on="date", validate="one_to_one")
        if len(joined) != test_days:
            raise ValueError("Incomplete aligned test coverage")
        joined = joined.rename(columns={"predicted_demand": "lightgbm_prediction", "quantity_requested": "actual"})
        joined["hospital_id"], joined["medicine_id"] = hospital, medicine
        joined["horizon_day"] = range(1, test_days + 1)
        rows.append(joined)
    predictions = pd.concat(rows, ignore_index=True).sort_values(["date", *KEYS]).reset_index(drop=True)
    metadata = dict(dataset_origin=origin, train_date_range=date_range(train_data),
        validation_date_range=date_range(val_data), test_date_range=date_range(test_data),
        training_feature_date_range=date_range(train_rows),
        evaluation_fit_date_range=date_range(pretest), serving_training_date_range=date_range(data),
        number_of_hospitals=data.hospital_id.nunique(), number_of_medicines=data.medicine_id.nunique(),
        number_of_observations=len(data), model_version=MODEL_VERSION,
        model_hyperparameters=final_parameters, selection_hyperparameters=PARAMETERS,
        random_seed=SEED, best_iteration=iterations,
        test_horizon_days=test_days,
        validation_method="Chronological one-step validation for early stopping (past observed lags).",
        test_method="Fixed-origin recursive holdout; only predicted targets are appended.",
        serving_refit="Full dataset after evaluation; held-out results belong to the pre-test model.")
    report = save_evaluation(predictions, metadata, report_dir)
    serving_rows = build_features(data, categories).dropna(subset=FEATURE_NAMES)
    serving_model = lgb.LGBMRegressor(**final_parameters)
    serving_model.fit(serving_rows[FEATURE_NAMES], serving_rows.quantity_requested)
    history = data.groupby(KEYS, observed=True).tail(14)[
        ["date", *KEYS, "quantity_requested", "outbreak_flag"]
    ].copy()
    artifact = dict(model=serving_model, model_version=MODEL_VERSION,
        feature_names=FEATURE_NAMES.copy(), categories=categories,
        training_date_range=date_range(data), history=history,
        evaluation_metadata=metadata, evaluation=report, evaluation_predictions=predictions)
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    print(f"Trained {MODEL_VERSION} on {len(data)} {origin} observations; best iteration={iterations}")
    print(f"Recursive {test_days}-day holdout: LightGBM {report['lightgbm_metrics']}; baseline {report['baseline_metrics']}")
    print(f"LightGBM outperformed baseline: {report['lightgbm_outperformed_baseline']}; artifact: {path}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--test-days", type=int, choices=(7, 14), default=14)
    args = parser.parse_args()
    train_model(args.data, args.model, args.reports, args.test_days)


if __name__ == "__main__":
    main()
