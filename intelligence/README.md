# PulseGrid AI: outbreak-aware demand forecasting V2

Standalone Python 3.12 LightGBM demand forecasts and a FastAPI service. Nothing connects to Supabase or changes the main backend. The target is **quantity_requested**, and series use canonical **medicine_id**. This version uses the historical `outbreak_flag` and a causal seven-day outbreak rate as demand features. Inventory, risk labels, and quantity dispensed remain reporting data until they are explicitly joined and validated.

## Windows PowerShell: run the demo

From the repository root:

```powershell
cd intelligence
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.ml.synthetic_data
python -m src.ml.train
python -m src.ml.evaluate
python -m pytest tests -q
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

All module commands run **inside intelligence/** so the standalone `src` package is selected. The generator writes `../data/processed/demand_history_synthetic.csv`; training writes `models/lightgbm_v1.joblib` and `reports/`. Paths are resolved relative to the modules rather than the shell. The existing observed CSV, preparation scripts and backend files are preserved.

The synthetic CSV is already generated in this checkout. The generator refuses to overwrite existing files, including the observed CSV. Skip generation when reusing the supplied synthetic dataset. To generate a separate reproducibility sample, use `python -m src.ml.synthetic_data --output ../data/processed/demand_history_synthetic_demo.csv`; train it with `python -m src.ml.train --data ../data/processed/demand_history_synthetic_demo.csv`. Training/evaluation intentionally refresh their generated artifact/reports. The small trained V1 demo artifact is included in Git; you can retrain after cloning with the pinned dependencies.

If PowerShell blocks activation, use the virtual environment's interpreter directly, e.g. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m src.ml.train`. No system execution-policy change is necessary.

In another PowerShell window:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/model/info' | ConvertTo-Json -Depth 12
$body = @{ hospital_id = 'HOSP_A'; medicine_id = 'PGMED001'; horizon_days = 7 } | ConvertTo-Json
$result = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict' -ContentType 'application/json' -Body $body
$result | ConvertTo-Json -Depth 6
$result.predictions.Count
```

For local Swagger, open `http://127.0.0.1:8000/docs`, expand POST `/predict`, click **Try it out**, enter the same JSON and execute. A 7-day request returns exactly 7 dates; change `horizon_days` to 14 for 14 dates. These local instructions are for your Windows machine, not a cloud web preview.

GET `/health` checks service liveness. GET `/model/info` reports training dates, known IDs/pairs, feature order and synthetic evaluation metrics. POST `/predict` also returns the recent seven-day baseline, average predicted demand, percentage change, and a transparent risk label:

- `normal`: predicted demand is not above the recent baseline
- `elevated`: predicted demand is up to 10% above baseline
- `high_surge`: predicted demand is 10-25% above baseline
- `critical_surge`: predicted demand is at least 25% above baseline

These are demand-surge signals, not a clinical outbreak diagnosis. `outbreak_flag` is an input scenario signal and should be populated from a validated surveillance workflow before production use. Both model-dependent endpoints return 503 if the artifact is missing. Unknown pairs return 400; unsupported horizons or malformed requests return 422. The artifact is loaded once per service instance; restart Uvicorn after replacing it. Never load joblib artifacts from untrusted sources.

Linux/cloud equivalent (from `intelligence/`):

```bash
source /workspace/.venvs/lightgbm-forecasting/bin/activate
python -m pip install -r requirements.txt
python -m src.ml.train
python -m src.ml.evaluate
python -m pytest tests -q
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The original preparation tests can be run separately from the repository root with `python -m pytest src/test_data.py -q`.

## Data and evaluation

The observed `demand_history_clean.csv` contains 300 observations over 15 dates for four hospitals and five medicines. It is validated but **not used for a claimed 7/14-day evaluation**: there is insufficient history for warm-up, training, validation and holdout. Passing this CSV to training produces a clear insufficient-history error.

The generator simulates 180 days ending October 8, 2026: hospital and medicine levels, weekly seasonality, trends, noise and occasional outbreak demand increases. Seed 42 makes generation reproducible. Every synthetic row carries `data_origin=synthetic`. IDs and medicine names come from the observed catalog. This is a demonstration dataset, not additional observations.

All series share the same boundaries: earlier dates for training, the next 14 dates for validation, and the final 14 dates for testing (or `--test-days 7`). Features need 14 historical warm-up days. Early stopping uses chronological **one-step** validation: past observed validation demand is available for subsequent validation dates. After choosing the tree count, an evaluation model is refitted on train plus validation dates. The final test is a **fixed-origin recursive forecast**: each series is seeded with pre-test history only and predicted demand is appended for each subsequent day. The baseline repeats the most recent seven pre-test observed demands. Both models are scored on identical pair/date rows. The report includes the first 7 and full 14 recursive forecast days, not two independent test periods.

After evaluation, the serving model is refitted with the selected tree count on the entire dataset. Its saved 14-day history per pair ends October 8, so demo forecasts start October 9. Its training range differs from the model used to compute holdout metrics; both ranges are explicit in the report. Serving historical demand is simulated in this V1 demo. Reproducibility is supported for the pinned stack and fixed seed; results may differ on another platform or dependency stack.

MAE and RMSE are in daily demand units. WAPE is a ratio (`0.07` means 7%); it is JSON `null` when total actual demand is zero and error is nonzero, or zero for an exact all-zero forecast. The improvement flag requires lower MAE and no worse WAPE/RMSE. There are no confidence intervals, clinical accuracy claims, shortage calculations or transfer recommendations.

## Modules and artifacts

| Module | Purpose |
| --- | --- |
| `src/ml/data_loader.py` | Validate required columns, finite nonnegative demand, dates, canonical IDs, unique keys and daily continuity; return chronological records. |
| `src/ml/synthetic_data.py` | Generate clearly labeled simulations without overwriting datasets. |
| `src/ml/features.py` | Define the feature order and categories once; build grouped lags and shifted rolling means for training and inference. |
| `src/ml/baseline.py` | Repeat the latest seven historical daily demands for 7/14 days. |
| `src/ml/train.py` | Split dates, select tree count, recursively evaluate, refit the serving model and persist it. |
| `src/ml/evaluate.py` | Compute actual metrics, write JSON and prediction-vs-actual CSV; regenerate reports from saved holdout results without retraining. |
| `src/ml/predict.py` | Load artifacts, validate pairs/horizons, and recursively predict finite nonnegative daily demand. |
| `src/api/main.py` | Expose health, model information and prediction through typed standalone FastAPI schemas. |
| `tests/` | Cover validation, leakage, training, metrics, baseline, persistence, predictions and API errors. |

`models/lightgbm_v1.joblib` contains the estimator, version, feature order, categorical mappings, serving training date range, latest 14 days of demand per pair, evaluation metadata, metrics and aligned held-out predictions. `reports/evaluation.json` contains split ranges, counts, parameters, seed, metrics, limitations and comparison outcome. `reports/prediction_vs_actual.csv` contains actual demand and both forecasts by date/pair/horizon day.

For later backend integration, import `predict_demand()` and supply a cached artifact, or call the standalone endpoint. No training takes place during a prediction request.

## Hackathon demonstration

1. Explain that V1 forecasts requested demand; show the synthetic label and model metadata first.
2. Show `reports/evaluation.json` and compare both models honestly on the same recursive holdout.
3. Use Swagger or PowerShell to request 7 and 14 days for `HOSP_A` / `PGMED001`, showing exact counts and dates after the saved history.
4. Demonstrate the unknown-ID error and explain why absent histories are rejected.
5. Describe inventory simulation and safe MOU-based redistribution as future integrations. More observed history, rolling-origin backtests and domain review are needed before operational use.
