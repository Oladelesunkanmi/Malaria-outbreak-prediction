# Changelog (CHANGES.md)

All enhancements, modifications, and evaluations performed on the Malaria Outbreak Prediction System.

---

### [2026-11-06] Full System Retrain & Artifact Replacement

#### 1. Archive & Backup (`archive/previous_run_20261106/`)
- Archived all previous model files (`models/*.pkl`, `models/*.json`), diagnostic visualizations (`static/images/*.png`), active database (`malaria_predict.db`), and changelog (`CHANGES.md`) into `archive/previous_run_20261106/`.

#### 2. Clean Deletion of Generated Artifacts
- Deleted generated files in `models/` and `static/images/` to eliminate stale training outputs.
- Confirmed zero impact on raw dataset (`data/`), application codebase, or database.

#### 3. Complete Pipeline Execution
- Retrained univariate baseline models (`best_model.pkl`, `model_metrics.json`, `horizon_evaluation.json`, `seasonal_baselines.json`, `preprocessing_report.json`).
- Retrained rainfall-enhanced models (`best_model_rainfall.pkl`, `model_metrics_rainfall.json`, `horizon_evaluation_rainfall.json`).
- Retrained outbreak classification pipeline (`outbreak_classifier.pkl`, `outbreak_metrics.json`, `outbreak_baselines.json`).
- Regenerated all 4 diagnostic plots (`model_comparison.png`, `actual_vs_predicted.png`, `prediction_error.png`, `outbreak_confusion_matrix.png`).
- Re-seeded `ModelMetric` records into `malaria_predict.db`.

#### 4. Audit & Verification (`RETRAIN_LOG.md`)
- Verified that 1-month holdout MAE equals 1-month horizon evaluation MAE across all models.
- Verified that RMSE equals $\sqrt{\text{MSE}}$ without discrepancies.
- Verified outbreak training (258 samples) and test (66 samples) split counts.
- Purged temporary verification predictions from `malaria_predict.db`, returning it to 0 predictions and 360 historical records.
- All 19 tests passing (`pytest -v`). Full execution log recorded in `RETRAIN_LOG.md`.

---

### [2026-11-06] Removal of 3-Month Forecast Horizon Cap

#### 1. Prediction Engine & Service (`ml/predict.py`, `services/prediction_service.py`)
- Removed the 3-month forecast cap exception (`months_ahead > 3` check) in `ml/predict.py`. Chained autoregressive forecasts can now proceed sequentially beyond 3 months.
- Retained dynamic calculation of `months_ahead` and `horizon_warning` (`horizon > 1`) so users remain informed regarding multi-step recursive uncertainty without blocking forecasts.
- Updated `services/prediction_service.py` to keep `'can_predict': True` and `'horizon_limit_reached': False` regardless of forecast horizon.

#### 2. User Interface & Dynamic Scripting (`templates/predict.html`, `static/js/prediction.js`)
- Removed the alert banner text *"Maximum Horizon Limit: Forecasts are capped at 3 months ahead of historical records."*
- Removed the disabled state on the submission button in `templates/predict.html` and `static/js/prediction.js`.
- Preserved chained multi-step warnings linking to recursive evaluation benchmarks.

#### 3. Test Suite (`tests/test_prediction.py`)
- Replaced `test_predict_horizon_limit_exceeded()` with `test_predict_beyond_3_months_allowed()`, verifying that chaining forecasts into month 4 and beyond executes cleanly and sets `horizon_months == 4`.
- All 19 tests passing (`pytest -v`).

---

### [2026-11-06] Comprehensive Software Verification & End-to-End Audit (SOFTWARE_CHECK.md)

#### 1. Recursive Multi-Step Horizon Evaluation Verification (`ml/train.py`)
- Verified that `models/horizon_evaluation.json` (baseline) and `models/horizon_evaluation_rainfall.json` (rainfall-enhanced) use the exact same test origins and sample counts ($N = 66$ for $h=1$, $60$ for $h=3$, $51$ for $h=6$, and $33$ for $h=12$).
- Verified that recursive rainfall forecasts use strictly past rainfall initialized up to origin month $t$, holding rainfall flat at the last known level during future rollouts without leaking observed future ground truth.

#### 2. Feature Shift & Null Check Verification (`ml/feature_engineering.py`)
- Confirmed that all rainfall features (`rainfall_lag_1`, `rainfall_lag_2`, `rainfall_lag_3`, `rainfall_lag_12`, `rainfall_cum_3`, `rainfall_mean_3`) use pandas `.shift(k)` with $k \ge 1$.
- Verified that dropping initial rows results in exactly 108 observations per state (324 total rows across 2016–2024) with zero missing values across all features.

#### 3. Test Suite & Database Isolation
- Verified test suite passes completely (19 passed out of 19 tests in 15.94s).
- Confirmed that `malaria_predict.db` maintains unchanged row counts before and after test execution (360 historical records, 0 predictions, 5 model metrics).

#### 4. End-to-End Audit & Security Checks
- Tested authenticated access to all views (`/dashboard`, `/dataset`, `/analytics`, `/predict`, `/evaluation`, `/history`) – all returned HTTP 200 OK.
- Tested sequential predictions for Kwara, Nasarawa, and Zamfara; verified 2-month and 3-month chained warning notices.
- Confirmed `malaria_prediction_system/` does not exist, zero active `Admin123` passwords exist in code, and `config.py` raises `ValueError` when `ADMIN_PASSWORD` is missing.
- Documented complete raw outputs in `SOFTWARE_CHECK.md`.

---

### [2026-11-06] Historical Climate & Rainfall Integration

#### 1. Dataset & Climate Integration (`data/rainfall.csv`, `config.py`, `ml/preprocessing.py`)
- Integrated `climate_monthly.csv` (historical NASA POWER monthly temperature, rainfall, and relative humidity data for Kwara, Nasarawa, and Zamfara, 2015–2024) into `data/rainfall.csv`.
- Configured `RAINFALL_PATH`, `BEST_MODEL_RAINFALL_PATH`, and `METRICS_RAINFALL_PATH` in `config.py`.
- Updated `preprocess_malaria_data()` in `ml/preprocessing.py` to perform an inner/left merge on `(State, year, month)`, standardizing precipitation to `rainfall_mm` and updating `models/preprocessing_report.json` with merge metrics (0 missing rainfall values across 360 observations).

#### 2. Feature Engineering (`ml/feature_engineering.py`)
- Created `BASE_FEATURE_COLUMNS` and `RAINFALL_FEATURE_COLUMNS` (`rainfall_lag_1`, `rainfall_lag_2`, `rainfall_lag_3`, `rainfall_lag_12`, `rainfall_cum_3`, `rainfall_mean_3`).
- All rainfall features are strictly built from past observations prior to month $t$, eliminating data leakage and ensuring no future observed rainfall is used at inference time.
- Updated `compute_single_row_features()` with `recent_history_rainfall` input argument.

#### 3. Dual Model Pipelines & Benchmarking (`ml/train.py`)
- Preserved baseline model pipeline (`best_model.pkl`, `models/model_metrics.json`, `models/horizon_evaluation.json`) trained on univariate autoregressive lag features.
- Implemented `train_rainfall_enhanced_model()` producing `best_model_rainfall.pkl`, `models/model_metrics_rainfall.json`, and `models/horizon_evaluation_rainfall.json`.
- Evaluated models side-by-side using exact chronological 86-month train / 22-month test partitions:
  - Linear Regression (Baseline): MAE = 9,730.60 | RMSE = 19,709.93 | $R^2$ = 0.7497
  - Linear Regression (Rainfall): MAE = 9,930.16 | RMSE = 19,791.02 | $R^2$ = 0.7476
  - Random Forest Regressor (Baseline): MAE = 10,513.63 | RMSE = 22,488.05 | $R^2$ = 0.6741
  - Random Forest Regressor (Rainfall): MAE = 10,349.71 | RMSE = 23,691.73 | $R^2$ = 0.6383 (MAE improved by -163.92 cases)
  - Multi-step recursive forecasts: at longer horizons ($h = 3, 6, 12$), rainfall features substantially reduced cumulative error drift for Linear Regression (3-month MAE: 20,626.35 vs. 21,995.95 baseline; 6-month MAE: 22,855.98 vs. 25,528.68 baseline; 12-month MAE: 26,651.20 vs. 28,496.43 baseline).

#### 4. Prediction Service & UI (`ml/predict.py`, `templates/predict.html`)
- Added `get_state_chronological_rainfall_series()` in `ml/predict.py` to extract past rainfall values dynamically.
- `predict_next_month()` automatically utilizes the rainfall-enhanced model pipeline while preserving full backwards compatibility if only baseline models exist.
- Updated `templates/predict.html` to display `rainfall_lag_1` and `rainfall_cum_3` in the technical feature inspection table.

#### 5. Verification & Test Suite (`tests/test_evaluation_baselines.py`)
- Updated test suite to verify baseline isolation and rainfall-extended features.
- All 19 tests passing (`pytest -v`).

---

### [2026-11-06] Academic & Methodological Improvements

#### 1. `ml/train.py`
- **Exact Chronological Split**: Added explicit tracking and reporting of split dates:
  - 108 eligible months per state (Jan 2016 – Dec 2024; 2015 provides initial 12 lags).
  - Training Set: 86 months per state (Jan 2016 – Feb 2023; 258 total observations).
  - Test Set: 22 months per state (Mar 2023 – Dec 2024; 66 total observations).
- **Naive Baseline Models Added**: Implemented evaluation for:
  - `Baseline: Last Month` ($\hat{y}_t = y_{t-1}$)
  - `Baseline: Same Month Last Year` ($\hat{y}_t = y_{t-12}$)
  - Evaluated on the identical 66 test observations alongside Linear Regression, Random Forest, and Decision Tree.
- **Multi-Step Recursive Evaluation**: Added `run_multistep_evaluation()` testing horizons of 1, 3, 6, and 12 months using recursive autoregressive feature updates. Outputs saved to `models/horizon_evaluation.json`.
- **Expanding-Window Time-Series Cross-Validation**: Added `run_timeseries_cv()` evaluating 10 expanding folds (min window 48 months, step 6 months), computing mean MAE and standard deviation.
- **Per-State Metrics Breakdown**: Computed MAE, RMSE, and $R^2$ separately for Kwara, Nasarawa, and Zamfara. Identified Zamfara as contributing 76.16% of total absolute error.
- **Risk Tier Validation**: Implemented `validate_risk_tiers_on_test()` computing Hit Rate (Sensitivity: 66.67%), False Alarm Rate (4.76%), and 3-tier exact match accuracy (80.30%).

#### 2. `ml/evaluate.py`
- Updated plotting logic to support naive baselines in the performance comparison bar chart.
- Dynamically handled subplots in actual vs. predicted scatters to prevent index out-of-range errors.

#### 3. `ml/risk_classification.py`
- Softened claims: reworded methodology and disclaimer to explicitly describe risk tiers as a transparent statistical early-warning heuristic ($\mu + 2\sigma$) rather than claiming compliance with WHO or government outbreak standards. Preserved non-clinical diagnosis disclaimer.

#### 4. `app.py`
- Security cleanup: removed plain-text administrative password printing in console output.

#### 5. `templates/login.html`
- Security cleanup: removed plain-text default password from the login card footer. Added sanitized notice directing users to `.env` / environment variables.

#### 6. `templates/evaluation.html`
- Added chronological split verification banner with exact month counts.
- Updated metrics table to display candidate ML models alongside naive baselines.
- Added multi-step recursive horizon evaluation table (1, 3, 6, 12 months).
- Added expanding-window time-series cross-validation table (10 folds).
- Added per-state performance breakdown table highlighting Zamfara's error share.
- Added risk tier validation card displaying Hit Rate, False Alarm Rate, and confusion matrix metrics.

#### 7. `tests/test_prediction.py`
- Updated `test_predict_skip_month_raises_error()` to dynamically compute future skipped periods relative to the latest recorded prediction in the database.

#### 8. `tests/test_evaluation_baselines.py`
- Added comprehensive unit and integration tests covering:
  - Exact chronological month counts (86 train / 22 test per state).
  - Naive baseline error computations.
  - Risk tier validation metrics.
  - Multi-step recursive forecasting output format.

#### 9. `pytest.ini`
- Configured `testpaths = tests` to ensure deterministic test collection.

#### 10. `README.md` & Implementation Report Artifact
- Updated documentation with verified empirical results from script runs.
- Converted all mathematical expressions to standard LaTeX format ($...$ and $$...$$).
- Removed hardcoded plain-text passwords, documenting environment variable configuration.

---

### [2026-11-06] Outbreak Classifier, Horizon Constraints, Security, & Evaluation Fixes

#### 1. `ml/train.py`
- **Model Selection Parity (Task 1)**: Clarified model selection logic to state that Linear Regression wins on RMSE and $R^2$ among all models and on MAE among candidate ML models, while acknowledging the "Baseline: Last Month" persistence baseline achieves a lower MAE (9,623.08 vs 9,730.60).
- **Consistent One-Step Horizon Error (Task 2)**: Updated `run_multistep_evaluation()` to evaluate each horizon $h$ across all valid test origins ($N = len(s\_test) - h + 1$). Used the exact row population rather than fixed train-end population. For $h=1$, this evaluates all 66 test holdout observations, making 1-month-ahead MAE (9,730.60) strictly identical to the 1-step test holdout MAE.
- **Model Object Isolation in Cross-Validation**: Added `from sklearn.base import clone` in `run_timeseries_cv()` to prevent CV folds from mutating estimators in `fitted_pipelines`, ensuring that serialized `best_model.pkl` reflects the genuine model fit on `train_df`.
- **Exact MSE Values (Task 3)**: Recalculated and verified MSE metrics across all models directly from code execution, ensuring JSON files and report match without discrepancies.

#### 2. `ml/outbreak.py` (New File - Task 7)
- **Training-Only Outbreak Baselines**: Computed seasonal mean ($\mu$) and standard deviation ($\sigma$) per state-month using training data only (86 months/state; Jan 2016 – Feb 2023). Outbreak threshold defined as $\mu + 2\sigma$. Saved to `models/outbreak_baselines.json`.
- **Prediction Target**: Binary target `outbreak_next` (1 if month $t+1$ is an outbreak, 0 otherwise) built strictly from past features available up to month $t$.
- **Supervised Classifiers**: Trained and evaluated Balanced Logistic Regression and Balanced Random Forest alongside two naive baselines ("Always No Outbreak" and "Last Month Was Outbreak") on the 258 train / 66 test split.
- **Cross-Validation & Visualizations**: Implemented 5-fold expanding-window cross-validation; generated confusion matrix heatmaps saved to `static/images/outbreak_confusion_matrix.png`; serialized metrics to `models/outbreak_metrics.json` and best classifier pipeline to `models/outbreak_classifier.pkl`.

#### 3. `database/db.py` & `tests/conftest.py` (Task 4)
- **Engine Rebinding**: Added `set_engine_and_session()` in `database/db.py` to allow dynamic re-binding of SQLAlchemy engine and sessions.
- **Test Database Isolation**: Created `tests/conftest.py` with session-scoped fixture creating a fresh temporary SQLite database (`tempfile.mkstemp`), seeding it, and tearing it down after test runs. Tests never write to or read from `malaria_predict.db`.
- **Live DB Restored**: Cleaned 25 test prediction records from `malaria_predict.db`, restoring it to pristine state (0 predictions, 360 historical surveillance records, 1 admin user).

#### 4. `services/prediction_service.py` & `ml/predict.py` (Task 5, 7)
- **3-Month Maximum Horizon Guardrail**: Enforced maximum chained prediction horizon of at most 3 months ahead of the last historical record in `MalariaRecord` (2024-12). Predictions beyond 3 months raise a descriptive `ValueError`.
- **Outbreak Probability Output**: Integrated `models/outbreak_classifier.pkl` into `predict_next_month()` to return next-month outbreak probability alongside case count forecasts.
- **Horizon Warning Metadata**: Added `horizon_months`, `horizon_warning`, and `horizon_message` alerting users to increasing forecast uncertainty for multi-step horizons ($h > 1$).

#### 5. `templates/predict.html` & `static/js/prediction.js` (Task 5, 7)
- Displayed dynamic warning alert for horizons $> 1$ month linking to `/evaluation#horizonSection`.
- Added disabled submit button state if the 3-month forecast horizon is reached.
- Added display for Outbreak Probability percentage alongside case count forecasts and seasonal benchmarks.

#### 6. `routes/evaluation.py` & `templates/evaluation.html` (Task 6, 7)
- **Honest Risk-Tier Discussion (Task 6)**: Reported raw counts (2 hits, 1 miss, 3 false alarms, 60 true negatives) alongside percentages, and added explicit notice that 3 positive cases are too few to estimate sensitivity reliably.
- **Supervised Outbreak Classification Section (Task 7)**: Added dedicated card displaying the 4-model classification benchmark table, 5-fold expanding-window CV table, per-state breakdown, and confusion matrix heatmap image.

#### 7. `config.py` & Security Cleanup (Task 8)
- Removed hardcoded `'Admin123!'` fallback default. Missing `ADMIN_PASSWORD` now fails fast with a descriptive `ValueError`.
- Removed unused duplicate project folder `malaria_prediction_system/`.
- Sanitized `.env.example` placeholder and verified zero occurrences of passwords in the repository.

#### 8. `tests/test_outbreak.py` & `tests/test_prediction.py` (Task 4, 5, 7)
- Created `tests/test_outbreak.py` testing past-only label construction, chronological split (258 train / 66 test), confusion matrix sums, and output probability bounds $[0, 1]$.
- Added `test_predict_horizon_limit_exceeded()` in `tests/test_prediction.py` verifying 4th month chained predictions raise `ValueError`.
- All 19 tests passing.
- Softened all public health/WHO claims.

---

### [2026-11-06] Prediction Interface Null-Safety & Error Resolution

#### 1. `templates/predict.html`
- **Null-Safe String Formatting**: Added inline conditional checks (`if ... is not none else 'N/A'`) to all numeric format strings (`prediction.predicted_cases`, `prediction.baseline_mean`, `prediction.threshold_high`, and variance spread calculation).
- **Nested Feature Vector Protection**: Wrapped the feature parameter inspection table in `{% if prediction.features_used %}` with individual `is not none` guards on each lag and rolling mean to prevent `TypeError: unsupported format string passed to NoneType.__format__` during multi-step chained forecast rendering.
- **Period Info Fallback**: Added safe formatting for `period_info.latest_month` and `period_info.next_month`.

#### 2. `ml/risk_classification.py`
- Added fallback `"explanation"` string inside `classify_risk()` when baseline seasonal parameters are missing, preventing `KeyError: 'explanation'` in caller workflows.

#### 3. `routes/predictions.py`
- Added input presence validation for `year` and `month` parameters on `POST /predict`.
- Hardened exception handling around `PredictionService.get_eligible_period()` re-fetch to ensure failed state updates gracefully reset `prediction_result` without breaking view rendering.

#### 4. `app.py`
- Added `app.logger.error(..., exc_info=True)` to `@app.errorhandler(500)` to ensure full tracebacks are emitted to application logs if unhandled runtime exceptions occur.
- Added dynamic `FLASK_DEBUG` environment variable inspection in `__main__` entry point (`debug=debug_mode`).

#### 5. `malaria_predict.db`
- Cleared manual prediction test records from the live SQLite database, returning it to pristine initial state (0 predictions, 360 historical surveillance records, 1 administrative user).

