# Changelog (CHANGES.md)

All enhancements, modifications, and evaluations performed on the Malaria Outbreak Prediction System.

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

