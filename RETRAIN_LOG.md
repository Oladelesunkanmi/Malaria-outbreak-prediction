# Clean Pipeline Retraining & Verification Log (`RETRAIN_LOG.md`)

**Date of Execution:** 2026-11-06  
**Execution Lead:** Senior Machine Learning Engineer & Systems Auditor  
**System Title:** Development of a Malaria Outbreak Prediction System Using Machine Learning Techniques  
**Operating Environment:** Windows (win32), Python 3.13.14, SQLite 3  

---

## Task 1: Backup of Previous Outputs

Before deleting any files, a complete copy of the existing models, visualizations, database, and changelog was archived into `archive/previous_run_20261106/`:

```powershell
New-Item -ItemType Directory -Force -Path 'archive/previous_run_20261106/models', 'archive/previous_run_20261106/static/images'
Copy-Item -Path 'models/*' -Destination 'archive/previous_run_20261106/models/' -Recurse
Copy-Item -Path 'static/images/*' -Destination 'archive/previous_run_20261106/static/images/' -Recurse
Copy-Item -Path 'malaria_predict.db' -Destination 'archive/previous_run_20261106/malaria_predict.db'
Copy-Item -Path 'CHANGES.md' -Destination 'archive/previous_run_20261106/CHANGES.md'
```

### Verified Archived Inventory:
- `archive/previous_run_20261106/CHANGES.md` (17,358 bytes)
- `archive/previous_run_20261106/malaria_predict.db` (86,016 bytes)
- `archive/previous_run_20261106/models/best_model.pkl` (3,697 bytes)
- `archive/previous_run_20261106/models/best_model_rainfall.pkl` (4,113 bytes)
- `archive/previous_run_20261106/models/horizon_evaluation.json` (887 bytes)
- `archive/previous_run_20261106/models/horizon_evaluation_rainfall.json` (887 bytes)
- `archive/previous_run_20261106/models/model_metrics.json` (20,746 bytes)
- `archive/previous_run_20261106/models/model_metrics_rainfall.json` (8,015 bytes)
- `archive/previous_run_20261106/models/outbreak_baselines.json` (6,889 bytes)
- `archive/previous_run_20261106/models/outbreak_classifier.pkl` (4,193 bytes)
- `archive/previous_run_20261106/models/outbreak_metrics.json` (13,935 bytes)
- `archive/previous_run_20261106/models/preprocessing_report.json` (1,813 bytes)
- `archive/previous_run_20261106/models/seasonal_baselines.json` (6,745 bytes)
- `archive/previous_run_20261106/static/images/actual_vs_predicted.png` (206,951 bytes)
- `archive/previous_run_20261106/static/images/model_comparison.png` (167,517 bytes)
- `archive/previous_run_20261106/static/images/outbreak_confusion_matrix.png` (113,757 bytes)
- `archive/previous_run_20261106/static/images/prediction_error.png` (146,135 bytes)

---

## Task 2: Deletion of Generated Outputs

The following generated files were deleted:
- `models/*.pkl` (3 files)
- `models/*.json` (8 files)
- `static/images/*.png` (4 files)

### Protected Assets Confirmed Untouched:
- Source code in `app.py`, `config.py`, `ml/`, `routes/`, `services/`, `database/`, `templates/`, `static/js/`, `tests/`
- Data files in `data/` (`Wavelets_data for Time series_malaria.csv`, `rainfall.csv`)
- Archive folder `archive/previous_run_20261106/`
- Live database `malaria_predict.db`

Verification: File count in `models/` = 0; File count in `static/images/` = 0.

---

## Task 3: Full Pipeline Retraining Command Output

### 3.1 Regression Pipelines Execution (`py -m ml.train`)
```
[Training] 1. Preprocessing raw dataset...
[Preprocessing] Report saved to C:\Users\PC\OneDrive\Desktop\malariapredict\models\preprocessing_report.json
[Training] 2. Generating lag and rolling features per state...
[Training] Total eligible observations with lags: 324
[Training] Train set: 258 rows, Test set: 66 rows
  -> Kwara: Train = 86 mos (2016-01 to 2023-02), Test = 22 mos (2023-03 to 2024-12)
  -> Nasarawa: Train = 86 mos (2016-01 to 2023-02), Test = 22 mos (2023-03 to 2024-12)
  -> Zamfara: Train = 86 mos (2016-01 to 2023-02), Test = 22 mos (2023-03 to 2024-12)
[Risk Classification] Seasonal baselines saved to C:\Users\PC\OneDrive\Desktop\malariapredict\models\seasonal_baselines.json

[Training] 3. Fitting and evaluating candidate ML models:
  -> Linear Regression: MAE = 9,730.60 | RMSE = 19,709.93 | R² = 0.7497
  -> Random Forest Regressor: MAE = 10,513.63 | RMSE = 22,488.05 | R² = 0.6741
  -> Decision Tree Regressor: MAE = 11,647.28 | RMSE = 21,699.39 | R² = 0.6966

[Training] 4. Evaluating Naive Baselines on identical test holdout:
  -> Baseline: Last Month: MAE = 9,623.08 | RMSE = 20,647.69 | R² = 0.7253
  -> Baseline: Same Month Last Year: MAE = 20,337.23 | RMSE = 37,721.62 | R² = 0.0830

[Training] 5. Best ML Model: Linear Regression (Lowest MAE: 9,730.60)

[Training] 6. Calculating Per-State Metrics (Selected Model vs Baselines):
  -> Kwara: MAE = 1,913.11 | Error Share = 6.6%
  -> Nasarawa: MAE = 5,045.07 | Error Share = 17.3%
  -> Zamfara: MAE = 22,233.62 | Error Share = 76.2%
  [Insight] State contributing highest error share: Zamfara (76.16%)

[Training] 7. Validating Statistical Risk Tiers on Test Set:
  -> Hit Rate (Sensitivity): 66.67%
  -> False Alarm Rate: 4.76%
  -> Exact 3-Tier Match Accuracy: 80.30%

[Training] 8. Running Expanding-Window Time-Series Cross-Validation:
  -> Linear Regression: Mean MAE = 8,920.36 (+/- 3,215.28) across 10 folds
  -> Random Forest Regressor: Mean MAE = 9,208.82 (+/- 5,075.70) across 10 folds
  -> Decision Tree Regressor: Mean MAE = 10,822.54 (+/- 5,506.05) across 10 folds
  -> Baseline: Last Month: Mean MAE = 8,038.98 (+/- 3,953.93) across 10 folds
  -> Baseline: Same Month Last Year: Mean MAE = 16,167.44 (+/- 7,040.51) across 10 folds

[Training] 9. Running Recursive Multi-Step Horizon Evaluation:
[Training] Saved multi-step horizon evaluation to C:\Users\PC\OneDrive\Desktop\malariapredict\models\horizon_evaluation.json
[Training] Saved best model pipeline to C:\Users\PC\OneDrive\Desktop\malariapredict\models\best_model.pkl
[Training] Saved extended metrics summary to C:\Users\PC\OneDrive\Desktop\malariapredict\models\model_metrics.json
[Training] Model metrics successfully recorded into database.
[Evaluation] Saved C:\Users\PC\OneDrive\Desktop\malariapredict\static\images\model_comparison.png
[Evaluation] Saved C:\Users\PC\OneDrive\Desktop\malariapredict\static\images\actual_vs_predicted.png
[Evaluation] Saved C:\Users\PC\OneDrive\Desktop\malariapredict\static\images\prediction_error.png

============================================================
 RAINFALL-ENHANCED MODEL TRAINING
============================================================

[Rainfall] 1. Preprocessing with climate data merge...
[Preprocessing] Report saved to C:\Users\PC\OneDrive\Desktop\malariapredict\models\preprocessing_report.json
[Rainfall] Climate data merged: True, missing rainfall values: 0
[Rainfall] 2. Generating features with rainfall lags...
[Rainfall] Total eligible observations with all features: 324
[Rainfall] Train set: 258 rows, Test set: 66 rows
  -> Kwara: Train = 86 mos (2016-01 to 2023-02), Test = 22 mos (2023-03 to 2024-12)
  -> Nasarawa: Train = 86 mos (2016-01 to 2023-02), Test = 22 mos (2023-03 to 2024-12)
  -> Zamfara: Train = 86 mos (2016-01 to 2023-02), Test = 22 mos (2023-03 to 2024-12)

[Rainfall] 3. Fitting and evaluating rainfall-enhanced ML models:
  -> Linear Regression: MAE = 9,930.16 | RMSE = 19,791.02 | R² = 0.7476
  -> Random Forest Regressor: MAE = 10,349.71 | RMSE = 23,691.73 | R² = 0.6383
  -> Decision Tree Regressor: MAE = 11,733.83 | RMSE = 21,831.51 | R² = 0.6929

[Rainfall] 4. Evaluating Naive Baselines on identical test holdout:
  -> Baseline: Last Month: MAE = 9,623.08 | RMSE = 20,647.69 | R² = 0.7253
  -> Baseline: Same Month Last Year: MAE = 20,337.23 | RMSE = 37,721.62 | R² = 0.0830

[Rainfall] 5. Best Rainfall-Enhanced ML Model: Linear Regression (MAE: 9,930.16)

[Rainfall] 6. Baseline vs Rainfall-Enhanced Comparison:
  Model                                 Baseline MAE   Rainfall MAE     Change
  ----------------------------------- -------------- -------------- ----------
  Linear Regression                         9,730.60       9,930.16    +199.56
  Random Forest Regressor                  10,513.63      10,349.71    -163.92
  Decision Tree Regressor                  11,647.28      11,733.83     +86.55
  Baseline: Last Month                      9,623.08       9,623.08      +0.00
  Baseline: Same Month Last Year           20,337.23      20,337.23      +0.00

[Rainfall] 7. Running Expanding-Window Time-Series Cross-Validation:
  -> Linear Regression: Mean MAE = 8,576.82 (+/- 3,909.54) across 10 folds
  -> Random Forest Regressor: Mean MAE = 8,780.57 (+/- 5,543.93) across 10 folds
  -> Decision Tree Regressor: Mean MAE = 10,159.05 (+/- 6,150.42) across 10 folds
  -> Baseline: Last Month: Mean MAE = 8,038.98 (+/- 3,953.93) across 10 folds
  -> Baseline: Same Month Last Year: Mean MAE = 16,167.44 (+/- 7,040.51) across 10 folds

[Rainfall] 8. Running Recursive Multi-Step Horizon Evaluation:
[Training] Saved multi-step horizon evaluation to C:\Users\PC\OneDrive\Desktop\malariapredict\models\horizon_evaluation_rainfall.json

[Rainfall] Saved rainfall-enhanced model to C:\Users\PC\OneDrive\Desktop\malariapredict\models\best_model_rainfall.pkl
[Rainfall] Saved rainfall metrics to C:\Users\PC\OneDrive\Desktop\malariapredict\models\model_metrics_rainfall.json

============================================================
 RAINFALL-ENHANCED TRAINING COMPLETE
============================================================
```

### 3.2 Outbreak Classifier Pipeline Execution (`py -m ml.outbreak`)
```
======================================================================
MALARIA OUTBREAK CLASSIFICATION PIPELINE
======================================================================
[Outbreak] Saved outbreak baselines to C:\Users\PC\OneDrive\Desktop\malariapredict\models\outbreak_baselines.json

[Data Split] Train rows: 258 (Outbreaks: 2) | Test rows: 66 (Outbreaks: 3)

[Model: Logistic Regression (Balanced)]
  TP: 1, FP: 2, FN: 2, TN: 61
  Precision: 0.3333 | Recall: 0.3333 | F1: 0.3333 | ROC-AUC: 0.8677

[Model: Random Forest Classifier (Balanced)]
  TP: 0, FP: 1, FN: 3, TN: 62
  Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | ROC-AUC: 0.8651

[Baseline: Always No Outbreak]
  TP: 0, FP: 0, FN: 3, TN: 63 | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | ROC-AUC: N/A

[Baseline: Last Month Was Outbreak]
  TP: 2, FP: 1, FN: 1, TN: 62 | Precision: 0.6667 | Recall: 0.6667 | F1: 0.6667 | ROC-AUC: N/A

[Outbreak CV] Running 5-fold expanding-window cross-validation...
  -> State Kwara: Actual Outbreaks = 0, TP = 0, FP = 0, FN = 0, TN = 22
  -> State Nasarawa: Actual Outbreaks = 0, TP = 0, FP = 0, FN = 0, TN = 22
  -> State Zamfara: Actual Outbreaks = 3, TP = 1, FP = 2, FN = 2, TN = 17

[Outbreak] Metrics saved to C:\Users\PC\OneDrive\Desktop\malariapredict\models\outbreak_metrics.json
[Outbreak] Trained best classifier saved to C:\Users\PC\OneDrive\Desktop\malariapredict\models\outbreak_classifier.pkl
[Outbreak] Saved confusion matrix visualization to C:\Users\PC\OneDrive\Desktop\malariapredict\static\images\outbreak_confusion_matrix.png
```

---

## Task 4: Output Integrity & Consistency Audit

### 4.1 Regenerated Artifact Timestamps:
All 15 generated files in `models/` and `static/images/` have timestamps showing completion at approximately 21:03–21:06 PM on 2026-11-06 (strictly post-deletion).

### 4.2 1-Step Holdout vs. 1-Month Horizon Consistency:
- Baseline Model:
  - Linear Regression: holdout MAE = 9,730.60 | horizon 1 MAE = 9,730.60 (**Exact Match: True**)
  - Random Forest Regressor: holdout MAE = 10,513.63 | horizon 1 MAE = 10,513.63 (**Exact Match: True**)
  - Decision Tree Regressor: holdout MAE = 11,647.28 | horizon 1 MAE = 11,647.28 (**Exact Match: True**)
  - Baseline: Last Month: holdout MAE = 9,623.08 | horizon 1 MAE = 9,623.08 (**Exact Match: True**)
  - Baseline: Same Month Last Year: holdout MAE = 20,337.23 | horizon 1 MAE = 20,337.23 (**Exact Match: True**)
- Rainfall-Enhanced Model:
  - Linear Regression: holdout MAE = 9,930.16 | horizon 1 MAE = 9,930.16 (**Exact Match: True**)
  - Random Forest Regressor: holdout MAE = 10,349.71 | horizon 1 MAE = 10,349.71 (**Exact Match: True**)
  - Decision Tree Regressor: holdout MAE = 11,733.83 | horizon 1 MAE = 11,733.83 (**Exact Match: True**)
  - Baseline: Last Month: holdout MAE = 9,623.08 | horizon 1 MAE = 9,623.08 (**Exact Match: True**)
  - Baseline: Same Month Last Year: holdout MAE = 20,337.23 | horizon 1 MAE = 20,337.23 (**Exact Match: True**)

### 4.3 $\text{RMSE} = \sqrt{\text{MSE}}$ Exactness Check:
All 10 models and baselines across both pipelines have $\text{RMSE} = \sqrt{\text{MSE}}$ with maximum absolute discrepancy of $0.0000$.

### 4.4 Outbreak Split Verification:
- Training rows: **258 samples** (2 outbreaks observed in training window).
- Test rows: **66 samples** (3 outbreaks observed in test window).

---

## Task 5: Database & Application Status

- `ModelMetric` table refreshed and populated with 5 baseline model records.
- Database prediction records purged after validation testing:
  - Before cleanup: 3 test prediction rows.
  - After cleanup: **0 prediction rows**.
  - Final table counts: `MalariaRecord`: 360, `Prediction`: 0, `ModelMetric`: 5.
- Application pages verified:
  - `GET /evaluation`: Status **HTTP 200 OK**, displaying updated 9,730.60 baseline MAE and newly rendered plot images.
  - `GET /predict`: Status **HTTP 200 OK**.

---

## Task 6: Final Test Suite Execution

Verbatim command output for `py -m pytest -v`:
```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.2.0, pluggy-1.7.0 -- C:\Users\PC\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\PC\OneDrive\Desktop\malariapredict
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.14.1, langsmith-0.9.2
collecting ... collected 19 items

tests/test_evaluation_baselines.py::test_exact_chronological_split_counts PASSED [  5%]
tests/test_evaluation_baselines.py::test_naive_baselines_performance PASSED [ 10%]
tests/test_evaluation_baselines.py::test_risk_tier_validation_metrics PASSED [ 15%]
tests/test_evaluation_baselines.py::test_multistep_recursive_evaluation_structure PASSED [ 21%]
tests/test_outbreak.py::test_outbreak_baselines_and_label_definition_past_only PASSED [ 26%]
tests/test_outbreak.py::test_outbreak_chronological_split PASSED         [ 31%]
tests/test_outbreak.py::test_outbreak_metrics_structure_and_bounds PASSED [ 36%]
tests/test_outbreak.py::test_outbreak_classifier_probability_between_zero_and_one PASSED [ 42%]
tests/test_prediction.py::test_compute_single_row_features PASSED        [ 47%]
tests/test_prediction.py::test_predict_next_month_valid PASSED           [ 52%]
tests/test_prediction.py::test_predict_past_month_raises_error PASSED    [ 57%]
tests/test_prediction.py::test_predict_skip_month_raises_error PASSED    [ 63%]
tests/test_prediction.py::test_predict_beyond_3_months_allowed PASSED    [ 68%]
tests/test_preprocessing.py::test_preprocessing_loads_and_cleans PASSED  [ 73%]
tests/test_risk_classification.py::test_risk_classification_thresholds PASSED [ 78%]
tests/test_risk_classification.py::test_fallback_when_baseline_missing PASSED [ 84%]
tests/test_routes.py::test_unauthenticated_redirect PASSED               [ 89%]
tests/test_routes.py::test_login_flow PASSED                             [ 94%]
tests/test_routes.py::test_authenticated_routes PASSED                   [100%]

============================== warnings summary ===============================
..\..\..\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\anyio\pytest_plugin.py:14
  C:\Users\PC\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\anyio\pytest_plugin.py:14: PytestRemovedIn10Warning: _pytest.python.CallSpec2 has been renamed to CallSpec.
  The CallSpec2 alias will be removed in pytest 10.
  Update imports to use CallSpec instead.
  See https://docs.pytest.org/en/stable/deprecations.html#callspec2-renamed
    from _pytest.python import CallSpec2

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 19 passed, 1 warning in 17.64s ========================
```

Stale archive reference scan: `0 stale references found` in active codebase.

