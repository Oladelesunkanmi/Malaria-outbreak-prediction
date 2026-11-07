# Software Verification & System Audit Report (`SOFTWARE_CHECK.md`)

**Date of Execution:** 2026-11-06  
**Auditor / Engineering Lead:** Senior Machine Learning Engineer & Systems Auditor  
**Application Target:** Malaria Outbreak Prediction System (`malariapredict`)  
**Operating Environment:** Python 3.13.14 (win32), Windows, SQLite 3  

---

## Task 1: Verify the Rainfall Horizon Evaluation

### 1.1 Source Code Verification (`ml/train.py` - `run_multistep_evaluation`)
The evaluation function `run_multistep_evaluation()` in `ml/train.py` computes recursive multi-step forecasts for horizons $h \in \{1, 3, 6, 12\}$.
Both models—the baseline univariate model (`models/horizon_evaluation.json`) and the rainfall-enhanced model (`models/horizon_evaluation_rainfall.json`)—use identical test origins and sample counts.

- **Origin Count Formula:** $N_{\text{origins}} = \text{len}(s_{\text{test}}) - h + 1$ per state.
  Given $22$ test holdout months per state (March 2023 to December 2024):
  - Horizon $h=1$: $22 - 1 + 1 = 22$ origins per state $\times 3$ states = **66 samples**
  - Horizon $h=3$: $22 - 3 + 1 = 20$ origins per state $\times 3$ states = **60 samples**
  - Horizon $h=6$: $22 - 6 + 1 = 17$ origins per state $\times 3$ states = **51 samples**
  - Horizon $h=12$: $22 - 12 + 1 = 11$ origins per state $\times 3$ states = **33 samples**

### 1.2 Past-Only Rainfall Values in Recursive Step Forecasts
- **Initialization at origin month $t$ (lines 66–68):**
  ```python
  clean_start_idx = 12 + n_train + start_idx
  prior_rainfall = s_clean['rainfall_mm'].iloc[:clean_start_idx].tolist() if has_rainfall else None
  ```
  `prior_rainfall` contains only observations available strictly prior to the forecast window.
- **Handling future unobserved rainfall months ($t+1, \dots, t+h$) (lines 89–90):**
  ```python
  if sim_rainfall is not None:
      sim_rainfall.append(sim_rainfall[-1] if sim_rainfall else 0.0)
  ```
  **Exact value used:** Flat persistence of the last known historical rainfall value from the origin month ($r_{\text{sim}} = r_t$).  
  **Source:** `s_clean['rainfall_mm']` at index `clean_start_idx - 1`. Under no circumstance is observed ground-truth future rainfall ($r_{t+k}$) leaked into the recursive simulator.

### 1.3 Exact Generated Metric Values (Old vs. New Side-by-Side)

#### Baseline Model Horizons (`models/horizon_evaluation.json`):
```json
{
  "Linear Regression": {
    "1_month_mae": 9730.6,
    "3_month_mae": 21995.95,
    "6_month_mae": 25528.68,
    "12_month_mae": 28496.43
  },
  "Random Forest Regressor": {
    "1_month_mae": 10513.63,
    "3_month_mae": 15747.05,
    "6_month_mae": 16892.7,
    "12_month_mae": 21677.13
  },
  "Decision Tree Regressor": {
    "1_month_mae": 11647.28,
    "3_month_mae": 17012.49,
    "6_month_mae": 18311.98,
    "12_month_mae": 16997.39
  },
  "Baseline: Last Month": {
    "1_month_mae": 9623.08,
    "3_month_mae": 23316.95,
    "6_month_mae": 26355.51,
    "12_month_mae": 23228.85
  },
  "Baseline: Same Month Last Year": {
    "1_month_mae": 20337.23,
    "3_month_mae": 20911.85,
    "6_month_mae": 21309.69,
    "12_month_mae": 23228.85
  }
}
```

#### Rainfall-Enhanced Model Horizons (`models/horizon_evaluation_rainfall.json`):
```json
{
  "Linear Regression": {
    "1_month_mae": 9930.16,
    "3_month_mae": 20626.35,
    "6_month_mae": 22855.98,
    "12_month_mae": 26651.2
  },
  "Random Forest Regressor": {
    "1_month_mae": 10349.71,
    "3_month_mae": 15193.72,
    "6_month_mae": 17347.52,
    "12_month_mae": 21925.6
  },
  "Decision Tree Regressor": {
    "1_month_mae": 11733.83,
    "3_month_mae": 16413.09,
    "6_month_mae": 17569.04,
    "12_month_mae": 20805.17
  },
  "Baseline: Last Month": {
    "1_month_mae": 9623.08,
    "3_month_mae": 23316.95,
    "6_month_mae": 26355.51,
    "12_month_mae": 23228.85
  },
  "Baseline: Same Month Last Year": {
    "1_month_mae": 20337.23,
    "3_month_mae": 20911.85,
    "6_month_mae": 21309.69,
    "12_month_mae": 23228.85
  }
}
```

---

## Task 2: Confirm Rainfall Features Are Past-Only

### 2.1 Feature Definitions and Code Lines (`ml/feature_engineering.py`)
In `ml/feature_engineering.py`, every rainfall feature is computed within a per-state grouping (`for state_name, group in df.groupby('State'):`) using strictly past observations:

| Feature Identifier | Line in Code | Exact Source Code | Shift Parameter ($k$) |
| :--- | :---: | :--- | :---: |
| `rainfall_lag_1` | 66 | `g['rainfall_lag_1'] = g['rainfall_mm'].shift(1)` | $k = 1$ |
| `rainfall_lag_2` | 67 | `g['rainfall_lag_2'] = g['rainfall_mm'].shift(2)` | $k = 2$ |
| `rainfall_lag_3` | 68 | `g['rainfall_lag_3'] = g['rainfall_mm'].shift(3)` | $k = 3$ |
| `rainfall_lag_12` | 69 | `g['rainfall_lag_12'] = g['rainfall_mm'].shift(12)` | $k = 12$ |
| `rainfall_cum_3` | 70–74 | `(g['rainfall_mm'].shift(1) + g['rainfall_mm'].shift(2) + g['rainfall_mm'].shift(3))` | $k \in \{1, 2, 3\}$ |
| `rainfall_mean_3` | 75 | `g['rainfall_mean_3'] = g['rainfall_cum_3'] / 3.0` | $k \in \{1, 2, 3\}$ |

Every single shift parameter $k \ge 1$. No concurrent ($k=0$) or forward ($k < 0$) rainfall observations enter the feature space.

### 2.2 Dropping Initial 12 Rows and Null Checks
- **Code verification (lines 84–86):**
  ```python
  if drop_na:
      result_df = result_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
  ```
- **Execution verification:**
  - Raw records: $360$ (120 per state).
  - Dropped records: Exactly $12$ records per state (Jan 2015 – Dec 2015) where `lag_12` and `rainfall_lag_12` are undefined.
  - Final eligible records: Exactly $324$ records (108 per state: Jan 2016 – Dec 2024).
  - Null count verification across all $324$ rows:
    - `Kwara`: 0 NaNs across all 18 features.
    - `Nasarawa`: 0 NaNs across all 18 features.
    - `Zamfara`: 0 NaNs across all 18 features.

---

## Task 3: Full Test Suite on Clean Database

### 3.1 Pre-Test Database Counts (`malaria_predict.db`)
- `MalariaRecord`: 360
- `Prediction`: 0
- `ModelMetric`: 5

### 3.2 Complete Pytest Command Output
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
tests/test_prediction.py::test_predict_horizon_limit_exceeded PASSED     [ 68%]
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
======================= 19 passed, 1 warning in 15.94s ========================
```

### 3.3 Post-Test Database Row Counts (`malaria_predict.db`)
- `MalariaRecord`: 360
- `Prediction`: 0
- `ModelMetric`: 5
**Status:** Pristine state preserved. Isolated session-scoped test database in `tests/conftest.py` prevented any reads from or writes to the live database file.

---

## Task 4: End-to-End Application Audit

### 4.1 Route Availability (Authenticated HTTP 200 Checks)
- `GET /dashboard` -> **HTTP 200 OK**
- `GET /dataset` -> **HTTP 200 OK**
- `GET /analytics` -> **HTTP 200 OK**
- `GET /predict` -> **HTTP 200 OK**
- `GET /evaluation` -> **HTTP 200 OK**
- `GET /history` -> **HTTP 200 OK**

### 4.2 State Predictions (1-Month Ahead: 2025-01)
- **Kwara State:**
  - Case Forecast: **14,886** expected cases
  - Risk Level: **LOW RISK**
  - Outbreak Probability: **0.1%**
  - Displayed Rainfall Features: `rainfall_lag_1` = **0.00 mm**, `rainfall_cum_3` = **285.95 mm**
  - Model: Linear Regression (Rainfall-Enhanced)
- **Nasarawa State:**
  - Case Forecast: **33,817** expected cases
  - Risk Level: **LOW RISK**
  - Outbreak Probability: **0.1%**
  - Displayed Rainfall Features: `rainfall_lag_1` = **0.00 mm**, `rainfall_cum_3` = **73.78 mm**
  - Model: Linear Regression (Rainfall-Enhanced)
- **Zamfara State:**
  - Case Forecast: **49,049** expected cases
  - Risk Level: **LOW RISK**
  - Outbreak Probability: **1.5%**
  - Displayed Rainfall Features: `rainfall_lag_1` = **0.00 mm**, `rainfall_cum_3` = **18.91 mm**
  - Model: Linear Regression (Rainfall-Enhanced)

### 4.3 Multi-Step Horizon Warning Verification
- **2-Month Horizon Warning (Kwara, 2025-02):**
  - **Warning Displayed:** True
  - **Rendered Message:** *"Chained Multi-Step Notice: This prediction is 2 months ahead of recorded historical surveillance data. Forecast accuracy decreases as horizon increases. Review recursive benchmarks"*
- **3-Month Horizon Warning (Kwara, 2025-03):**
  - **Warning Displayed:** True
  - **Rendered Message:** *"Chained Multi-Step Notice: This prediction is 3 months ahead of recorded historical surveillance data. Forecast accuracy decreases as horizon increases. Review recursive benchmarks"*

### 4.4 Live Database State Restored
- Test predictions generated during audit: 5 records total.
- Records deleted following verification: 5 records.
- **Row counts before vs. after:**
  - `MalariaRecord`: 360 -> **360**
  - `Prediction`: 0 -> **0**
  - `ModelMetric`: 5 -> **5**

---

## Task 5: Leftover Risk Audit

### 5.1 Redundant Directory Check
- Command: `Test-Path malaria_prediction_system`
- Output: `False` (Directory does not exist).

### 5.2 Repository Secret Scan (`Admin123`)
- Command: `Get-ChildItem -Recurse -File -Exclude *.git*,*.pkl,*.db | Select-String -Pattern 'Admin123'`
- Output:
  - Exact 1 reference found in `CHANGES.md` on line 131: documenting the historical remediation entry (*"- Removed hardcoded 'Admin123!' fallback default"*).
  - No active passwords, code references, or default configurations exist in the repository.

### 5.3 Missing `ADMIN_PASSWORD` Environment Variable Error Check
- Verified via isolated invocation without `ADMIN_PASSWORD`:
  ```
  CRITICAL SECURITY CONFIGURATION ERROR: The 'ADMIN_PASSWORD' environment variable is not set. 
  Please configure ADMIN_PASSWORD in your .env file or environment variables before launching the application.
  ```
  `config.py` immediately fails fast with a descriptive `ValueError`.

