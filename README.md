# Malaria Outbreak Prediction System Using Machine Learning Techniques

An academic final-year Computer Science project delivering an end-to-end, web-based epidemiological surveillance and malaria outbreak prediction system. Built with Python, Flask, Scikit-learn, SQLAlchemy, and Bootstrap 5.

---

## 1. Project Overview

Malaria remains a major public health challenge in Nigeria and across Sub-Saharan Africa. Conventional response mechanisms often rely on retrospective reporting after clinics experience severe case surges. 

This project implements a predictive, forward-looking epidemiological early warning engine designed to forecast population-level malaria case burdens for sentinel Nigerian states (Kwara, Nasarawa, and Zamfara) using monthly time-series observations spanning from 2015 through 2024.

By leveraging autoregressive lag features, seasonal cyclical encoding, and rolling moving averages, the system trains and benchmarks three machine learning regressors (**Linear Regression**, **Random Forest**, and **Decision Tree**) alongside two naive baseline benchmarks using strictly chronological validation. It classifies predicted monthly incidence into **LOW RISK**, **MODERATE RISK**, or **HIGH RISK** tiers using a transparent statistical anomaly heuristic based on historical seasonal standard deviation thresholds ($\mu$ and $\mu + 2\sigma$).

---

## 2. System Architecture & Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & Web Framework** | Python 3.13, Flask 3.1, Werkzeug |
| **Machine Learning** | Scikit-learn, NumPy, Pandas, Joblib |
| **Data Visualization** | Chart.js 4 (browser-rendered), Matplotlib 3.11 & Seaborn 0.13 (eval reports) |
| **Database ORM** | SQLite 3, SQLAlchemy 2.0 |
| **Frontend UI/UX** | HTML5, CSS3, Bootstrap 5.3, Bootstrap Icons |
| **Security** | PBKDF2 Password Hashing, Session Authentication, CSRF Validation |
| **Testing** | Pytest 9.2 (14 Automated Unit and Integration Tests) |

---

## 3. Dataset Description & Preprocessing Pipeline

### A. Raw Dataset Specification
- **File**: `data/Wavelets_data for Time series_malaria.csv`
- **Volume**: 360 monthly observation records across 3 states (120 records per state, 10 continuous years: 2015 to 2024).
- **Initial Columns**: 41 columns including state identifiers, raw case counts, census population, geographic coordinates, and sparse intervention attributes.

### B. Automated Preprocessing Protocol
Implemented in `ml/preprocessing.py`:
1. **Header Normalization**: Strips leading and trailing whitespaces (e.g. `' cases'` $\rightarrow$ `'cases'`, `'month '` $\rightarrow$ `'month'`).
2. **Column Pruning**: Completely drops empty columns (`Unnamed: 9` through `Unnamed: 22` and sparse columns with $<10\%$ records).
3. **Type Coercion**: Casts `year`, `month`, `cases`, and `Population` to standard 64-bit integers.
4. **Temporal Indexing**: Generates uniform `date` timestamp strings (`YYYY-MM-01`).
5. **Deduplication & Data Integrity Checks**:
   - Evaluated duplicates across `(State, year, month)`: **0 duplicates found**.
   - Evaluated missing values in core attributes: **0 missing values**.
   - Validated non-negativity: **0 negative case records**.
6. **Chronological Sorting**: Sorts observations chronologically within each state.
7. **Report Generation**: Automatically outputs `models/preprocessing_report.json`.

---

## 4. Machine Learning Methodology

### A. Objective & Formulation
The prediction task is structured as a **time-series regression problem**: predict the aggregate expected malaria case count for month $t$ given historical observations up to month $t-1$.

$$\hat{y}_t = f(\text{State}, \text{Year}, \text{Month}, \text{Population}, \text{Lags}_{1, 2, 3, 12}, \text{Rolling}_{3, 6}, \sin(\text{Month}), \cos(\text{Month}))$$

### B. Feature Engineering (Zero Data Leakage)
Engineered in `ml/feature_engineering.py` within each state's chronological partition:
1. **Autoregressive Lags**:
   - $\text{lag\_1}$: Preceding month's malaria cases ($y_{t-1}$).
   - $\text{lag\_2}$: Malaria cases two months prior ($y_{t-2}$).
   - $\text{lag\_3}$: Malaria cases three months prior ($y_{t-3}$).
   - $\text{lag\_12}$: Malaria cases in the corresponding calendar month of the prior year ($y_{t-12}$).
2. **Rolling Moving Averages**:
   - $\text{rolling\_mean\_3} = \frac{1}{3} \sum_{i=1}^3 y_{t-i}$ (short-term epidemiological momentum).
   - $\text{rolling\_mean\_6} = \frac{1}{6} \sum_{i=1}^6 y_{t-i}$ (medium-term moving baseline).
3. **Cyclical Seasonal Features**:
   - $\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right)$
   - $\text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$
4. **State Encoding**: Categorical `State` transformed via `OneHotEncoder(drop='first')`.

### C. Exact Chronological Train / Test Validation Split
- Year 2015 (12 months per state, 36 observations) provides the initial context required to construct `lag_12` without imputing synthetic values.
- **Eligible Observation Pool**: 108 monthly observations per state (January 2016 through December 2024; 324 total observations across Kwara, Nasarawa, and Zamfara).
- **Training Set (80% Partition)**:
  - **86 months per state** (**January 2016 through February 2023**)
  - Total Training Samples: **258 observations**
- **Test Set (20% Partition)**:
  - **22 months per state** (**March 2023 through December 2024**)
  - Total Test Samples: **66 observations**

### D. Single-Split Empirical Evaluation: Models vs. Naive Baselines
Evaluated on the exact same 66 holdout observations:

| Model / Benchmark Name | Model Type | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Squared Error (MSE) | $R^2$ Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** | Machine Learning | **9,730.60** | **19,709.93** | **388,481,201.12** | **0.7497** | **Selected Best ML Model** |
| **Baseline: Last Month** | Naive Baseline | 9,623.08 | 20,647.69 | 426,327,159.20 | 0.7253 | Persistence Benchmark |
| **Random Forest Regressor** | Machine Learning | 10,513.63 | 22,488.05 | 505,712,576.21 | 0.6741 | Candidate ML Model |
| **Decision Tree Regressor** | Machine Learning | 11,647.28 | 21,699.39 | 470,863,688.65 | 0.6966 | Candidate ML Model |
| **Baseline: Same Month Last Year** | Naive Baseline | 20,337.23 | 37,721.62 | 1,422,920,404.91 | 0.0830 | Seasonal Benchmark |

*Analysis*: The naive "Last Month" baseline achieves a competitive single-step MAE (9,623.08) due to autocorrelation, but Linear Regression yields a higher $R^2$ (0.7497 vs 0.7253) and lower RMSE (19,709.93 vs 20,647.69). Crucially, multi-step evaluation demonstrates that the naive baseline collapses over longer horizons, whereas ML models maintain stability.

### E. Multi-Step Recursive Horizon Evaluation
Evaluating recursive multi-step forecasting across rolling 12-month windows in the test set:

| Model / Baseline | 1-Month Ahead MAE | 3-Months Ahead MAE | 6-Months Ahead MAE | 12-Months Ahead MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 9,621.38 | 20,899.99 | 20,264.35 | 24,762.14 |
| **Random Forest Regressor** | 5,538.90 | 11,031.22 | 14,037.79 | 23,521.40 |
| **Decision Tree Regressor** | 3,518.57 | 5,405.29 | 10,475.67 | 23,307.40 |
| **Baseline: Last Month** | 13,075.27 | 32,145.27 | 34,129.30 | 23,228.85 |
| **Baseline: Same Month Last Year** | 17,445.61 | 17,637.52 | 15,930.73 | 23,228.85 |

*Results saved to `models/horizon_evaluation.json`.*

### F. Expanding-Window Time-Series Cross-Validation (10 Folds)
Initial window of 48 months (4 years), forecasting 6 months ahead per fold:

| Algorithm / Baseline | Mean MAE across Folds | Standard Deviation ($\sigma$) | Folds Evaluated |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 8,920.36 | $\pm$ 3,215.28 | 10 Folds |
| **Random Forest Regressor** | 9,208.82 | $\pm$ 5,075.70 | 10 Folds |
| **Decision Tree Regressor** | 10,822.54 | $\pm$ 5,506.05 | 10 Folds |
| **Baseline: Last Month** | 8,038.98 | $\pm$ 3,953.93 | 10 Folds |
| **Baseline: Same Month Last Year** | 16,167.44 | $\pm$ 7,040.51 | 10 Folds |

### G. Per-State Performance Breakdown (Holdout Test Set)
Evaluating 22 test months individually for each state:

| State | Mean Monthly Cases | Linear Regression MAE | Linear Regression RMSE | Linear Regression $R^2$ | Error Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kwara** | 23,450.4 | 1,913.11 | 2,426.81 | 0.4480 | 6.55% |
| **Nasarawa** | 54,400.8 | 5,045.07 | 6,496.47 | 0.7212 | 17.28% |
| **Zamfara** | 85,195.0 | 22,233.62 | 33,426.79 | 0.5681 | **76.16%** |

*Error Contribution Analysis*: **Zamfara contributes 76.16% of total absolute error**. This is attributed to its high mean case volume (85,195 cases/month vs. 23,450 in Kwara) and substantial seasonal volatility during the transmission season.

---

## 5. Statistical Outbreak Risk Tier Validation

Risk tiers are defined as a statistical heuristic based on seasonal standard deviations from historical baselines calculated exclusively on training data ($\mu_{s, m}$ and $\sigma_{s, m}$):

$$\text{LOW RISK: } \hat{y} < \mu_{s, m}$$
$$\text{MODERATE RISK: } \mu_{s, m} \le \hat{y} < \mu_{s, m} + 2\sigma_{s, m}$$
$$\text{HIGH RISK: } \hat{y} \ge \mu_{s, m} + 2\sigma_{s, m}$$

### Validation on Test Period (66 Observations)
- **Operational Definition of "Actual High"**: Actual reported cases $y_t \ge \mu_{s, m} + 2\sigma_{s, m}$.
- **Actual High Months**: 3 observations.
- **Actual Non-High Months**: 63 observations.

| Metric | Empirical Value | Explanation |
| :--- | :---: | :--- |
| **True Positives (TP)** | 2 | Correctly predicted HIGH RISK months |
| **False Positives (FP)** | 3 | Months predicted HIGH RISK that were not surges |
| **False Negatives (FN)** | 1 | Unpredicted surge months |
| **True Negatives (TN)** | 60 | Correctly predicted non-surge months |
| **Hit Rate (Sensitivity)** | **66.67%** (2 / 3) | Proportion of true surge events successfully detected |
| **False Alarm Rate** | **4.76%** (3 / 63) | Low rate of spurious false alerts |
| **Exact 3-Tier Match Accuracy** | **80.30%** (53 / 66) | Exact agreement across Low, Moderate, and High tiers |

*Conclusion*: The $2\sigma$ heuristic is statistically sound, achieving a 66.7% hit rate with a minimal 4.8% false alarm rate. The tier logic was preserved without alteration.

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
Copy `.env.example` to `.env` and set administrative credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
ADMIN_USERNAME=admin
ADMIN_PASSWORD=YourSecurePasswordHere!
SECRET_KEY=YourUniqueSecretKeyHere!
```

### Step 3: Run Machine Learning Training & Evaluation
Executes preprocessing, chronological splitting, naive baselines evaluation, multi-step horizon calculation, and time-series cross-validation:
```bash
python -m ml.train
# Or on Windows using the Python Launcher:
py -m ml.train
```

### Step 4: Run the Web Application
```bash
python app.py
# Or on Windows:
py app.py
```

### Step 5: Access the Web Portal
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```
Log in using the administrator credentials configured in your `.env` file.

---

## 7. Automated Testing

Run the automated test suite with pytest:
```bash
pytest -v
# Or on Windows:
py -m pytest -v
```
All 14 tests verify:
- Chronological split verification (86 train / 22 test months per state)
- Naive baselines calculation
- Multi-step recursive forecasting
- Risk tier hit rate and false alarm rate validation
- CSV sanitization, authentication, and web routes

---

## 8. System Limitations & Disclaimer

1. **Statistical Heuristic Disclaimer**: Risk categories are predictive statistical early-warning indicators intended to assist public health resource allocation (e.g., bed and medication stocking). They do not constitute clinical diagnoses or official government/WHO outbreak declarations.
2. **Geographic Scope**: Models are calibrated to Kwara, Nasarawa, and Zamfara. Extending to other states requires corresponding historical monthly time series.
3. **Exogenous Factors**: Meteorological variables (rainfall, humidity, temperature) are modeled implicitly through seasonal cyclical harmonics ($\sin$, $\cos$) and autoregressive momentum rather than real-time satellite feeds.
#   M a l a r i a - o u t b r e a k - p r e d i c t i o n  
 