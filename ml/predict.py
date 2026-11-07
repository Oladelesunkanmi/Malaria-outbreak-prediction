import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from config import Config
from ml.feature_engineering import (
    compute_single_row_features,
    FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    RAINFALL_FEATURE_COLUMNS
)
from ml.risk_classification import classify_risk, load_seasonal_baselines
from database.db import get_db
from database.models import MalariaRecord, Prediction

def get_latest_recorded_period(state):
    """
    Determine the latest recorded period (historical or predicted) for a given state.
    Returns:
        tuple: (latest_year, latest_month, next_year, next_month)
    """
    with get_db() as db:
        # Check latest prediction first
        latest_pred = db.query(Prediction).filter_by(state=state)\
            .order_by(Prediction.prediction_year.desc(), Prediction.prediction_month.desc()).first()
            
        latest_hist = db.query(MalariaRecord).filter_by(state=state)\
            .order_by(MalariaRecord.year.desc(), MalariaRecord.month.desc()).first()
            
        if not latest_hist:
            return None, None, None, None

        latest_year = latest_hist.year
        latest_month = latest_hist.month
        
        if latest_pred:
            if (latest_pred.prediction_year > latest_year) or (
                latest_pred.prediction_year == latest_year and latest_pred.prediction_month > latest_month
            ):
                latest_year = latest_pred.prediction_year
                latest_month = latest_pred.prediction_month
                
        # Calculate next chronological period
        if latest_month == 12:
            next_year = latest_year + 1
            next_month = 1
        else:
            next_year = latest_year
            next_month = latest_month + 1
            
        return latest_year, latest_month, next_year, next_month

def get_state_chronological_case_series(state):
    """
    Retrieve full sequence of chronological cases (historical records + sequential predictions)
    for lag and rolling feature calculations.
    """
    with get_db() as db:
        hist_records = db.query(MalariaRecord).filter_by(state=state)\
            .order_by(MalariaRecord.year.asc(), MalariaRecord.month.asc()).all()
            
        if not hist_records:
            return [], None
            
        population = hist_records[-1].population
        
        # Build series: [(year, month, cases)]
        series = [(r.year, r.month, float(r.cases)) for r in hist_records]
        
        # Append any subsequent saved predictions in chronological order
        preds = db.query(Prediction).filter_by(state=state)\
            .order_by(Prediction.prediction_year.asc(), Prediction.prediction_month.asc()).all()
            
        latest_hist_year = hist_records[-1].year
        latest_hist_month = hist_records[-1].month
        
        for p in preds:
            if (p.prediction_year > latest_hist_year) or (
                p.prediction_year == latest_hist_year and p.prediction_month > latest_hist_month
            ):
                series.append((p.prediction_year, p.prediction_month, float(p.predicted_cases)))
                
        return series, population

def get_state_chronological_rainfall_series(state):
    """
    Retrieve full sequence of chronological rainfall observations from data/rainfall.csv.
    Returns:
        list of float: rainfall measurements in chronological order
    """
    rainfall_path = Config.RAINFALL_PATH
    if not os.path.exists(rainfall_path):
        return None
    try:
        rain_df = pd.read_csv(rainfall_path)
        rain_df.columns = rain_df.columns.str.strip()
        rain_col = 'rain_mm' if 'rain_mm' in rain_df.columns else ('rainfall_mm' if 'rainfall_mm' in rain_df.columns else None)
        if not rain_col:
            numeric_cols = [c for c in rain_df.columns if pd.api.types.is_numeric_dtype(rain_df[c])]
            rain_col = numeric_cols[-1] if numeric_cols else None
        if not rain_col:
            return None
            
        s_rain = rain_df[rain_df['State'].str.strip() == state.strip()].copy()
        s_rain['date'] = pd.to_datetime(s_rain['year'].astype(str) + '-' + s_rain['month'].astype(str).str.zfill(2) + '-01')
        s_rain = s_rain.sort_values('date').reset_index(drop=True)
        return [float(x) for x in s_rain[rain_col].values]
    except Exception:
        return None

def predict_next_month(state, target_year, target_month, save_to_db=False):
    """
    Generate next-month malaria case prediction for a specific state.
    Enforces chronological sequence and prevents historical prediction.
    Automatically uses rainfall-enhanced model pipeline if available.
    
    Returns:
        dict: Complete prediction result with risk classification
    """
    # 1. Determine Model Path
    use_rainfall_model = (
        hasattr(Config, 'BEST_MODEL_RAINFALL_PATH') and 
        os.path.exists(Config.BEST_MODEL_RAINFALL_PATH)
    )
    active_model_path = Config.BEST_MODEL_RAINFALL_PATH if use_rainfall_model else Config.BEST_MODEL_PATH

    if not os.path.exists(active_model_path):
        raise FileNotFoundError(
            "Trained model not found. Please run model training first (py -m ml.train)."
        )
        
    target_year = int(target_year)
    target_month = int(target_month)
    
    if target_month < 1 or target_month > 12:
        raise ValueError(f"Invalid month: {target_month}. Month must be between 1 and 12.")

    # 2. Check Chronological Validity
    latest_yr, latest_mo, next_yr, next_mo = get_latest_recorded_period(state)
    if latest_yr is None:
        raise ValueError(f"No historical records found for state '{state}'. Insufficient data.")

    # Prevent predicting past or current historical months
    if (target_year < latest_yr) or (target_year == latest_yr and target_month <= latest_mo):
        raise ValueError(
            f"Cannot predict for {target_year}-{target_month:02d}. "
            f"Data for this period already exists. Next eligible prediction period is {next_yr}-{next_mo:02d}."
        )

    # Validate that it immediately follows the latest period
    if target_year != next_yr or target_month != next_mo:
        raise ValueError(
            f"Sequential prediction required. The next eligible prediction period for {state} is "
            f"{next_yr}-{next_mo:02d}. You cannot skip months."
        )

    # Check chained forecast horizon relative to last historical record
    with get_db() as db:
        latest_hist = db.query(MalariaRecord).filter_by(state=state)\
            .order_by(MalariaRecord.year.desc(), MalariaRecord.month.desc()).first()
    if latest_hist:
        months_ahead = (target_year - latest_hist.year) * 12 + (target_month - latest_hist.month)
    else:
        months_ahead = 1

    # 3. Retrieve Historical & Predicted Series
    series, population = get_state_chronological_case_series(state)
    if len(series) < 12:
        raise ValueError(
            f"Insufficient historical data for state '{state}'. At least 12 months of observations required."
        )
        
    case_values = [item[2] for item in series]
    rainfall_values = get_state_chronological_rainfall_series(state)
    
    # 4. Compute Engineered Features
    feature_dict = compute_single_row_features(
        state=state,
        year=target_year,
        month=target_month,
        population=population,
        recent_history_cases=case_values,
        recent_history_rainfall=rainfall_values
    )
    
    # Select feature columns based on which model is being used
    expected_feature_cols = FEATURE_COLUMNS if use_rainfall_model else BASE_FEATURE_COLUMNS
    X_single = pd.DataFrame([feature_dict])[expected_feature_cols]
    
    # 5. Load Trained Model and Predict
    model_pipeline = joblib.load(active_model_path)
    raw_pred = model_pipeline.predict(X_single)[0]
    
    # Non-negativity constraint
    predicted_cases = float(max(0.0, raw_pred))
    
    # Identify model name from pipeline
    model_name = type(model_pipeline.named_steps['regressor']).__name__
    if model_name == 'LinearRegression':
        model_name = 'Linear Regression (Rainfall-Enhanced)' if use_rainfall_model else 'Linear Regression'
    elif model_name == 'RandomForestRegressor':
        model_name = 'Random Forest Regressor (Rainfall-Enhanced)' if use_rainfall_model else 'Random Forest Regressor'
    elif model_name == 'DecisionTreeRegressor':
        model_name = 'Decision Tree Regressor (Rainfall-Enhanced)' if use_rainfall_model else 'Decision Tree Regressor'

    # 6. Outbreak Risk Classification
    risk_info = classify_risk(predicted_cases, state, target_month)
    
    # 7. Outbreak Probability from Classifier
    outbreak_model_path = os.path.join(Config.MODELS_DIR, 'outbreak_classifier.pkl')
    outbreak_prob = None
    if os.path.exists(outbreak_model_path):
        try:
            outbreak_pipe = joblib.load(outbreak_model_path)
            # Use appropriate feature slice if outbreak classifier was trained on base features
            clf_cols = getattr(outbreak_pipe, 'feature_names_in_', None)
            if clf_cols is not None:
                X_clf = pd.DataFrame([feature_dict])[list(clf_cols)]
            else:
                X_clf = pd.DataFrame([feature_dict])[BASE_FEATURE_COLUMNS]
            probs = outbreak_pipe.predict_proba(X_clf)[0]
            outbreak_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        except Exception:
            pass

    # Extract climate context if available
    recent_rain_1 = feature_dict.get('rainfall_lag_1')
    recent_rain_cum3 = feature_dict.get('rainfall_cum_3')

    prediction_record = {
        'state': state,
        'prediction_year': target_year,
        'prediction_month': target_month,
        'prediction_period': f"{target_year}-{target_month:02d}",
        'predicted_cases': round(predicted_cases, 1),
        'risk_level': risk_info['risk_level'],
        'badge_class': risk_info['badge_class'],
        'baseline_mean': risk_info['baseline_mean'],
        'threshold_high': risk_info['threshold_high'],
        'model_name': model_name,
        'model_variant': 'Rainfall-Enhanced' if use_rainfall_model else 'Univariate Baseline',
        'rainfall_lag_1': round(recent_rain_1, 2) if (recent_rain_1 is not None and not np.isnan(recent_rain_1)) else None,
        'rainfall_cum_3': round(recent_rain_cum3, 2) if (recent_rain_cum3 is not None and not np.isnan(recent_rain_cum3)) else None,
        'horizon_months': months_ahead,
        'horizon_warning': bool(months_ahead > 1),
        'horizon_message': (
            f"Notice: This prediction is {months_ahead} months ahead of historical records. "
            f"Forecast accuracy decreases as horizon increases."
        ) if months_ahead > 1 else None,
        'outbreak_probability': round(outbreak_prob, 4) if outbreak_prob is not None else None,
        'features_used': feature_dict,
        'explanation': risk_info['explanation'],
        'methodology': risk_info['methodology'],
        'disclaimer': risk_info['disclaimer'],
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 8. Optionally Save to Database
    if save_to_db:
        with get_db() as db:
            pred_obj = Prediction(
                state=state,
                prediction_year=target_year,
                prediction_month=target_month,
                predicted_cases=predicted_cases,
                risk_level=risk_info['risk_level'],
                model_name=model_name,
                generated_at=datetime.now()
            )
            db.add(pred_obj)
            db.flush()
            prediction_record['id'] = pred_obj.id
            
    return prediction_record
