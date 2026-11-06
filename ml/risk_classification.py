import json
import os
import pandas as pd
import numpy as np
from config import Config

BASELINE_CACHE_PATH = os.path.join(Config.MODELS_DIR, 'seasonal_baselines.json')

def compute_seasonal_baselines(df=None, save=True):
    """
    Compute historical seasonal mean and standard deviation for each (State, Month) pair.
    Uses training/historical period observations only to prevent data leakage.
    
    Returns:
        dict: Baselines dictionary with keys as "{state}_{month}"
    """
    if df is None:
        from ml.preprocessing import preprocess_malaria_data
        df, _ = preprocess_malaria_data(save_report=False)
        
    baselines = {}
    grouped = df.groupby(['State', 'month'])['cases']
    
    for (state, month), cases_series in grouped:
        count = int(cases_series.count())
        mean_val = float(cases_series.mean())
        std_val = float(cases_series.std(ddof=1)) if count > 1 else 0.0
        
        # Safe fallback if std is zero or insufficient observations
        if std_val == 0.0 or np.isnan(std_val):
            # Fallback to 15% of the mean as estimated variance
            std_val = max(1.0, mean_val * 0.15)
            
        threshold_high = mean_val + (2.0 * std_val)
        
        key = f"{state}_{month}"
        baselines[key] = {
            "state": state,
            "month": int(month),
            "count": count,
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "threshold_high": round(threshold_high, 2)
        }
        
    if save:
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        with open(BASELINE_CACHE_PATH, 'w') as f:
            json.dump(baselines, f, indent=4)
        print(f"[Risk Classification] Seasonal baselines saved to {BASELINE_CACHE_PATH}")
        
    return baselines

def load_seasonal_baselines():
    """Load cached seasonal baselines or compute them if missing."""
    if os.path.exists(BASELINE_CACHE_PATH):
        try:
            with open(BASELINE_CACHE_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return compute_seasonal_baselines()

def classify_risk(predicted_cases, state, month, baselines=None):
    """
    Classify predicted malaria cases into statistical risk categories:
    - LOW RISK: predicted_cases < seasonal_mean
    - MODERATE RISK: seasonal_mean <= predicted_cases < seasonal_mean + 2 * seasonal_std
    - HIGH RISK: predicted_cases >= seasonal_mean + 2 * seasonal_std
    
    Returns:
        dict: Detailed classification result
    """
    if baselines is None:
        baselines = load_seasonal_baselines()
        
    key = f"{state}_{month}"
    baseline = baselines.get(key)
    
    if not baseline:
        # Fallback if state or month is not directly in baseline
        return {
            "risk_level": "MODERATE RISK",
            "predicted_cases": float(predicted_cases),
            "baseline_mean": None,
            "threshold_high": None,
            "std": None,
            "badge_class": "warning",
            "explanation": f"Statistical risk evaluation for {state} (month {month}) using fallback baseline parameters.",
            "methodology": "Fallback statistical baseline applied due to missing seasonal parameters.",
            "disclaimer": "This is a statistical epidemiological risk indicator, not an official WHO/government outbreak declaration or clinical diagnosis."
        }
        
    mean_val = baseline["mean"]
    threshold_high = baseline["threshold_high"]
    std_val = baseline["std"]
    
    if predicted_cases < mean_val:
        risk_level = "LOW RISK"
        badge_class = "success"
        explanation = f"Predicted cases ({predicted_cases:,.0f}) are below the historical seasonal average ({mean_val:,.0f})."
    elif predicted_cases < threshold_high:
        risk_level = "MODERATE RISK"
        badge_class = "warning"
        explanation = f"Predicted cases ({predicted_cases:,.0f}) are at or above the seasonal average ({mean_val:,.0f}) but below the high-risk alert threshold ({threshold_high:,.0f})."
    else:
        risk_level = "HIGH RISK"
        badge_class = "danger"
        explanation = f"Predicted cases ({predicted_cases:,.0f}) exceed the +2 standard deviation alert threshold ({threshold_high:,.0f}) above the historical baseline ({mean_val:,.0f})."
        
    return {
        "risk_level": risk_level,
        "predicted_cases": float(predicted_cases),
        "baseline_mean": mean_val,
        "threshold_high": threshold_high,
        "std": std_val,
        "badge_class": badge_class,
        "explanation": explanation,
        "methodology": "Based on historical seasonal mean (μ) and two standard deviations (μ + 2σ) statistical heuristic.",
        "disclaimer": "This risk classification is a predictive statistical heuristic intended for public health resource planning, not an official government or WHO outbreak declaration or clinical diagnosis."
    }

