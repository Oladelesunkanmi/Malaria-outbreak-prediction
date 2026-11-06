import pytest
from ml.risk_classification import classify_risk, compute_seasonal_baselines

def test_risk_classification_thresholds():
    # Mock baselines where mean=10,000 and std=2,000 (threshold_high = 14,000)
    mock_baselines = {
        "Kwara_5": {
            "state": "Kwara",
            "month": 5,
            "count": 10,
            "mean": 10000.0,
            "std": 2000.0,
            "threshold_high": 14000.0
        }
    }
    
    # 1. Low risk (< mean)
    low_res = classify_risk(8000.0, "Kwara", 5, baselines=mock_baselines)
    assert low_res["risk_level"] == "LOW RISK"
    assert low_res["badge_class"] == "success"
    
    # 2. Moderate risk (mean <= cases < threshold_high)
    mod_res1 = classify_risk(10000.0, "Kwara", 5, baselines=mock_baselines)
    assert mod_res1["risk_level"] == "MODERATE RISK"
    assert mod_res1["badge_class"] == "warning"
    
    mod_res2 = classify_risk(13999.0, "Kwara", 5, baselines=mock_baselines)
    assert mod_res2["risk_level"] == "MODERATE RISK"
    assert mod_res2["badge_class"] == "warning"
    
    # 3. High risk (>= threshold_high)
    high_res1 = classify_risk(14000.0, "Kwara", 5, baselines=mock_baselines)
    assert high_res1["risk_level"] == "HIGH RISK"
    assert high_res1["badge_class"] == "danger"
    
    high_res2 = classify_risk(25000.0, "Kwara", 5, baselines=mock_baselines)
    assert high_res2["risk_level"] == "HIGH RISK"
    assert high_res2["badge_class"] == "danger"

def test_fallback_when_baseline_missing():
    res = classify_risk(15000.0, "UnknownState", 1, baselines={})
    assert res["risk_level"] == "MODERATE RISK"
    assert "Fallback" in res["methodology"]

