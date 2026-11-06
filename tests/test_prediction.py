import pytest
import os
from ml.predict import predict_next_month, get_latest_recorded_period
from ml.feature_engineering import compute_single_row_features
from database.db import init_db, seed_records_from_csv

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    seed_records_from_csv()

def test_compute_single_row_features():
    # 12 simulated monthly values
    history = [1000 + i * 100 for i in range(12)]
    feats = compute_single_row_features('Kwara', 2025, 1, 2396570, history)
    
    assert feats['State'] == 'Kwara'
    assert feats['year'] == 2025
    assert feats['month'] == 1
    assert feats['lag_1'] == history[-1]
    assert feats['lag_2'] == history[-2]
    assert feats['lag_3'] == history[-3]
    assert feats['lag_12'] == history[0]
    assert feats['rolling_mean_3'] == (history[-1] + history[-2] + history[-3]) / 3.0

def test_predict_next_month_valid():
    latest_yr, latest_mo, next_yr, next_mo = get_latest_recorded_period('Kwara')
    assert next_yr is not None
    assert next_mo is not None
    
    result = predict_next_month('Kwara', next_yr, next_mo, save_to_db=False)
    assert result['state'] == 'Kwara'
    assert result['prediction_year'] == next_yr
    assert result['prediction_month'] == next_mo
    assert result['predicted_cases'] >= 0.0
    assert result['risk_level'] in ['LOW RISK', 'MODERATE RISK', 'HIGH RISK']

def test_predict_past_month_raises_error():
    # Attempting to predict a past historical month (e.g. 2024-05) must fail
    with pytest.raises(ValueError, match="already exists"):
        predict_next_month('Kwara', 2024, 5, save_to_db=False)

def test_predict_skip_month_raises_error():
    # Attempting to skip ahead of next eligible period must fail with sequential required
    latest_yr, latest_mo, next_yr, next_mo = get_latest_recorded_period('Kwara')
    skip_yr = next_yr + 1
    with pytest.raises(ValueError, match="Sequential prediction required"):
        predict_next_month('Kwara', skip_yr, next_mo, save_to_db=False)

def test_predict_horizon_limit_exceeded():
    # Chaining up to 3 months is permitted, attempting a 4th month ahead of historical data must raise ValueError
    latest_yr, latest_mo, next_yr, next_mo = get_latest_recorded_period('Nasarawa')
    # Save 3 consecutive months in test database:
    p1 = predict_next_month('Nasarawa', next_yr, next_mo, save_to_db=True)
    ly, lm, ny, nm = get_latest_recorded_period('Nasarawa')
    p2 = predict_next_month('Nasarawa', ny, nm, save_to_db=True)
    ly, lm, ny, nm = get_latest_recorded_period('Nasarawa')
    p3 = predict_next_month('Nasarawa', ny, nm, save_to_db=True)
    
    # 4th month attempt must fail due to horizon > 3 limit
    ly, lm, ny, nm = get_latest_recorded_period('Nasarawa')
    with pytest.raises(ValueError, match="horizon limit exceeded"):
        predict_next_month('Nasarawa', ny, nm, save_to_db=False)

