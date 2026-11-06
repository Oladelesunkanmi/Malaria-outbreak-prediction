import pytest
import os
import json
import numpy as np
import pandas as pd
from config import Config
from ml.preprocessing import preprocess_malaria_data
from ml.feature_engineering import generate_features, FEATURE_COLUMNS, TARGET_COLUMN
from ml.train import (
    validate_risk_tiers_on_test,
    run_timeseries_cv,
    run_multistep_evaluation
)
from ml.risk_classification import compute_seasonal_baselines
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

@pytest.fixture(scope="module")
def dataset_and_split():
    df, _ = preprocess_malaria_data(save_report=False)
    df_feat = generate_features(df, drop_na=True)
    
    train_rows, test_rows = [], []
    split_info = {}
    for state, group in df_feat.groupby('State'):
        g = group.sort_values(['year', 'month']).reset_index(drop=True)
        n = len(g)
        n_test = int(np.round(n * 0.20))
        g_train = g.iloc[:-n_test]
        g_test = g.iloc[-n_test:]
        train_rows.append(g_train)
        test_rows.append(g_test)
        split_info[state] = {
            'eligible_months': n,
            'train_months': len(g_train),
            'train_start': (int(g_train.iloc[0]['year']), int(g_train.iloc[0]['month'])),
            'train_end': (int(g_train.iloc[-1]['year']), int(g_train.iloc[-1]['month'])),
            'test_months': len(g_test),
            'test_start': (int(g_test.iloc[0]['year']), int(g_test.iloc[0]['month'])),
            'test_end': (int(g_test.iloc[-1]['year']), int(g_test.iloc[-1]['month']))
        }
        
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    return df_feat, train_df, test_df, split_info

def test_exact_chronological_split_counts(dataset_and_split):
    _, train_df, test_df, split_info = dataset_and_split
    
    # Total counts
    assert len(train_df) == 258
    assert len(test_df) == 66
    
    for state in ['Kwara', 'Nasarawa', 'Zamfara']:
        info = split_info[state]
        assert info['eligible_months'] == 108
        assert info['train_months'] == 86
        assert info['test_months'] == 22
        # Train dates: 2016-01 to 2023-02
        assert info['train_start'] == (2016, 1)
        assert info['train_end'] == (2023, 2)
        # Test dates: 2023-03 to 2024-12
        assert info['test_start'] == (2023, 3)
        assert info['test_end'] == (2024, 12)

def test_naive_baselines_performance(dataset_and_split):
    _, _, test_df, _ = dataset_and_split
    y_test = test_df['cases'].values
    
    # 1. Baseline: Last Month
    preds_lm = test_df['lag_1'].values
    mae_lm = mean_absolute_error(y_test, preds_lm)
    r2_lm = r2_score(y_test, preds_lm)
    assert mae_lm > 0
    assert 9000 < mae_lm < 10500  # Verified ~9,623.08
    assert r2_lm > 0.65
    
    # 2. Baseline: Same Month Last Year
    preds_smy = test_df['lag_12'].values
    mae_smy = mean_absolute_error(y_test, preds_smy)
    assert mae_smy > mae_lm  # Annual lag exhibits higher single-step error than lag-1

def test_risk_tier_validation_metrics(dataset_and_split):
    _, train_df, test_df, _ = dataset_and_split
    baselines = compute_seasonal_baselines(train_df, save=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ['State']),
            ('num', 'passthrough', [c for c in FEATURE_COLUMNS if c != 'State'])
        ]
    )
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])
    pipe.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    y_pred = np.maximum(0, pipe.predict(test_df[FEATURE_COLUMNS]))
    
    tier_val = validate_risk_tiers_on_test(test_df, y_pred, baselines)
    
    assert tier_val['total_test_observations'] == 66
    assert tier_val['actual_high_count'] == 3
    assert tier_val['true_positives'] == 2
    assert tier_val['hit_rate'] == pytest.approx(0.6667, rel=1e-2)
    assert tier_val['false_alarm_rate'] == pytest.approx(0.0476, rel=1e-2)
    assert tier_val['tier_exact_match_accuracy'] == pytest.approx(0.8030, rel=1e-2)

def test_multistep_recursive_evaluation_structure(dataset_and_split):
    _, train_df, test_df, _ = dataset_and_split
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ['State']),
            ('num', 'passthrough', [c for c in FEATURE_COLUMNS if c != 'State'])
        ]
    )
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])
    pipe.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    
    horizons = (1, 3, 6, 12)
    res = run_multistep_evaluation(
        {'Linear Regression': pipe},
        train_df,
        test_df,
        horizons=horizons,
        save=False
    )
    
    assert 'Linear Regression' in res
    assert 'Baseline: Last Month' in res
    for h in horizons:
        assert f"{h}_month_mae" in res['Linear Regression']
        assert res['Linear Regression'][f"{h}_month_mae"] > 0

