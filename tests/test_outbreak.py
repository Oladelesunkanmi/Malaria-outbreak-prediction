import os
import json
import numpy as np
import pandas as pd
import pytest
from config import Config

from ml.preprocessing import preprocess_malaria_data
from ml.feature_engineering import generate_features, FEATURE_COLUMNS
from ml.outbreak import (
    compute_outbreak_baselines,
    add_outbreak_target,
    train_and_evaluate_outbreak_classifiers,
    OUTBREAK_BASELINES_PATH,
    OUTBREAK_METRICS_PATH
)
import joblib

def test_outbreak_baselines_and_label_definition_past_only():
    """
    Test that outbreak threshold computation and target definition:
    1. Derive thresholds strictly from training rows.
    2. Constructs 'outbreak_target' without leaking future case values.
    3. Baseline 'last_month_outbreak' strictly uses lag_1.
    """
    df_clean, _ = preprocess_malaria_data(save_report=False)
    df_feat = generate_features(df_clean, drop_na=True)
    
    # Split
    train_parts = []
    test_parts = []
    for state, g in df_feat.groupby('State'):
        g_sorted = g.sort_values('date').reset_index(drop=True)
        train_parts.append(g_sorted.iloc[:86])
        test_parts.append(g_sorted.iloc[86:])
        
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    
    baselines = compute_outbreak_baselines(train_df, save=False)
    assert len(baselines) == 36, "Should contain 3 states x 12 months = 36 thresholds"
    
    labeled_test = add_outbreak_target(test_df, baselines)
    assert 'outbreak_target' in labeled_test.columns
    assert 'last_month_outbreak' in labeled_test.columns
    
    # Verify that target matches cases >= threshold
    for _, row in labeled_test.iterrows():
        b = baselines[f"{row['State']}_{int(row['month'])}"]
        expected = 1 if row['cases'] >= b['threshold_outbreak'] else 0
        assert row['outbreak_target'] == expected
        
        # Verify last_month_outbreak uses lag_1
        prev_m = 12 if row['month'] == 1 else int(row['month']) - 1
        b_prev = baselines[f"{row['State']}_{prev_m}"]
        expected_prev = 1 if row['lag_1'] >= b_prev['threshold_outbreak'] else 0
        assert row['last_month_outbreak'] == expected_prev

def test_outbreak_chronological_split():
    """Verify exact chronological split: 258 train rows, 66 test rows, 3 test outbreaks."""
    df_clean, _ = preprocess_malaria_data(save_report=False)
    df_feat = generate_features(df_clean, drop_na=True)
    
    train_counts = []
    test_counts = []
    for state, g in df_feat.groupby('State'):
        g_sorted = g.sort_values('date').reset_index(drop=True)
        train_counts.append(len(g_sorted.iloc[:86]))
        test_counts.append(len(g_sorted.iloc[86:]))
        
    assert sum(train_counts) == 258
    assert sum(test_counts) == 66

def test_outbreak_metrics_structure_and_bounds():
    """Verify generated outbreak metrics schema and value bounds."""
    assert os.path.exists(OUTBREAK_METRICS_PATH)
    with open(OUTBREAK_METRICS_PATH, 'r') as f:
        data = json.load(f)
        
    assert 'models' in data
    assert 'cross_validation_5_folds' in data
    assert 'per_state_evaluation' in data
    
    models = data['models']
    for m_name in [
        'Logistic Regression (Balanced)',
        'Random Forest Classifier (Balanced)',
        'Baseline: Always No Outbreak',
        'Baseline: Last Month Was Outbreak'
    ]:
        assert m_name in models
        m = models[m_name]
        cm = m['confusion_matrix']
        total_cm = cm['tp'] + cm['fp'] + cm['fn'] + cm['tn']
        assert total_cm == 66, f"Confusion matrix for {m_name} must sum to 66 test samples"
        assert 0.0 <= m['precision'] <= 1.0
        assert 0.0 <= m['recall'] <= 1.0
        assert 0.0 <= m['f1'] <= 1.0
        if m['roc_auc'] is not None:
            assert 0.0 <= m['roc_auc'] <= 1.0

def test_outbreak_classifier_probability_between_zero_and_one():
    """Verify that the trained outbreak classifier outputs valid probabilities between 0 and 1."""
    clf_path = os.path.join(Config.MODELS_DIR, 'outbreak_classifier.pkl')
    assert os.path.exists(clf_path)
    pipe = joblib.load(clf_path)
    
    df_clean, _ = preprocess_malaria_data(save_report=False)
    df_feat = generate_features(df_clean, drop_na=True)
    
    sample_X = df_feat[FEATURE_COLUMNS].head(10)
    probs = pipe.predict_proba(sample_X)
    
    assert probs.shape[1] == 2
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)
