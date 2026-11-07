import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import Config
from ml.preprocessing import preprocess_malaria_data
from ml.feature_engineering import (
    generate_features,
    compute_single_row_features,
    FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    RAINFALL_FEATURE_COLUMNS,
    TARGET_COLUMN
)
from ml.risk_classification import compute_seasonal_baselines, classify_risk
from database.db import get_db, init_db
from database.models import ModelMetric

HORIZON_EVALUATION_PATH = os.path.join(Config.MODELS_DIR, 'horizon_evaluation.json')

def run_multistep_evaluation(pipelines, train_df, test_df, clean_df=None, horizons=(1, 3, 6, 12), save=True,
                              feature_cols=None, save_path=None):
    """
    Evaluate recursive multi-step forecasts for horizons 1, 3, 6, 12 months.
    Mimics ml.predict.predict_next_month sequential feature feeding.
    Evaluates each horizon h across all valid test origins (len(s_test) - h + 1).
    For h=1, evaluates all 22 test months for each state (66 total samples),
    ensuring 1-month-ahead MAE is strictly consistent with the 1-step holdout table.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    if save_path is None:
        save_path = HORIZON_EVALUATION_PATH
    models_to_eval = dict(pipelines)
    horizon_results = {
        m: {h: [] for h in horizons}
        for m in list(models_to_eval.keys()) + ['Baseline: Last Month', 'Baseline: Same Month Last Year']
    }

    if clean_df is None:
        from ml.preprocessing import preprocess_malaria_data
        clean_df, _ = preprocess_malaria_data(save_report=False)

    has_rainfall = 'rainfall_mm' in clean_df.columns and clean_df['rainfall_mm'].notna().any()

    for state, s_test in test_df.groupby('State'):
        s_clean = clean_df[clean_df['State'] == state].sort_values('date').reset_index(drop=True)
        n_train = 86
        
        for h in horizons:
            n_origins = len(s_test) - h + 1
            for start_idx in range(n_origins):
                actuals_window = s_test.iloc[start_idx : start_idx + h]
                clean_start_idx = 12 + n_train + start_idx
                prior_cases = s_clean['cases'].iloc[:clean_start_idx].tolist()
                prior_rainfall = s_clean['rainfall_mm'].iloc[:clean_start_idx].tolist() if has_rainfall else None
                
                # 1. ML Pipelines recursive rollout
                for m_name, pipe in models_to_eval.items():
                    sim_cases = list(prior_cases)
                    sim_rainfall = list(prior_rainfall) if prior_rainfall else None
                    for step in range(h):
                        row = actuals_window.iloc[step]
                        pop = int(row['Population'])
                        feat = compute_single_row_features(
                            state=state,
                            year=int(row['year']),
                            month=int(row['month']),
                            population=pop,
                            recent_history_cases=sim_cases,
                            recent_history_rainfall=sim_rainfall
                        )
                        feat_df = pd.DataFrame([feat])[feature_cols]
                        pred = max(0.0, float(pipe.predict(feat_df)[0]))
                        sim_cases.append(pred)  # Recursive autoregressive feedback
                        # For rainfall: we don't have future rainfall, so repeat last known value
                        if sim_rainfall is not None:
                            sim_rainfall.append(sim_rainfall[-1] if sim_rainfall else 0.0)
                        
                        if step == h - 1:
                            act = float(row['cases'])
                            horizon_results[m_name][h].append(float(abs(act - pred)))
                            
                # 2. Baseline: Last Month (repeats prior_cases[-1] flatly across horizon)
                last_val = float(prior_cases[-1])
                act_h = float(actuals_window['cases'].iloc[h - 1])
                horizon_results['Baseline: Last Month'][h].append(float(abs(act_h - last_val)))
                
                # 3. Baseline: Same Month Last Year (uses observation from 12 months prior)
                pred_smy = float(prior_cases[-12 + (h - 1)])
                horizon_results['Baseline: Same Month Last Year'][h].append(float(abs(act_h - pred_smy)))

    summary = {}
    for m, h_dict in horizon_results.items():
        summary[m] = {
            f"{h}_month_mae": round(float(np.mean(h_dict[h])), 2)
            for h in horizons
        }

    if save:
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"[Training] Saved multi-step horizon evaluation to {save_path}")

    return summary

def run_timeseries_cv(df_feat, preprocessor, candidate_algorithms, min_train=48, step=6, feature_cols=None):
    """
    Expanding-window time-series cross-validation.
    Trains on initial window of min_train months, forecasts next 'step' months, and expands.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    n_months = 108  # Eligible months per state
    models_to_eval = dict(candidate_algorithms)
    fold_maes = {
        m: [] for m in list(models_to_eval.keys()) + ['Baseline: Last Month', 'Baseline: Same Month Last Year']
    }

    fold_count = 0
    for train_end in range(min_train, n_months, step):
        test_end = min(train_end + step, n_months)
        if test_end <= train_end:
            break
        fold_count += 1
        
        train_parts, test_parts = [], []
        for state, g in df_feat.groupby('State'):
            g = g.sort_values(['year', 'month']).reset_index(drop=True)
            train_parts.append(g.iloc[:train_end])
            test_parts.append(g.iloc[train_end:test_end])
            
        f_train = pd.concat(train_parts, ignore_index=True)
        f_test = pd.concat(test_parts, ignore_index=True)
        y_test_f = f_test[TARGET_COLUMN]
        
        for m_name, reg in models_to_eval.items():
            pipe = Pipeline([('preprocessor', clone(preprocessor)), ('regressor', clone(reg))])
            pipe.fit(f_train[feature_cols], f_train[TARGET_COLUMN])
            preds = np.maximum(0, pipe.predict(f_test[feature_cols]))
            fold_maes[m_name].append(float(mean_absolute_error(y_test_f, preds)))
            
        fold_maes['Baseline: Last Month'].append(float(mean_absolute_error(y_test_f, f_test['lag_1'])))
        fold_maes['Baseline: Same Month Last Year'].append(float(mean_absolute_error(y_test_f, f_test['lag_12'])))

    cv_summary = {}
    for m, maes in fold_maes.items():
        cv_summary[m] = {
            'mean_mae': round(float(np.mean(maes)), 2),
            'std_mae': round(float(np.std(maes)), 2),
            'folds_evaluated': fold_count,
            'fold_maes': [round(x, 2) for x in maes]
        }
    return cv_summary

def validate_risk_tiers_on_test(test_df, y_pred, baselines):
    """
    Evaluate HIGH RISK classifications on the test holdout.
    'Actual High' is defined as actual cases >= seasonal mean + 2 std (mu + 2*sigma).
    Computes Hit Rate (Sensitivity) and False Alarm Rate.
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    tier_exact_matches = 0

    test_records = test_df.reset_index(drop=True)
    for idx, row in test_records.iterrows():
        state = row['State']
        month = int(row['month'])
        actual = float(row['cases'])
        pred = float(y_pred[idx])
        
        b = baselines.get(f"{state}_{month}")
        thresh = b['threshold_high']
        mean_val = b['mean']
        
        actual_is_high = actual >= thresh
        pred_is_high = pred >= thresh
        
        # Determine actual risk tier
        if actual < mean_val:
            act_tier = 'LOW RISK'
        elif actual < thresh:
            act_tier = 'MODERATE RISK'
        else:
            act_tier = 'HIGH RISK'
            
        pred_tier = classify_risk(pred, state, month, baselines=baselines)['risk_level']
        if act_tier == pred_tier:
            tier_exact_matches += 1
            
        if pred_is_high and actual_is_high:
            tp += 1
        elif pred_is_high and not actual_is_high:
            fp += 1
        elif not pred_is_high and actual_is_high:
            fn += 1
        else:
            tn += 1

    total_actual_high = tp + fn
    total_actual_not_high = fp + tn

    hit_rate = tp / total_actual_high if total_actual_high > 0 else 0.0
    false_alarm_rate = fp / total_actual_not_high if total_actual_not_high > 0 else 0.0
    tier_accuracy = tier_exact_matches / len(test_records) if len(test_records) > 0 else 0.0

    return {
        'actual_high_definition': 'Actual cases >= historical seasonal mean + 2 standard deviations (training baseline)',
        'total_test_observations': len(test_records),
        'actual_high_count': total_actual_high,
        'actual_not_high_count': total_actual_not_high,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'hit_rate': round(hit_rate, 4),
        'false_alarm_rate': round(false_alarm_rate, 4),
        'tier_exact_match_accuracy': round(tier_accuracy, 4)
    }

def train_and_evaluate_models(save_artifacts=True):
    """
    Train ML models and evaluate alongside naive baselines.
    Executes chronological split, time-series CV, multi-step horizon analysis,
    per-state breakdowns, and risk tier validation.
    """
    print("[Training] 1. Preprocessing raw dataset...")
    df, report = preprocess_malaria_data(save_report=save_artifacts)
    
    print("[Training] 2. Generating lag and rolling features per state...")
    df_feat = generate_features(df, drop_na=True)
    print(f"[Training] Total eligible observations with lags: {len(df_feat)}")
    
    # 3. Chronological train/test split (latest 20% from each state)
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
            'train_range': f"{g_train.iloc[0]['year']}-{int(g_train.iloc[0]['month']):02d} to {g_train.iloc[-1]['year']}-{int(g_train.iloc[-1]['month']):02d}",
            'test_months': len(g_test),
            'test_range': f"{g_test.iloc[0]['year']}-{int(g_test.iloc[0]['month']):02d} to {g_test.iloc[-1]['year']}-{int(g_test.iloc[-1]['month']):02d}"
        }
        
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    
    print(f"[Training] Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    for s, info in split_info.items():
        print(f"  -> {s}: Train = {info['train_months']} mos ({info['train_range']}), Test = {info['test_months']} mos ({info['test_range']})")
    
    # Compute historical seasonal baselines using training data only (no data leakage)
    baselines = compute_seasonal_baselines(train_df, save=save_artifacts)
    
    categorical_features = ['State']
    numerical_features = [col for col in BASE_FEATURE_COLUMNS if col not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )
    
    candidate_algorithms = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42, max_depth=5)
    }
    
    X_train, y_train = train_df[BASE_FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[BASE_FEATURE_COLUMNS], test_df[TARGET_COLUMN]
    
    metrics_summary = {}
    fitted_pipelines = {}
    test_predictions = {}
    
    print("\n[Training] 3. Fitting and evaluating candidate ML models:")
    for name, regressor in candidate_algorithms.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred = np.maximum(0, y_pred)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        
        metrics_summary[name] = {
            'mae': round(mae, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'model_type': 'Machine Learning'
        }
        fitted_pipelines[name] = pipeline
        test_predictions[name] = y_pred.tolist()
        
        print(f"  -> {name}: MAE = {mae:,.2f} | RMSE = {rmse:,.2f} | R² = {r2:.4f}")

    # 4. Naive Baselines Evaluation on the exact same test split
    print("\n[Training] 4. Evaluating Naive Baselines on identical test holdout:")
    baselines_spec = {
        'Baseline: Last Month': test_df['lag_1'].values,
        'Baseline: Same Month Last Year': test_df['lag_12'].values
    }
    for base_name, preds_base in baselines_spec.items():
        mae_b = float(mean_absolute_error(y_test, preds_base))
        mse_b = float(mean_squared_error(y_test, preds_base))
        rmse_b = float(np.sqrt(mse_b))
        r2_b = float(r2_score(y_test, preds_base))
        
        metrics_summary[base_name] = {
            'mae': round(mae_b, 2),
            'mse': round(mse_b, 2),
            'rmse': round(rmse_b, 2),
            'r2': round(r2_b, 4),
            'model_type': 'Naive Baseline'
        }
        test_predictions[base_name] = [float(x) for x in preds_base]
        print(f"  -> {base_name}: MAE = {mae_b:,.2f} | RMSE = {rmse_b:,.2f} | R² = {r2_b:.4f}")

    # 5. Select Best Model among candidate ML models based on lowest MAE
    ml_candidates = {k: v for k, v in metrics_summary.items() if v['model_type'] == 'Machine Learning'}
    best_model_name = min(ml_candidates.keys(), key=lambda k: ml_candidates[k]['mae'])
    best_pipeline = fitted_pipelines[best_model_name]
    best_preds = np.array(test_predictions[best_model_name])
    
    print(f"\n[Training] 5. Best ML Model: {best_model_name} (Lowest MAE: {metrics_summary[best_model_name]['mae']:,.2f})")

    # 6. Per-State Performance Breakdown
    print("\n[Training] 6. Calculating Per-State Metrics (Selected Model vs Baselines):")
    state_metrics = {}
    total_abs_error_sum = 0
    test_df_copy = test_df.copy()
    test_df_copy['best_pred'] = best_preds

    for state, s_test in test_df_copy.groupby('State'):
        y_true_s = s_test['cases'].values
        y_pred_s = s_test['best_pred'].values
        mae_s = float(mean_absolute_error(y_true_s, y_pred_s))
        rmse_s = float(np.sqrt(mean_squared_error(y_true_s, y_pred_s)))
        r2_s = float(r2_score(y_true_s, y_pred_s))
        abs_err_sum = float(np.sum(np.abs(y_true_s - y_pred_s)))
        total_abs_error_sum += abs_err_sum
        
        # Baselines per state
        mae_lm = float(mean_absolute_error(y_true_s, s_test['lag_1']))
        rmse_lm = float(np.sqrt(mean_squared_error(y_true_s, s_test['lag_1'])))
        r2_lm = float(r2_score(y_true_s, s_test['lag_1']))
        
        mae_smy = float(mean_absolute_error(y_true_s, s_test['lag_12']))
        rmse_smy = float(np.sqrt(mean_squared_error(y_true_s, s_test['lag_12'])))
        r2_smy = float(r2_score(y_true_s, s_test['lag_12']))
        
        state_metrics[state] = {
            'mean_cases': round(float(y_true_s.mean()), 1),
            'linear_regression': {
                'mae': round(mae_s, 2),
                'rmse': round(rmse_s, 2),
                'r2': round(r2_s, 4),
                'abs_error_sum': round(abs_err_sum, 1)
            },
            'baseline_last_month': {
                'mae': round(mae_lm, 2),
                'rmse': round(rmse_lm, 2),
                'r2': round(r2_lm, 4)
            },
            'baseline_same_month_last_year': {
                'mae': round(mae_smy, 2),
                'rmse': round(rmse_smy, 2),
                'r2': round(r2_smy, 4)
            }
        }

    # Add error share percentage
    for state, info in state_metrics.items():
        share = (info['linear_regression']['abs_error_sum'] / total_abs_error_sum) * 100
        info['linear_regression']['error_share_pct'] = round(share, 2)
        print(f"  -> {state}: MAE = {info['linear_regression']['mae']:,.2f} | Error Share = {share:.1f}%")

    highest_err_state = max(state_metrics.keys(), key=lambda s: state_metrics[s]['linear_regression']['error_share_pct'])
    print(f"  [Insight] State contributing highest error share: {highest_err_state} ({state_metrics[highest_err_state]['linear_regression']['error_share_pct']}%)")

    # 7. Risk Tier Validation on Test Set
    print("\n[Training] 7. Validating Statistical Risk Tiers on Test Set:")
    risk_tier_val = validate_risk_tiers_on_test(test_df, best_preds, baselines)
    print(f"  -> Hit Rate (Sensitivity): {risk_tier_val['hit_rate']:.2%}")
    print(f"  -> False Alarm Rate: {risk_tier_val['false_alarm_rate']:.2%}")
    print(f"  -> Exact 3-Tier Match Accuracy: {risk_tier_val['tier_exact_match_accuracy']:.2%}")

    # 8. Expanding-Window Time-Series Cross-Validation
    print("\n[Training] 8. Running Expanding-Window Time-Series Cross-Validation:")
    tscv_results = run_timeseries_cv(df_feat, preprocessor, candidate_algorithms, min_train=48, step=6, feature_cols=BASE_FEATURE_COLUMNS)
    for m, cv_m in tscv_results.items():
        print(f"  -> {m}: Mean MAE = {cv_m['mean_mae']:,.2f} (+/- {cv_m['std_mae']:,.2f}) across {cv_m['folds_evaluated']} folds")

    # 9. Multi-Step Horizon Evaluation (1, 3, 6, 12 months)
    print("\n[Training] 9. Running Recursive Multi-Step Horizon Evaluation:")
    horizon_results = run_multistep_evaluation(fitted_pipelines, train_df, test_df, clean_df=df, horizons=(1, 3, 6, 12), save=save_artifacts, feature_cols=BASE_FEATURE_COLUMNS)

    result = {
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_model': best_model_name,
        'metrics': metrics_summary,
        'split_details': {
            'total_observations': len(df),
            'eligible_observations': len(df_feat),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'states': split_info
        },
        'state_metrics': state_metrics,
        'highest_error_state': highest_err_state,
        'timeseries_cv': tscv_results,
        'risk_tier_validation': risk_tier_val,
        'horizon_evaluation': horizon_results,
        'feature_columns': BASE_FEATURE_COLUMNS,
        'test_actuals': y_test.tolist(),
        'test_states': test_df['State'].tolist(),
        'test_dates': (test_df['year'].astype(str) + '-' + test_df['month'].astype(str).str.zfill(2)).tolist(),
        'test_predictions': test_predictions
    }
    
    if save_artifacts:
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        joblib.dump(best_pipeline, Config.BEST_MODEL_PATH)
        print(f"[Training] Saved best model pipeline to {Config.BEST_MODEL_PATH}")
        
        with open(Config.METRICS_PATH, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"[Training] Saved extended metrics summary to {Config.METRICS_PATH}")
        
        try:
            init_db()
            with get_db() as db:
                db.query(ModelMetric).delete()
                for name, met in metrics_summary.items():
                    m_rec = ModelMetric(
                        model_name=name,
                        mae=met['mae'],
                        mse=met['mse'],
                        rmse=met['rmse'],
                        r2=met['r2'],
                        trained_at=datetime.now()
                    )
                    db.add(m_rec)
            print("[Training] Model metrics successfully recorded into database.")
        except Exception as e:
            print(f"[Training] Warning: Could not sync metrics to DB: {e}")
            
        try:
            from ml.evaluate import generate_evaluation_plots
            generate_evaluation_plots(test_df, y_test, test_predictions, metrics_summary)
        except Exception as e:
            print(f"[Training] Warning: Error generating evaluation plots: {e}")
            
    return result


def train_rainfall_enhanced_model(save_artifacts=True):
    """
    Train a rainfall-enhanced ML model and evaluate it alongside the baseline.
    Preserves the existing baseline model artefacts.
    """
    print("\n" + "=" * 60)
    print(" RAINFALL-ENHANCED MODEL TRAINING")
    print("=" * 60)

    # 1. Preprocessing (now includes rainfall merge)
    print("\n[Rainfall] 1. Preprocessing with climate data merge...")
    df, report = preprocess_malaria_data(save_report=save_artifacts)

    has_rainfall = report.get('rainfall_column_present', False)
    if not has_rainfall:
        print("[Rainfall] WARNING: No rainfall data found. Cannot train rainfall-enhanced model.")
        print("[Rainfall] Place a rainfall CSV at:", Config.RAINFALL_PATH)
        return None

    print(f"[Rainfall] Climate data merged: {report['rainfall_merged']}, missing rainfall values: {report['rainfall_missing_count']}")

    # 2. Generate features (now includes rainfall lags)
    print("[Rainfall] 2. Generating features with rainfall lags...")
    df_feat = generate_features(df, drop_na=True)
    print(f"[Rainfall] Total eligible observations with all features: {len(df_feat)}")

    # 3. Chronological train/test split (same logic as baseline)
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
            'train_range': f"{g_train.iloc[0]['year']}-{int(g_train.iloc[0]['month']):02d} to {g_train.iloc[-1]['year']}-{int(g_train.iloc[-1]['month']):02d}",
            'test_months': len(g_test),
            'test_range': f"{g_test.iloc[0]['year']}-{int(g_test.iloc[0]['month']):02d} to {g_test.iloc[-1]['year']}-{int(g_test.iloc[-1]['month']):02d}"
        }

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    print(f"[Rainfall] Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    for s, info in split_info.items():
        print(f"  -> {s}: Train = {info['train_months']} mos ({info['train_range']}), Test = {info['test_months']} mos ({info['test_range']})")

    # 4. Build preprocessor with full feature set (base + rainfall)
    categorical_features = ['State']
    numerical_features = [col for col in FEATURE_COLUMNS if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )

    candidate_algorithms = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42, max_depth=5)
    }

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    metrics_summary = {}
    fitted_pipelines = {}
    test_predictions = {}

    print("\n[Rainfall] 3. Fitting and evaluating rainfall-enhanced ML models:")
    for name, regressor in candidate_algorithms.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred = np.maximum(0, y_pred)

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        metrics_summary[name] = {
            'mae': round(mae, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'model_type': 'Machine Learning (Rainfall)'
        }
        fitted_pipelines[name] = pipeline
        test_predictions[name] = y_pred.tolist()
        print(f"  -> {name}: MAE = {mae:,.2f} | RMSE = {rmse:,.2f} | R² = {r2:.4f}")

    # Naive baselines (unchanged – rainfall doesn't affect them)
    print("\n[Rainfall] 4. Evaluating Naive Baselines on identical test holdout:")
    baselines_spec = {
        'Baseline: Last Month': test_df['lag_1'].values,
        'Baseline: Same Month Last Year': test_df['lag_12'].values
    }
    for base_name, preds_base in baselines_spec.items():
        mae_b = float(mean_absolute_error(y_test, preds_base))
        mse_b = float(mean_squared_error(y_test, preds_base))
        rmse_b = float(np.sqrt(mse_b))
        r2_b = float(r2_score(y_test, preds_base))
        metrics_summary[base_name] = {
            'mae': round(mae_b, 2),
            'mse': round(mse_b, 2),
            'rmse': round(rmse_b, 2),
            'r2': round(r2_b, 4),
            'model_type': 'Naive Baseline'
        }
        test_predictions[base_name] = [float(x) for x in preds_base]
        print(f"  -> {base_name}: MAE = {mae_b:,.2f} | RMSE = {rmse_b:,.2f} | R² = {r2_b:.4f}")

    # 5. Select best rainfall-enhanced ML model
    ml_candidates = {k: v for k, v in metrics_summary.items() if 'Machine Learning' in v['model_type']}
    best_model_name = min(ml_candidates.keys(), key=lambda k: ml_candidates[k]['mae'])
    best_pipeline = fitted_pipelines[best_model_name]
    best_preds = np.array(test_predictions[best_model_name])
    print(f"\n[Rainfall] 5. Best Rainfall-Enhanced ML Model: {best_model_name} (MAE: {metrics_summary[best_model_name]['mae']:,.2f})")

    # 6. Load baseline metrics for side-by-side comparison
    baseline_metrics = {}
    if os.path.exists(Config.METRICS_PATH):
        with open(Config.METRICS_PATH, 'r') as f:
            baseline_data = json.load(f)
            baseline_metrics = baseline_data.get('metrics', {})

    comparison = {}
    for model_name in metrics_summary:
        rain_m = metrics_summary[model_name]
        base_m = baseline_metrics.get(model_name, {})
        comparison[model_name] = {
            'baseline_mae': base_m.get('mae', 'N/A'),
            'baseline_rmse': base_m.get('rmse', 'N/A'),
            'baseline_r2': base_m.get('r2', 'N/A'),
            'rainfall_mae': rain_m['mae'],
            'rainfall_rmse': rain_m['rmse'],
            'rainfall_r2': rain_m['r2'],
            'mae_change': round(rain_m['mae'] - base_m['mae'], 2) if isinstance(base_m.get('mae'), (int, float)) else 'N/A',
            'rmse_change': round(rain_m['rmse'] - base_m['rmse'], 2) if isinstance(base_m.get('rmse'), (int, float)) else 'N/A',
        }

    print("\n[Rainfall] 6. Baseline vs Rainfall-Enhanced Comparison:")
    print(f"  {'Model':<35} {'Baseline MAE':>14} {'Rainfall MAE':>14} {'Change':>10}")
    print(f"  {'-'*35} {'-'*14} {'-'*14} {'-'*10}")
    for m, comp in comparison.items():
        b_mae = f"{comp['baseline_mae']:>14,.2f}" if isinstance(comp['baseline_mae'], (int, float)) else f"{'N/A':>14}"
        r_mae = f"{comp['rainfall_mae']:>14,.2f}"
        chg = f"{comp['mae_change']:>+10,.2f}" if isinstance(comp['mae_change'], (int, float)) else f"{'N/A':>10}"
        print(f"  {m:<35} {b_mae} {r_mae} {chg}")

    # 7. Expanding-Window CV
    print("\n[Rainfall] 7. Running Expanding-Window Time-Series Cross-Validation:")
    tscv_results = run_timeseries_cv(df_feat, preprocessor, candidate_algorithms, min_train=48, step=6,
                                     feature_cols=FEATURE_COLUMNS)
    for m, cv_m in tscv_results.items():
        print(f"  -> {m}: Mean MAE = {cv_m['mean_mae']:,.2f} (+/- {cv_m['std_mae']:,.2f}) across {cv_m['folds_evaluated']} folds")

    # 8. Multi-Step Horizon Evaluation
    print("\n[Rainfall] 8. Running Recursive Multi-Step Horizon Evaluation:")
    horizon_save_path = os.path.join(Config.MODELS_DIR, 'horizon_evaluation_rainfall.json')
    horizon_results = run_multistep_evaluation(
        fitted_pipelines, train_df, test_df, clean_df=df,
        horizons=(1, 3, 6, 12), save=save_artifacts,
        feature_cols=FEATURE_COLUMNS, save_path=horizon_save_path
    )

    # Build result summary
    result = {
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_variant': 'rainfall_enhanced',
        'best_model': best_model_name,
        'metrics': metrics_summary,
        'comparison_to_baseline': comparison,
        'split_details': {
            'total_observations': len(df),
            'eligible_observations': len(df_feat),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'states': split_info
        },
        'timeseries_cv': tscv_results,
        'horizon_evaluation': horizon_results,
        'feature_columns': FEATURE_COLUMNS,
        'rainfall_features': RAINFALL_FEATURE_COLUMNS,
    }

    if save_artifacts:
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        joblib.dump(best_pipeline, Config.BEST_MODEL_RAINFALL_PATH)
        print(f"\n[Rainfall] Saved rainfall-enhanced model to {Config.BEST_MODEL_RAINFALL_PATH}")

        with open(Config.METRICS_RAINFALL_PATH, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"[Rainfall] Saved rainfall metrics to {Config.METRICS_RAINFALL_PATH}")

    print("\n" + "=" * 60)
    print(" RAINFALL-ENHANCED TRAINING COMPLETE")
    print("=" * 60)

    return result


if __name__ == '__main__':
    train_and_evaluate_models()
    train_rainfall_enhanced_model()
