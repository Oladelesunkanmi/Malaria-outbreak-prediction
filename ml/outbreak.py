import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import joblib

from config import Config
from ml.preprocessing import preprocess_malaria_data
from ml.feature_engineering import generate_features, FEATURE_COLUMNS

OUTBREAK_BASELINES_PATH = os.path.join(Config.MODELS_DIR, 'outbreak_baselines.json')
OUTBREAK_METRICS_PATH = os.path.join(Config.MODELS_DIR, 'outbreak_metrics.json')
OUTBREAK_MODEL_PATH = os.path.join(Config.MODELS_DIR, 'outbreak_classifier.pkl')
OUTBREAK_CM_PLOT_PATH = os.path.join(Config.STATIC_IMAGES_DIR, 'outbreak_confusion_matrix.png')

def compute_outbreak_baselines(train_df, save=True):
    """
    Compute training-only seasonal mean and standard deviation per (State, month).
    Outbreak threshold = mean + 2 * std.
    Applies fallback std = max(1.0, 0.15 * mean) if std == 0 or count <= 1.
    """
    baselines = {}
    grouped = train_df.groupby(['State', 'month'])['cases']
    
    for (state, month), cases_series in grouped:
        count = int(cases_series.count())
        mean_val = float(cases_series.mean())
        std_val = float(cases_series.std(ddof=1)) if count > 1 else 0.0
        
        if std_val == 0.0 or np.isnan(std_val):
            std_val = max(1.0, mean_val * 0.15)
            
        threshold_high = mean_val + (2.0 * std_val)
        key = f"{state}_{month}"
        baselines[key] = {
            "state": state,
            "month": int(month),
            "count": count,
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "threshold_outbreak": round(threshold_high, 2)
        }
        
    if save:
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        with open(OUTBREAK_BASELINES_PATH, 'w') as f:
            json.dump(baselines, f, indent=4)
        print(f"[Outbreak] Saved outbreak baselines to {OUTBREAK_BASELINES_PATH}")
        
    return baselines

def add_outbreak_target(df_feat, baselines):
    """
    Construct outbreak target for each observation.
    Target: outbreak_target = 1 if cases >= threshold_outbreak for that state and month, else 0.
    Also constructs baseline feature: 'last_month_outbreak' using only lag_1 and previous month threshold.
    """
    df = df_feat.copy()
    targets = []
    last_month_outbreak = []
    
    for _, row in df.iterrows():
        st = row['State']
        m = int(row['month'])
        actual_cases = float(row['cases'])
        
        # Outbreak status of current target month
        b = baselines.get(f"{st}_{m}")
        thresh = b['threshold_outbreak'] if b else 1e9
        is_ob = 1 if actual_cases >= thresh else 0
        targets.append(is_ob)
        
        # Prior month outbreak status (for naive baseline)
        prev_m = 12 if m == 1 else m - 1
        b_prev = baselines.get(f"{st}_{prev_m}")
        thresh_prev = b_prev['threshold_outbreak'] if b_prev else 1e9
        lag1_cases = float(row['lag_1'])
        is_prev_ob = 1 if lag1_cases >= thresh_prev else 0
        last_month_outbreak.append(is_prev_ob)
        
    df['outbreak_target'] = targets
    df['last_month_outbreak'] = last_month_outbreak
    return df

def train_and_evaluate_outbreak_classifiers(save_artifacts=True):
    """
    Train and evaluate outbreak classifiers and naive baselines on the exact
    same chronological split (258 train rows, 66 test rows).
    """
    print("=" * 70)
    print("MALARIA OUTBREAK CLASSIFICATION PIPELINE")
    print("=" * 70)
    
    # 1. Load cleaned data and engineer features
    df_clean, _ = preprocess_malaria_data(save_report=False)
    df_feat = generate_features(df_clean, drop_na=True)
    
    # 2. Chronological Train/Test Split (86 train / 22 test per state)
    train_parts, test_parts = [], []
    for state, g in df_feat.groupby('State'):
        g_sorted = g.sort_values('date').reset_index(drop=True)
        n_train = int(len(g_sorted) * 0.8)  # 86 rows
        train_parts.append(g_sorted.iloc[:n_train])
        test_parts.append(g_sorted.iloc[n_train:])
        
    train_feat = pd.concat(train_parts, ignore_index=True)
    test_feat = pd.concat(test_parts, ignore_index=True)
    
    # 3. Compute outbreak thresholds strictly from training set
    baselines = compute_outbreak_baselines(train_feat, save=save_artifacts)
    
    # 4. Construct labels
    train_df = add_outbreak_target(train_feat, baselines)
    test_df = add_outbreak_target(test_feat, baselines)
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['outbreak_target'].values
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['outbreak_target'].values
    
    total_test_positives = int(np.sum(y_test))
    total_test_negatives = int(len(y_test) - total_test_positives)
    total_train_positives = int(np.sum(y_train))
    
    print(f"\n[Data Split] Train rows: {len(X_train)} (Outbreaks: {total_train_positives}) | Test rows: {len(X_test)} (Outbreaks: {total_test_positives})")
    
    # 5. Preprocessing Pipeline for Classifiers
    num_cols = [c for c in FEATURE_COLUMNS if c != 'State']
    classifier_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), ['State']),
            ('num', StandardScaler(), num_cols)
        ]
    )
    
    # 6. Candidate Classifiers
    candidate_classifiers = {
        'Logistic Regression (Balanced)': Pipeline([
            ('preprocessor', classifier_preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
        ]),
        'Random Forest Classifier (Balanced)': Pipeline([
            ('preprocessor', ColumnTransformer(
                transformers=[('cat', OneHotEncoder(drop='first', sparse_output=False), ['State'])],
                remainder='passthrough'
            )),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])
    }
    
    results = {}
    fitted_models = {}
    
    # Evaluate ML Classifiers
    for name, pipe in candidate_classifiers.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else None
        
        results[name] = {
            'model_type': 'Machine Learning Classifier',
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'roc_auc': round(auc, 4) if auc is not None else None,
            'predictions': y_pred.tolist(),
            'probabilities': [round(float(p), 4) for p in y_prob]
        }
        fitted_models[name] = pipe
        print(f"\n[Model: {name}]")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    # Evaluate Baselines
    # Baseline 1: Always No Outbreak
    y_pred_always_zero = np.zeros(len(y_test), dtype=int)
    cm_zero = confusion_matrix(y_test, y_pred_always_zero, labels=[0, 1])
    tn_z, fp_z, fn_z, tp_z = int(cm_zero[0, 0]), int(cm_zero[0, 1]), int(cm_zero[1, 0]), int(cm_zero[1, 1])
    results['Baseline: Always No Outbreak'] = {
        'model_type': 'Rule-Based Baseline',
        'confusion_matrix': {'tp': tp_z, 'fp': fp_z, 'fn': fn_z, 'tn': tn_z},
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'roc_auc': None,
        'predictions': y_pred_always_zero.tolist(),
        'probabilities': None
    }
    print("\n[Baseline: Always No Outbreak]")
    print(f"  TP: {tp_z}, FP: {fp_z}, FN: {fn_z}, TN: {tn_z} | Precision: 0.0000 | Recall: 0.0000 | F1: 0.0000 | ROC-AUC: N/A")

    # Baseline 2: Last Month Was Outbreak
    y_pred_last_ob = test_df['last_month_outbreak'].values.astype(int)
    cm_last = confusion_matrix(y_test, y_pred_last_ob, labels=[0, 1])
    tn_l, fp_l, fn_l, tp_l = int(cm_last[0, 0]), int(cm_last[0, 1]), int(cm_last[1, 0]), int(cm_last[1, 1])
    prec_l = float(precision_score(y_test, y_pred_last_ob, zero_division=0))
    rec_l = float(recall_score(y_test, y_pred_last_ob, zero_division=0))
    f1_l = float(f1_score(y_test, y_pred_last_ob, zero_division=0))
    results['Baseline: Last Month Was Outbreak'] = {
        'model_type': 'Rule-Based Baseline',
        'confusion_matrix': {'tp': tp_l, 'fp': fp_l, 'fn': fn_l, 'tn': tn_l},
        'precision': round(prec_l, 4),
        'recall': round(rec_l, 4),
        'f1': round(f1_l, 4),
        'roc_auc': None,
        'predictions': y_pred_last_ob.tolist(),
        'probabilities': None
    }
    print("\n[Baseline: Last Month Was Outbreak]")
    print(f"  TP: {tp_l}, FP: {fp_l}, FN: {fn_l}, TN: {tn_l} | Precision: {prec_l:.4f} | Recall: {rec_l:.4f} | F1: {f1_l:.4f} | ROC-AUC: N/A")

    # 7. Expanding-Window Time-Series Cross-Validation (5 Folds)
    print("\n[Outbreak CV] Running 5-fold expanding-window cross-validation...")
    cv_metrics = run_outbreak_timeseries_cv(df_feat, baselines, n_folds=5)

    # 8. Per-State Outbreak Evaluation
    state_outbreak_metrics = {}
    best_clf_name = 'Logistic Regression (Balanced)' if results['Logistic Regression (Balanced)']['f1'] >= results['Random Forest Classifier (Balanced)']['f1'] else 'Random Forest Classifier (Balanced)'
    best_pipe = fitted_models[best_clf_name]
    
    for st, s_test in test_df.groupby('State'):
        X_st = s_test[FEATURE_COLUMNS]
        y_st = s_test['outbreak_target'].values
        y_st_pred = best_pipe.predict(X_st)
        cm_st = confusion_matrix(y_st, y_st_pred, labels=[0, 1])
        tn_s, fp_s, fn_s, tp_s = int(cm_st[0, 0]), int(cm_st[0, 1]), int(cm_st[1, 0]), int(cm_st[1, 1])
        prec_s = float(precision_score(y_st, y_st_pred, zero_division=0))
        rec_s = float(recall_score(y_st, y_st_pred, zero_division=0))
        f1_s = float(f1_score(y_st, y_st_pred, zero_division=0))
        state_outbreak_metrics[st] = {
            'total_test_months': len(y_st),
            'actual_outbreaks': int(np.sum(y_st)),
            'tp': tp_s, 'fp': fp_s, 'fn': fn_s, 'tn': tn_s,
            'precision': round(prec_s, 4),
            'recall': round(rec_s, 4),
            'f1': round(f1_s, 4)
        }
        print(f"  -> State {st}: Actual Outbreaks = {int(np.sum(y_st))}, TP = {tp_s}, FP = {fp_s}, FN = {fn_s}, TN = {tn_s}")

    # 9. Save Artifacts
    full_output = {
        'test_set_summary': {
            'total_test_samples': len(y_test),
            'actual_outbreaks': total_test_positives,
            'actual_non_outbreaks': total_test_negatives,
            'test_period': '2023-03 to 2024-12'
        },
        'training_set_summary': {
            'total_train_samples': len(y_train),
            'actual_outbreaks': total_train_positives,
            'train_period': '2016-01 to 2023-02'
        },
        'models': results,
        'cross_validation_5_folds': cv_metrics,
        'per_state_evaluation': state_outbreak_metrics,
        'selected_best_classifier': best_clf_name
    }

    if save_artifacts:
        with open(OUTBREAK_METRICS_PATH, 'w') as f:
            json.dump(full_output, f, indent=4)
        print(f"\n[Outbreak] Metrics saved to {OUTBREAK_METRICS_PATH}")
        
        joblib.dump(best_pipe, OUTBREAK_MODEL_PATH)
        print(f"[Outbreak] Trained best classifier saved to {OUTBREAK_MODEL_PATH}")
        
        # 10. Generate Confusion Matrix Plot
        generate_confusion_matrix_plot(results, OUTBREAK_CM_PLOT_PATH)

    return full_output

def run_outbreak_timeseries_cv(df_feat, baselines, n_folds=5):
    """
    Run expanding-window cross-validation for outbreak classification across n_folds.
    """
    n_months = 108  # Eligible months per state
    min_train = 48  # Initial window 4 years (2016-2019)
    step = (n_months - min_train) // n_folds
    
    cv_records = {
        'Logistic Regression (Balanced)': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
        'Random Forest Classifier (Balanced)': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
        'Baseline: Always No Outbreak': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
        'Baseline: Last Month Was Outbreak': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    }
    
    for fold in range(n_folds):
        train_end = min_train + (fold * step)
        test_end = min(train_end + step, n_months)
        if test_end <= train_end:
            break
            
        train_parts, test_parts = [], []
        for state, g in df_feat.groupby('State'):
            g = g.sort_values('date').reset_index(drop=True)
            train_parts.append(g.iloc[:train_end])
            test_parts.append(g.iloc[train_end:test_end])
            
        f_train = pd.concat(train_parts, ignore_index=True)
        f_test = pd.concat(test_parts, ignore_index=True)
        
        # Compute fold-specific baselines
        fold_baselines = compute_outbreak_baselines(f_train, save=False)
        f_train = add_outbreak_target(f_train, fold_baselines)
        f_test = add_outbreak_target(f_test, fold_baselines)
        
        X_tr, y_tr = f_train[FEATURE_COLUMNS], f_train['outbreak_target'].values
        X_te, y_te = f_test[FEATURE_COLUMNS], f_test['outbreak_target'].values
        
        # If fold training has only one class, add a dummy or handle safely
        num_cols = [c for c in FEATURE_COLUMNS if c != 'State']
        
        # 1. Logistic Regression
        pipe_lr = Pipeline([
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), ['State']),
                    ('num', StandardScaler(), num_cols)
                ]
            )),
            ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
        ])
        
        try:
            pipe_lr.fit(X_tr, y_tr)
            y_p_lr = pipe_lr.predict(X_te)
            y_prob_lr = pipe_lr.predict_proba(X_te)[:, 1] if len(pipe_lr.classes_) > 1 else np.zeros(len(X_te))
            auc_lr = float(roc_auc_score(y_te, y_prob_lr)) if len(np.unique(y_te)) > 1 else 0.5
        except Exception:
            y_p_lr = np.zeros(len(y_te))
            auc_lr = 0.5
            
        cv_records['Logistic Regression (Balanced)']['precision'].append(float(precision_score(y_te, y_p_lr, zero_division=0)))
        cv_records['Logistic Regression (Balanced)']['recall'].append(float(recall_score(y_te, y_p_lr, zero_division=0)))
        cv_records['Logistic Regression (Balanced)']['f1'].append(float(f1_score(y_te, y_p_lr, zero_division=0)))
        cv_records['Logistic Regression (Balanced)']['roc_auc'].append(auc_lr)
        
        # 2. Random Forest
        pipe_rf = Pipeline([
            ('preprocessor', ColumnTransformer(
                transformers=[('cat', OneHotEncoder(drop='first', sparse_output=False), ['State'])],
                remainder='passthrough'
            )),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])
        try:
            pipe_rf.fit(X_tr, y_tr)
            y_p_rf = pipe_rf.predict(X_te)
            y_prob_rf = pipe_rf.predict_proba(X_te)[:, 1] if len(pipe_rf.classes_) > 1 else np.zeros(len(X_te))
            auc_rf = float(roc_auc_score(y_te, y_prob_rf)) if len(np.unique(y_te)) > 1 else 0.5
        except Exception:
            y_p_rf = np.zeros(len(y_te))
            auc_rf = 0.5
            
        cv_records['Random Forest Classifier (Balanced)']['precision'].append(float(precision_score(y_te, y_p_rf, zero_division=0)))
        cv_records['Random Forest Classifier (Balanced)']['recall'].append(float(recall_score(y_te, y_p_rf, zero_division=0)))
        cv_records['Random Forest Classifier (Balanced)']['f1'].append(float(f1_score(y_te, y_p_rf, zero_division=0)))
        cv_records['Random Forest Classifier (Balanced)']['roc_auc'].append(auc_rf)
        
        # 3. Always No Outbreak
        y_p_zero = np.zeros(len(y_te))
        cv_records['Baseline: Always No Outbreak']['precision'].append(0.0)
        cv_records['Baseline: Always No Outbreak']['recall'].append(0.0)
        cv_records['Baseline: Always No Outbreak']['f1'].append(0.0)
        cv_records['Baseline: Always No Outbreak']['roc_auc'].append(0.5)
        
        # 4. Last Month Was Outbreak
        y_p_last = f_test['last_month_outbreak'].values
        cv_records['Baseline: Last Month Was Outbreak']['precision'].append(float(precision_score(y_te, y_p_last, zero_division=0)))
        cv_records['Baseline: Last Month Was Outbreak']['recall'].append(float(recall_score(y_te, y_p_last, zero_division=0)))
        cv_records['Baseline: Last Month Was Outbreak']['f1'].append(float(f1_score(y_te, y_p_last, zero_division=0)))
        cv_records['Baseline: Last Month Was Outbreak']['roc_auc'].append(0.5)
        
    cv_summary = {}
    for m_name, met_dict in cv_records.items():
        cv_summary[m_name] = {}
        for metric, vals in met_dict.items():
            cv_summary[m_name][metric] = {
                'mean': round(float(np.mean(vals)), 4),
                'min': round(float(np.min(vals)), 4),
                'max': round(float(np.max(vals)), 4)
            }
            
    return cv_summary

def generate_confusion_matrix_plot(results, output_path):
    """Generate side-by-side confusion matrix heatmaps."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models_to_plot = [
        'Logistic Regression (Balanced)',
        'Random Forest Classifier (Balanced)',
        'Baseline: Last Month Was Outbreak'
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for idx, name in enumerate(models_to_plot):
        cm_data = results[name]['confusion_matrix']
        matrix = np.array([
            [cm_data['tn'], cm_data['fp']],
            [cm_data['fn'], cm_data['tp']]
        ])
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=axes[idx],
            xticklabels=['No Outbreak (0)', 'Outbreak (1)'],
            yticklabels=['No Outbreak (0)', 'Outbreak (1)']
        )
        axes[idx].set_title(f"{name}\nF1: {results[name]['f1']:.2f} | Rec: {results[name]['recall']:.2f}", fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Outbreak] Saved confusion matrix visualization to {output_path}")

if __name__ == '__main__':
    train_and_evaluate_outbreak_classifiers(save_artifacts=True)

