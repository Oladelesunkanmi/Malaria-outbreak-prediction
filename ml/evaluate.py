import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from config import Config

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def generate_evaluation_plots(test_df, y_test, test_predictions, metrics_summary):
    """
    Generate three high-quality evaluation plots:
    1. Model performance comparison (MAE, RMSE, R2) for all models & baselines
    2. Actual vs Predicted malaria cases for candidate ML models
    3. Prediction error distribution (residuals) for the best ML model
    """
    os.makedirs(Config.STATIC_IMAGES_DIR, exist_ok=True)
    
    y_test_arr = np.array(y_test)
    
    # -------------------------------------------------------------
    # Plot 1: Model Performance Comparison Bar Chart (All Models & Baselines)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    models = list(metrics_summary.keys())
    maes = [metrics_summary[m]['mae'] for m in models]
    rmses = [metrics_summary[m]['rmse'] for m in models]
    r2s = [metrics_summary[m]['r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Errors chart
    axes[0].bar(x - width/2, maes, width, label='MAE', color='#0284c7')
    axes[0].bar(x + width/2, rmses, width, label='RMSE', color='#0d9488')
    axes[0].set_title('Prediction Error Metrics (Lower is Better)', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel('Cases (Count)', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    axes[0].legend(frameon=True)
    
    for i, v in enumerate(maes):
        axes[0].text(i - width/2, v + 250, f"{v:,.0f}", ha='center', fontsize=8, rotation=30)
    for i, v in enumerate(rmses):
        axes[0].text(i + width/2, v + 250, f"{v:,.0f}", ha='center', fontsize=8, rotation=30)
        
    # R2 chart
    colors = ['#10b981' if r > 0.7 else ('#f59e0b' if r > 0.0 else '#ef4444') for r in r2s]
    bars = axes[1].bar(models, r2s, width=0.45, color=colors)
    axes[1].set_title('Goodness of Fit ($R^2$ Score)', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_ylabel('$R^2$ Score', fontsize=11)
    axes[1].set_ylim(min(-0.1, min(r2s) - 0.1), 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    for bar in bars:
        h = bar.get_height()
        y_pos = h + 0.02 if h >= 0 else h - 0.06
        axes[1].text(bar.get_x() + bar.get_width()/2., y_pos, f"{h:.4f}", ha='center', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    comp_path = os.path.join(Config.STATIC_IMAGES_DIR, 'model_comparison.png')
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Saved {comp_path}")
    
    # -------------------------------------------------------------
    # Plot 2: Actual vs Predicted Malaria Cases (Scatter & Diagonal)
    # -------------------------------------------------------------
    ml_models = [m for m in ['Linear Regression', 'Random Forest Regressor', 'Decision Tree Regressor'] if m in test_predictions]
    if not ml_models:
        ml_models = list(test_predictions.keys())[:3]
        
    fig, axes = plt.subplots(1, len(ml_models), figsize=(6 * len(ml_models), 5), sharey=True, sharex=True)
    if len(ml_models) == 1:
        axes = [axes]
        
    for idx, m_name in enumerate(ml_models):
        preds_arr = np.array(test_predictions[m_name])
        ax = axes[idx]
        
        scatter = ax.scatter(y_test_arr, preds_arr, alpha=0.65, edgecolors='none', c='#0284c7', s=45)
        max_val = max(y_test_arr.max(), preds_arr.max()) * 1.05
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Ideal Fit (y = x)')
        
        r2_val = metrics_summary[m_name]['r2'] if m_name in metrics_summary else 0.0
        ax.set_title(f"{m_name}\n$R^2$ = {r2_val:.4f}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Actual Cases', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Predicted Cases', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        
    plt.suptitle('Actual vs. Predicted Malaria Cases on Chronological Test Set', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    avp_path = os.path.join(Config.STATIC_IMAGES_DIR, 'actual_vs_predicted.png')
    plt.savefig(avp_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Evaluation] Saved {avp_path}")
    
    # -------------------------------------------------------------
    # Plot 3: Prediction Error Residuals Visualization (Best ML Model)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    ml_candidates = {k: v for k, v in metrics_summary.items() if v.get('model_type') == 'Machine Learning'}
    if not ml_candidates:
        ml_candidates = metrics_summary
    best_model_name = min(ml_candidates.keys(), key=lambda k: ml_candidates[k]['mae'])
    
    best_preds = np.array(test_predictions[best_model_name])
    residuals = y_test_arr - best_preds
    
    # Residuals vs Predicted
    axes[0].scatter(best_preds, residuals, alpha=0.7, color='#0f766e', s=45)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title(f'Residuals vs Predicted Cases ({best_model_name})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Cases', fontsize=11)
    axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=11)
    
    # Distribution of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1], color='#0284c7', bins=15)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Error (Actual - Predicted)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    
    plt.tight_layout()
    res_path = os.path.join(Config.STATIC_IMAGES_DIR, 'prediction_error.png')
    plt.savefig(res_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Saved {res_path}")
