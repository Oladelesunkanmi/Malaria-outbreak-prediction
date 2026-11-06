import os
import json
from flask import Blueprint, render_template, flash, redirect, url_for
from config import Config
from routes.auth import login_required
from database.db import get_db
from database.models import ModelMetric
from ml.train import train_and_evaluate_models

evaluation_bp = Blueprint('evaluation', __name__)

@evaluation_bp.route('/evaluation')
@login_required
def index():
    metrics_data = {}
    if os.path.exists(Config.METRICS_PATH):
        try:
            with open(Config.METRICS_PATH, 'r') as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    # Fallback to DB if metrics file wasn't found
    if not metrics_data:
        with get_db() as db:
            db_metrics = db.query(ModelMetric).all()
            if db_metrics:
                summary = {m.model_name: m.to_dict() for m in db_metrics}
                best = min(summary.keys(), key=lambda k: summary[k]['mae']) if summary else "N/A"
                metrics_data = {
                    'metrics': summary,
                    'best_model': best,
                    'trained_at': db_metrics[0].trained_at.strftime('%Y-%m-%d %H:%M:%S') if db_metrics else "N/A"
                }

    outbreak_metrics = {}
    outbreak_path = os.path.join(Config.MODELS_DIR, 'outbreak_metrics.json')
    if os.path.exists(outbreak_path):
        try:
            with open(outbreak_path, 'r') as f:
                outbreak_metrics = json.load(f)
        except Exception:
            pass

    return render_template(
        'evaluation.html',
        metrics_data=metrics_data,
        outbreak_metrics=outbreak_metrics,
        active_page='evaluation'
    )

@evaluation_bp.route('/evaluation/retrain', methods=['POST'])
@login_required
def retrain():
    try:
        results = train_and_evaluate_models(save_artifacts=True)
        flash(f"Models successfully retrained! Best model: {results['best_model']}.", "success")
    except Exception as e:
        flash(f"Model retraining failed: {str(e)}", "danger")
    return redirect(url_for('evaluation.index'))

