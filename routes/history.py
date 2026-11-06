import io
import csv
from flask import Blueprint, render_template, request, flash, redirect, url_for, Response
from routes.auth import login_required
from database.db import get_db
from database.models import Prediction, MalariaRecord
from services.prediction_service import PredictionService

history_bp = Blueprint('history', __name__)

@history_bp.route('/history')
@login_required
def index():
    state_filter = request.args.get('state', '').strip()
    risk_filter = request.args.get('risk', '').strip()
    page = request.args.get('page', 1, type=int)
    per_page = 20

    with get_db() as db:
        query = db.query(Prediction)
        if state_filter:
            query = query.filter_by(state=state_filter)
        if risk_filter:
            query = query.filter_by(risk_level=risk_filter)

        total_predictions = query.count()
        predictions = query.order_by(Prediction.generated_at.desc())\
            .offset((page - 1) * per_page).limit(per_page).all()

        available_states = [s[0] for s in db.query(MalariaRecord.state).distinct().all()]

    total_pages = (total_predictions + per_page - 1) // per_page or 1

    return render_template(
        'history.html',
        predictions=predictions,
        total_predictions=total_predictions,
        page=page,
        total_pages=total_pages,
        available_states=sorted(available_states),
        state_filter=state_filter,
        risk_filter=risk_filter,
        active_page='history'
    )

@history_bp.route('/history/delete/<int:pred_id>', methods=['POST'])
@login_required
def delete(pred_id):
    success = PredictionService.delete_prediction(pred_id)
    if success:
        flash('Prediction record removed.', 'info')
    else:
        flash('Prediction record not found.', 'danger')
    return redirect(url_for('history.index'))

@history_bp.route('/history/export')
@login_required
def export_csv():
    with get_db() as db:
        preds = db.query(Prediction).order_by(Prediction.generated_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'ID', 'State', 'Target Year', 'Target Month', 'Predicted Cases', 
        'Risk Level', 'Model Used', 'Generated At'
    ])

    for p in preds:
        writer.writerow([
            p.id, p.state, p.prediction_year, p.prediction_month,
            round(p.predicted_cases, 1), p.risk_level, p.model_name,
            p.generated_at.strftime('%Y-%m-%d %H:%M:%S') if p.generated_at else ''
        ])

    output.seek(0)
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=malaria_predictions_history.csv"}
    )

