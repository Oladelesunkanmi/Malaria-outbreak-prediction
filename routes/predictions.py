from flask import Blueprint, render_template, request, flash, jsonify, redirect, url_for
from routes.auth import login_required, validate_csrf, generate_csrf_token
from services.prediction_service import PredictionService
from database.db import get_db
from database.models import MalariaRecord

predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    csrf_token = generate_csrf_token()
    
    with get_db() as db:
        states = [s[0] for s in db.query(MalariaRecord.state).distinct().all()]
        states = sorted(states)
        
    default_state = states[0] if states else 'Kwara'
    period_info = PredictionService.get_eligible_period(default_state)
    
    prediction_result = None
    
    if request.method == 'POST':
        token = request.form.get('csrf_token')
        if not validate_csrf(token):
            flash('Invalid session token. Please try again.', 'danger')
            return redirect(url_for('predictions.predict'))
            
        selected_state = request.form.get('state') or default_state
        target_year = request.form.get('year')
        target_month = request.form.get('month')
        
        try:
            if not target_year or not target_month:
                raise ValueError("Prediction year and month are required.")
                
            prediction_result = PredictionService.make_prediction(
                state=selected_state,
                target_year=int(target_year),
                target_month=int(target_month),
                save=True
            )
            flash(
                f"Prediction successfully generated for {selected_state} ({target_year}-{int(target_month):02d}): "
                f"{prediction_result['predicted_cases']:,.0f} expected cases.",
                "success"
            )
            # Re-fetch period info after saving prediction
            period_info = PredictionService.get_eligible_period(selected_state)
            default_state = selected_state
        except Exception as e:
            flash(f"Prediction Error: {str(e)}", "danger")
            prediction_result = None
            try:
                period_info = PredictionService.get_eligible_period(selected_state)
            except Exception:
                pass
            default_state = selected_state
            
    return render_template(
        'predict.html',
        states=states,
        default_state=default_state,
        period_info=period_info,
        prediction=prediction_result,
        csrf_token=csrf_token,
        active_page='predict'
    )

@predictions_bp.route('/api/predict/eligible/<state>')
@login_required
def eligible_period_api(state):
    info = PredictionService.get_eligible_period(state)
    return jsonify(info)

