from flask import Blueprint, render_template, jsonify
from routes.auth import login_required
from services.analytics_service import AnalyticsService
from services.prediction_service import PredictionService

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@dashboard_bp.route('/dashboard')
@login_required
def index():
    summary = AnalyticsService.get_dashboard_summary()
    recent_predictions = PredictionService.get_history(limit=5)['items']
    return render_template(
        'dashboard.html',
        summary=summary,
        recent_predictions=recent_predictions,
        active_page='dashboard'
    )

@dashboard_bp.route('/api/dashboard/charts')
@login_required
def charts_api():
    monthly_trend = AnalyticsService.get_monthly_trend()
    state_distribution = AnalyticsService.get_cases_by_state()
    yearly_trend = AnalyticsService.get_yearly_trend()
    
    return jsonify({
        'monthly_trend': monthly_trend,
        'state_distribution': state_distribution,
        'yearly_trend': yearly_trend
    })

