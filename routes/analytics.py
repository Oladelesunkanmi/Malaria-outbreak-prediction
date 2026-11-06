from flask import Blueprint, render_template, jsonify
from routes.auth import login_required
from services.analytics_service import AnalyticsService

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/analytics')
@login_required
def index():
    summary = AnalyticsService.get_dashboard_summary()
    return render_template(
        'analytics.html',
        summary=summary,
        active_page='analytics'
    )

@analytics_bp.route('/api/analytics/data')
@login_required
def analytics_data():
    return jsonify({
        'monthly_trend': AnalyticsService.get_monthly_trend(),
        'state_distribution': AnalyticsService.get_cases_by_state(),
        'yearly_trend': AnalyticsService.get_yearly_trend(),
        'seasonal_patterns': AnalyticsService.get_seasonal_patterns()
    })

