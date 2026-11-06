from routes.auth import auth_bp
from routes.dashboard import dashboard_bp
from routes.dataset import dataset_bp
from routes.analytics import analytics_bp
from routes.predictions import predictions_bp
from routes.evaluation import evaluation_bp
from routes.history import history_bp

__all__ = [
    'auth_bp',
    'dashboard_bp',
    'dataset_bp',
    'analytics_bp',
    'predictions_bp',
    'evaluation_bp',
    'history_bp'
]

