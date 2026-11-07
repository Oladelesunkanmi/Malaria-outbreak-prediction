import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'malaria-predict-secure-prod-key-2026')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', 
        f"sqlite:///{os.path.join(BASE_DIR, 'malaria_predict.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATASET_PATH = os.path.join(DATA_DIR, 'Wavelets_data for Time series_malaria.csv')
    RAINFALL_PATH = os.path.join(DATA_DIR, 'rainfall.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
    BEST_MODEL_RAINFALL_PATH = os.path.join(MODELS_DIR, 'best_model_rainfall.pkl')
    METRICS_PATH = os.path.join(MODELS_DIR, 'model_metrics.json')
    METRICS_RAINFALL_PATH = os.path.join(MODELS_DIR, 'model_metrics_rainfall.json')
    REPORT_PATH = os.path.join(MODELS_DIR, 'preprocessing_report.json')
    UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
    STATIC_IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')
    
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {'csv'}
    
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD')
    if not ADMIN_PASSWORD:
        raise ValueError(
            "CRITICAL SECURITY CONFIGURATION ERROR: The 'ADMIN_PASSWORD' environment variable is not set. "
            "Please configure ADMIN_PASSWORD in your .env file or environment variables before launching the application."
        )
