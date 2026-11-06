import os
from flask import Flask, render_template, session
from config import Config
from database.db import init_db, seed_admin, seed_records_from_csv
from routes import (
    auth_bp,
    dashboard_bp,
    dataset_bp,
    analytics_bp,
    predictions_bp,
    evaluation_bp,
    history_bp
)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize Database and seed tables
    with app.app_context():
        init_db()
        seed_admin()
        seed_records_from_csv()

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(evaluation_bp)
    app.register_blueprint(history_bp)

    @app.context_processor
    def inject_global_vars():
        return {
            'current_admin': session.get('admin_username'),
            'app_name': 'MalariaOutbreak Sentinel AI',
            'app_version': '1.0.0'
        }

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('base.html', not_found=True), 404

    @app.errorhandler(500)
    def server_error(e):
        app.logger.error(f"Internal Server Error: {e}", exc_info=True)
        return render_template('base.html', server_error=True), 500

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')
    print(f"\n==================================================")
    print(f" MALARIA OUTBREAK PREDICTION SYSTEM")
    print(f" Web application running at: http://127.0.0.1:{port}")
    print(f" Default Admin User: {Config.ADMIN_USERNAME}")
    print(f" Administrative Authentication: Set via .env or environment")
    print(f"==================================================\n")
    app.run(host='127.0.0.1', port=port, debug=debug_mode)

