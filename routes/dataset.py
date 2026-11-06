import os
import json
import pandas as pd
from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from werkzeug.utils import secure_filename

from config import Config
from routes.auth import login_required, validate_csrf, generate_csrf_token
from database.db import get_db, seed_records_from_csv
from database.models import MalariaRecord
from ml.preprocessing import preprocess_malaria_data

dataset_bp = Blueprint('dataset', __name__)

@dataset_bp.route('/dataset')
@login_required
def index():
    state_filter = request.args.get('state', '').strip()
    year_filter = request.args.get('year', '').strip()
    month_filter = request.args.get('month', '').strip()
    page = request.args.get('page', 1, type=int)
    per_page = 20

    with get_db() as db:
        query = db.query(MalariaRecord)
        if state_filter:
            query = query.filter(MalariaRecord.state.ilike(f"%{state_filter}%"))
        if year_filter and year_filter.isdigit():
            query = query.filter(MalariaRecord.year == int(year_filter))
        if month_filter and month_filter.isdigit():
            query = query.filter(MalariaRecord.month == int(month_filter))

        total_records = query.count()
        records = query.order_by(
            MalariaRecord.year.desc(), MalariaRecord.month.desc(), MalariaRecord.state.asc()
        ).offset((page - 1) * per_page).limit(per_page).all()

        available_states = [s[0] for s in db.query(MalariaRecord.state).distinct().all()]
        available_years = [y[0] for y in db.query(MalariaRecord.year).distinct().order_by(MalariaRecord.year.desc()).all()]

    # Load preprocessing report
    report = {}
    if os.path.exists(Config.REPORT_PATH):
        try:
            with open(Config.REPORT_PATH, 'r') as f:
                report = json.load(f)
        except Exception:
            pass

    total_pages = (total_records + per_page - 1) // per_page or 1
    csrf_token = generate_csrf_token()

    return render_template(
        'dataset.html',
        records=records,
        total_records=total_records,
        page=page,
        total_pages=total_pages,
        available_states=available_states,
        available_years=available_years,
        state_filter=state_filter,
        year_filter=year_filter,
        month_filter=month_filter,
        report=report,
        csrf_token=csrf_token,
        active_page='dataset'
    )

@dataset_bp.route('/dataset/upload', methods=['POST'])
@login_required
def upload():
    token = request.form.get('csrf_token')
    if not validate_csrf(token):
        flash('Invalid session token. Please try again.', 'danger')
        return redirect(url_for('dataset.index'))

    if 'csv_file' not in request.files:
        flash('No file was uploaded.', 'danger')
        return redirect(url_for('dataset.index'))

    file = request.files['csv_file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('dataset.index'))

    if not file.filename.lower().endswith('.csv'):
        flash('Only CSV files (.csv) are allowed.', 'danger')
        return redirect(url_for('dataset.index'))

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Validate dataset structure
    try:
        df_test = pd.read_csv(save_path)
        df_test.columns = df_test.columns.str.strip()
        required_cols = {'State', 'cases', 'year', 'month', 'Population'}
        missing = required_cols - set(df_test.columns)
        if missing:
            os.remove(save_path)
            flash(f'Incompatible dataset: Missing required columns: {", ".join(missing)}', 'danger')
            return redirect(url_for('dataset.index'))

        if len(df_test) < 12:
            os.remove(save_path)
            flash('Incompatible dataset: At least 12 monthly observations are required.', 'danger')
            return redirect(url_for('dataset.index'))

        # Replace main dataset and reprocess
        import shutil
        shutil.copyfile(save_path, Config.DATASET_PATH)
        
        # Reprocess
        _, new_report = preprocess_malaria_data(Config.DATASET_PATH, save_report=True)
        imported_count = seed_records_from_csv(Config.DATASET_PATH)
        
        flash(f'Dataset uploaded and validated successfully! {imported_count} records updated.', 'success')
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        flash(f'Failed to process CSV: {str(e)}', 'danger')

    return redirect(url_for('dataset.index'))

