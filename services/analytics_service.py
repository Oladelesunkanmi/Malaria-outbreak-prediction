import os
import json
import pandas as pd
from sqlalchemy import func
from config import Config
from database.db import get_db
from database.models import MalariaRecord, Prediction, ModelMetric

class AnalyticsService:
    @staticmethod
    def get_dashboard_summary():
        """Retrieve key KPI metrics for dashboard display."""
        with get_db() as db:
            total_cases = db.query(func.sum(MalariaRecord.cases)).scalar() or 0
            total_observations = db.query(MalariaRecord).count()
            states = [s[0] for s in db.query(MalariaRecord.state).distinct().all()]
            
            # Latest historical record
            latest_record = db.query(MalariaRecord).order_by(
                MalariaRecord.year.desc(), MalariaRecord.month.desc()
            ).first()
            latest_month_str = f"{latest_record.year}-{latest_record.month:02d}" if latest_record else "N/A"
            
            # Latest prediction
            latest_pred = db.query(Prediction).order_by(Prediction.generated_at.desc()).first()
            
            # Best model name
            best_model_name = "N/A"
            if os.path.exists(Config.METRICS_PATH):
                try:
                    with open(Config.METRICS_PATH, 'r') as f:
                        metrics_data = json.load(f)
                        best_model_name = metrics_data.get('best_model', 'N/A')
                except Exception:
                    pass
            
            if best_model_name == "N/A":
                metric_rec = db.query(ModelMetric).order_by(ModelMetric.mae.asc()).first()
                if metric_rec:
                    best_model_name = metric_rec.model_name
                    
            return {
                'total_cases': int(total_cases),
                'total_observations': total_observations,
                'total_states': len(states),
                'states': sorted(states),
                'latest_reporting_month': latest_month_str,
                'selected_model': best_model_name,
                'latest_prediction': latest_pred.to_dict() if latest_pred else None
            }

    @staticmethod
    def get_monthly_trend():
        """Get monthly case counts across all dates, grouped by state."""
        with get_db() as db:
            records = db.query(
                MalariaRecord.date, MalariaRecord.state, MalariaRecord.cases
            ).order_by(MalariaRecord.date.asc()).all()
            
            if not records:
                return {'labels': [], 'datasets': []}
                
            df = pd.DataFrame(records, columns=['date', 'state', 'cases'])
            dates = sorted(df['date'].unique().tolist())
            
            states = sorted(df['state'].unique().tolist())
            colors = {
                'Kwara': '#0284c7',      # Sky blue
                'Nasarawa': '#0d9488',   # Teal
                'Zamfara': '#f59e0b'     # Amber
            }
            
            datasets = []
            for state in states:
                state_df = df[df['state'] == state].set_index('date')
                cases_list = [int(state_df.loc[d, 'cases']) if d in state_df.index else 0 for d in dates]
                datasets.append({
                    'label': state,
                    'data': cases_list,
                    'borderColor': colors.get(state, '#64748b'),
                    'backgroundColor': colors.get(state, '#64748b') + '20',
                    'tension': 0.25,
                    'fill': False
                })
                
            return {
                'labels': dates,
                'datasets': datasets
            }

    @staticmethod
    def get_cases_by_state():
        """Get total cases and percentage share by state."""
        with get_db() as db:
            results = db.query(
                MalariaRecord.state, func.sum(MalariaRecord.cases)
            ).group_by(MalariaRecord.state).all()
            
            states = [r[0] for r in results]
            cases = [int(r[1]) for r in results]
            total = sum(cases) or 1
            
            return {
                'labels': states,
                'data': cases,
                'percentages': [round((c / total) * 100, 1) for c in cases],
                'backgroundColor': ['#0284c7', '#0d9488', '#f59e0b']
            }

    @staticmethod
    def get_yearly_trend():
        """Get total yearly malaria cases by state."""
        with get_db() as db:
            results = db.query(
                MalariaRecord.year, MalariaRecord.state, func.sum(MalariaRecord.cases)
            ).group_by(MalariaRecord.year, MalariaRecord.state).order_by(MalariaRecord.year.asc()).all()
            
            if not results:
                return {'labels': [], 'datasets': []}
                
            df = pd.DataFrame(results, columns=['year', 'state', 'cases'])
            years = sorted(df['year'].unique().tolist())
            states = sorted(df['state'].unique().tolist())
            
            colors = {
                'Kwara': '#0284c7',
                'Nasarawa': '#0d9488',
                'Zamfara': '#f59e0b'
            }
            
            datasets = []
            for state in states:
                state_df = df[df['state'] == state].set_index('year')
                cases_list = [int(state_df.loc[y, 'cases']) if y in state_df.index else 0 for y in years]
                datasets.append({
                    'label': state,
                    'data': cases_list,
                    'backgroundColor': colors.get(state, '#64748b')
                })
                
            return {
                'labels': [str(y) for y in years],
                'datasets': datasets
            }

    @staticmethod
    def get_seasonal_patterns():
        """Calculate historical seasonal average cases for each month (Jan-Dec)."""
        with get_db() as db:
            results = db.query(
                MalariaRecord.month, MalariaRecord.state, func.avg(MalariaRecord.cases)
            ).group_by(MalariaRecord.month, MalariaRecord.state).order_by(MalariaRecord.month.asc()).all()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            if not results:
                return {'labels': month_names, 'datasets': []}
                
            df = pd.DataFrame(results, columns=['month', 'state', 'avg_cases'])
            states = sorted(df['state'].unique().tolist())
            
            colors = {
                'Kwara': '#0284c7',
                'Nasarawa': '#0d9488',
                'Zamfara': '#f59e0b'
            }
            
            datasets = []
            for state in states:
                state_df = df[df['state'] == state].set_index('month')
                avg_list = [round(float(state_df.loc[m, 'avg_cases']), 1) if m in state_df.index else 0 for m in range(1, 13)]
                datasets.append({
                    'label': state,
                    'data': avg_list,
                    'borderColor': colors.get(state, '#64748b'),
                    'backgroundColor': colors.get(state, '#64748b') + '33',
                    'tension': 0.3,
                    'fill': True
                })
                
            return {
                'labels': month_names,
                'datasets': datasets
            }

    @staticmethod
    def get_predicted_vs_historical_series():
        """
        Retrieve historical trend for the last 24 months plus recent predictions.
        """
        with get_db() as db:
            # Last 24 months of historical records
            hist_records = db.query(MalariaRecord).order_by(
                MalariaRecord.year.desc(), MalariaRecord.month.desc()
            ).limit(72).all()  # 24 months * 3 states = 72 records
            
            hist_records = sorted(hist_records, key=lambda r: (r.date, r.state))
            
            preds = db.query(Prediction).order_by(
                Prediction.prediction_year.asc(), Prediction.prediction_month.asc()
            ).all()
            
            dates_set = set(r.date for r in hist_records)
            for p in preds:
                dates_set.add(f"{p.prediction_year}-{p.prediction_month:02d}-01")
                
            all_dates = sorted(list(dates_set))
            
            return {
                'dates': all_dates,
                'historical_count': len(hist_records),
                'prediction_count': len(preds)
            }

