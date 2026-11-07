from database.db import get_db
from database.models import Prediction, MalariaRecord
from ml.predict import predict_next_month, get_latest_recorded_period

class PredictionService:
    @staticmethod
    def get_eligible_period(state):
        """Get the latest recorded period and next eligible period for a state."""
        latest_yr, latest_mo, next_yr, next_mo = get_latest_recorded_period(state)
        
        with get_db() as db:
            latest_hist = db.query(MalariaRecord).filter_by(state=state)\
                .order_by(MalariaRecord.year.desc(), MalariaRecord.month.desc()).first()
                
        hist_yr = latest_hist.year if latest_hist else latest_yr
        hist_mo = latest_hist.month if latest_hist else latest_mo
        
        horizon = (next_yr - hist_yr) * 12 + (next_mo - hist_mo) if (next_yr and hist_yr) else 1
        can_predict = True
        horizon_warning = horizon > 1
        
        return {
            'state': state,
            'latest_year': latest_yr,
            'latest_month': latest_mo,
            'historical_year': hist_yr,
            'historical_month': hist_mo,
            'next_year': next_yr,
            'next_month': next_mo,
            'next_period_label': f"{next_yr}-{next_mo:02d}" if next_yr else None,
            'horizon_months': horizon,
            'horizon_warning': horizon_warning,
            'can_predict': can_predict,
            'horizon_limit_reached': False
        }

    @staticmethod
    def make_prediction(state, target_year, target_month, save=True):
        """Execute next-month prediction."""
        return predict_next_month(state, target_year, target_month, save_to_db=save)

    @staticmethod
    def get_history(state=None, limit=100, page=1):
        """Retrieve prediction history with optional state filtering and pagination."""
        with get_db() as db:
            query = db.query(Prediction).order_by(Prediction.generated_at.desc())
            if state:
                query = query.filter_by(state=state)
                
            total = query.count()
            offset = (page - 1) * limit
            items = query.offset(offset).limit(limit).all()
            
            return {
                'total': total,
                'page': page,
                'limit': limit,
                'items': [item.to_dict() for item in items]
            }

    @staticmethod
    def delete_prediction(prediction_id):
        """Delete a prediction record by ID."""
        with get_db() as db:
            pred = db.query(Prediction).filter_by(id=prediction_id).first()
            if pred:
                db.delete(pred)
                return True
            return False

