from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Admin(Base):
    __tablename__ = 'admins'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

class MalariaRecord(Base):
    __tablename__ = 'malaria_records'
    __table_args__ = (
        UniqueConstraint('state', 'year', 'month', name='uq_state_year_month'),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String(50), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, nullable=False, index=True)
    date = Column(String(20), nullable=False)  # Stored as YYYY-MM-01
    cases = Column(Integer, nullable=False)
    population = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'state': self.state,
            'year': self.year,
            'month': self.month,
            'date': self.date,
            'cases': self.cases,
            'population': self.population,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String(50), nullable=False, index=True)
    prediction_year = Column(Integer, nullable=False)
    prediction_month = Column(Integer, nullable=False)
    predicted_cases = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)  # LOW RISK, MODERATE RISK, HIGH RISK
    model_name = Column(String(100), nullable=False)
    generated_at = Column(DateTime, default=datetime.now, nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'state': self.state,
            'prediction_year': self.prediction_year,
            'prediction_month': self.prediction_month,
            'predicted_cases': round(self.predicted_cases, 1),
            'risk_level': self.risk_level,
            'model_name': self.model_name,
            'generated_at': self.generated_at.strftime('%Y-%m-%d %H:%M:%S') if self.generated_at else None
        }

class ModelMetric(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    mae = Column(Float, nullable=False)
    mse = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    r2 = Column(Float, nullable=False)
    trained_at = Column(DateTime, default=datetime.now, nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_name': self.model_name,
            'mae': round(self.mae, 2),
            'mse': round(self.mse, 2),
            'rmse': round(self.rmse, 2),
            'r2': round(self.r2, 4),
            'trained_at': self.trained_at.strftime('%Y-%m-%d %H:%M:%S') if self.trained_at else None
        }
