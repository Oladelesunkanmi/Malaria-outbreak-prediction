import os
import pandas as pd
from contextlib import contextmanager
from werkzeug.security import generate_password_hash
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from config import Config
from database.models import Base, Admin, MalariaRecord, Prediction, ModelMetric

engine = create_engine(
    Config.SQLALCHEMY_DATABASE_URI, 
    connect_args={"check_same_thread": False} if "sqlite" in Config.SQLALCHEMY_DATABASE_URI else {}
)
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
)

def set_engine_and_session(new_uri):
    """Rebind database engine and session to a new URI (e.g. For isolated test databases)."""
    global engine, SessionLocal
    engine = create_engine(
        new_uri,
        connect_args={"check_same_thread": False} if "sqlite" in new_uri else {}
    )
    SessionLocal = scoped_session(
        sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
    )
    return engine

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)

def seed_admin(username=None, password=None):
    """Ensure an administrator account exists."""
    username = username or Config.ADMIN_USERNAME
    password = password or Config.ADMIN_PASSWORD
    
    with get_db() as db:
        existing = db.query(Admin).filter_by(username=username).first()
        if not existing:
            hashed = generate_password_hash(password)
            admin = Admin(username=username, password_hash=hashed)
            db.add(admin)
            print(f"[DB] Initialized default admin account: {username}")
        else:
            print(f"[DB] Admin account '{username}' already exists.")

def seed_records_from_csv(csv_path=None):
    """Seed MalariaRecord table from cleaned CSV dataset without duplicates."""
    csv_path = csv_path or Config.DATASET_PATH
    if not os.path.exists(csv_path):
        print(f"[DB] Dataset not found at: {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    required_cols = {'State', 'cases', 'year', 'month', 'Population'}
    if not required_cols.issubset(set(df.columns)):
        print(f"[DB] CSV missing required columns: {required_cols - set(df.columns)}")
        return 0

    df = df.dropna(subset=list(required_cols)).copy()
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['cases'] = df['cases'].astype(int)
    df['Population'] = df['Population'].astype(int)
    df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'

    inserted_count = 0
    with get_db() as db:
        # Fetch existing keys to avoid duplicates
        existing_keys = set(
            db.query(MalariaRecord.state, MalariaRecord.year, MalariaRecord.month).all()
        )
        
        new_records = []
        for _, row in df.iterrows():
            key = (row['State'], int(row['year']), int(row['month']))
            if key not in existing_keys:
                rec = MalariaRecord(
                    state=row['State'],
                    year=int(row['year']),
                    month=int(row['month']),
                    date=row['date'],
                    cases=int(row['cases']),
                    population=int(row['Population'])
                )
                new_records.append(rec)
                existing_keys.add(key)
        
        if new_records:
            db.bulk_save_objects(new_records)
            inserted_count = len(new_records)
            print(f"[DB] Imported {inserted_count} new malaria records from CSV.")
        else:
            print("[DB] All records already present in database.")
            
    return inserted_count
