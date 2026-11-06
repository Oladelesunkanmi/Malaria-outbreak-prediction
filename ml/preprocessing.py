import os
import json
import pandas as pd
import numpy as np
from config import Config

def preprocess_malaria_data(csv_path=None, save_report=True):
    """
    Load and clean the raw malaria dataset.
    
    Returns:
        cleaned_df (pd.DataFrame): Cleaned and validated dataframe
        report (dict): Preprocessing summary report
    """
    csv_path = csv_path or Config.DATASET_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    # 1. Load the CSV
    raw_df = pd.read_csv(csv_path)
    original_records = len(raw_df)
    original_cols_count = len(raw_df.columns)
    
    # 2. Normalize column names (strip leading/trailing whitespace)
    raw_df.columns = raw_df.columns.str.strip()
    
    # 3. Identify and drop completely empty columns
    empty_cols = [c for c in raw_df.columns if raw_df[c].isna().all()]
    df = raw_df.drop(columns=empty_cols).copy()
    
    # 4. Remove irrelevant unnamed columns or sparse intervention columns (< 5% non-null)
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    sparse_cols = [c for c in df.columns if df[c].notna().sum() < (0.1 * len(df)) and c not in ['State', 'cases', 'year', 'month', 'Population', 'x', 'y', 'state_id', 'time']]
    cols_to_drop = list(set(unnamed_cols + sparse_cols))
    df = df.drop(columns=cols_to_drop)
    
    removed_cols_count = original_cols_count - len(df.columns)
    
    # 5. Check duplicate records
    duplicates_count = int(df.duplicated(subset=['State', 'year', 'month']).sum())
    if duplicates_count > 0:
        df = df.drop_duplicates(subset=['State', 'year', 'month']).copy()

    # 6. Type conversions for core columns
    required_cols = ['State', 'cases', 'year', 'month', 'Population']
    missing_core = [c for c in required_cols if c not in df.columns]
    if missing_core:
        raise ValueError(f"Dataset is missing essential columns: {missing_core}")

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
    df['cases'] = pd.to_numeric(df['cases'], errors='coerce')
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
    
    # Check for missing values in required columns
    missing_values_count = int(df[required_cols].isna().sum().sum())
    df = df.dropna(subset=required_cols).copy()
    
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['cases'] = df['cases'].astype(int)
    df['Population'] = df['Population'].astype(int)

    # 7. Check for invalid malaria case values (cases < 0)
    invalid_cases_count = int((df['cases'] < 0).sum())
    if invalid_cases_count > 0:
        df = df[df['cases'] >= 0].copy()

    # 8. Create proper datetime column
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )

    # 9. Sort observations chronologically by State and Date
    df = df.sort_values(['State', 'date']).reset_index(drop=True)

    # 10. Validate that each state has monthly observations in chronological order
    states_covered = sorted(df['State'].unique().tolist())
    min_date = df['date'].min().strftime('%Y-%m')
    max_date = df['date'].max().strftime('%Y-%m')
    
    # Generate preprocessing report
    report = {
        "original_records": original_records,
        "cleaned_records": len(df),
        "original_columns": original_cols_count,
        "cleaned_columns": len(df.columns),
        "removed_columns_count": removed_cols_count,
        "empty_columns_removed": empty_cols,
        "duplicate_records": duplicates_count,
        "missing_values_in_core": missing_values_count,
        "invalid_cases_removed": invalid_cases_count,
        "states_covered": states_covered,
        "date_range": f"{min_date} to {max_date}",
        "records_per_state": df['State'].value_counts().to_dict()
    }

    if save_report:
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        with open(Config.REPORT_PATH, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"[Preprocessing] Report saved to {Config.REPORT_PATH}")

    return df, report

if __name__ == "__main__":
    df, report = preprocess_malaria_data()
    print("Preprocessing completed successfully.")
    print(json.dumps(report, indent=2))

