import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'State',
    'year',
    'month',
    'Population',
    'month_sin',
    'month_cos',
    'lag_1',
    'lag_2',
    'lag_3',
    'lag_12',
    'rolling_mean_3',
    'rolling_mean_6'
]

TARGET_COLUMN = 'cases'

def generate_features(df, drop_na=True):
    """
    Generate lag, rolling average, and cyclical seasonal features
    grouped by state separately to prevent data leakage.
    
    Parameters:
        df (pd.DataFrame): Preprocessed dataframe containing State, year, month, cases, Population
        drop_na (bool): If True, drops initial records where lags are unavailable (first 12 months)
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    df = df.copy()
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
        )
    
    # Sort chronologically within each state
    df = df.sort_values(['State', 'date']).reset_index(drop=True)
    
    state_frames = []
    for state_name, group in df.groupby('State'):
        g = group.copy().sort_values('date')
        
        # 1. Seasonal features: sine and cosine transforms for month (1-12)
        g['month_sin'] = np.sin(2 * np.pi * g['month'] / 12.0)
        g['month_cos'] = np.cos(2 * np.pi * g['month'] / 12.0)
        
        # 2. Lag features (cases from previous months)
        g['lag_1'] = g['cases'].shift(1)
        g['lag_2'] = g['cases'].shift(2)
        g['lag_3'] = g['cases'].shift(3)
        g['lag_12'] = g['cases'].shift(12)
        
        # 3. Rolling averages using strictly prior observations (no data leakage)
        # 3-month rolling mean of previous 3 months: (lag_1 + lag_2 + lag_3) / 3
        g['rolling_mean_3'] = (g['cases'].shift(1) + g['cases'].shift(2) + g['cases'].shift(3)) / 3.0
        
        # 6-month rolling mean of previous 6 months:
        g['rolling_mean_6'] = g['cases'].shift(1).rolling(window=6, min_periods=6).mean()
        
        state_frames.append(g)
        
    result_df = pd.concat(state_frames, ignore_index=True)
    
    if drop_na:
        result_df = result_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
        
    return result_df

def compute_single_row_features(state, year, month, population, recent_history_cases):
    """
    Compute features for a single next-month prediction row given the recent historical cases.
    
    Parameters:
        state (str): Name of state
        year (int): Prediction year
        month (int): Prediction month (1-12)
        population (int): Population of the state
        recent_history_cases (list or pd.Series): Cases ordered chronologically up to the preceding month,
                                                  with at least 12 months of values.
                                                  recent_history_cases[-1] is lag_1.
    
    Returns:
        dict: Feature dictionary ready for model inference
    """
    if len(recent_history_cases) < 12:
        raise ValueError(f"At least 12 months of prior cases required to compute lags. Provided {len(recent_history_cases)}.")
        
    lag_1 = float(recent_history_cases[-1])
    lag_2 = float(recent_history_cases[-2])
    lag_3 = float(recent_history_cases[-3])
    lag_12 = float(recent_history_cases[-12])
    
    rolling_mean_3 = (lag_1 + lag_2 + lag_3) / 3.0
    rolling_mean_6 = float(np.mean(recent_history_cases[-6:]))
    
    month_sin = float(np.sin(2 * np.pi * month / 12.0))
    month_cos = float(np.cos(2 * np.pi * month / 12.0))
    
    return {
        'State': state,
        'year': int(year),
        'month': int(month),
        'Population': int(population),
        'month_sin': month_sin,
        'month_cos': month_cos,
        'lag_1': lag_1,
        'lag_2': lag_2,
        'lag_3': lag_3,
        'lag_12': lag_12,
        'rolling_mean_3': rolling_mean_3,
        'rolling_mean_6': rolling_mean_6
    }

