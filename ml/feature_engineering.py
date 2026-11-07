import numpy as np
import pandas as pd

# ---------- Base feature columns (without rainfall) ----------
BASE_FEATURE_COLUMNS = [
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

# ---------- Rainfall feature columns ----------
RAINFALL_FEATURE_COLUMNS = [
    'rainfall_lag_1',
    'rainfall_lag_2',
    'rainfall_lag_3',
    'rainfall_lag_12',
    'rainfall_cum_3',
    'rainfall_mean_3',
]

# ---------- Full feature set (base + rainfall) ----------
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + RAINFALL_FEATURE_COLUMNS

TARGET_COLUMN = 'cases'

def generate_features(df, drop_na=True):
    """
    Generate lag, rolling average, cyclical seasonal, and rainfall features
    grouped by state separately to prevent data leakage.
    
    Parameters:
        df (pd.DataFrame): Preprocessed dataframe containing State, year, month, cases, Population,
                           and optionally rainfall_mm.
        drop_na (bool): If True, drops initial records where lags are unavailable (first 12 months)
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    df = df.copy()
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
        )
    
    has_rainfall = 'rainfall_mm' in df.columns and df['rainfall_mm'].notna().any()
    
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
        g['rolling_mean_3'] = (g['cases'].shift(1) + g['cases'].shift(2) + g['cases'].shift(3)) / 3.0
        g['rolling_mean_6'] = g['cases'].shift(1).rolling(window=6, min_periods=6).mean()
        
        # 4. Rainfall features (only from past observations)
        if has_rainfall:
            g['rainfall_lag_1'] = g['rainfall_mm'].shift(1)
            g['rainfall_lag_2'] = g['rainfall_mm'].shift(2)
            g['rainfall_lag_3'] = g['rainfall_mm'].shift(3)
            g['rainfall_lag_12'] = g['rainfall_mm'].shift(12)
            g['rainfall_cum_3'] = (
                g['rainfall_mm'].shift(1) +
                g['rainfall_mm'].shift(2) +
                g['rainfall_mm'].shift(3)
            )
            g['rainfall_mean_3'] = g['rainfall_cum_3'] / 3.0
        else:
            # Fill with NaN so column layout stays consistent
            for rc in RAINFALL_FEATURE_COLUMNS:
                g[rc] = np.nan
        
        state_frames.append(g)
        
    result_df = pd.concat(state_frames, ignore_index=True)
    
    if drop_na:
        # Drop rows where any required feature is NaN (first 12 months per state)
        result_df = result_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
        
    return result_df

def compute_single_row_features(state, year, month, population, recent_history_cases,
                                 recent_history_rainfall=None):
    """
    Compute features for a single next-month prediction row given recent historical data.
    
    Parameters:
        state (str): Name of state
        year (int): Prediction year
        month (int): Prediction month (1-12)
        population (int): Population of the state
        recent_history_cases (list or pd.Series): Cases ordered chronologically up to the preceding month,
                                                  with at least 12 months of values.
                                                  recent_history_cases[-1] is lag_1.
        recent_history_rainfall (list or None): Rainfall values ordered chronologically up to the
                                                preceding month, with at least 12 values.
                                                recent_history_rainfall[-1] is rainfall_lag_1.
    
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
    
    features = {
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
    
    # Rainfall features
    if recent_history_rainfall is not None and len(recent_history_rainfall) >= 12:
        r_lag_1 = float(recent_history_rainfall[-1])
        r_lag_2 = float(recent_history_rainfall[-2])
        r_lag_3 = float(recent_history_rainfall[-3])
        r_lag_12 = float(recent_history_rainfall[-12])
        r_cum_3 = r_lag_1 + r_lag_2 + r_lag_3
        r_mean_3 = r_cum_3 / 3.0

        features['rainfall_lag_1'] = r_lag_1
        features['rainfall_lag_2'] = r_lag_2
        features['rainfall_lag_3'] = r_lag_3
        features['rainfall_lag_12'] = r_lag_12
        features['rainfall_cum_3'] = r_cum_3
        features['rainfall_mean_3'] = r_mean_3
    else:
        # Default to NaN when rainfall history is not available
        for rc in RAINFALL_FEATURE_COLUMNS:
            features[rc] = np.nan
    
    return features
