import pytest
import os
import pandas as pd
from ml.preprocessing import preprocess_malaria_data
from config import Config

def test_preprocessing_loads_and_cleans():
    df, report = preprocess_malaria_data(save_report=False)
    
    # Assert records count
    assert len(df) == 360
    assert report['original_records'] == 360
    assert report['cleaned_records'] == 360
    
    # Assert 3 states
    assert sorted(report['states_covered']) == ['Kwara', 'Nasarawa', 'Zamfara']
    assert df['State'].nunique() == 3
    
    # Assert each state has 120 observations
    for state in ['Kwara', 'Nasarawa', 'Zamfara']:
        assert (df['State'] == state).sum() == 120
        
    # Assert columns removed
    assert report['removed_columns_count'] >= 28
    assert 'Unnamed: 9' not in df.columns
    
    # Assert date column created and sorted
    assert 'date' in df.columns
    assert df['year'].min() == 2015
    assert df['year'].max() == 2024
    assert df['month'].min() == 1
    assert df['month'].max() == 12
    
    # Assert no missing in core
    assert report['missing_values_in_core'] == 0
    assert report['duplicate_records'] == 0
    assert report['invalid_cases_removed'] == 0

