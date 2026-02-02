"""
Machine Learning model training and prediction functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from modules.config import (
    TEST_SIZE, RANDOM_STATE, RF_N_ESTIMATORS, RF_MAX_DEPTH, 
    HGB_MAX_ITER, MIN_RECORDS_FOR_TRAINING
)


def prepare_model_data(df):
    """Prepare data for machine learning models"""
    # Select features - check for available columns
    base_cols = ['person_no', 'fam_no', 'provinceid', 'municipality_id']
    feature_cols = [col for col in base_cols if col in df.columns]
    
    # Handle month and year columns (check for suffixed versions)
    if 'month' in df.columns:
        feature_cols.append('month')
    elif 'month_affected' in df.columns:
        df['month'] = df['month_affected']
        feature_cols.append('month')
    elif 'month_assistance' in df.columns:
        df['month'] = df['month_assistance']
        feature_cols.append('month')
    
    if 'year' in df.columns:
        feature_cols.append('year')
    elif 'year_affected' in df.columns:
        df['year'] = df['year_affected']
        feature_cols.append('year')
    elif 'year_assistance' in df.columns:
        df['year'] = df['year_assistance']
        feature_cols.append('year')
    
    # Encode disaster_name
    le_disaster = LabelEncoder()
    df['disaster_name_encoded'] = le_disaster.fit_transform(df['disaster_name'].astype(str))
    feature_cols.append('disaster_name_encoded')
    
    # Prepare X and y
    X = df[feature_cols].fillna(0)
    y = df['total_amount'].fillna(0)
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], 0)
    y = y.replace([np.inf, -np.inf], 0)
    
    return X, y, le_disaster, feature_cols


def train_models(X, y):
    """Train and compare multiple models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    models = {
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, 
            random_state=RANDOM_STATE, 
            max_depth=RF_MAX_DEPTH
        ),
        'Hist Gradient Boosting Regressor': HistGradientBoostingRegressor(
            random_state=RANDOM_STATE, 
            max_iter=HGB_MAX_ITER
        ),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    return results, X_train, X_test, y_train, y_test


def validate_training_data(data):
    """Check if there's enough data for model training"""
    merged_clean = data['merged'].copy()
    merged_clean = merged_clean[merged_clean['total_amount'] > 0]
    return len(merged_clean) >= MIN_RECORDS_FOR_TRAINING, merged_clean

