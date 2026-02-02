"""
Machine Learning model training and prediction functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from modules.config import (
    TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE, RF_N_ESTIMATORS, RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF, RF_MAX_FEATURES,
    HGB_MAX_ITER, HGB_LEARNING_RATE, HGB_EARLY_STOPPING,
    HGB_VALIDATION_FRACTION, HGB_N_ITER_NO_CHANGE,
    MIN_RECORDS_FOR_TRAINING, OUTLIER_METHOD, OUTLIER_Z_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER, USE_LOG_TRANSFORM, USE_FEATURE_SCALING,
    USE_ENHANCED_FEATURES, RIDGE_ALPHA
)


def remove_outliers_iqr(y, multiplier=1.5):
    """Remove outliers using IQR method"""
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (y >= lower_bound) & (y <= upper_bound)


def remove_outliers_zscore(y, threshold=3.0):
    """Remove outliers using z-score method"""
    z_scores = np.abs((y - y.mean()) / y.std())
    return z_scores < threshold


def prepare_model_data(df, remove_outliers=True, enhanced_features=None):
    """Prepare data for machine learning models with noise reduction and enhanced feature engineering"""
    # Select base features - check for available columns
    base_cols = ['person_no', 'fam_no', 'provinceid', 'municipality_id']
    feature_cols = [col for col in base_cols if col in df.columns]
    
    # CRITICAL: Add quantity if available (highly correlated with total_amount)
    if 'quantity' in df.columns:
        feature_cols.append('quantity')
    
    # Handle month and year columns
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
    
    # Enhanced feature engineering for better predictions
    if enhanced_features is None:
        enhanced_features = USE_ENHANCED_FEATURES
    
    if enhanced_features:
        # Season encoding (if available)
        if 'season' in df.columns:
            le_season = LabelEncoder()
            df['season_encoded'] = le_season.fit_transform(df['season'].astype(str))
            feature_cols.append('season_encoded')
        
        # Province name encoding (if available)
        if 'province_name' in df.columns:
            le_province = LabelEncoder()
            df['province_name_encoded'] = le_province.fit_transform(df['province_name'].astype(str))
            feature_cols.append('province_name_encoded')
        
        # Municipality name encoding (if available)
        if 'municipality_name' in df.columns:
            le_municipality = LabelEncoder()
            df['municipality_name_encoded'] = le_municipality.fit_transform(df['municipality_name'].astype(str))
            feature_cols.append('municipality_name_encoded')
        
        # Interaction features (multiplicative relationships)
        if 'person_no' in df.columns and 'fam_no' in df.columns:
            df['person_fam_ratio'] = df['person_no'] / (df['fam_no'] + 1)  # Persons per family
            feature_cols.append('person_fam_ratio')
            df['person_fam_product'] = df['person_no'] * df['fam_no']
            feature_cols.append('person_fam_product')
        
        # Quantity-based features (if quantity exists)
        if 'quantity' in df.columns:
            if 'person_no' in df.columns:
                df['quantity_per_person'] = df['quantity'] / (df['person_no'] + 1)
                feature_cols.append('quantity_per_person')
            if 'fam_no' in df.columns:
                df['quantity_per_family'] = df['quantity'] / (df['fam_no'] + 1)
                feature_cols.append('quantity_per_family')
        
        # Time-based features
        if 'month' in df.columns:
            # Cyclical encoding for month (captures seasonality)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            feature_cols.extend(['month_sin', 'month_cos'])
            
            # Quarter encoding
            df['quarter'] = ((df['month'] - 1) // 3) + 1
            feature_cols.append('quarter')
        
        # Year-based features
        if 'year' in df.columns:
            # Years since a reference year (normalize year)
            if df['year'].min() > 0:
                df['years_normalized'] = df['year'] - df['year'].min()
                feature_cols.append('years_normalized')
        
        # Historical aggregated features (mean assistance by various groups)
        # Note: These use full dataset statistics - for production, calculate on training set only
        # Province-level averages
        if 'provinceid' in df.columns and 'total_amount' in df.columns:
            province_avg = df.groupby('provinceid')['total_amount'].mean().to_dict()
            df['province_avg_assistance'] = df['provinceid'].map(province_avg).fillna(0)
            feature_cols.append('province_avg_assistance')
        
        # Disaster type averages
        if 'disaster_name' in df.columns and 'total_amount' in df.columns:
            disaster_avg = df.groupby('disaster_name')['total_amount'].mean().to_dict()
            df['disaster_avg_assistance'] = df['disaster_name'].map(disaster_avg).fillna(0)
            feature_cols.append('disaster_avg_assistance')
        
        # Year-month combinations
        if 'year' in df.columns and 'month' in df.columns:
            df['year_month'] = df['year'] * 12 + df['month']
            feature_cols.append('year_month')
        
        # Disaster severity indicators (if person_no and fam_no are high, disaster is severe)
        if 'person_no' in df.columns:
            df['severity_person'] = pd.cut(df['person_no'], bins=5, labels=False).fillna(0)
            feature_cols.append('severity_person')
        
        if 'fam_no' in df.columns:
            df['severity_family'] = pd.cut(df['fam_no'], bins=5, labels=False).fillna(0)
            feature_cols.append('severity_family')
    
    # Prepare X and y
    # Remove any feature columns that don't exist or are the target
    feature_cols = [col for col in feature_cols if col in df.columns and col != 'total_amount']
    X = df[feature_cols].copy()
    y = df['total_amount'].copy()
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], 0)
    y = y.replace([np.inf, -np.inf], 0)
    
    # Remove outliers from target variable
    outlier_info = {'removed': 0, 'percentage': 0.0}
    if remove_outliers and len(y) > 0:
        if OUTLIER_METHOD == 'iqr':
            outlier_mask = remove_outliers_iqr(y, OUTLIER_IQR_MULTIPLIER)
        else:  # zscore
            outlier_mask = remove_outliers_zscore(y, OUTLIER_Z_THRESHOLD)
        
        removed_count = (~outlier_mask).sum()
        if removed_count > 0:
            outlier_info['removed'] = removed_count
            outlier_info['percentage'] = removed_count / len(df) * 100
            X = X[outlier_mask].copy()
            y = y[outlier_mask].copy()
    
    # Handle missing values (use median for numeric, 0 for others)
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)
        else:
            X[col] = X[col].fillna(0)
    
    y = y.fillna(0)
    
    # Feature scaling
    scaler = None
    if USE_FEATURE_SCALING:
        scaler = StandardScaler()
        # Only scale numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Log transform target variable if enabled
    log_transform_applied = False
    if USE_LOG_TRANSFORM and (y > 0).all():
        # Only apply if all values are positive
        y = np.log1p(y)  # log1p to handle zeros: log(1+x)
        log_transform_applied = True
    
    return X, y, le_disaster, feature_cols, scaler, log_transform_applied, outlier_info


def train_models(X, y, scaler=None, log_transform_applied=False):
    """Train and compare multiple models with train/validation/test splits"""
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Second split: separate train and validation from remaining data
    val_size_relative = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_relative, random_state=RANDOM_STATE
    )
    
    # Prepare models with optimized hyperparameters
    models = {
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Hist Gradient Boosting Regressor': HistGradientBoostingRegressor(
            max_iter=HGB_MAX_ITER,
            learning_rate=HGB_LEARNING_RATE,
            random_state=RANDOM_STATE,
            early_stopping=HGB_EARLY_STOPPING if HGB_EARLY_STOPPING else False,
            validation_fraction=HGB_VALIDATION_FRACTION if HGB_EARLY_STOPPING else None,
            n_iter_no_change=HGB_N_ITER_NO_CHANGE if HGB_EARLY_STOPPING else None
        ),
        'Ridge Regression': Ridge(alpha=RIDGE_ALPHA)  # Regularized linear regression
    }
    
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predictions on all three sets
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # If log transform was applied, convert predictions back to original scale for RMSE
        if log_transform_applied:
            y_train_orig = np.expm1(y_train)
            y_val_orig = np.expm1(y_val)
            y_test_orig = np.expm1(y_test)
            y_pred_train_orig = np.expm1(y_pred_train)
            y_pred_val_orig = np.expm1(y_pred_val)
            y_pred_test_orig = np.expm1(y_pred_test)
        else:
            y_train_orig = y_train
            y_val_orig = y_val
            y_test_orig = y_test
            y_pred_train_orig = y_pred_train
            y_pred_val_orig = y_pred_val
            y_pred_test_orig = y_pred_test
        
        # Calculate metrics for each set (using original scale for RMSE)
        r2_train = r2_score(y_train, y_pred_train)
        r2_val = r2_score(y_val, y_pred_val)
        r2_test = r2_score(y_test, y_pred_test)
        
        rmse_train = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
        rmse_val = np.sqrt(mean_squared_error(y_val_orig, y_pred_val_orig))
        rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
        
        results[name] = {
            'model': model,
            # Training metrics
            'r2_train': r2_train,
            'rmse_train': rmse_train,
            # Validation metrics
            'r2_val': r2_val,
            'rmse_val': rmse_val,
            # Test metrics (for backward compatibility)
            'r2': r2_test,
            'rmse': rmse_test,
            'y_pred': y_pred_test_orig,
            'y_test': y_test_orig,
            # Additional data for analysis
            'y_pred_train': y_pred_train_orig,
            'y_train': y_train_orig,
            'y_pred_val': y_pred_val_orig,
            'y_val': y_val_orig
        }
    
    return results, X_train, X_test, y_train, y_test, X_val, y_val


def validate_training_data(data):
    """Check if there's enough data for model training"""
    merged_clean = data['merged'].copy()
    merged_clean = merged_clean[merged_clean['total_amount'] > 0]
    return len(merged_clean) >= MIN_RECORDS_FOR_TRAINING, merged_clean

