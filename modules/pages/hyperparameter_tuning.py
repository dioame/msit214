"""
Hyperparameter tuning page display functions
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from modules.models import prepare_model_data, validate_training_data
from modules.config import (
    TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE, USE_LOG_TRANSFORM
)


def test_hyperparameter_combinations(X, y, scaler=None, log_transform_applied=False):
    """Test 10 different hyperparameter combinations for each model type"""
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    val_size_relative = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_relative, random_state=RANDOM_STATE
    )
    
    all_results = []
    
    # Random Forest hyperparameter combinations (10 variations)
    rf_combinations = [
        {'n_estimators': 30, 'max_depth': 3, 'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 7, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
        {'n_estimators': 75, 'max_depth': 4, 'min_samples_split': 15, 'min_samples_leaf': 7, 'max_features': 'log2'},
        {'n_estimators': 50, 'max_depth': 6, 'min_samples_split': 8, 'min_samples_leaf': 4, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 12, 'min_samples_leaf': 6, 'max_features': 'sqrt'},
        {'n_estimators': 40, 'max_depth': 4, 'min_samples_split': 18, 'min_samples_leaf': 9, 'max_features': 'log2'},
        {'n_estimators': 80, 'max_depth': 6, 'min_samples_split': 6, 'min_samples_leaf': 3, 'max_features': 'sqrt'},
        {'n_estimators': 60, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'log2'},
        {'n_estimators': 90, 'max_depth': 7, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
    ]
    
    st.subheader("🌳 Random Forest Regressor - Hyperparameter Testing")
    rf_results = []
    
    for i, params in enumerate(rf_combinations, 1):
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Convert back to original scale if log transform was applied
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
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_val = r2_score(y_val, y_pred_val)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_val = np.sqrt(mean_squared_error(y_val_orig, y_pred_val_orig))
        rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
        
        result = {
            'Model': 'Random Forest',
            'Config': f"#{i}",
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'min_samples_leaf': params['min_samples_leaf'],
            'max_features': params['max_features'],
            'Train R²': r2_train,
            'Validation R²': r2_val,
            'Test R²': r2_test,
            'Validation RMSE': rmse_val,
            'Test RMSE': rmse_test,
            'model_obj': model
        }
        rf_results.append(result)
        all_results.append(result)
    
    # Display RF results
    rf_df = pd.DataFrame(rf_results)
    rf_display = rf_df[['Config', 'n_estimators', 'max_depth', 'min_samples_split', 
                        'min_samples_leaf', 'max_features', 'Train R²', 'Validation R²', 
                        'Test R²', 'Validation RMSE', 'Test RMSE']].copy()
    
    # Highlight best validation R²
    best_rf_idx = rf_df['Validation R²'].idxmax()
    st.dataframe(
        rf_display.style.highlight_max(subset=['Validation R²', 'Test R²'], axis=0),
        use_container_width=True,
        hide_index=True
    )
    
    best_rf = rf_results[best_rf_idx]  # Get from original list to preserve model object
    st.success(f"🏆 **Best Random Forest Config**: {best_rf['Config']} - Validation R²: {best_rf['Validation R²']:.4f}, Test R²: {best_rf['Test R²']:.4f}")
    
    # Hist Gradient Boosting hyperparameter combinations (10 variations)
    hgb_combinations = [
        {'max_iter': 100, 'learning_rate': 0.01},
        {'max_iter': 150, 'learning_rate': 0.05},
        {'max_iter': 200, 'learning_rate': 0.1},
        {'max_iter': 100, 'learning_rate': 0.001},
        {'max_iter': 250, 'learning_rate': 0.15},
        {'max_iter': 150, 'learning_rate': 0.02},
        {'max_iter': 200, 'learning_rate': 0.08},
        {'max_iter': 300, 'learning_rate': 0.1},
        {'max_iter': 120, 'learning_rate': 0.03},
        {'max_iter': 180, 'learning_rate': 0.05},
    ]
    
    st.subheader("⚡ Hist Gradient Boosting Regressor - Hyperparameter Testing")
    hgb_results = []
    
    for i, params in enumerate(hgb_combinations, 1):
        model = HistGradientBoostingRegressor(
            max_iter=params['max_iter'],
            learning_rate=params['learning_rate'],
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Convert back to original scale if log transform was applied
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
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_val = r2_score(y_val, y_pred_val)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_val = np.sqrt(mean_squared_error(y_val_orig, y_pred_val_orig))
        rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
        
        result = {
            'Model': 'Hist Gradient Boosting',
            'Config': f"#{i}",
            'max_iter': params['max_iter'],
            'learning_rate': params['learning_rate'],
            'Train R²': r2_train,
            'Validation R²': r2_val,
            'Test R²': r2_test,
            'Validation RMSE': rmse_val,
            'Test RMSE': rmse_test,
            'model_obj': model
        }
        hgb_results.append(result)
        all_results.append(result)
    
    # Display HGB results
    hgb_df = pd.DataFrame(hgb_results)
    hgb_display = hgb_df[['Config', 'max_iter', 'learning_rate', 'Train R²', 
                          'Validation R²', 'Test R²', 'Validation RMSE', 'Test RMSE']].copy()
    
    st.dataframe(
        hgb_display.style.highlight_max(subset=['Validation R²', 'Test R²'], axis=0),
        use_container_width=True,
        hide_index=True
    )
    
    best_hgb_idx = hgb_df['Validation R²'].idxmax()
    best_hgb = hgb_results[best_hgb_idx]  # Get from original list to preserve model object
    st.success(f"🏆 **Best Hist Gradient Boosting Config**: {best_hgb['Config']} - Validation R²: {best_hgb['Validation R²']:.4f}, Test R²: {best_hgb['Test R²']:.4f}")
    
    # Ridge Regression hyperparameter combinations (10 variations)
    ridge_combinations = [
        {'alpha': 0.1},
        {'alpha': 0.5},
        {'alpha': 1.0},
        {'alpha': 2.0},
        {'alpha': 5.0},
        {'alpha': 10.0},
        {'alpha': 0.01},
        {'alpha': 0.2},
        {'alpha': 3.0},
        {'alpha': 7.5},
    ]
    
    st.subheader("📊 Ridge Regression - Hyperparameter Testing")
    ridge_results = []
    
    for i, params in enumerate(ridge_combinations, 1):
        model = Ridge(alpha=params['alpha'])
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Convert back to original scale if log transform was applied
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
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_val = r2_score(y_val, y_pred_val)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_val = np.sqrt(mean_squared_error(y_val_orig, y_pred_val_orig))
        rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
        
        result = {
            'Model': 'Ridge Regression',
            'Config': f"#{i}",
            'alpha': params['alpha'],
            'Train R²': r2_train,
            'Validation R²': r2_val,
            'Test R²': r2_test,
            'Validation RMSE': rmse_val,
            'Test RMSE': rmse_test,
            'model_obj': model
        }
        ridge_results.append(result)
        all_results.append(result)
    
    # Display Ridge results
    ridge_df = pd.DataFrame(ridge_results)
    ridge_display = ridge_df[['Config', 'alpha', 'Train R²', 'Validation R²', 
                              'Test R²', 'Validation RMSE', 'Test RMSE']].copy()
    
    st.dataframe(
        ridge_display.style.highlight_max(subset=['Validation R²', 'Test R²'], axis=0),
        use_container_width=True,
        hide_index=True
    )
    
    best_ridge_idx = ridge_df['Validation R²'].idxmax()
    best_ridge = ridge_results[best_ridge_idx]  # Get from original list to preserve model object
    st.success(f"🏆 **Best Ridge Regression Config**: {best_ridge['Config']} - Validation R²: {best_ridge['Validation R²']:.4f}, Test R²: {best_ridge['Test R²']:.4f}")
    
    # Overall best model comparison
    st.subheader("🏆 Overall Best Model Comparison")
    
    best_models = [
        {'Model': 'Random Forest', 'Config': best_rf['Config'], 'Validation R²': best_rf['Validation R²'], 
         'Test R²': best_rf['Test R²'], 'Validation RMSE': best_rf['Validation RMSE'], 
         'Test RMSE': best_rf['Test RMSE'], 'model_obj': best_rf['model_obj']},
        {'Model': 'Hist Gradient Boosting', 'Config': best_hgb['Config'], 'Validation R²': best_hgb['Validation R²'], 
         'Test R²': best_hgb['Test R²'], 'Validation RMSE': best_hgb['Validation RMSE'], 
         'Test RMSE': best_hgb['Test RMSE'], 'model_obj': best_hgb['model_obj']},
        {'Model': 'Ridge Regression', 'Config': best_ridge['Config'], 'Validation R²': best_ridge['Validation R²'], 
         'Test R²': best_ridge['Test R²'], 'Validation RMSE': best_ridge['Validation RMSE'], 
         'Test RMSE': best_ridge['Test RMSE'], 'model_obj': best_ridge['model_obj']},
    ]
    
    best_models_df = pd.DataFrame(best_models)
    best_display = best_models_df[['Model', 'Config', 'Validation R²', 'Test R²', 
                                    'Validation RMSE', 'Test RMSE']].copy()
    
    st.dataframe(
        best_display.style.highlight_max(subset=['Validation R²', 'Test R²'], axis=0),
        use_container_width=True,
        hide_index=True
    )
    
    overall_best_idx = best_models_df['Validation R²'].idxmax()
    overall_best = best_models_df.loc[overall_best_idx]
    st.success(f"🎯 **Overall Best Model**: {overall_best['Model']} ({overall_best['Config']}) - Validation R²: {overall_best['Validation R²']:.4f}, Test R²: {overall_best['Test R²']:.4f}")
    
    return all_results, best_models


def show_hyperparameter_tuning(data):
    """Display hyperparameter tuning page"""
    st.header("🔧 Hyperparameter Tuning & Optimization")
    st.markdown("""
    **Objective**: Test 10 different hyperparameter combinations for each model type to find the optimal configuration.
    
    This page systematically tests various hyperparameter settings to identify the best-performing model configurations:
    - **Random Forest**: Tests combinations of n_estimators, max_depth, min_samples_split, min_samples_leaf, and max_features
    - **Hist Gradient Boosting**: Tests combinations of max_iter and learning_rate
    - **Ridge Regression**: Tests different alpha (regularization) values
    
    Models are evaluated on validation and test sets to ensure generalization.
    """)
    
    # Validate and prepare data
    has_enough_data, merged_clean = validate_training_data(data)
    
    if not has_enough_data:
        st.warning("Insufficient data for model training. Need at least 50 records with assistance data.")
        return
    
    st.info("🔧 **Note**: Using the same data preprocessing (outlier removal, feature scaling, log transformation) as the main models page.")
    
    X, y, le_disaster, feature_cols, scaler, log_transform_applied, outlier_info = prepare_model_data(merged_clean)
    
    if outlier_info['removed'] > 0:
        st.success(f"✅ Removed {outlier_info['removed']} outliers ({outlier_info['percentage']:.1f}% of data)")
    if log_transform_applied:
        st.success("✅ Applied log transformation to target variable")
    if scaler is not None:
        st.success("✅ Applied feature scaling")
    
    if st.button("🚀 Start Hyperparameter Tuning", type="primary"):
        with st.spinner("Testing 30 hyperparameter combinations (10 per model type)..."):
            all_results, best_models = test_hyperparameter_combinations(X, y, scaler, log_transform_applied)
        
        st.markdown("---")
        st.markdown("### 📋 Summary")
        st.markdown(f"✅ Tested **30 total configurations** (10 per model type)")
        st.markdown(f"✅ Best model identified based on Validation R² score")
        st.markdown(f"✅ All models evaluated on train, validation, and test sets")
        
        # Store best overall model in session state
        overall_best_idx = pd.DataFrame(best_models)['Validation R²'].idxmax()
        overall_best_model = best_models[overall_best_idx]
        st.session_state['best_tuned_model'] = overall_best_model['model_obj']
        st.session_state['best_tuned_model_name'] = overall_best_model['Model']
        st.session_state['best_tuned_model_config'] = overall_best_model['Config']
    else:
        st.info("👆 Click the button above to start hyperparameter tuning. This will test 30 different configurations across all model types.")

