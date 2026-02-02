"""
Predictive models page display functions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from modules.models import prepare_model_data, train_models, validate_training_data


def show_predictive_models(data):
    st.header("🤖 Predictive Model Comparison")
    st.markdown("""
    **Objective**: Predict total assistance cost based on disaster characteristics.
    
    We compare three machine learning algorithms:
    - **Random Forest Regressor**: Ensemble method capturing non-linear relationships (optimized to prevent overfitting)
    - **Hist Gradient Boosting Regressor**: Advanced gradient boosting with early stopping
    - **Ridge Regression**: Regularized linear regression
    """)
    
    # Validate and prepare data
    has_enough_data, merged_clean = validate_training_data(data)
    
    if not has_enough_data:
        st.warning("Insufficient data for model training. Need at least 50 records with assistance data.")
        return
    
    # Show optimization info
    st.info("🔧 **Optimizations Applied**: Outlier removal, feature scaling, log transformation, and reduced model complexity to prevent overfitting.")
    
    X, y, le_disaster, feature_cols, scaler, log_transform_applied, outlier_info = prepare_model_data(merged_clean)
    
    # Display data preprocessing info
    if outlier_info['removed'] > 0:
        st.success(f"✅ Removed {outlier_info['removed']} outliers ({outlier_info['percentage']:.1f}% of data) to reduce noise")
    if log_transform_applied:
        st.success("✅ Applied log transformation to target variable to handle skewness")
    if scaler is not None:
        st.success("✅ Applied feature scaling for better model performance")
    
    with st.spinner("Training models with optimized hyperparameters..."):
        results, X_train, X_test, y_train, y_test, X_val, y_val = train_models(X, y, scaler, log_transform_applied)
    
    # Display results
    st.subheader("📊 Model Performance Comparison")
    
    # Create comparison table for train/validation/test metrics
    st.subheader("📋 Accuracy Comparison: Training vs Validation vs Test")
    
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Train R²': f"{result['r2_train']:.4f}",
            'Validation R²': f"{result['r2_val']:.4f}",
            'Test R²': f"{result['r2']:.4f}",
            'Train RMSE': f"₱{result['rmse_train']:,.2f}",
            'Validation RMSE': f"₱{result['rmse_val']:,.2f}",
            'Test RMSE': f"₱{result['rmse']:,.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Add insights about overfitting
    st.markdown("#### 🔍 Model Analysis")
    for name, result in results.items():
        train_r2 = result['r2_train']
        val_r2 = result['r2_val']
        test_r2 = result['r2']
        
        # Check for overfitting (large gap between train and validation/test)
        train_val_gap = train_r2 - val_r2
        train_test_gap = train_r2 - test_r2
        
        if train_val_gap > 0.1 or train_test_gap > 0.1:
            st.warning(f"⚠️ **{name}**: Potential overfitting detected. Train R² ({train_r2:.4f}) is significantly higher than Validation R² ({val_r2:.4f}) or Test R² ({test_r2:.4f}).")
        elif abs(train_val_gap) < 0.05 and abs(train_test_gap) < 0.05:
            st.success(f"✅ **{name}**: Good generalization. Train, Validation, and Test R² scores are similar.")
        else:
            st.info(f"ℹ️ **{name}**: Train R²: {train_r2:.4f}, Validation R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
    
    col1, col2, col3 = st.columns(3)
    
    # Sort by R2 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_results):
        with [col1, col2, col3][i]:
            st.metric(
                name,
                f"R²: {result['r2']:.2%}",
                f"RMSE: ₱{result['rmse']:,.2f}"
            )
    
    # Best model highlight
    best_model_name = sorted_results[0][0]
    best_model = sorted_results[0][1]
    
    st.success(f"🏆 **Best Model: {best_model_name}** - R² Score: {best_model['r2']:.2%}, RMSE: ₱{best_model['rmse']:,.2f}")
    
    # Visualization
    st.subheader("📈 Prediction vs Actual (Test Set)")
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(results.keys()))
    
    for idx, (name, result) in enumerate(results.items(), 1):
        fig.add_trace(
            go.Scatter(
                x=result['y_test'],
                y=result['y_pred'],
                mode='markers',
                name=name,
                showlegend=False
            ),
            row=1, col=idx
        )
        # Perfect prediction line
        max_val = max(result['y_test'].max(), result['y_pred'].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Perfect Prediction',
                showlegend=(idx == 1)
            ),
            row=1, col=idx
        )
        fig.update_xaxes(title_text="Actual Cost (₱)", row=1, col=idx)
        fig.update_yaxes(title_text="Predicted Cost (₱)", row=1, col=idx)
    
    fig.update_layout(height=400, title_text="Model Predictions Comparison")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (for Random Forest)
    if 'Random Forest Regressor' in results:
        st.subheader("🔍 Feature Importance (Random Forest)")
        rf_model = results['Random Forest Regressor']['model']
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance, x='importance', y='feature', 
                    orientation='h', title="Feature Importance",
                    labels={'importance': 'Importance Score', 'feature': 'Feature'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Store best model in session state
    st.session_state['best_model'] = best_model['model']
    st.session_state['le_disaster'] = le_disaster
    st.session_state['feature_cols'] = feature_cols
    st.session_state['scaler'] = scaler
    st.session_state['log_transform_applied'] = log_transform_applied

