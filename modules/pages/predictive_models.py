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
    - **Random Forest Regressor**: Ensemble method capturing non-linear relationships
    - **Hist Gradient Boosting Regressor**: Advanced gradient boosting
    - **Linear Regression**: Baseline linear model
    """)
    
    # Validate and prepare data
    has_enough_data, merged_clean = validate_training_data(data)
    
    if not has_enough_data:
        st.warning("Insufficient data for model training. Need at least 50 records with assistance data.")
        return
    
    X, y, le_disaster, feature_cols = prepare_model_data(merged_clean)
    
    with st.spinner("Training models..."):
        results, X_train, X_test, y_train, y_test = train_models(X, y)
    
    # Display results
    st.subheader("📊 Model Performance Comparison")
    
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

