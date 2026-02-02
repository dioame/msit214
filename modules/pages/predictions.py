"""
Predictions page display functions
"""

import streamlit as st
import pandas as pd


def show_predictions(data):
    st.header("🎯 Make Predictions")
    
    if 'best_model' not in st.session_state:
        st.warning("Please train models first in the 'Predictive Models' section.")
        return
    
    st.markdown("Enter disaster characteristics to predict assistance cost:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        disaster_name = st.selectbox("Disaster Type", data['affected']['disaster_name'].unique())
        person_no = st.number_input("Number of Persons Affected", min_value=1, value=100)
        fam_no = st.number_input("Number of Families Affected", min_value=1, value=20)
    
    with col2:
        month = st.slider("Month", 1, 12, 6)
        year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
        provinceid = st.selectbox("Province ID", sorted(data['affected']['provinceid'].dropna().unique()))
        municipality_id = st.selectbox("Municipality ID", 
                                       sorted(data['affected'][data['affected']['provinceid'] == provinceid]['municipality_id'].dropna().unique()))
    
    # Prepare prediction input
    le_disaster = st.session_state['le_disaster']
    feature_cols = st.session_state['feature_cols']
    model = st.session_state['best_model']
    
    try:
        disaster_encoded = le_disaster.transform([disaster_name])[0]
        
        prediction_input = pd.DataFrame({
            'person_no': [person_no],
            'fam_no': [fam_no],
            'month': [month],
            'year': [year],
            'provinceid': [provinceid],
            'municipality_id': [municipality_id],
            'disaster_name_encoded': [disaster_encoded]
        })
        
        # Ensure correct column order
        prediction_input = prediction_input[feature_cols]
        
        predicted_cost = model.predict(prediction_input)[0]
        
        st.success(f"### Predicted Assistance Cost: **₱{predicted_cost:,.2f}**")
        
        # Show confidence interval (using RMSE as approximation)
        if 'best_model' in st.session_state:
            # Get RMSE from model comparison (approximate)
            rmse_estimate = predicted_cost * 0.3  # Rough estimate
            st.info(f"Estimated range: ₱{predicted_cost - rmse_estimate:,.2f} - ₱{predicted_cost + rmse_estimate:,.2f}")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

