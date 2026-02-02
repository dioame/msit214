"""
Disaster Assistance Predictive Analysis Web Application (Refactored)
A modular Streamlit app for analyzing disaster data and predicting assistance costs.
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import modules
from modules.config import PAGE_TITLE, PAGE_ICON, LAYOUT, CSS_STYLES
from modules.data_processing import load_data, clean_and_prepare_data
from modules.pages import (
    show_overview,
    show_data_exploration,
    show_predictive_models,
    show_actionable_insights,
    show_predictions,
    show_documentation
)

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(CSS_STYLES, unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">🌊 Disaster Assistance Predictive Analysis</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        sheets = load_data()
        if sheets is None:
            st.stop()
        
        data = clean_and_prepare_data(sheets)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        ["📈 Overview", "🔍 Data Exploration", "🤖 Predictive Models", 
         "💡 Actionable Insights", "🎯 Predictions", "📄 Documentation"]
    )
    
    # Route to appropriate page
    if page == "📈 Overview":
        show_overview(data)
    elif page == "🔍 Data Exploration":
        show_data_exploration(data)
    elif page == "🤖 Predictive Models":
        show_predictive_models(data)
    elif page == "💡 Actionable Insights":
        show_actionable_insights(data)
    elif page == "🎯 Predictions":
        show_predictions(data)
    elif page == "📄 Documentation":
        show_documentation(data)


if __name__ == "__main__":
    main()

