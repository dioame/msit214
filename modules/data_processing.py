"""
Data loading, cleaning, and preprocessing functions
"""

import pandas as pd
import streamlit as st
from modules.config import EXCEL_FILE, SHEET_NAMES


@st.cache_data
def load_data():
    """Load and cache the Excel data"""
    try:
        sheets = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAMES)
        return sheets
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_data
def clean_and_prepare_data(sheets):
    """Clean dates, merge data, and create features"""
    affected = sheets['affected'].copy()
    assistance = sheets['assistance'].copy()
    evacuation = sheets['evacuation'].copy()
    
    # Clean disaster_date in affected
    affected['disaster_date'] = pd.to_datetime(affected['disaster_date'], errors='coerce')
    affected = affected.dropna(subset=['disaster_date'])
    
    # Clean disaster_date in assistance (handle Excel serial dates)
    def convert_date(date_val):
        if pd.isna(date_val):
            return pd.NaT
        if isinstance(date_val, (int, float)):
            # Excel serial date
            return pd.to_datetime('1899-12-30') + pd.Timedelta(days=int(date_val))
        return pd.to_datetime(date_val, errors='coerce')
    
    assistance['disaster_date'] = assistance['disaster_date'].apply(convert_date)
    assistance = assistance.dropna(subset=['disaster_date'])
    
    # Feature engineering for affected
    affected['year'] = affected['disaster_date'].dt.year
    affected['month'] = affected['disaster_date'].dt.month
    affected['season'] = affected['month'].apply(lambda x: 
        'Q1' if x in [1,2,3] else 'Q2' if x in [4,5,6] else 'Q3' if x in [7,8,9] else 'Q4')
    
    # Feature engineering for assistance
    assistance['year'] = assistance['disaster_date'].dt.year
    assistance['month'] = assistance['disaster_date'].dt.month
    assistance['season'] = assistance['month'].apply(lambda x: 
        'Q1' if x in [1,2,3] else 'Q2' if x in [4,5,6] else 'Q3' if x in [7,8,9] else 'Q4')
    
    # Aggregate assistance by incident
    assistance_agg = assistance.groupby(['incident_id', 'disaster_name', 'disaster_date', 
                                         'provinceid', 'municipality_id', 'year', 'month', 'season']).agg({
        'total_amount': 'sum',
        'quantity': 'sum',
        'fnfi_name': lambda x: ', '.join(x.unique()[:5])  # Top items
    }).reset_index()
    
    # Merge affected with assistance
    merged = affected.merge(
        assistance_agg,
        on=['incident_id', 'disaster_name', 'disaster_date', 'provinceid', 'municipality_id'],
        how='left',
        suffixes=('_affected', '_assistance')
    )
    
    # Fill missing assistance data with 0
    merged['total_amount'] = merged['total_amount'].fillna(0)
    merged['quantity'] = merged['quantity'].fillna(0)
    
    # Ensure unified year, month, and season columns exist (prefer affected version)
    if 'year_affected' in merged.columns:
        merged['year'] = merged['year_affected']
    elif 'year_assistance' in merged.columns:
        merged['year'] = merged['year_assistance']
    elif 'year' in merged.columns:
        pass  # Already exists
    else:
        # Extract from disaster_date if available
        merged['year'] = merged['disaster_date'].dt.year
        merged['month'] = merged['disaster_date'].dt.month
    
    if 'month_affected' in merged.columns:
        merged['month'] = merged['month_affected']
    elif 'month_assistance' in merged.columns:
        merged['month'] = merged['month_assistance']
    elif 'month' not in merged.columns:
        # Extract from disaster_date if available
        if 'disaster_date' in merged.columns:
            merged['month'] = merged['disaster_date'].dt.month
    
    if 'season_affected' in merged.columns:
        merged['season'] = merged['season_affected']
    elif 'season_assistance' in merged.columns:
        merged['season'] = merged['season_assistance']
    elif 'season' not in merged.columns and 'month' in merged.columns:
        merged['season'] = merged['month'].apply(lambda x: 
            'Q1' if x in [1,2,3] else 'Q2' if x in [4,5,6] else 'Q3' if x in [7,8,9] else 'Q4')
    
    return {
        'affected': affected,
        'assistance': assistance,
        'evacuation': evacuation,
        'merged': merged
    }

