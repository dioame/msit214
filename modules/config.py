"""
Configuration constants and settings for the Disaster Analysis App
"""

# File paths
EXCEL_FILE = 'disaster_data_latest (1).xlsx'
SHEET_NAMES = ['affected', 'assistance', 'evacuation']

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_RECORDS_FOR_TRAINING = 50

# Random Forest parameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# Hist Gradient Boosting parameters
HGB_MAX_ITER = 100

# Association Rule Mining defaults
DEFAULT_MIN_SUPPORT = 0.01  # 1%
DEFAULT_MIN_CONFIDENCE = 0.5  # 50%
MAX_ITEMSET_LENGTH = 3

# UI Configuration
PAGE_TITLE = "Disaster Assistance Predictive Analysis"
PAGE_ICON = "🌊"
LAYOUT = "wide"

# CSS Styles
CSS_STYLES = """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .doc-title {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .doc-section {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .doc-subsection {
        font-size: 1.2rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .doc-text {
        text-align: justify;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    </style>
"""

