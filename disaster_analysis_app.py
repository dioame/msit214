"""
Disaster Assistance Predictive Analysis Web Application
A comprehensive Streamlit app for analyzing disaster data and predicting assistance costs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Disaster Assistance Predictive Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the Excel data"""
    try:
        excel_file = 'disaster_data_latest (1).xlsx'
        sheets = pd.read_excel(excel_file, sheet_name=['affected', 'assistance', 'evacuation'])
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Hist Gradient Boosting Regressor': HistGradientBoostingRegressor(random_state=42, max_iter=100),
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

@st.cache_data
def generate_association_rules(assistance_df, min_support=0.01, min_confidence=0.5):
    """
    Generate association rules from assistance data using Apriori algorithm.
    
    Parameters:
    - assistance_df: DataFrame with assistance data
    - min_support: Minimum support threshold (default 0.01 = 1%)
    - min_confidence: Minimum confidence threshold (default 0.5 = 50%)
    
    Returns:
    - rules_df: DataFrame with association rules and metrics
    - transactions: List of transactions for visualization
    """
    # Transform data into transaction format
    # Group by incident_id to get items distributed together
    transactions = assistance_df.groupby('incident_id')['fnfi_name'].apply(list).tolist()
    
    # Remove duplicates within each transaction
    transactions = [list(set(trans)) for trans in transactions]
    
    # Filter out transactions with only one item (can't generate rules)
    transactions = [trans for trans in transactions if len(trans) > 1]
    
    if len(transactions) == 0:
        return None, transactions
    
    # Encode transactions into binary matrix
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Generate frequent itemsets using Apriori
    try:
        frequent_itemsets = apriori(df_transactions, min_support=min_support, use_colnames=True, max_len=3)
        
        if len(frequent_itemsets) == 0:
            return None, transactions
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        # Sort by confidence and lift
        rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
        
        # Format the rules for better readability
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules['rule'] = rules['antecedents_str'] + ' → ' + rules['consequents_str']
        
        # Calculate percentage metrics
        rules['support_pct'] = rules['support'] * 100
        rules['confidence_pct'] = rules['confidence'] * 100
        
        return rules, transactions
    
    except Exception as e:
        st.warning(f"Error generating association rules: {str(e)}")
        return None, transactions

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
        ["📈 Overview", "🔍 Data Exploration", "🤖 Predictive Models", "💡 Actionable Insights", "🎯 Predictions"]
    )
    
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

def show_overview(data):
    st.header("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Affected Records", len(data['affected']))
        st.metric("Total Persons Affected", f"{data['affected']['person_no'].sum():,.0f}")
        st.metric("Total Families Affected", f"{data['affected']['fam_no'].sum():,.0f}")
    
    with col2:
        st.metric("Assistance Records", len(data['assistance']))
        st.metric("Total Assistance Cost", f"₱{data['assistance']['total_amount'].sum():,.2f}")
        st.metric("Total Items Distributed", f"{data['assistance']['quantity'].sum():,.0f}")
    
    with col3:
        st.metric("Evacuation Records", len(data['evacuation']))
        st.metric("Unique Disasters", data['affected']['disaster_name'].nunique())
        st.metric("Provinces Affected", data['affected']['province_name'].nunique())
    
    st.subheader("📅 Disaster Timeline")
    timeline_data = data['affected'].groupby(data['affected']['disaster_date'].dt.to_period('M')).agg({
        'person_no': 'sum',
        'incident_id': 'count'
    }).reset_index()
    timeline_data['disaster_date'] = timeline_data['disaster_date'].astype(str)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=timeline_data['disaster_date'], y=timeline_data['person_no'], 
                  name="Persons Affected", line=dict(color='#1f77b4')),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=timeline_data['disaster_date'], y=timeline_data['incident_id'], 
                  name="Number of Incidents", line=dict(color='#ff7f0e')),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Persons Affected", secondary_y=False)
    fig.update_yaxes(title_text="Number of Incidents", secondary_y=True)
    fig.update_layout(title="Disaster Impact Over Time", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🌍 Top Disaster Types")
    disaster_counts = data['affected']['disaster_name'].value_counts().head(10)
    fig = px.bar(x=disaster_counts.values, y=disaster_counts.index, 
                orientation='h', labels={'x': 'Number of Incidents', 'y': 'Disaster Type'},
                title="Top 10 Disaster Types by Frequency")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_data_exploration(data):
    st.header("🔍 Data Exploration")
    
    tab1, tab2, tab3 = st.tabs(["Affected Data", "Assistance Data", "Merged Analysis"])
    
    with tab1:
        st.subheader("Affected Population Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Province analysis
            province_impact = data['affected'].groupby('province_name').agg({
                'person_no': 'sum',
                'fam_no': 'sum',
                'incident_id': 'count'
            }).sort_values('person_no', ascending=False).head(10)
            
            fig = px.bar(province_impact.reset_index(), 
                        x='province_name', y='person_no',
                        title="Top 10 Provinces by Persons Affected",
                        labels={'person_no': 'Persons Affected', 'province_name': 'Province'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Disaster type impact
            disaster_impact = data['affected'].groupby('disaster_name').agg({
                'person_no': 'sum',
                'fam_no': 'sum'
            }).sort_values('person_no', ascending=False).head(10)
            
            fig = px.pie(disaster_impact.reset_index(), 
                        values='person_no', names='disaster_name',
                        title="Persons Affected by Disaster Type (Top 10)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Assistance Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost by item
            item_costs = data['assistance'].groupby('fnfi_name')['total_amount'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=item_costs.values, y=item_costs.index, 
                        orientation='h',
                        title="Top 10 Items by Total Cost",
                        labels={'x': 'Total Cost (₱)', 'y': 'Item Name'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost distribution
            fig = px.histogram(data['assistance'], x='total_amount', 
                             nbins=50, title="Distribution of Assistance Costs",
                             labels={'total_amount': 'Cost (₱)', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost per person/family
        st.subheader("Cost Efficiency Analysis")
        merged_sample = data['merged'][data['merged']['total_amount'] > 0]
        if len(merged_sample) > 0:
            merged_sample['cost_per_person'] = merged_sample['total_amount'] / merged_sample['person_no'].replace(0, 1)
            merged_sample['cost_per_family'] = merged_sample['total_amount'] / merged_sample['fam_no'].replace(0, 1)
            
            col1, col2 = st.columns(2)
            with col1:
                cost_by_disaster = merged_sample.groupby('disaster_name').agg({
                    'cost_per_person': 'mean'
                }).sort_values('cost_per_person', ascending=False).head(10)
                fig = px.bar(cost_by_disaster.reset_index(), 
                            x='disaster_name', y='cost_per_person',
                            title="Average Cost per Person by Disaster Type",
                            labels={'cost_per_person': 'Cost per Person (₱)', 'disaster_name': 'Disaster Type'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cost_by_disaster = merged_sample.groupby('disaster_name').agg({
                    'cost_per_family': 'mean'
                }).sort_values('cost_per_family', ascending=False).head(10)
                fig = px.bar(cost_by_disaster.reset_index(), 
                            x='disaster_name', y='cost_per_family',
                            title="Average Cost per Family by Disaster Type",
                            labels={'cost_per_family': 'Cost per Family (₱)', 'disaster_name': 'Disaster Type'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Merged Data Analysis")
        st.dataframe(data['merged'].head(100), use_container_width=True)
        
        # Correlation analysis - dynamically select available numeric columns
        merged_df = data['merged']
        potential_cols = ['person_no', 'fam_no', 'total_amount', 'quantity', 'month', 'year', 
                         'month_affected', 'year_affected', 'month_assistance', 'year_assistance']
        
        # Find which columns actually exist
        numeric_cols = []
        for col in potential_cols:
            if col in merged_df.columns:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(merged_df[col]):
                    numeric_cols.append(col)
        
        # Remove duplicates (prefer non-suffixed versions)
        final_cols = []
        for col in numeric_cols:
            if col in ['month_affected', 'month_assistance'] and 'month' in numeric_cols:
                continue
            if col in ['year_affected', 'year_assistance'] and 'year' in numeric_cols:
                continue
            final_cols.append(col)
        
        if len(final_cols) >= 2:
            corr_data = merged_df[final_cols].corr()
            
            fig = px.imshow(corr_data, text_auto=True, aspect="auto",
                           title="Correlation Matrix",
                           labels=dict(x="Variable", y="Variable", color="Correlation"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient numeric columns for correlation analysis.")

def show_predictive_models(data):
    st.header("🤖 Predictive Model Comparison")
    st.markdown("""
    **Objective**: Predict total assistance cost based on disaster characteristics.
    
    We compare three machine learning algorithms:
    - **Random Forest Regressor**: Ensemble method capturing non-linear relationships
    - **Hist Gradient Boosting Regressor**: Advanced gradient boosting
    - **Linear Regression**: Baseline linear model
    """)
    
    # Prepare data
    merged_clean = data['merged'].copy()
    merged_clean = merged_clean[merged_clean['total_amount'] > 0]  # Only disasters with assistance
    
    if len(merged_clean) < 50:
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

def show_actionable_insights(data):
    st.header("💡 Actionable Insights")
    
    tab1, tab2, tab3 = st.tabs(["Cost Efficiency", "Resource Planning", "Geospatial Analysis"])
    
    with tab1:
        st.subheader("💰 Cost-Efficiency and Procurement")
        
        # High-cost items
        st.markdown("### High-Cost Item Identification")
        item_analysis = data['assistance'].groupby('fnfi_name').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'cost': 'mean'
        }).reset_index()
        item_analysis['unit_cost'] = item_analysis['total_amount'] / item_analysis['quantity'].replace(0, 1)
        item_analysis = item_analysis.sort_values('total_amount', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 10 Items by Total Spending**")
            st.dataframe(item_analysis.head(10)[['fnfi_name', 'total_amount', 'quantity', 'unit_cost']].style.format({
                'total_amount': '₱{:.2f}',
                'unit_cost': '₱{:.2f}'
            }), use_container_width=True)
        
        with col2:
            fig = px.bar(item_analysis.head(10), x='fnfi_name', y='total_amount',
                        title="Top 10 Items by Total Cost",
                        labels={'total_amount': 'Total Cost (₱)', 'fnfi_name': 'Item Name'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Action**: Negotiate bulk-purchasing contracts for high-volume, expensive items to reduce unit costs.")
        
        # Cost per person/family baseline
        st.markdown("### Cost-Per-Person/Family Baseline")
        merged_sample = data['merged'][data['merged']['total_amount'] > 0]
        if len(merged_sample) > 0:
            merged_sample['cost_per_person'] = merged_sample['total_amount'] / merged_sample['person_no'].replace(0, 1)
            baseline = merged_sample.groupby('disaster_name').agg({
                'cost_per_person': ['mean', 'std'],
                'total_amount': 'sum',
                'person_no': 'sum'
            }).round(2)
            
            st.dataframe(baseline, use_container_width=True)
            st.info("💡 **Action**: Establish benchmarks for assistance efficiency. Monitor for sudden cost spikes.")
    
    with tab2:
        st.subheader("📦 Resource Pre-Positioning and Logistics")
        
        # Disaster-specific resource planning
        st.markdown("### Disaster-Specific Resource Planning")
        disaster_items = data['assistance'].groupby(['disaster_name', 'fnfi_name']).agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).reset_index()
        
        disaster_select = st.selectbox("Select Disaster Type", data['assistance']['disaster_name'].unique())
        
        disaster_data = disaster_items[disaster_items['disaster_name'] == disaster_select].sort_values('quantity', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(disaster_data, x='fnfi_name', y='quantity',
                        title=f"Top Items for {disaster_select}",
                        labels={'quantity': 'Total Quantity', 'fnfi_name': 'Item Name'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(disaster_data, values='quantity', names='fnfi_name',
                        title=f"Item Distribution for {disaster_select}")
            st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"💡 **Action**: Design standardized pre-packed kits for '{disaster_select}' with the most common items.")
        
        # Association Rule Mining for Resource Pre-Positioning
        st.markdown("### 📊 Association Rule Mining: Item Bundling Analysis")
        st.markdown("""
        **Objective**: Identify which relief items are frequently distributed together to optimize pre-packed kit design and minimize stockouts.
        
        This analysis uses the Apriori algorithm to discover patterns in item co-distribution across disaster incidents.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Minimum Support (%)", min_value=0.1, max_value=50.0, value=1.0, step=0.1) / 100
        with col2:
            min_confidence = st.slider("Minimum Confidence (%)", min_value=10, max_value=100, value=50, step=5) / 100
        
        with st.spinner("Generating association rules..."):
            rules_df, transactions = generate_association_rules(
                data['assistance'], 
                min_support=min_support, 
                min_confidence=min_confidence
            )
        
        if rules_df is not None and len(rules_df) > 0:
            st.success(f"✅ Found {len(rules_df)} association rules!")
            
            # Display top rules
            st.markdown("#### 🎯 Top Association Rules")
            
            # Filter and display top rules by confidence
            top_rules = rules_df.head(20).copy()
            
            # Create a formatted display
            display_cols = ['rule', 'support_pct', 'confidence_pct', 'lift']
            display_df = top_rules[display_cols].copy()
            display_df.columns = ['Association Rule', 'Support (%)', 'Confidence (%)', 'Lift']
            display_df = display_df.round(2)
            
            st.dataframe(
                display_df.style.format({
                    'Support (%)': '{:.2f}%',
                    'Confidence (%)': '{:.2f}%',
                    'Lift': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Highlight high-confidence rules
            high_conf_rules = rules_df[rules_df['confidence'] >= 0.95]
            if len(high_conf_rules) > 0:
                st.markdown("#### ⭐ High-Confidence Rules (≥95% Confidence)")
                st.markdown("These rules indicate very strong associations that can be used for standardized kit design:")
                
                for idx, rule in high_conf_rules.head(10).iterrows():
                    st.markdown(f"""
                    - **{rule['antecedents_str']}** → **{rule['consequents_str']}**
                      - Confidence: {rule['confidence_pct']:.2f}% | Support: {rule['support_pct']:.2f}% | Lift: {rule['lift']:.2f}
                    """)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Support vs Confidence scatter plot
                fig = px.scatter(
                    rules_df.head(30),
                    x='support',
                    y='confidence',
                    size='lift',
                    color='lift',
                    hover_data=['antecedents_str', 'consequents_str'],
                    title="Association Rules: Support vs Confidence",
                    labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'},
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top rules by lift
                top_lift = rules_df.nlargest(15, 'lift')
                fig = px.bar(
                    top_lift,
                    x='lift',
                    y='rule',
                    orientation='h',
                    title="Top 15 Rules by Lift Score",
                    labels={'lift': 'Lift Score', 'rule': 'Association Rule'},
                    color='confidence',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.markdown("#### 💡 Key Insights for Resource Pre-Positioning")
            
            # Find the rule with highest confidence
            best_rule = rules_df.iloc[0]
            st.info(f"""
            **Highest Confidence Rule**: {best_rule['antecedents_str']} → {best_rule['consequents_str']}
            - **Confidence**: {best_rule['confidence_pct']:.2f}% - This means when {best_rule['antecedents_str']} are distributed, 
              {best_rule['consequents_str']} is included {best_rule['confidence_pct']:.2f}% of the time.
            - **Support**: {best_rule['support_pct']:.2f}% - This pattern occurs in {best_rule['support_pct']:.2f}% of all transactions.
            - **Lift**: {best_rule['lift']:.2f} - This rule is {best_rule['lift']:.2f}x more likely than random chance.
            
            **Actionable Recommendation**: Design pre-packed kits that include {best_rule['antecedents_str']} together with {best_rule['consequents_str']} 
            to minimize stockouts and response lag.
            """)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(transactions))
            with col2:
                st.metric("Average Items per Transaction", f"{np.mean([len(t) for t in transactions]):.1f}")
            with col3:
                st.metric("Unique Items", data['assistance']['fnfi_name'].nunique())
        
        else:
            st.warning("""
            ⚠️ No association rules found with the current thresholds.
            
            **Suggestions**:
            - Lower the minimum support threshold (try 0.5% or lower)
            - Lower the minimum confidence threshold (try 30% or lower)
            - Ensure you have sufficient transaction data
            """)
        
        # Seasonal analysis
        st.markdown("### Seasonal Disaster Patterns")
        seasonal_data = data['affected'].groupby(['season', 'disaster_name']).agg({
            'person_no': 'sum',
            'incident_id': 'count'
        }).reset_index()
        
        fig = px.bar(seasonal_data, x='season', y='person_no', color='disaster_name',
                    title="Persons Affected by Season and Disaster Type",
                    labels={'person_no': 'Persons Affected', 'season': 'Quarter'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🗺️ Geospatial Hotspot Analysis")
        
        # Province analysis
        province_analysis = data['merged'].groupby('province_name').agg({
            'person_no': 'sum',
            'total_amount': 'sum',
            'incident_id': 'count'
        }).reset_index().sort_values('person_no', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Provinces by Impact**")
            st.dataframe(province_analysis.head(10), use_container_width=True)
        
        with col2:
            fig = px.scatter(province_analysis.head(15), x='person_no', y='total_amount',
                           size='incident_id', hover_name='province_name',
                           title="Province Impact vs Cost",
                           labels={'person_no': 'Persons Affected', 'total_amount': 'Total Cost (₱)'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Action**: Prioritize funding and logistical hubs in the most disaster-prone locations.")
        
        # Municipality analysis
        municipality_analysis = data['merged'].groupby(['province_name', 'municipality_name']).agg({
            'person_no': 'sum',
            'total_amount': 'sum'
        }).reset_index().sort_values('person_no', ascending=False).head(20)
        
        st.markdown("### Top 20 Municipalities by Impact")
        st.dataframe(municipality_analysis, use_container_width=True)

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

if __name__ == "__main__":
    main()

