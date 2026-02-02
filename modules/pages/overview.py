"""
Overview page display functions
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


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

