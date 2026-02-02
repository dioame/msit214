"""
Data exploration page display functions
"""

import streamlit as st
import pandas as pd
import plotly.express as px


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

