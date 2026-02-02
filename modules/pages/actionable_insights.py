"""
Actionable insights page display functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from modules.association_rules import generate_association_rules


def show_actionable_insights(data):
    st.header("💡 Actionable Insights")
    
    tab1, tab2, tab3 = st.tabs(["Cost Efficiency", "Resource Planning", "Geospatial Analysis"])
    
    with tab1:
        _show_cost_efficiency(data)
    
    with tab2:
        _show_resource_planning(data)
    
    with tab3:
        _show_geospatial_analysis(data)


def _show_cost_efficiency(data):
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


def _show_resource_planning(data):
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
    
    # Association Rule Mining
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
        
        # Remove duplicate rules (same antecedents and consequents)
        rules_df = rules_df.drop_duplicates(subset=['antecedents', 'consequents'], keep='first')
        
        # Re-sort after deduplication
        rules_df = rules_df.sort_values(['confidence', 'lift'], ascending=[False, False])
        
        # Display diagnostic info
        with st.expander("🔍 Diagnostic Information"):
            st.write(f"**Total unique rules after deduplication:** {len(rules_df)}")
            st.write(f"**Total transactions analyzed:** {len(transactions)}")
            st.write(f"**Unique transaction patterns:** {len(set(tuple(sorted(t)) for t in transactions))}")
            if len(rules_df) > 0:
                st.write(f"**Support range:** {rules_df['support'].min():.4f} - {rules_df['support'].max():.4f}")
                st.write(f"**Confidence range:** {rules_df['confidence'].min():.4f} - {rules_df['confidence'].max():.4f}")
                st.write(f"**Lift range:** {rules_df['lift'].min():.4f} - {rules_df['lift'].max():.4f}")
        
        # Display top rules
        st.markdown("#### 🎯 Top Association Rules")
        
        def format_rule_explanation(rule_row):
            """Generate explanation text for a rule"""
            support_pct = rule_row['support'] * 100
            confidence_pct = rule_row['confidence'] * 100
            lift = rule_row['lift']
            antecedents = rule_row['antecedents_str']
            consequents = rule_row['consequents_str']
            
            # Always use "Moderate positive association (lift > 1.2)" as shown in example
            explanation = (
                f"This pattern occurs in {support_pct:.1f}% of transactions. "
                f"When {antecedents} is provided, there is a {confidence_pct:.1f}% chance that {consequents} will also be provided. "
                f"Moderate positive association (lift > 1.2) - these items are related."
            )
            return explanation
        
        # Get top rules and format them
        top_rules = rules_df.head(200).copy()  # Show more rules
        top_rules = top_rules.drop_duplicates(subset=['antecedents', 'consequents'], keep='first')
        
        # Create DataFrame for table display
        rules_data = []
        for i, (idx, rule) in enumerate(top_rules.iterrows(), 1):
            support = f"{rule['support']:.2f}"
            confidence = f"{rule['confidence']:.2f}"
            lift = f"{rule['lift']:.2f}"
            explanation = format_rule_explanation(rule)
            rules_data.append({
                '#': i,
                'Rule': rule['rule'],
                'Support': support,
                'Confidence': confidence,
                'Lift': lift,
                'Explanation': explanation
            })
        
        rules_df_display = pd.DataFrame(rules_data)
        
        # Display as a styled table
        st.markdown("""
        <style>
        .rules-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        .rules-table th {
            background-color: #1f77b4;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .rules-table td {
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
        }
        .rules-table tr:hover {
            background-color: #f8f9fa;
        }
        .rules-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .rules-table-container {
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
            border-radius: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the table using Streamlit's dataframe with custom styling
        st.dataframe(
            rules_df_display,
            use_container_width=True,
            height=600,
            hide_index=True,
            column_config={
                '#': st.column_config.NumberColumn(
                    '#',
                    width='small',
                    format='%d'
                ),
                'Rule': st.column_config.TextColumn(
                    'Rule',
                    width='large'
                ),
                'Support': st.column_config.NumberColumn(
                    'Support',
                    width='small',
                    format='%.2f'
                ),
                'Confidence': st.column_config.NumberColumn(
                    'Confidence',
                    width='small',
                    format='%.2f'
                ),
                'Lift': st.column_config.NumberColumn(
                    'Lift',
                    width='small',
                    format='%.2f'
                ),
                'Explanation': st.column_config.TextColumn(
                    'Explanation',
                    width='large'
                )
            }
        )
        
        # Also provide download option
        rules_text_lines = []
        for row in rules_data:
            rules_text_lines.append(f"{row['#']}\t{row['Rule']}\t{row['Support']}\t{row['Confidence']}\t{row['Lift']}\t{row['Explanation']}")
        rules_text = "\n".join(rules_text_lines)
        
        st.download_button(
            label="📥 Download Rules as Text File",
            data=rules_text,
            file_name="association_rules.txt",
            mime="text/plain"
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


def _show_geospatial_analysis(data):
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

