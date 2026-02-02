"""
Association rule mining functions using Apriori algorithm
"""

import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from modules.config import DEFAULT_MIN_SUPPORT, DEFAULT_MIN_CONFIDENCE, MAX_ITEMSET_LENGTH


@st.cache_data
def generate_association_rules(assistance_df, min_support=DEFAULT_MIN_SUPPORT, min_confidence=DEFAULT_MIN_CONFIDENCE):
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
    # Filter out rows with missing incident_id or fnfi_name
    assistance_df = assistance_df.dropna(subset=['incident_id', 'fnfi_name'])
    
    # Convert fnfi_name to string and clean
    assistance_df['fnfi_name'] = assistance_df['fnfi_name'].astype(str).str.strip()
    
    # Remove empty strings, whitespace-only values, and 'nan' strings
    assistance_df = assistance_df[
        (assistance_df['fnfi_name'] != '') & 
        (assistance_df['fnfi_name'] != 'nan') &
        (assistance_df['fnfi_name'].notna())
    ]
    
    # Transform data into transaction format
    # Group by incident_id to get items distributed together
    transactions = assistance_df.groupby('incident_id')['fnfi_name'].apply(
        lambda x: list(x.unique())  # Get unique items per incident
    ).tolist()
    
    # Remove duplicates within each transaction and filter out empty transactions
    transactions = [list(set(trans)) for trans in transactions if trans and len(trans) > 0]
    
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
        frequent_itemsets = apriori(
            df_transactions, 
            min_support=min_support, 
            use_colnames=True, 
            max_len=MAX_ITEMSET_LENGTH
        )
        
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

