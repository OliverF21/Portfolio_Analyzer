import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# IMPORT OUR NEW MODULES
from extraction import extract_holdings
from processing import create_portfolio_df
from analysis import calculate_risk_metrics

# --- CONFIG ---
st.set_page_config(page_title="Modular Portfolio Analyzer", layout="wide")
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

# --- APP UI ---
st.title("🧩 Modular Portfolio Analyzer")

uploaded_file = st.file_uploader("Upload Robinhood PDF", type="pdf")

if uploaded_file:
    # 1. EXTRACT
    with st.status("Phase 1: Extraction...", expanded=True) as status:
        if not api_key:
            st.error("API Key Missing")
            st.stop()
            
        raw_holdings = extract_holdings(uploaded_file, api_key=api_key)
        
        if not raw_holdings:
            st.error("Extraction Failed. Is the PDF a scan?")
            st.stop()
        
        status.update(label=f"✅ Extracted {len(raw_holdings)} positions", state="complete")

    # 2. PROCESS
    with st.status("Phase 2: Data Structuring...", expanded=True) as status:
        df = create_portfolio_df(raw_holdings)
        if df.empty:
            st.error("Failed to fetch market data.")
            st.stop()
        status.update(label="✅ Market Data Fetched", state="complete")

    # 3. ANALYZE
    with st.status("Phase 3: Risk Analysis...", expanded=True) as status:
        risk_df = calculate_risk_metrics(df)
        
        # Merge Risk data back into Main DF
        final_df = pd.merge(df, risk_df, on='ticker', how='left').fillna(0.0)
        
        status.update(label="✅ Risk Metrics Calculated", state="complete")

    # --- DISPLAY ---
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Holdings")
        st.dataframe(final_df[['ticker', 'quantity', 'value', 'weight']].style.format({
            "value": "${:,.2f}",
            "weight": "{:.1%}"
        }))
        
    with col2:
        st.subheader("Allocation")
        fig = px.pie(final_df, values='value', names='ticker', title="Portfolio Weight")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Risk Analysis")
    
    # Helper to calculate Weighted Sharpe safely
    weighted_sharpe = (final_df['weight'] * final_df['sharpe']).sum()
    st.metric("Portfolio Sharpe Ratio", f"{weighted_sharpe:.2f}")

    st.dataframe(
        final_df[['ticker', 'weight', 'annual_return', 'volatility', 'sharpe']]
        .sort_values('sharpe', ascending=False)
        .style.format({
            "weight": "{:.1%}",
            "annual_return": "{:.1%}",
            "volatility": "{:.1%}",
            "sharpe": "{:.2f}"
        })
        .background_gradient(subset=['sharpe'], cmap="RdYlGn")
    )