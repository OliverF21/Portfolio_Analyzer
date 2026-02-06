import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# IMPORT MODULES
from extraction import extract_holdings_from_pdf, parse_manual_data, get_example_csv
from processing import create_portfolio_df
from analysis import calculate_risk_metrics

# --- CONFIG ---
st.set_page_config(page_title="Modular Portfolio Analyzer", layout="wide")
load_dotenv()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    manual_mode = st.toggle("📝 Manual Data / CSV Mode", value=True)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    
    if not manual_mode and not api_key:
        st.warning("⚠️ No API Key found for PDF extraction.")

# --- MAIN UI ---
st.title("Sharpe Ratio Analysis Dashboard")

raw_holdings = []

# PHASE 1: DATA ENTRY
if manual_mode:
    st.info("Edit your portfolio CSV below (Ticker, Quantity).")
    csv_input = st.text_area("CSV Data", value=get_example_csv(), height=200)
    
    if st.button("🚀 Load Manual Data"):
        with st.spinner("Parsing CSV..."):
            raw_holdings = parse_manual_data(csv_input)
            if not raw_holdings:
                st.error("Could not parse CSV. Please check format.")
                st.stop()
            st.success(f"Loaded {len(raw_holdings)} rows from CSV.")
else:
    uploaded_file = st.file_uploader("Upload Robinhood PDF", type="pdf")
    if uploaded_file:
        if not api_key:
            st.error("API Key required for PDF Mode.")
            st.stop()
            
        with st.spinner("Extracting data from PDF..."):
            raw_holdings = extract_holdings_from_pdf(uploaded_file, api_key=api_key)
            if not raw_holdings:
                st.error("Extraction Failed. Try Manual Mode.")
                st.stop()
            st.success(f"Extracted {len(raw_holdings)} positions.")

# PHASE 2 & 3: PROCESSING & ANALYSIS
if raw_holdings:
    # 2. PROCESS
    df = create_portfolio_df(raw_holdings)
    
    # 3. ANALYZE
    with st.spinner("Calculating Risk Metrics (Live)..."):
        risk_df = calculate_risk_metrics(df)
        final_df = pd.merge(df, risk_df, on='ticker', how='left').fillna(0.0)

    # --- DASHBOARD ---
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Holdings")
        st.dataframe(final_df[['ticker', 'quantity', 'value', 'weight']].style.format({
            "value": "${:,.2f}",
            "weight": "{:.1%}"
        }))
        
    with col2:
        st.subheader("🍰 Allocation")
        fig = px.pie(final_df, values='value', names='ticker', title="Portfolio Weight")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📉 Risk Analysis")
    
    weighted_sharpe = (final_df['weight'] * final_df['sharpe']).sum()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Portfolio Sharpe", f"{weighted_sharpe:.2f}")
    
    if not final_df.empty:
        best = final_df.loc[final_df['sharpe'].idxmax()]
        k2.metric("Best Asset", best['ticker'], f"{best['sharpe']:.2f}")
        worst = final_df.loc[final_df['sharpe'].idxmin()]
        k3.metric("Laggard", worst['ticker'], f"{worst['sharpe']:.2f}")

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