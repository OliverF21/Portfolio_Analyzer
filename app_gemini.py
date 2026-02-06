import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# MODULE IMPORTS
from ui import apply_custom_style, display_header
from extraction import extract_holdings_from_pdf, parse_manual_data, get_example_csv
from processing import create_portfolio_df
from analysis import calculate_risk_metrics, get_comparative_performance

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Portfolio Analyst Pro", layout="wide", page_icon="📈")
load_dotenv()
apply_custom_style() # INJECT CSS

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Source")
    manual_mode = st.toggle("📝 Manual Mode", value=True)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not manual_mode and not api_key:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            st.warning("⚠️ No API Key found.")

# --- MAIN APP ---
display_header()

# 1. DATA INGESTION
raw_holdings = []
if manual_mode:
    with st.expander("📝 Data Entry (CSV)", expanded=True):
        csv_input = st.text_area("Paste Holdings", value=get_example_csv(), height=150)
        if st.button("🚀 Analyze Portfolio", type="primary"):
            raw_holdings = parse_manual_data(csv_input)
else:
    uploaded_file = st.file_uploader("Upload Robinhood PDF", type="pdf")
    if uploaded_file and api_key:
        with st.spinner("✨ AI is reading your document..."):
            raw_holdings = extract_holdings_from_pdf(uploaded_file, api_key=api_key)

if raw_holdings:
    # 2. PROCESS
    df = create_portfolio_df(raw_holdings)
    
    # 3. ANALYZE
    with st.spinner("🔮 Crunching conservative risk models..."):
        risk_df = calculate_risk_metrics(df)
        final_df = pd.merge(df, risk_df, on='ticker', how='left').fillna(0.0)
        perf_data = get_comparative_performance(final_df['ticker'].tolist())

    # --- THE DASHBOARD ---
    
    # Top Level KPIs
    k1, k2, k3, k4 = st.columns(4)
    weighted_sharpe = (final_df['weight'] * final_df['sharpe']).sum()
    weighted_cagr = (final_df['weight'] * final_df['cagr']).sum()
    total_val = final_df['value'].sum()
    worst_dd = final_df['max_drawdown'].min()

    k1.metric("Total Value", f"${total_val:,.0f}")
    k2.metric("Portfolio Sharpe", f"{weighted_sharpe:.2f}", help="Risk-Adj Return (Target > 1.0)")
    k3.metric("Exp. Annual Return", f"{weighted_cagr:.1%}", help="CAGR (Total Return)")
    k4.metric("Max Drawdown", f"{worst_dd:.1%}", help="Worst crash in portfolio")

    st.markdown("---")

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Price Performance", "🔬 Deep Dive"])

    with tab1:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.subheader("Holdings Breakdown")
            st.dataframe(
                final_df[['ticker', 'quantity', 'value', 'weight']].style.format({
                    "value": "${:,.2f}", "weight": "{:.1%}"
                }), use_container_width=True, height=400
            )
        with c2:
            st.subheader("Allocation")
            fig = px.pie(final_df, values='value', names='ticker', hole=0.4, color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Time Machine: Compare Performance (2 Years)")
        st.caption("This chart normalizes all assets to start at 0%, showing true relative growth (Total Return).")
        
        if not perf_data.empty:
            fig_perf = px.line(perf_data, x=perf_data.index, y=perf_data.columns)
            fig_perf.update_layout(
                xaxis_title="Date", 
                yaxis_title="Total Return (%)",
                hovermode="x unified",
                legend_title="Asset",
                height=500,
                yaxis=dict(tickformat=".0%")
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.warning("Could not fetch historical data for comparison.")

    with tab3:
        st.subheader("Risk Lab")
        st.caption("Conservative Metrics: 5% Risk-Free Rate, Total Return (Dividends Included).")
        st.dataframe(
            final_df[['ticker', 'weight', 'cagr', 'volatility', 'max_drawdown', 'sharpe']]
            .sort_values('sharpe', ascending=False)
            .style.format({
                "weight": "{:.1%}", "cagr": "{:.1%}", "volatility": "{:.1%}", 
                "max_drawdown": "{:.1%}", "sharpe": "{:.2f}"
            })
            .background_gradient(subset=['sharpe'], cmap="RdYlGn")
            .background_gradient(subset=['max_drawdown'], cmap="Reds_r"),
            use_container_width=True
        )