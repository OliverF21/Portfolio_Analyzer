import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# --- IMPORTS (The 4 Modules) ---
from ui import apply_custom_style, display_header, display_top_assets
from extraction import extract_holdings_from_pdf, parse_manual_data, get_example_csv
from processing import create_portfolio_df
from analysis import calculate_risk_metrics, get_portfolio_history

# --- 1. SETUP ---
st.set_page_config(page_title="Portfolio Analyst Pro", layout="wide", page_icon="📈")
load_dotenv()
apply_custom_style()

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Source")
    manual_mode = st.toggle("📝 Manual Mode", value=True)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not manual_mode and not api_key:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            st.warning("⚠️ No API Key found.")

# --- 3. HEADER & DATA LOADING ---
display_header()

raw_holdings = []
if manual_mode:
    with st.expander("📝 Data Entry (CSV)", expanded=False):
        csv_input = st.text_area("Paste Holdings", value=get_example_csv(), height=150)
        if st.button("🚀 Analyze Portfolio", type="primary"):
            raw_holdings = parse_manual_data(csv_input)
else:
    uploaded_file = st.file_uploader("Upload Robinhood PDF", type="pdf")
    if uploaded_file and api_key:
        with st.spinner("✨ AI is reading your document..."):
            raw_holdings = extract_holdings_from_pdf(uploaded_file, api_key=api_key)

# --- 4. MAIN DASHBOARD ---
if raw_holdings:
    # A. Processing
    df = create_portfolio_df(raw_holdings)
    
    # B. Analysis
    with st.spinner("🔮 Crunching conservative risk models..."):
        risk_df = calculate_risk_metrics(df)
        final_df = pd.merge(df, risk_df, on='ticker', how='left').fillna(0.0)
        history_series = get_portfolio_history(final_df)

    # C. Top Level KPIs
    k1, k2, k3, k4 = st.columns(4)
    weighted_sharpe = (final_df['weight'] * final_df['sharpe']).sum()
    weighted_cagr = (final_df['weight'] * final_df['cagr']).sum()
    total_val = final_df['value'].sum()
    worst_dd = final_df['max_drawdown'].min()

    k1.metric("Total Value", f"${total_val:,.0f}")
    k2.metric("Portfolio Sharpe", f"{weighted_sharpe:.2f}", help="Target > 1.0")
    k3.metric("Exp. Annual Return", f"{weighted_cagr:.1%}", help="CAGR (Total Return)")
    k4.metric("Max Drawdown", f"{worst_dd:.1%}", help="Worst crash in portfolio")

    st.markdown("---")

    # D. Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "💰 Value History", "🔬 Deep Dive"])

    with tab1:
        st.caption("🏆 Top Positions")
        display_top_assets(final_df)
        st.markdown("<br>", unsafe_allow_html=True) 

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("All Holdings")
            st.dataframe(
                final_df,
                column_config={
                    "ticker": st.column_config.TextColumn("Asset", help="Stock Ticker"),
                    "value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
                    "weight": st.column_config.ProgressColumn("Allocation", format="%.1f%%", min_value=0, max_value=1),
                    "quantity": st.column_config.NumberColumn("Shares", format="%.4f"),
                    "price": st.column_config.NumberColumn("Price", format="$%.2f")
                },
                column_order=("ticker", "weight", "value", "quantity", "price"),
                hide_index=True,
                use_container_width=True,
                height=450
            )
        
        with c2:
            st.subheader("Allocation")
            fig = px.pie(final_df, values='value', names='ticker', hole=0.5, color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_layout(showlegend=False) 
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Portfolio Value Over Time (2 Years)")
        st.caption("Simulated value of your CURRENT holdings over the last 2 years (Total Return).")
        
        if not history_series.empty:
            hist_df = history_series.to_frame(name="Total Value")
            fig_hist = px.area(hist_df, x=hist_df.index, y="Total Value")
            fig_hist.update_layout(
                xaxis_title="Date", 
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified",
                height=500,
                yaxis=dict(tickformat="$,.0f"),
                showlegend=False
            )
            fig_hist.update_traces(line_color='#6366f1', fillcolor='rgba(99, 102, 241, 0.2)')
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Could not calculate historical value.")

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