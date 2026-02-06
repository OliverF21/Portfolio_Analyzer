import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# MODULE IMPORTS
from ui import apply_custom_style, display_header, display_top_assets
from extraction import extract_holdings_from_pdf, parse_manual_data, get_example_csv
from processing import create_portfolio_df
from analysis import calculate_risk_metrics, get_portfolio_history, get_correlation_matrix, get_optimization_suggestions

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

# --- 3. HEADER & INPUT ---
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
    
    # B. Analysis (Calling ALL Analysis Functions)
    with st.spinner("🔮 Crunching Industry Standard Risk Models..."):
        risk_df = calculate_risk_metrics(df)
        final_df = pd.merge(df, risk_df, on='ticker', how='left').fillna(0.0)
        history_series = get_portfolio_history(final_df)
        corr_matrix = get_correlation_matrix(final_df)
        trim_df, boost_df = get_optimization_suggestions(final_df)

    # C. KPIs
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

    # D. Tabs (Sharpe Optimizer First)
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Sharpe Optimizer", "📊 Overview", "💰 Value History", "🔬 Deep Dive"])

    with tab1:
        st.subheader("Sharpe Maximization Engine")
        st.caption("Identify drags on your portfolio efficiency and correlation risks.")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            st.markdown("#### 1. Eliminate 'Portfolio Drag'")
            st.info("Assets with Sharpe < Portfolio Average. Reallocating from here increases efficiency.")
            if not trim_df.empty:
                st.dataframe(
                    trim_df[['ticker', 'weight', 'sharpe', 'cagr', 'volatility']],
                    column_config={
                        "weight": st.column_config.ProgressColumn("Allocation", format="%.1f%%", min_value=0, max_value=1),
                        "sharpe": st.column_config.NumberColumn("Sharpe (Low)", format="%.2f"),
                        "volatility": st.column_config.NumberColumn("Vol", format="%.1f%%")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.success("✅ No inefficient assets found.")

        with col_opt2:
            st.markdown("#### 2. Fund 'Efficiency Leaders'")
            st.success("Assets with Sharpe > Portfolio Average. Increasing weight here improves score.")
            if not boost_df.empty:
                st.dataframe(
                    boost_df[['ticker', 'weight', 'sharpe', 'cagr', 'volatility']],
                    column_config={
                        "weight": st.column_config.ProgressColumn("Allocation", format="%.1f%%", min_value=0, max_value=1),
                        "sharpe": st.column_config.NumberColumn("Sharpe (High)", format="%.2f"),
                        "volatility": st.column_config.NumberColumn("Vol", format="%.1f%%")
                    },
                    hide_index=True,
                    use_container_width=True
                )
        
        st.markdown("---")
        st.markdown("#### 3. Correlation Heatmap")
        st.caption("Maximize the Denominator: Find Low Correlation (Blue) to reduce Volatility.")
        
        if not corr_matrix.empty:
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                aspect="auto"
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Need multiple assets to calculate correlation.")

    with tab2:
        st.caption("🏆 Top Positions")
        display_top_assets(final_df)
        st.markdown("<br>", unsafe_allow_html=True) 

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("All Holdings")
            st.dataframe(
                final_df,
                column_config={
                    "ticker": st.column_config.TextColumn("Asset"),
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

    with tab3:
        st.subheader("Portfolio Value Over Time (2 Years)")
        if not history_series.empty:
            hist_df = history_series.to_frame(name="Total Value")
            fig_hist = px.area(hist_df, x=hist_df.index, y="Total Value")
            fig_hist.update_layout(
                xaxis_title="Date", yaxis_title="Value ($)",
                hovermode="x unified", height=500,
                yaxis=dict(tickformat="$,.0f"), showlegend=False
            )
            fig_hist.update_traces(line_color='#6366f1', fillcolor='rgba(99, 102, 241, 0.2)')
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab4:
        st.subheader("Risk Lab (Industry Standard)")
        st.caption("3-Year Monthly Arithmetic Mean | Risk-Free: 4.26%")
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