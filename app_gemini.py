import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# MODULE IMPORTS
from ui import apply_custom_style, display_header, display_risk_hero
from extraction import extract_holdings_from_pdf, parse_manual_data, get_example_csv
from processing import create_portfolio_df
from analysis import calculate_risk_metrics, get_portfolio_history, get_ai_reallocation_strategy

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Portfolio Analyst Pro", layout="wide", page_icon="📈")
load_dotenv()
apply_custom_style()

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
    with st.expander("📝 Data Entry (CSV)", expanded=False):
        csv_input = st.text_area("Paste Holdings", value=get_example_csv(), height=100)
        if st.button("🚀 Load Data", type="primary"):
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
        
    # --- SECTION 1: RISK INTELLIGENCE (The Focus) ---
    st.subheader("🎯 Risk Intelligence")
    
    if not final_df.empty:
        # Find Best/Worst
        best_asset = final_df.loc[final_df['sharpe'].idxmax()]
        worst_asset = final_df.loc[final_df['sharpe'].idxmin()]
        
        # DISPLAY HERO CARDS
        display_risk_hero(best_asset, worst_asset)
        
        # AI REALLOCATION ENGINE
        st.markdown("### 🤖 Algorithmic Reallocation")
        col_ai, col_btn = st.columns([4, 1])
        
        with col_ai:
            st.info("Ask the AI to analyze your Sharpe Ratios and suggest specific trades to improve efficiency.")
        with col_btn:
            if st.button("✨ Generate Plan", type="primary", use_container_width=True):
                with st.spinner("Consulting Strategy Engine..."):
                    if api_key:
                        suggestion = get_ai_reallocation_strategy(final_df, api_key)
                        st.session_state['ai_suggestion'] = suggestion
                    else:
                        st.error("API Key required for AI.")
        
        if 'ai_suggestion' in st.session_state:
            st.markdown(f"""
            <div class="ai-box">
                <b>💡 AI Strategy Note:</b><br><br>
                {st.session_state['ai_suggestion']}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SECTION 2: PORTFOLIO HEALTH ---
    st.subheader("📊 Portfolio Health")
    k1, k2, k3, k4 = st.columns(4)
    weighted_sharpe = (final_df['weight'] * final_df['sharpe']).sum()
    weighted_cagr = (final_df['weight'] * final_df['cagr']).sum()
    total_val = final_df['value'].sum()
    
    k1.metric("Total Value", f"${total_val:,.0f}")
    k2.metric("Portfolio Sharpe", f"{weighted_sharpe:.2f}", help="Target > 1.0")
    k3.metric("Exp. Annual Return", f"{weighted_cagr:.1%}")
    k4.metric("Active Assets", len(final_df))

    # --- SECTION 3: DEEP DIVE TABS ---
    tab1, tab2 = st.tabs(["💰 Holdings & Risk", "📈 Value History"])

    with tab1:
        st.dataframe(
            final_df,
            column_config={
                "ticker": st.column_config.TextColumn("Asset"),
                "sharpe": st.column_config.ProgressColumn("Sharpe Ratio", min_value=-1, max_value=3, format="%.2f"),
                "weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=1, format="%.1f%%"),
                "cagr": st.column_config.NumberColumn("Return (CAGR)", format="%.1f%%"),
                "volatility": st.column_config.NumberColumn("Volatility", format="%.1f%%"),
                "value": st.column_config.NumberColumn("Value", format="$%.0f")
            },
            column_order=("ticker", "sharpe", "weight", "cagr", "volatility", "value"),
            hide_index=True,
            use_container_width=True
        )

    with tab2:
        history = get_portfolio_history(final_df)
        if not history.empty:
            hist_df = history.to_frame(name="Value")
            fig = px.area(hist_df, x=hist_df.index, y="Value")
            fig.update_layout(height=400, showlegend=False, yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)
            