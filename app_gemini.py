import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import pdfplumber
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. CORE CONFIGURATION ---
st.set_page_config(page_title="Core Principles Analyzer", layout="wide")
load_dotenv()

# --- 2. THE ENGINE (PURE MATH & TYPE SAFETY) ---
def safe_float(value):
    """Guard: Forces any input to be a float or 0.0."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def calculate_single_ticker_risk(ticker):
    """
    Calculates Annualized Return, Volatility, and Sharpe for ONE ticker.
    Returns (0.0, 0.0, 0.0) on failure to prevent app crashes.
    """
    try:
        # 1. Fetch Data (2y history)
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="2y", auto_adjust=False)
        
        if len(hist) < 50:
            return 0.0, 0.0, 0.0
            
        # 2. Extract Closes
        closes = hist['Close']
        
        # 3. Calculate Returns (Drop NaNs created by pct_change)
        returns = closes.pct_change().dropna()
        if returns.empty: return 0.0, 0.0, 0.0

        # 4. Annualize (252 trading days)
        ann_vol = returns.std() * np.sqrt(252)
        ann_return = returns.mean() * 252
        
        # 5. Sharpe (Risk Free Rate = 4%)
        if ann_vol > 0:
            sharpe = (ann_return - 0.04) / ann_vol
        else:
            sharpe = 0.0
            
        return safe_float(sharpe), safe_float(ann_vol), safe_float(ann_return)

    except Exception as e:
        print(f"Calc failed for {ticker}: {e}")
        return 0.0, 0.0, 0.0

# --- 3. DATA INGESTION ---
def get_example_portfolio():
    """Generates the Test Portfolio (AAPL/MSFT) for verification."""
    st.info("ℹ️ Running Core Calculation Test on: AAPL (50), MSFT (20)")
    
    df = pd.DataFrame([
        {"ticker": "AAPL", "quantity": 50},
        {"ticker": "MSFT", "quantity": 20}
    ])
    
    # Enrich with Live Price
    current_prices = []
    for t in df['ticker']:
        try:
            p = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
            current_prices.append(safe_float(p))
        except:
            current_prices.append(0.0)
            
    df['Price'] = current_prices
    df['Value'] = df['quantity'] * df['Price']
    
    # Calculate Weights
    total_value = df['Value'].sum()
    df['Weight'] = df['Value'] / total_value if total_value > 0 else 0.0
        
    return df

# --- 4. MAIN APPLICATION ---
st.title("🛡️ Core Principles Analyzer")

uploaded_file = st.file_uploader("Upload Monthly Statement (PDF)", type="pdf")

if uploaded_file:
    # PDF Logic (Simplified for stability)
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join([page.extract_text() or "" for page in pdf.pages])
        
        # Use simple extraction (Stub for now to focus on Math verification)
        if len(text) < 50:
            st.error("No text found in PDF (Scanned?). Using Example Portfolio.")
            df = get_example_portfolio()
        else:
            # Here we would call Gemini. For safety, we default to Example if API fails.
            # (In production, insert Gemini call here)
            st.warning("PDF uploaded. Swapping to Example Portfolio to verify Risk Engine logic first.")
            df = get_example_portfolio()
    except:
        df = get_example_portfolio()
else:
    df = get_example_portfolio()

# --- 5. EXECUTION & DISPLAY ---
if not df.empty:
    st.divider()
    
    # A. Display Raw Data
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("📋 Portfolio")
        st.dataframe(df[['ticker', 'quantity', 'Value']].style.format({
            "quantity": "{:.0f}",
            "Value": "${:,.2f}"
        }))
        
    with c2:
        st.subheader("Allocation")
        fig = px.pie(df, values='Value', names='ticker')
        st.plotly_chart(fig, use_container_width=True)

    # B. The Risk Engine
    st.divider()
    st.subheader("📉 Risk Calculation (Live)")
    
    with st.spinner("Applying Core Principles..."):
        risk_data = []
        
        # Iterate tickers (Atomic Operations)
        for ticker in df['ticker']:
            sharpe, vol, ret = calculate_single_ticker_risk(ticker)
            
            risk_data.append({
                "Ticker": ticker,
                "Sharpe": sharpe,       
                "Volatility": vol,      
                "Return": ret           
            })
            
        risk_df = pd.DataFrame(risk_data)
        
        # Merge with Portfolio
        final_df = pd.merge(df, risk_df, left_on='ticker', right_on='Ticker')
        
        # Portfolio-Level Weighted Sharpe
        final_df['Weighted_Sharpe'] = final_df['Weight'] * final_df['Sharpe'].fillna(0.0)
        portfolio_sharpe = final_df['Weighted_Sharpe'].sum()
        
        # C. KPI Display
        k1, k2 = st.columns(2)
        k1.metric("Portfolio Sharpe Ratio", f"{portfolio_sharpe:.2f}")
        
        if not final_df.empty:
            top_asset = final_df.loc[final_df['Sharpe'].idxmax()]
            k2.metric(f"Best Asset ({top_asset['Ticker']})", f"{top_asset['Sharpe']:.2f}")

        # D. Detailed Table
        st.dataframe(
            final_df[['Ticker', 'Weight', 'Return', 'Volatility', 'Sharpe']]
            .style.format({
                "Weight": "{:.1%}",
                "Return": "{:.1%}",
                "Volatility": "{:.1%}",
                "Sharpe": "{:.2f}"
            })
            .background_gradient(subset=['Sharpe'], cmap="RdYlGn")
        )