import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import os
import tempfile
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Portfolio Analyzer", layout="wide")
load_dotenv()

# --- 1. SECURE API KEY RETRIEVAL ---
def get_api_key():
    # Check Streamlit Cloud Secrets first
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    # Fallback to local .env
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    return None

api_key = get_api_key()

if not api_key:
    st.error("🚨 API Key Not Found!")
    st.info("Please add GEMINI_API_KEY to your Streamlit Secrets or local .env file.")
    st.stop()

# Initialize Client (Using the Secure Key)
client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-2.0-flash"

# --- 2. DATA STRUCTURES ---
class Holding(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL)")
    quantity: float = Field(description="The total number of shares owned")

class Portfolio(BaseModel):
    holdings: list[Holding]

# --- 3. MAIN APP INTERFACE ---
st.title("🤖 AI Portfolio Analyzer")
st.markdown("Upload your **Robinhood PDF** to extract holdings, visualize allocation, and calculate risk.")

uploaded_file = st.file_uploader("Upload Monthly Statement", type="pdf")

if uploaded_file:
    # Save uploaded file to temp path for Gemini
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # --- STEP 1: EXTRACTION ---
        with st.status("Processing Document...", expanded=True) as status:
            st.write("📤 Uploading to Gemini...")
            statement_file = client.files.upload(file=tmp_path)
            
            st.write("🧠 Extracting holdings...")
            prompt = """
            Extract every stock and ETF holding from this monthly statement. 
            Look at BOTH 'Securities Held' and 'Loaned Securities' sections. 
            Combine quantities if the same ticker appears in both. 
            Return a clean list of Tickers and Quantities.
            """

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[statement_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Portfolio
                )
            )
            
            holdings_list = response.parsed.holdings
            df = pd.DataFrame([h.dict() for h in holdings_list])
            
            if df.empty:
                st.error("No holdings found!")
                st.stop()
                
            status.update(label="✅ Extraction Complete!", state="complete", expanded=False)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # --- STEP 2: VISUALIZATION ---
    st.divider()
    
    with st.spinner("Fetching live market prices..."):
        tickers = df['ticker'].tolist()
        
        # Batch download for speed
        try:
            batch_data = yf.download(tickers, period="1d")['Close'].iloc[-1]
        except:
            batch_data = pd.Series()

        # Map prices
        def get_price(t):
            try:
                # Handle cases where batch_data might be a float (single ticker) or Series
                if isinstance(batch_data, (float, np.float64)):
                    return batch_data
                if t in batch_data:
                    return batch_data[t]
                return yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
            except:
                return 0.0

        df['Price'] = df['ticker'].apply(get_price)
        df['Value'] = df['quantity'] * df['Price']
        
        # Clean Data
        df = df[df['Value'] > 0].copy()
        total_value = df['Value'].sum()
        df['Weight'] = df['Value'] / total_value

    # Dashboard Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Holdings")
        st.dataframe(df[['ticker', 'quantity', 'Value']].style.format({
            "quantity": "{:.4f}",
            "Value": "${:,.2f}"
        }))
        st.metric("Total Value", f"${total_value:,.2f}")

    with col2:
        st.subheader("🍰 Allocation")
        if not df.empty:
            fig = px.pie(df, values='Value', names='ticker', title='Portfolio Weighting')
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 3: RISK ANALYSIS ---
    st.divider()
    st.header("📉 Risk Analysis (Sharpe Ratio)")
    
    with st.spinner("Calculating 10-year risk metrics..."):
        hist_data = yf.download(tickers, period="10y", group_by='ticker')
        
        metrics = []
        for ticker in tickers:
            try:
                # Handle yfinance data structures safely
                if len(tickers) > 1:
                    stock_hist = hist_data[ticker]['Close'].dropna()
                else:
                    stock_hist = hist_data['Close'].dropna()

                if len(stock_hist) < 252:
                    continue 

                returns = stock_hist.pct_change()
                ann_return = returns.mean() * 252
                ann_vol = returns.std() * np.sqrt(252)
                
                sharpe = (ann_return - 0.04) / ann_vol if ann_vol > 0 else 0
                
                metrics.append({
                    'Ticker': ticker,
                    'Sharpe': sharpe,
                    'Return': ann_return,
                    'Volatility': ann_vol
                })
            except:
                pass
        
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            final_df = pd.merge(df[['ticker', 'Weight']], metrics_df, left_on='ticker', right_on='Ticker')
            
            portfolio_sharpe = (final_df['Weight'] * final_df['Sharpe']).sum()
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Portfolio Sharpe Ratio", f"{portfolio_sharpe:.2f}", delta="Target: > 1.0")
            
            best_stock = final_df.loc[final_df['Sharpe'].idxmax()]
            worst_vol = final_df.loc[final_df['Volatility'].idxmax()]
            
            kpi2.metric("Best Efficiency", best_stock['Ticker'], f"{best_stock['Sharpe']:.2f} Sharpe")
            kpi3.metric("Highest Volatility", worst_vol['Ticker'], f"{worst_vol['Volatility']:.1%}")

            st.subheader("Detailed Risk Table")
            st.dataframe(final_df[['Ticker', 'Weight', 'Return', 'Volatility', 'Sharpe']].sort_values('Sharpe', ascending=False).style.format({
                "Weight": "{:.1%}",
                "Return": "{:.1%}",
                "Volatility": "{:.1%}",
                "Sharpe": "{:.2f}"
            }).background_gradient(subset=['Sharpe'], cmap="RdYlGn"))