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
load_dotenv() # Load local .env (ignored on Cloud if not found)

# --- 1. SECURE API KEY RETRIEVAL ---
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    return None

api_key = get_api_key()

if not api_key:
    st.error("🚨 API Key Not Found!")
    st.info("Please add GEMINI_API_KEY to your Streamlit Secrets or local .env file.")
    st.stop()

# Initialize Client ONCE
client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-2.0-flash"

# --- 2. DATA STRUCTURES ---
class Holding(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL)")
    quantity: float = Field(description="The total number of shares owned")

class Portfolio(BaseModel):
    holdings: list[Holding]

# --- 3. MAIN APP LOGIC ---
st.title("🤖 AI Portfolio Analyzer")
st.markdown("Upload your **Robinhood PDF** to extract holdings, visualize allocation, and calculate risk (Sharpe Ratio).")

uploaded_file = st.file_uploader("Upload Monthly Statement", type="pdf")

if uploaded_file:
    with st.spinner("Step 1: AI is reading your PDF..."):
        # Save uploaded file to temp path for Gemini
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            # A. Extract Data using Gemini Vision
            statement_file = client.files.upload(file=tmp_path)
            
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
            
            # Parse into DataFrame
            holdings_list = response.parsed.holdings
            df = pd.DataFrame([h.dict() for h in holdings_list])

            if df.empty:
                st.error("No holdings found. Please check the PDF.")
                st.stop()

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # --- VISUALIZATION STEP ---
    st.success("✅ Extraction Complete!")
    
    # B. Fetch Market Data & Calculate Weights
    with st.spinner("Step 2: Fetching live market data..."):
        tickers = df['ticker'].tolist()
        
        # Safe Batch Download
        try:
            batch_data = yf.download(tickers, period="1d")['Close'].iloc[-1]
        except:
            st.warning("Batch download failed. Trying individual tickers...")
            batch_data = pd.Series()

        def get_price(t):
            try:
                if t in batch_data: return batch_data[t]
                return yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
            except:
                return 0.0

        df['Price'] = df['ticker'].apply(get_price)
        df['Value'] = df['quantity'] * df['Price']
        
        # Filter out zero value/failed tickers
        df = df[df['Value'] > 0].copy()
        
        total_value = df['Value'].sum()
        df['Weight'] = df['Value'] / total_value

    # C. Display Visualizations
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Holdings Table")
        st.dataframe(df[['ticker', 'quantity', 'Value']].style.format({
            "quantity": "{:.4f}",
            "Value": "${:,.2f}"
        }))
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

    with col2:
        st.subheader("🍰 Allocation Pie Chart")
        fig = px.pie(df, values='Value', names='ticker', title='Portfolio Weighting')
        st.plotly_chart(fig, use_container_width=True)

    # --- ANALYSIS STEP ---
    st.divider()
    st.header("📉 Risk Analysis (Sharpe Ratio)")
    
    with st.spinner("Step 3: Calculating 10-Year Risk Metrics..."):
        # Fetch 10y history
        hist_data = yf.download(tickers, period="10y", group_by='ticker')
        
        metrics = []
        for ticker in tickers:
            try:
                # Handle yfinance multi-index
                stock_hist = hist_data[ticker] if len(tickers) > 1 else hist_data
                stock_hist = stock_hist['Close'].dropna()
                
                if len(stock_hist) < 252:
                    metrics.append({'Ticker': ticker, 'Sharpe': 0.0, 'Return': 0.0})
                    continue

                returns = stock_hist.pct_change()
                ann_return = returns.mean() * 252
                ann_vol = returns.std() * np.sqrt(252)
                
                # Sharpe (Risk Free Rate = 4%)
                sharpe = (ann_return - 0.04) / ann_vol if ann_vol != 0 else 0
                
                metrics.append({
                    'Ticker': ticker,
                    'Sharpe': sharpe,
                    'Annual Return': ann_return,
                    'Volatility': ann_vol
                })
            except:
                pass
        
        metrics_df = pd.DataFrame(metrics)
        
        # Merge with Weights
        final_df = pd.merge(df[['ticker', 'Weight']], metrics_df, left_on='ticker', right_on='Ticker')
        
        # Calculate Weighted Sharpe
        portfolio_sharpe = (final_df['Weight'] * final_df['Sharpe']).sum()
        
        # Display Scorecard
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio Sharpe Ratio", f"{portfolio_sharpe:.2f}", help="> 1.0 is Good, > 2.0 is Excellent")
        c2.metric("Top Performer", final_df.sort_values('Sharpe', ascending=False).iloc[0]['Ticker'])
        c3.metric("Highest Risk (Vol)", final_df.sort_values('Volatility', ascending=False).iloc[0]['Ticker'])

        st.subheader("Detailed Risk Metrics")
        st.dataframe(final_df[['Ticker', 'Weight', 'Annual Return', 'Volatility', 'Sharpe']].sort_values('Sharpe', ascending=False).style.format({
            "Weight": "{:.1%}",
            "Annual Return": "{:.1%}",
            "Volatility": "{:.1%}",
            "Sharpe": "{:.2f}"
        }).background_gradient(subset=['Sharpe'], cmap="RdYlGn"))

        