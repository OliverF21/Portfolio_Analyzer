import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import os
import tempfile
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Portfolio Analyzer", layout="wide")
load_dotenv()

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
    st.stop()

# Initialize Client
client = genai.Client(api_key=api_key)

# Switch to 1.5 Flash (Often has better free tier availability than 2.0)
MODEL_ID = "gemini-1.5-flash" 

# --- 2. DATA STRUCTURES ---
class Holding(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL)")
    quantity: float = Field(description="The total number of shares owned")

class Portfolio(BaseModel):
    holdings: list[Holding]

# --- 3. RETRY LOGIC (The Fix for Error 429) ---
# This function tries 3 times, waiting 30 seconds between tries if it hits a limit
@retry(
    retry=retry_if_exception_type(Exception), 
    stop=stop_after_attempt(3), 
    wait=wait_fixed(30)
)
def call_gemini_with_retry(model_id, contents, config):
    return client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config
    )

# --- 4. MAIN APP INTERFACE ---
st.title("🤖 AI Portfolio Analyzer")
st.markdown(f"**Status:** Running on `{MODEL_ID}` with auto-retry enabled.")

uploaded_file = st.file_uploader("Upload Monthly Statement", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # --- STEP 1: EXTRACTION ---
        with st.status("Processing Document...", expanded=True) as status:
            st.write("📤 Uploading to Gemini...")
            statement_file = client.files.upload(file=tmp_path)
            
            st.write("⏳ Analyzing (Auto-retry enabled)...")
            prompt = """
            Extract every stock and ETF holding from this monthly statement. 
            Look at BOTH 'Securities Held' and 'Loaned Securities' sections. 
            Combine quantities if the same ticker appears in both. 
            Return a clean list of Tickers and Quantities.
            """

            try:
                # Call the retry-wrapped function
                response = call_gemini_with_retry(
                    model_id=MODEL_ID,
                    contents=[statement_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=Portfolio
                    )
                )
            except Exception as e:
                st.error(f"❌ Analysis Failed after retries: {str(e)}")
                st.stop()
            
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
        try:
            batch_data = yf.download(tickers, period="1d")['Close'].iloc[-1]
        except:
            batch_data = pd.Series()

        def get_price(t):
            try:
                if isinstance(batch_data, (float, np.float64)):
                    return batch_data
                if t in batch_data:
                    return batch_data[t]
                return yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
            except:
                return 0.0

        df['Price'] = df['ticker'].apply(get_price)
        df['Value'] = df['quantity'] * df['Price']
        df = df[df['Value'] > 0].copy()
        total_value = df['Value'].sum()
        if total_value > 0:
            df['Weight'] = df['Value'] / total_value
        else:
            df['Weight'] = 0

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📋 Holdings")
        st.dataframe(df[['ticker', 'quantity', 'Value']].style.format({"quantity": "{:.4f}", "Value": "${:,.2f}"}))
        st.metric("Total Value", f"${total_value:,.2f}")

    with col2:
        st.subheader("🍰 Allocation")
        if not df.empty:
            fig = px.pie(df, values='Value', names='ticker', title='Portfolio Weighting')
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 3: RISK ANALYSIS ---
    st.divider()
    st.header("📉 Risk Analysis")
    
    with st.spinner("Calculating Risk Metrics..."):
        hist_data = yf.download(tickers, period="10y", group_by='ticker')
        metrics = []
        for ticker in tickers:
            try:
                if len(tickers) > 1:
                    stock_hist = hist_data[ticker]['Close'].dropna()
                else:
                    stock_hist = hist_data['Close'].dropna()
                
                if len(stock_hist) > 200:
                    returns = stock_hist.pct_change()
                    ann_vol = returns.std() * np.sqrt(252)
                    ann_return = returns.mean() * 252
                    sharpe = (ann_return - 0.04) / ann_vol if ann_vol > 0 else 0
                    metrics.append({'Ticker': ticker, 'Sharpe': sharpe, 'Volatility': ann_vol})
            except:
                pass
        
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            final_df = pd.merge(df[['ticker', 'Weight']], metrics_df, left_on='ticker', right_on='Ticker')
            portfolio_sharpe = (final_df['Weight'] * final_df['Sharpe']).sum()
            st.metric("Portfolio Sharpe Ratio", f"{portfolio_sharpe:.2f}")