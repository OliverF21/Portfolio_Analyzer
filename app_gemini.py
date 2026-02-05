import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import pdfplumber
import os
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Portfolio Analyzer (Final)", layout="wide")
load_dotenv()

# --- 1. SETUP ---
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()
if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None

# --- 2. ROBUST DATA FETCHING (Fixes the Risk Metric Crash) ---
def get_safe_history(ticker):
    """Fetches history 1-by-1 to avoid MultiIndex crashes."""
    try:
        # Fetch 2 years of data
        df = yf.Ticker(ticker).history(period="2y")
        if df.empty: return None
        return df['Close']
    except:
        return None

def get_mock_data():
    return pd.DataFrame([
        {"ticker": "AAPL", "quantity": 50, "Value": 8500},
        {"ticker": "NVDA", "quantity": 10, "Value": 9000},
        {"ticker": "MSFT", "quantity": 20, "Value": 8000},
        {"ticker": "JPM", "quantity": 30, "Value": 5000}
    ])

# --- 3. DATA MODELS ---
class Holding(BaseModel):
    ticker: str
    quantity: float

class Portfolio(BaseModel):
    holdings: list[Holding]

# --- 4. MAIN APP ---
st.title("🛡️ AI Portfolio Analyzer (Robust)")
st.markdown("""
**Status:** Running. 
If your PDF is a **scan (image)**, text extraction will fail. 
Switch to **'Force Vision Mode'** below if that happens.
""")

use_vision = st.toggle("Force Vision Mode (Best for Scanned PDFs)", value=False)
uploaded_file = st.file_uploader("Upload Monthly Statement", type="pdf")

if uploaded_file:
    df = pd.DataFrame()
    
    with st.status("Processing...", expanded=True) as status:
        try:
            if not client: raise Exception("No API Key found.")

            # PATH A: VISION MODE (For Scans)
            if use_vision:
                st.write("👁️ Using Vision API (Slower but reads images)...")
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    f = client.files.upload(file=tmp_path)
                    time.sleep(2) # Give it a second to process
                    
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=[f, "Extract all stock holdings (Ticker, Quantity) as JSON."],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=Portfolio
                        )
                    )
                    data = response.parsed.holdings
                    df = pd.DataFrame([h.dict() for h in data])
                finally:
                    os.remove(tmp_path)

            # PATH B: TEXT MODE (Cheaper, fails on scans)
            else:
                st.write("📄 Extracting text locally...")
                full_text = ""
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text: full_text += text + "\n"
                
                if len(full_text) < 50:
                    raise Exception("No text found. PDF is likely a scan. Enable 'Force Vision Mode'.")

                st.write(f"🧠 Analyzing {len(full_text)} characters...")
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=f"Extract holdings from this text to JSON: {full_text[:30000]}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=Portfolio
                    )
                )
                data = response.parsed.holdings
                df = pd.DataFrame([h.dict() for h in data])

            if df.empty: raise Exception("AI found no data.")
            status.update(label="✅ Success!", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.warning("⚠️ Using DEMO DATA due to error.")
            df = get_mock_data()
            status.update(label="⚠️ Using Demo Data", state="error", expanded=False)

    # --- VISUALIZATION ---
    st.divider()
    if not df.empty:
        # 1. Get Prices (Simple Loop)
        with st.spinner("Fetching Prices..."):
            if 'Value' not in df.columns: # Calculate if not mock
                prices = []
                for t in df['ticker']:
                    try:
                        p = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
                        prices.append(p)
                    except:
                        prices.append(0.0)
                df['Price'] = prices
                df['Value'] = df['quantity'] * df['Price']
            
            # Clean
            df = df[df['Value'] > 0].copy()
            total_val = df['Value'].sum()

        # 2. Display
        c1, c2 = st.columns([1,2])
        c1.dataframe(df[['ticker', 'quantity', 'Value']].style.format({"Value": "${:,.2f}"}))
        c1.metric("Total Value", f"${total_val:,.2f}")
        c2.plotly_chart(px.pie(df, values='Value', names='ticker', title="Allocation"), use_container_width=True)

        # 3. Risk Metrics (The Fix)
        st.subheader("📉 Risk Analysis")
        with st.spinner("Calculating Risk (1-by-1)..."):
            risk_rows = []
            for t in df['ticker'].unique():
                hist = get_safe_history(t)
                if hist is not None and len(hist) > 100:
                    ret = hist.pct_change().mean() * 252
                    vol = hist.pct_change().std() * np.sqrt(252)
                    sharpe = (ret - 0.04) / vol if vol > 0 else 0
                    risk_rows.append({"Ticker": t, "Sharpe": sharpe, "Volatility": vol})
                else:
                    risk_rows.append({"Ticker": t, "Sharpe": 0.0, "Volatility": 0.0})
            
            risk_df = pd.DataFrame(risk_rows)
            st.dataframe(risk_df.style.format("{:.2f}"))