import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import pdfplumber
import os
import tempfile
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Portfolio Analyzer (Lite)", layout="wide")
load_dotenv()

# --- 1. SETUP & AUTH ---
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()
if not api_key:
    st.error("🚨 API Key Not Found!")
    st.stop()

client = genai.Client(api_key=api_key)
# 1.5-Flash is the most efficient model for text analysis
MODEL_ID = "gemini-1.5-flash"

# --- 2. DATA SCHEMA ---
class Holding(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL)")
    quantity: float = Field(description="The total number of shares owned")

class Portfolio(BaseModel):
    holdings: list[Holding]

# --- 3. RETRY LOGIC (Safety Net) ---
@retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(3), wait=wait_fixed(10))
def analyze_text_with_gemini(text_content):
    prompt = f"""
    Analyze the following text extracted from a brokerage statement.
    
    TEXT CONTENT:
    {text_content}
    
    INSTRUCTIONS:
    1. Identify the 'Holdings' or 'Positions' section.
    2. Extract Ticker Symbols and Quantities.
    3. Ignore 'Cash', 'Pending', or 'Sweep' entries.
    4. Combine duplicates if the same ticker appears twice.
    5. Return valid JSON matching the schema.
    """
    
    return client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Portfolio
        )
    )

# --- 4. MAIN APP ---
st.title("⚡ AI Portfolio Analyzer (Lite Mode)")
st.markdown("Uses **Local Text Extraction** to minimize API usage and bypass limits.")

uploaded_file = st.file_uploader("Upload Monthly Statement", type="pdf")

if uploaded_file:
    # --- PHASE 1: LOCAL EXTRACTION ---
    with st.status("Reading Document...", expanded=True) as status:
        st.write("📄 Extracting text locally (0% API usage)...")
        
        try:
            full_text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n"
            
            char_count = len(full_text)
            st.write(f"✅ Extracted {char_count} characters.")
            
            if char_count < 50:
                st.error("Could not read text. Is this a scanned image? This mode requires selectable text.")
                st.stop()

            # --- PHASE 2: AI ANALYSIS ---
            st.write("🧠 Sending text to Gemini (Lite Request)...")
            
            response = analyze_text_with_gemini(full_text)
            
            holdings_list = response.parsed.holdings
            df = pd.DataFrame([h.dict() for h in holdings_list])
            
            if df.empty:
                st.error("AI found no holdings. Check the PDF format.")
                st.stop()
                
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

    # --- PHASE 3: FINANCIAL ANALYSIS (No AI - Uses yfinance) ---
    st.divider()
    
    with st.spinner("Fetching live market data..."):
        tickers = df['ticker'].tolist()
        
        # Batch download
        try:
            batch_data = yf.download(tickers, period="1d")['Close'].iloc[-1]
        except:
            batch_data = pd.Series()

        def get_price(t):
            try:
                if isinstance(batch_data, (float, np.float64)): return batch_data
                if t in batch_data: return batch_data[t]
                return yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
            except: return 0.0

        df['Price'] = df['ticker'].apply(get_price)
        df['Value'] = df['quantity'] * df['Price']
        df = df[df['Value'] > 0].copy()
        
        total_value = df['Value'].sum()
        df['Weight'] = df['Value'] / total_value if total_value > 0 else 0

    # Dashboard
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Holdings")
        st.dataframe(df[['ticker', 'quantity', 'Value']].style.format({
            "quantity": "{:.4f}",
            "Value": "${:,.2f}"
        }))
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

    with col2:
        st.subheader("🍰 Allocation")
        if not df.empty:
            fig = px.pie(df, values='Value', names='ticker', title='Portfolio Weighting')
            st.plotly_chart(fig, use_container_width=True)

    # --- PHASE 4: RISK METRICS ---
    st.divider()
    st.header("📉 Risk Analysis")
    
    with st.spinner("Calculating 10-Year Risk Metrics..."):
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
                    
                    metrics.append({
                        'Ticker': ticker,
                        'Sharpe': sharpe,
                        'Volatility': ann_vol,
                        'Return': ann_return
                    })
            except: pass
        
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            final = pd.merge(df[['ticker', 'Weight']], metrics_df, left_on='ticker', right_on='Ticker')
            
            weighted_sharpe = (final['Weight'] * final['Sharpe']).sum()
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Portfolio Sharpe", f"{weighted_sharpe:.2f}", delta="Target > 1.0")
            
            best = final.loc[final['Sharpe'].idxmax()]
            kpi2.metric("Best Asset", best['Ticker'], f"{best['Sharpe']:.2f} Sharpe")
            
            st.dataframe(final[['Ticker', 'Weight', 'Return', 'Volatility', 'Sharpe']].style.format({
                "Weight": "{:.1%}", "Return": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}"
            }))