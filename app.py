
import streamlit as st
import pdfplumber
import pandas as pd
import yfinance as yf
import numpy as np

# --- CONFIGURATION ---
RISK_FREE_RATE = 0.04
LOOKBACK_YEARS = 10

st.set_page_config(page_title="Robinhood Portfolio Analyzer", layout="wide")

def parse_pdf(file):
    holdings = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text: continue
            for line in text.split('\n'):
                parts = line.split()
                if len(parts) < 4: continue
                # Basic check for Ticker format (ALL CAPS, 2-5 chars) and a '$' sign
                if parts[0].isupper() and 2 <= len(parts[0]) <= 5 and '$' in line:
                    try:
                        # Grab the last item as Market Value
                        value_str = parts[-1].replace('$', '').replace(',', '')
                        holdings.append({'Ticker': parts[0], 'Market Value': float(value_str)})
                    except:
                        continue
    return pd.DataFrame(holdings)

def get_sharpe(ticker, years):
    try:
        # Ticker adjustments for Crypto
        if ticker in ['BTC', 'ETH', 'DOGE', 'SOL']: ticker = f"{ticker}-USD"
        
        hist = yf.Ticker(ticker).history(period=f"{years}y")
        if hist.empty: return None, "No Data"
        
        # Calc Metrics
        hist['Returns'] = hist['Close'].pct_change()
        annual_ret = hist['Returns'].mean() * 252
        volatility = hist['Returns'].std() * np.sqrt(252)
        sharpe = (annual_ret - RISK_FREE_RATE) / volatility
        
        return sharpe, len(hist)
    except:
        return None, "Error"

# --- THE APP UI ---
st.title("📊 Portfolio Sharpe Analyzer")
st.write("Upload your Robinhood Monthly Statement (PDF) to analyze risk-adjusted returns.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Extracting holdings..."):
        df = parse_pdf(uploaded_file)
    
    if df.empty:
        st.error("Could not find holdings. Try uploading a different PDF or checking the format.")
    else:
        # Dedup and aggregate (in case multiple lots of same stock)
        df = df.groupby('Ticker')['Market Value'].sum().reset_index()
        total_value = df['Market Value'].sum()
        df['Allocation'] = df['Market Value'] / total_value
        
        st.success(f"Found {len(df)} holdings. Total Value: ${total_value:,.2f}")
        
        # Run Analysis
        results = []
        progress_bar = st.progress(0)
        
        for i, row in df.iterrows():
            # Update progress bar
            progress_bar.progress((i + 1) / len(df))
            
            sharpe, count = get_sharpe(row['Ticker'], LOOKBACK_YEARS)
            results.append({
                'Ticker': row['Ticker'],
                'Allocation': row['Allocation'],
                'Value': row['Market Value'],
                'Sharpe Ratio (10y)': sharpe if sharpe else 0,
                'Data Points': count
            })
            
        final_df = pd.DataFrame(results).sort_values(by='Allocation', ascending=False)
        
        # Formatting for display
        display_df = final_df.copy()
        display_df['Allocation'] = display_df['Allocation'].map('{:.1%}'.format)
        display_df['Value'] = display_df['Value'].map('${:,.2f}'.format)
        display_df['Sharpe Ratio (10y)'] = display_df['Sharpe Ratio (10y)'].map('{:.2f}'.format)
        
        st.table(display_df)
        
        # Calculate Weighted Portfolio Sharpe
        weighted_sharpe = (final_df['Sharpe Ratio (10y)'] * final_df['Allocation']).sum()
        st.metric(label="Total Portfolio Weighted Sharpe Ratio", value=f"{weighted_sharpe:.2f}")