import yfinance as yf
import pandas as pd
import numpy as np
import json
from google import genai
from google.genai import types

def calculate_risk_metrics(df):
    """Calculates Conservative Risk Metrics (3y Total Return, 5% Hurdle)."""
    if df.empty or 'ticker' not in df.columns: return pd.DataFrame()
    metrics = []
    tickers = df['ticker'].tolist()
    
    try:
        data = yf.download(tickers, period="3y", group_by='ticker', auto_adjust=True, progress=False)
    except: return pd.DataFrame()

    for t in tickers:
        try:
            if len(tickers) > 1: hist = data[t]['Close']
            else: hist = data['Close']
            hist_monthly = hist.resample('ME').last().dropna()
            if len(hist_monthly) < 12: continue

            start = hist_monthly.iloc[0]; end = hist_monthly.iloc[-1]
            years = (hist_monthly.index[-1] - hist_monthly.index[0]).days / 365.25
            cagr = (end/start)**(1/years)-1 if start>0 else 0
            
            ann_vol = hist_monthly.pct_change().dropna().std() * np.sqrt(12)
            sharpe = (cagr - 0.05) / ann_vol if ann_vol > 0 else 0
            
            metrics.append({"ticker": t, "sharpe": float(sharpe), "volatility": float(ann_vol), "cagr": float(cagr)})
        except: continue
    return pd.DataFrame(metrics)

def get_portfolio_history(df):
    """Calculates historical value of current holdings."""
    if df.empty: return pd.Series()
    tickers = df['ticker'].tolist()
    quantities = dict(zip(df['ticker'], df['quantity']))
    try:
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
        data = data.ffill().dropna()
        portfolio_history = pd.Series(0.0, index=data.index)
        for t in tickers:
            if t in data.columns: portfolio_history += data[t] * quantities.get(t, 0)
        return portfolio_history
    except: return pd.Series()

def get_ai_reallocation_strategy(df, api_key):
    """
    Uses Gemini to analyze the DataFrame and suggest specific moves.
    """
    if df.empty or not api_key: return "Error: Missing data or API key."
    
    # Convert dataframe to a lean CSV string for the prompt
    context_data = df[['ticker', 'weight', 'sharpe', 'cagr', 'volatility']].to_csv(index=False)
    
    prompt = f"""
    Act as a Quantitative Portfolio Manager. 
    Your goal is to MAXIMIZE the Portfolio Sharpe Ratio.
    
    Here is the current portfolio data:
    {context_data}
    
    TASK:
    1. Identify the 'Dead Weight' (Low Sharpe assets).
    2. Identify the 'Efficiency Leaders' (High Sharpe assets).
    3. Suggest specific Reallocation Moves (e.g., "Sell 50% of X to buy Y").
    4. Explain WHY based on the volatility and CAGR metrics provided.
    
    Keep the response concise, actionable, and professional. Use bullet points.
    """
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"
    