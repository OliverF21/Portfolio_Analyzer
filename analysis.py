import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    """Calculates CAGR (True Annual Return) and Volatility."""
    if df.empty or 'ticker' not in df.columns:
        return pd.DataFrame()

    metrics = []
    tickers = df['ticker'].tolist()
    
    try:
        data = yf.download(tickers, period="2y", group_by='ticker', progress=False)
    except:
        return pd.DataFrame()

    for t in tickers:
        try:
            # Handle Data Structure
            if len(tickers) > 1:
                hist = data[t]['Close']
            else:
                hist = data['Close']
            
            hist = hist.dropna()
            if len(hist) < 60: continue

            # --- MATH FIX ---
            # 1. Volatility (Standard Deviation of Daily Returns)
            daily_rets = hist.pct_change().dropna()
            ann_vol = daily_rets.std() * np.sqrt(252)

            # 2. CAGR (Geometric Mean / Compound Growth) - The "True" Return
            # Formula: (End_Price / Start_Price) ^ (365.25 / Days) - 1
            start_price = hist.iloc[0]
            end_price = hist.iloc[-1]
            days = (hist.index[-1] - hist.index[0]).days
            
            if days > 0 and start_price > 0:
                cagr = (end_price / start_price) ** (365.25 / days) - 1
            else:
                cagr = 0.0

            # 3. Sharpe Ratio (Using CAGR as the Return metric)
            # Assuming 4% Risk-Free Rate
            if ann_vol > 0:
                sharpe = (cagr - 0.04) / ann_vol
            else:
                sharpe = 0.0
            
            metrics.append({
                "ticker": t,
                "sharpe": float(sharpe),
                "volatility": float(ann_vol),
                "annual_return": float(cagr)
            })
            
        except:
            metrics.append({"ticker": t, "sharpe": 0.0, "volatility": 0.0, "annual_return": 0.0})
            
    return pd.DataFrame(metrics)