import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    """
    Takes the Portfolio DF and enriches it with Risk Metrics (Sharpe, Vol, Return).
    """
    if df.empty or 'ticker' not in df.columns:
        return pd.DataFrame()

    metrics = []
    tickers = df['ticker'].tolist()
    
    # 1. Fetch History (2 Years)
    try:
        data = yf.download(tickers, period="2y", group_by='ticker', progress=False)
    except:
        return pd.DataFrame() # Return empty if fetch fails

    # 2. Iterate and Calculate
    for t in tickers:
        try:
            # Handle Multi-Index safely
            if len(tickers) > 1:
                hist = data[t]['Close']
            else:
                hist = data['Close']
            
            hist = hist.dropna()
            
            # Need at least 3 months of data
            if len(hist) < 60:
                raise ValueError("Insufficient Data")

            # Daily Returns
            returns = hist.pct_change().dropna()
            
            # Annualize
            ann_return = returns.mean() * 252
            ann_vol = returns.std() * np.sqrt(252)
            
            # Sharpe (4% Risk Free Rate)
            if ann_vol > 0:
                sharpe = (ann_return - 0.04) / ann_vol
            else:
                sharpe = 0.0
            
            metrics.append({
                "ticker": t,
                "sharpe": float(sharpe),
                "volatility": float(ann_vol),
                "annual_return": float(ann_return)
            })
            
        except:
            # Safe Fallback
            metrics.append({
                "ticker": t, 
                "sharpe": 0.0, 
                "volatility": 0.0, 
                "annual_return": 0.0
            })
            
    return pd.DataFrame(metrics)
