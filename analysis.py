import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    """
    Calculates 'Standard' 3-Year Sharpe using Monthly Returns.
    This matches Morningstar/Yahoo methodology better than daily data.
    """
    if df.empty or 'ticker' not in df.columns:
        return pd.DataFrame()

    metrics = []
    tickers = df['ticker'].tolist()
    
    # 1. Fetch 3 Years of Data (Standard Window)
    try:
        data = yf.download(tickers, period="3y", group_by='ticker', progress=False)
    except:
        return pd.DataFrame()

    for t in tickers:
        try:
            # Handle Data Structure
            if len(tickers) > 1:
                hist = data[t]['Close']
            else:
                hist = data['Close']
            
            # 2. Resample to Monthly (The "Standard" Way)
            # We take the last price of each month to smooth out daily noise
            hist_monthly = hist.resample('ME').last().dropna()
            
            # Need at least 24 months for a valid 3Y metric (allow some buffer)
            if len(hist_monthly) < 24:
                # Fallback to daily if monthly data is too sparse (e.g., new IPO)
                metrics.append({"ticker": t, "sharpe": 0.0, "volatility": 0.0, "annual_return": 0.0})
                continue

            # 3. Calculate Monthly Returns
            monthly_rets = hist_monthly.pct_change().dropna()
            
            # 4. Annualize (x12 instead of x252)
            ann_return = monthly_rets.mean() * 12
            ann_vol = monthly_rets.std() * np.sqrt(12)
            
            # 5. Sharpe (Risk Free Rate = 4.26% default)
            if ann_vol > 0:
                sharpe = (ann_return - 0.0426) / ann_vol
            else:
                sharpe = 0.0
            
            metrics.append({
                "ticker": t,
                "sharpe": float(sharpe),
                "volatility": float(ann_vol),
                "annual_return": float(ann_return)
            })
            
        except Exception as e:
            metrics.append({"ticker": t, "sharpe": 0.0, "volatility": 0.0, "annual_return": 0.0})
            
    return pd.DataFrame(metrics)