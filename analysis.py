import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    """
    Standard Industry Sharpe Algorithm:
    1. Returns: Arithmetic Mean of Daily Returns (Annualized).
    2. Volatility: Standard Deviation of Daily Returns (Annualized).
    3. Risk-Free Rate: 4.26% (Fixed User Rate).
       Note: Using a fixed high Rf rate against historical data (where rates were lower)
       will still result in a slightly more conservative score than Morningstar.
    """
    if df.empty or 'ticker' not in df.columns:
        return pd.DataFrame()

    metrics = []
    tickers = df['ticker'].tolist()
    
    # 1. Fetch 5 Years of DAILY Data
    try:
        data = yf.download(
            tickers, 
            period="5y", 
            group_by='ticker', 
            auto_adjust=True, # Total Return
            progress=False
        )
    except:
        return pd.DataFrame()

    for t in tickers:
        try:
            # Handle Data Structure safely
            if len(tickers) > 1:
                hist = data[t]['Close']
            else:
                hist = data['Close']
            
            # Drop missing data
            hist = hist.dropna()
            
            # Need meaningful history
            if len(hist) < 200: 
                raise ValueError("Insufficient Data")

            # --- STEP 1: DAILY RETURNS ---
            daily_rets = hist.pct_change().dropna()
            
            # --- STEP 2: ARITHMETIC MEAN (Standard Industry Practice) ---
            # Formula: Mean * 252
            daily_arith_mean = daily_rets.mean()
            annualized_return = daily_arith_mean * 252

            # --- STEP 3: VOLATILITY ---
            # Annualize: Daily Std * Sqrt(252)
            annualized_vol = daily_rets.std() * np.sqrt(252)
            
            # --- STEP 4: SHARPE RATIO ---
            # Formula: (E(R) - rf) / StdDev
            rf_rate = 0.0426 # User Specified 4.26%
            
            if annualized_vol > 0:
                sharpe = (annualized_return - rf_rate) / annualized_vol
            else:
                sharpe = 0.0

            # --- METRIC 5: MAX DRAWDOWN ---
            rolling_max = hist.cummax()
            drawdown = (hist - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            metrics.append({
                "ticker": t,
                "sharpe": float(sharpe),
                "volatility": float(annualized_vol),
                "cagr": float(annualized_return), # Labelled CAGR for consistency, but is Arithmetic Mean
                "max_drawdown": float(max_drawdown)
            })
            
        except:
            metrics.append({
                "ticker": t, "sharpe": 0.0, "volatility": 0.0, 
                "cagr": 0.0, "max_drawdown": 0.0
            })
            
    return pd.DataFrame(metrics)

def get_portfolio_history(df):
    """Calculates the historical TOTAL VALUE of the current portfolio."""
    if df.empty: return pd.Series()
    tickers = df['ticker'].tolist()
    quantities = dict(zip(df['ticker'], df['quantity']))
    try:
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
        data = data.ffill().dropna()
        portfolio_history = pd.Series(0.0, index=data.index)
        for t in tickers:
            if t in data.columns:
                portfolio_history += data[t] * quantities.get(t, 0)
        return portfolio_history
    except: return pd.Series()