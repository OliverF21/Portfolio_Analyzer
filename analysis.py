import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import gmean

def calculate_risk_metrics(df):
    """
    Strict Geometric Sharpe Algorithm:
    1. Returns: Geometric Mean of Daily Returns (Annualized).
    2. Volatility: Standard Deviation of Daily Returns (Annualized).
    3. Risk-Free Rate: Fixed at 4.26%.
    """
    if df.empty or 'ticker' not in df.columns:
        return pd.DataFrame()

    metrics = []
    tickers = df['ticker'].tolist()
    
    # 1. Fetch 3 Years of DAILY Data (Crucial for Daily Geom Mean)
    try:
        data = yf.download(
            tickers, 
            period="3y", 
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
            
            # --- STEP 2: GEOMETRIC MEAN (The "Conservatism" Component) ---
            # Formula: (Product(1 + r))^(1/n) - 1
            # We use scipy.stats.gmean for precision
            # We add 1 to returns because gmean requires positive numbers
            daily_geo_mean = gmean(daily_rets + 1) - 1
            
            # Annualize: (1 + daily_geo)^252 - 1
            annualized_return = (1 + daily_geo_mean)**252 - 1

            # --- STEP 3: VOLATILITY (Std Dev from Mean) ---
            # Annualize: Daily Std * Sqrt(252)
            annualized_vol = daily_rets.std() * np.sqrt(252)
            
            # --- STEP 4: SHARPE RATIO ---
            # Formula: (E(R) - rf) / StdDev
            rf_rate = 0.0426 # User Specified 4.26%
            
            if annualized_vol > 0:
                sharpe = (annualized_return - rf_rate) / annualized_vol
            else:
                sharpe = 0.0

            # --- METRIC 5: MAX DRAWDOWN (Kept for context) ---
            rolling_max = hist.cummax()
            drawdown = (hist - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            metrics.append({
                "ticker": t,
                "sharpe": float(sharpe),
                "volatility": float(annualized_vol),
                "cagr": float(annualized_return), # Labeling as CAGR for display consistency
                "max_drawdown": float(max_drawdown)
            })
            
        except:
            metrics.append({
                "ticker": t, "sharpe": 0.0, "volatility": 0.0, 
                "cagr": 0.0, "max_drawdown": 0.0
            })
            
    return pd.DataFrame(metrics)

def get_portfolio_history(df):
    """
    Calculates the historical TOTAL VALUE of the current portfolio.
    Formula: Sum(Historical Price_i * Current Quantity_i) for all assets.
    """
    if df.empty: return pd.Series()
    
    tickers = df['ticker'].tolist()
    quantities = dict(zip(df['ticker'], df['quantity']))
    
    try:
        # Fetch 2y daily history
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)['Close']
        
        # Handle single ticker case
        if isinstance(data, pd.Series): 
            data = data.to_frame(name=tickers[0])
            
        # Fill missing data
        data = data.ffill().dropna()

        # Calculate weighted value for each day
        portfolio_history = pd.Series(0.0, index=data.index)
        
        for t in tickers:
            if t in data.columns:
                portfolio_history += data[t] * quantities.get(t, 0)
                
        return portfolio_history
        
    except Exception as e:
        print(f"History fetch error: {e}")
        return pd.Series()