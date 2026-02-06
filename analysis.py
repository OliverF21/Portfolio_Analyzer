import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    """
    INDUSTRY STANDARD 3-YEAR SHARPE CALCULATOR
    ------------------------------------------
    1. Horizon: 3 Years.
    2. Frequency: Monthly Returns.
    3. Return Metric: Arithmetic Mean (Annualized).
    4. Risk-Free Rate: 4.26%.
    """
    if df.empty or 'ticker' not in df.columns:
        return pd.DataFrame()

    metrics = []
    tickers = df['ticker'].tolist()
    
    # 1. DATA INGESTION
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
            if len(tickers) > 1:
                hist = data[t]['Close']
            else:
                hist = data['Close']
            
            hist = hist.dropna()
            
            # Validation (Need ~2y of daily data to make a 3y monthly valid)
            if len(hist) < 500: 
                raise ValueError("Insufficient Data")

            # 2. RESAMPLING TO MONTHLY
            hist_monthly = hist.resample('ME').last().dropna()
            monthly_rets = hist_monthly.pct_change().dropna()
            
            if len(monthly_rets) < 24:
                raise ValueError("Insufficient Monthly Data")

            # 3. ARITHMETIC MEAN RETURN (ANNUALIZED)
            avg_monthly_ret = monthly_rets.mean()
            annualized_return = avg_monthly_ret * 12

            # 4. VOLATILITY (ANNUALIZED)
            std_dev_monthly = monthly_rets.std()
            annualized_vol = std_dev_monthly * np.sqrt(12)
            
            # 5. SHARPE RATIO
            rf_rate = 0.0426
            if annualized_vol > 0:
                sharpe = (annualized_return - rf_rate) / annualized_vol
            else:
                sharpe = 0.0

            # 6. DRAWDOWN
            rolling_max = hist.cummax()
            drawdown = (hist - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            metrics.append({
                "ticker": t,
                "sharpe": float(sharpe),
                "volatility": float(annualized_vol),
                "cagr": float(annualized_return),
                "max_drawdown": float(max_drawdown)
            })
            
        except:
            metrics.append({
                "ticker": t, "sharpe": 0.0, "volatility": 0.0, 
                "cagr": 0.0, "max_drawdown": 0.0
            })
            
    return pd.DataFrame(metrics)

def get_portfolio_history(df):
    """Reconstructs historical value of current portfolio."""
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

def get_correlation_matrix(df):
    """Fetches correlation data to identify diversification opportunities."""
    tickers = df['ticker'].tolist()
    try:
        data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)['Close']
        if len(tickers) == 1: return pd.DataFrame()
        # Calculate correlation of daily returns
        corr_matrix = data.pct_change().corr()
        return corr_matrix
    except:
        return pd.DataFrame()

def get_optimization_suggestions(df):
    """
    Identifies 'Drag' (Low Sharpe) and 'Boost' (High Sharpe) candidates.
    Returns two DataFrames: to_trim, to_boost
    """
    if df.empty or 'sharpe' not in df.columns: return pd.DataFrame(), pd.DataFrame()
    
    # Calculate Weighted Average Sharpe of the Portfolio
    avg_sharpe = df['sharpe'].mean()
    
    # Identify inefficient assets (Sharpe < Average) & (Weight > 1%)
    to_trim = df[(df['sharpe'] < avg_sharpe) & (df['weight'] > 0.01)].sort_values('sharpe', ascending=True)
    
    # Identify efficiency leaders (Sharpe > Average)
    to_boost = df[df['sharpe'] > avg_sharpe].sort_values('sharpe', ascending=False)
    
    return to_trim, to_boost