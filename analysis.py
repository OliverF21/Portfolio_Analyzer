import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    """ (Keep Industry Standard 3Y Sharpe Code) """
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
            hist = hist.dropna()
            if len(hist) < 500: raise ValueError("Insufficient Data")

            hist_monthly = hist.resample('ME').last().dropna()
            monthly_rets = hist_monthly.pct_change().dropna()
            if len(monthly_rets) < 24: raise ValueError("Insufficient Monthly Data")

            avg_monthly_ret = monthly_rets.mean()
            annualized_return = avg_monthly_ret * 12
            std_dev_monthly = monthly_rets.std()
            annualized_vol = std_dev_monthly * np.sqrt(12)
            
            rf_rate = 0.0426
            if annualized_vol > 0:
                sharpe = (annualized_return - rf_rate) / annualized_vol
            else:
                sharpe = 0.0

            rolling_max = hist.cummax()
            drawdown = (hist - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            metrics.append({
                "ticker": t, "sharpe": float(sharpe), "volatility": float(annualized_vol),
                "cagr": float(annualized_return), "max_drawdown": float(max_drawdown)
            })
        except:
            metrics.append({"ticker": t, "sharpe": 0.0, "volatility": 0.0, "cagr": 0.0, "max_drawdown": 0.0})     
    return pd.DataFrame(metrics)

def get_portfolio_history(df):
    """ (Keep History Code) """
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

def get_correlation_matrix(df):
    """
    Fetches correlation data, CLUSTERED BY SECTOR.
    This groups similar industries together in the heatmap (e.g. all Tech together).
    """
    if df.empty: return pd.DataFrame()
    
    # 1. Sort the DataFrame by Sector first, then Ticker
    if 'sector' in df.columns:
        sorted_df = df.sort_values(by=['sector', 'ticker'])
    else:
        sorted_df = df.sort_values(by='ticker')
        
    sorted_tickers = sorted_df['ticker'].tolist()
    
    try:
        data = yf.download(sorted_tickers, period="1y", auto_adjust=True, progress=False)['Close']
        if len(sorted_tickers) == 1: return pd.DataFrame()
        
        # Ensure we process columns in the SORTED order
        # Handle cases where some tickers might have failed download
        available_cols = [t for t in sorted_tickers if t in data.columns]
        data = data[available_cols]
        
        corr_matrix = data.pct_change().corr()
        return corr_matrix
    except:
        return pd.DataFrame()

def get_optimization_suggestions(df):
    """ (Keep Optimization Code) """
    if df.empty or 'sharpe' not in df.columns: return pd.DataFrame(), pd.DataFrame()
    avg_sharpe = df['sharpe'].mean()
    to_trim = df[(df['sharpe'] < avg_sharpe) & (df['weight'] > 0.01)].sort_values('sharpe', ascending=True)
    to_boost = df[df['sharpe'] > avg_sharpe].sort_values('sharpe', ascending=False)
    return to_trim, to_boost
