import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    if df.empty or 'ticker' not in df.columns: return pd.DataFrame()
    metrics = []
    tickers = df['ticker'].tolist()
    
    try:
        # Fetch 3 Years of TOTAL RETURN (Dividends included)
        data = yf.download(tickers, period="3y", group_by='ticker', auto_adjust=True, progress=False)
    except: return pd.DataFrame()

    for t in tickers:
        try:
            if len(tickers) > 1: hist = data[t]['Close']
            else: hist = data['Close']
            
            hist_monthly = hist.resample('ME').last().dropna()
            if len(hist_monthly) < 12: raise ValueError("Insufficient Data")

            # 1. CAGR (Geometric Mean)
            start_price = hist_monthly.iloc[0]
            end_price = hist_monthly.iloc[-1]
            years = (hist_monthly.index[-1] - hist_monthly.index[0]).days / 365.25
            cagr = (end_price / start_price) ** (1 / years) - 1 if start_price > 0 else 0

            # 2. Volatility
            monthly_rets = hist_monthly.pct_change().dropna()
            ann_vol = monthly_rets.std() * np.sqrt(12)
            
            # 3. Sharpe (5% Hurdle)
            sharpe = (cagr - 0.05) / ann_vol if ann_vol > 0 else 0

            # 4. Max Drawdown
            rolling_max = hist_monthly.cummax()
            drawdown = (hist_monthly - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            metrics.append({"ticker": t, "sharpe": float(sharpe), "volatility": float(ann_vol), "cagr": float(cagr), "max_drawdown": float(max_drawdown)})
        except:
            metrics.append({"ticker": t, "sharpe": 0.0, "volatility": 0.0, "cagr": 0.0, "max_drawdown": 0.0})
            
    return pd.DataFrame(metrics)

def get_portfolio_history(df):
    if df.empty: return pd.Series()
    tickers = df['ticker'].tolist()
    quantities = dict(zip(df['ticker'], df['quantity']))
    
    try:
        # Fetch 2y daily history
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
        data = data.ffill().dropna()

        # Calculate Total Portfolio Value over time
        portfolio_history = pd.Series(0.0, index=data.index)
        for t in tickers:
            if t in data.columns:
                portfolio_history += data[t] * quantities.get(t, 0)
        return portfolio_history
    except: return pd.Series()