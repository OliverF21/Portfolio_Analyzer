import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    """
    CAPM-Based Risk Analysis:
    1. Market Benchmark: SPY (S&P 500).
    2. Beta: Measure of systematic risk relative to SPY.
    3. Expected Return: Calculated via CAPM (Rf + Beta * (Rm - Rf)).
    4. Sharpe: (CAPM Return - Rf) / Volatility.
    """
    if df.empty or 'ticker' not in df.columns:
        return pd.DataFrame()

    metrics = []
    tickers = df['ticker'].tolist()
    
    # 1. Fetch 5 Years of Data for Tickers AND SPY (Market Proxy)
    try:
        # Deduplicate and add SPY
        download_list = list(set(tickers + ['SPY']))
        
        data = yf.download(
            download_list, 
            period="5y", 
            group_by='ticker', 
            auto_adjust=True, 
            progress=False
        )
    except:
        return pd.DataFrame()

    # 2. Process Market Data (SPY)
    try:
        spy_hist = data['SPY']['Close'].dropna()
        spy_rets = spy_hist.pct_change().dropna()
        
        # Calculate Market Return (Rm) - Annualized Geometric Mean of SPY
        spy_days = (spy_hist.index[-1] - spy_hist.index[0]).days
        spy_total_ret = (spy_hist.iloc[-1] / spy_hist.iloc[0])
        rm_annual = spy_total_ret ** (365.25 / spy_days) - 1
        
        # Market Variance (Annualized) for Beta calc
        market_var = spy_rets.var() * 252
    except:
        # Fallback if SPY fails
        rm_annual = 0.10 # Historical average approx
        market_var = 0.04 
        spy_rets = pd.Series()

    rf_rate = 0.0426 # User Specified 4.26%

    for t in tickers:
        try:
            # Handle Data Structure
            if len(download_list) > 1:
                hist = data[t]['Close']
            else:
                hist = data['Close']
            
            hist = hist.dropna()
            if len(hist) < 200: raise ValueError("Insufficient Data")

            # --- STEP 1: RETURNS & VOLATILITY ---
            daily_rets = hist.pct_change().dropna()
            
            # Align with SPY for Beta Calc
            # We join the asset returns with market returns to ensure dates match
            aligned = pd.concat([daily_rets, spy_rets], axis=1, join='inner').dropna()
            aligned.columns = ['Asset', 'Market']
            
            # Annualized Volatility
            annualized_vol = daily_rets.std() * np.sqrt(252)

            # --- STEP 2: CALCULATE BETA ---
            if not aligned.empty:
                # Covariance(Asset, Market) / Variance(Market)
                covariance = np.cov(aligned['Asset'], aligned['Market'])[0, 1]
                market_variance_daily = np.var(aligned['Market'], ddof=1)
                beta = covariance / market_variance_daily
            else:
                beta = 1.0 # Fallback

            # --- STEP 3: CAPM EXPECTED RETURN ---
            # E(R) = Rf + Beta * (Rm - Rf)
            capm_return = rf_rate + beta * (rm_annual - rf_rate)

            # --- STEP 4: SHARPE RATIO ---
            if annualized_vol > 0:
                sharpe = (capm_return - rf_rate) / annualized_vol
            else:
                sharpe = 0.0

            # --- DRAWDOWN ---
            rolling_max = hist.cummax()
            drawdown = (hist - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            metrics.append({
                "ticker": t,
                "sharpe": float(sharpe),
                "volatility": float(annualized_vol),
                "cagr": float(capm_return), # Storing CAPM E(R) in the 'return' column
                "max_drawdown": float(max_drawdown)
            })
            
        except Exception as e:
            metrics.append({
                "ticker": t, "sharpe": 0.0, "volatility": 0.0, 
                "cagr": 0.0, "max_drawdown": 0.0
            })
            
    return pd.DataFrame(metrics)

def get_portfolio_history(df):
    """Calculates historical value (Standard implementation)."""
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