import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    """
    Calculates Conservative Risk Metrics:
    1. Total Return (Dividends Reinvested via auto_adjust=True)
    2. 5% Hurdle Rate for Sharpe
    3. Max Drawdown
    """
    if df.empty or 'ticker' not in df.columns:
        return pd.DataFrame()

    metrics = []
    tickers = df['ticker'].tolist()
    
    # 1. Fetch 3 Years of TOTAL RETURN Data
    try:
        data = yf.download(
            tickers, 
            period="3y", 
            group_by='ticker', 
            auto_adjust=True, # Critical: Includes Dividends
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
            
            # 2. Resample to Monthly (Smoother, Conservative View)
            hist_monthly = hist.resample('ME').last().dropna()
            
            if len(hist_monthly) < 12: 
                raise ValueError("Insufficient Data")

            # --- METRIC 1: REALITY (CAGR) ---
            start_price = hist_monthly.iloc[0]
            end_price = hist_monthly.iloc[-1]
            years = (hist_monthly.index[-1] - hist_monthly.index[0]).days / 365.25
            
            if start_price > 0 and years > 0:
                cagr = (end_price / start_price) ** (1 / years) - 1
            else:
                cagr = 0.0

            # --- METRIC 2: RISK (Volatility) ---
            monthly_rets = hist_monthly.pct_change().dropna()
            ann_vol = monthly_rets.std() * np.sqrt(12)
            
            # --- METRIC 3: CONSERVATISM (Sharpe w/ High Hurdle) ---
            rf_rate = 0.05
            if ann_vol > 0:
                sharpe = (cagr - rf_rate) / ann_vol
            else:
                sharpe = 0.0

            # --- METRIC 4: GUT CHECK (Max Drawdown) ---
            rolling_max = hist_monthly.cummax()
            drawdown = (hist_monthly - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            metrics.append({
                "ticker": t,
                "sharpe": float(sharpe),
                "volatility": float(ann_vol),
                "cagr": float(cagr),
                "max_drawdown": float(max_drawdown)
            })
            
        except:
            metrics.append({
                "ticker": t, "sharpe": 0.0, "volatility": 0.0, 
                "cagr": 0.0, "max_drawdown": 0.0
            })
            
    return pd.DataFrame(metrics)

def get_comparative_performance(tickers):
    """
    Fetches 2-year daily history for all tickers.
    Normalizes them to % Return (starts at 0%) for visual comparison.
    """
    if not tickers: return pd.DataFrame()
    
    try:
        # Download daily data
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)['Close']
        
        # Handle single ticker case
        if isinstance(data, pd.Series): 
            data = data.to_frame(name=tickers[0])
            
        # Normalize: (Price / StartPrice) - 1
        normalized = (data / data.iloc[0]) - 1
        return normalized
    except Exception as e:
        print(f"Performance fetch error: {e}")
        return pd.DataFrame()
    