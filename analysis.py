import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_metrics(df):
    """
    INDUSTRY STANDARD 3-YEAR SHARPE CALCULATOR
    ------------------------------------------
    Methodology:
    1. Horizon: 3 Years (36 Months).
    2. Frequency: Monthly Returns (Resampled from Daily).
       - Why? Daily data is too noisy for strategic ratios. Standard fact sheets use monthly.
    3. Return Metric: Arithmetic Mean (Annualized).
       - Why? Geometric mean is for wealth compounding; Arithmetic is for statistical expectation (Sharpe standard).
    4. Risk-Free Rate: 4.26% (Fixed Hurdle).
    """
    
    # --- 1. SAFETY CHECKS ---
    # Ensure we have a valid dataframe with a 'ticker' column
    if df.empty or 'ticker' not in df.columns:
        return pd.DataFrame()

    metrics = []
    tickers = df['ticker'].tolist()
    
    # --- 2. DATA INGESTION (NO SHORTCUTS) ---
    # We fetch daily data first, then resample. This is more accurate than fetching "1mo" 
    # intervals directly because yfinance monthly data can sometimes behave oddly with dividends.
    try:
        data = yf.download(
            tickers, 
            period="3y",      # Explicit 3-Year Lookback
            group_by='ticker', 
            auto_adjust=True, # Critical: Includes Dividends in the price (Total Return)
            progress=False
        )
    except:
        return pd.DataFrame()

    # --- 3. TICKER-BY-TICKER ANALYSIS ---
    for t in tickers:
        try:
            # A. Extract Series
            # Handle multi-index dataframe structure safely
            if len(tickers) > 1:
                hist = data[t]['Close']
            else:
                hist = data['Close']
            
            # B. Clean Data
            # Remove NaN values that might exist at the start/end
            hist = hist.dropna()
            
            # C. Validation
            # A 3-Year monthly calculation needs ~36 months. 
            # If we have less than 24 months (2 years) of daily data, the metric is statistically junk.
            if len(hist) < 500: # approx 2 trading years
                raise ValueError("Insufficient Data (Less than 2y history)")

            # --- 4. RESAMPLING (THE STANDARD TECHNIQUE) ---
            # We convert Daily Price -> Monthly Ending Price.
            # 'ME' stands for Month End.
            hist_monthly = hist.resample('ME').last().dropna()

            # --- 5. CALCULATE RETURNS ---
            # Percentage change between months.
            monthly_rets = hist_monthly.pct_change().dropna()
            
            if len(monthly_rets) < 24:
                raise ValueError("Insufficient Monthly Data")

            # --- 6. ARITHMETIC MEAN RETURN (ANNUALIZED) ---
            # Standard Formula: Average Monthly Return * 12
            # Note: We do NOT use Geometric mean here, as standard Sharpe uses Arithmetic expected return.
            avg_monthly_ret = monthly_rets.mean()
            annualized_return = avg_monthly_ret * 12

            # --- 7. VOLATILITY (ANNUALIZED) ---
            # Standard Formula: Std Dev of Monthly Returns * Sqrt(12)
            std_dev_monthly = monthly_rets.std()
            annualized_vol = std_dev_monthly * np.sqrt(12)
            
            # --- 8. SHARPE RATIO CALCULATION ---
            # Formula: (Expected Return - Risk Free Rate) / Volatility
            rf_rate = 0.0426 # User Constraint: 4.26%
            
            if annualized_vol > 0:
                sharpe = (annualized_return - rf_rate) / annualized_vol
            else:
                sharpe = 0.0

            # --- 9. CONTEXTUAL METRICS (Drawdown) ---
            # We use the daily data for this to catch intra-month crashes
            rolling_max = hist.cummax()
            drawdown = (hist - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            # --- 10. PACKAGING ---
            metrics.append({
                "ticker": t,
                "sharpe": float(sharpe),
                "volatility": float(annualized_vol),
                "cagr": float(annualized_return), # Labeled CAGR for UI consistency, but acts as Exp. Return
                "max_drawdown": float(max_drawdown)
            })
            
        except:
            # Fallback for new IPOs or errors
            metrics.append({
                "ticker": t, "sharpe": 0.0, "volatility": 0.0, 
                "cagr": 0.0, "max_drawdown": 0.0
            })
            
    return pd.DataFrame(metrics)

def get_portfolio_history(df):
    """
    Reconstructs the 2-Year historical value of the *current* portfolio 
    quantity held constant.
    """
    if df.empty: return pd.Series()
    
    tickers = df['ticker'].tolist()
    quantities = dict(zip(df['ticker'], df['quantity']))
    
    try:
        # Fetch 2y daily history for the visualization
        data = yf.download(tickers, period="2y", auto_adjust=True, progress=False)['Close']
        
        if isinstance(data, pd.Series): 
            data = data.to_frame(name=tickers[0])
            
        data = data.ffill().dropna()

        portfolio_history = pd.Series(0.0, index=data.index)
        
        for t in tickers:
            if t in data.columns:
                portfolio_history += data[t] * quantities.get(t, 0)
                
        return portfolio_history
        
    except Exception as e:
        print(f"History fetch error: {e}")
        return pd.Series()