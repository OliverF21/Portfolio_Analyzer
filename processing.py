import pandas as pd
import yfinance as yf

def create_portfolio_df(holdings_list):
    """
    Converts raw list -> Clean DataFrame with Prices, Values, and Weights.
    """
    if not holdings_list:
        return pd.DataFrame()

    df = pd.DataFrame(holdings_list)
    
    # 1. Normalize Columns
    df.columns = [c.lower() for c in df.columns] # ensure 'ticker', 'quantity'
    
    # 2. Force Numeric
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0.0)
    
    # 3. Fetch Live Prices (Batch fetch for speed)
    tickers = df['ticker'].tolist()
    if not tickers:
        return df

    try:
        # Fetch last close for all tickers
        batch_data = yf.download(tickers, period="1d", group_by='ticker', progress=False)
        
        current_prices = []
        for t in tickers:
            try:
                # Handle single vs multi-ticker result structures
                if len(tickers) > 1:
                    price = batch_data[t]['Close'].iloc[-1]
                else:
                    price = batch_data['Close'].iloc[-1]
                current_prices.append(float(price))
            except:
                current_prices.append(0.0)
    except:
        current_prices = [0.0] * len(tickers)

    # 4. Calculate Financials
    df['price'] = current_prices
    df['value'] = df['quantity'] * df['price']
    
    # Filter out zero values to keep table clean
    df = df[df['value'] > 0].copy()
    
    total_value = df['value'].sum()
    if total_value > 0:
        df['weight'] = df['value'] / total_value
    else:
        df['weight'] = 0.0
        
    return df