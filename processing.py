import pandas as pd
import yfinance as yf

def fetch_sector_map(tickers):
    """
    Fetches Sector information for a list of tickers.
    Note: This involves network calls to Yahoo Finance's metadata API.
    """
    mapping = {}
    if not tickers: return {}
    
    try:
        # Using yf.Tickers is generally more efficient for bulk metadata
        t_objs = yf.Tickers(" ".join(tickers))
        
        for t in tickers:
            try:
                # accessing .info triggers the fetch
                sector = t_objs.tickers[t].info.get('sector', 'Unknown')
                mapping[t] = sector
            except:
                mapping[t] = 'Unknown'
    except:
        # Fallback if bulk fetch fails
        for t in tickers: mapping[t] = 'Unknown'
        
    return mapping

def create_portfolio_df(holdings_list):
    """Converts raw list -> Clean DataFrame with Prices, Values, Weights, AND SECTORS."""
    if not holdings_list:
        return pd.DataFrame()

    df = pd.DataFrame(holdings_list)
    
    # 1. Normalize
    df.columns = [c.lower() for c in df.columns]
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0.0)
    
    # 2. Fetch Live Prices
    tickers = df['ticker'].tolist()
    if not tickers: return df

    try:
        batch_data = yf.download(tickers, period="1d", group_by='ticker', progress=False)
        current_prices = []
        for t in tickers:
            try:
                if len(tickers) > 1: price = batch_data[t]['Close'].iloc[-1]
                else: price = batch_data['Close'].iloc[-1]
                current_prices.append(float(price))
            except:
                current_prices.append(0.0)
    except:
        current_prices = [0.0] * len(tickers)

    df['price'] = current_prices
    df['value'] = df['quantity'] * df['price']
    
    # 3. Filter and Weight
    df = df[df['value'] > 0].copy()
    total_value = df['value'].sum()
    df['weight'] = df['value'] / total_value if total_value > 0 else 0.0

    # 4. FETCH SECTORS (The New Addition)
    # We do this after filtering to save API calls on zero-value assets
    if not df.empty:
        valid_tickers = df['ticker'].tolist()
        sector_map = fetch_sector_map(valid_tickers)
        df['sector'] = df['ticker'].map(sector_map).fillna('Unknown')
        
    return df
