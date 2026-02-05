from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import pandas as pd
import yfinance as yf
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv() # This loads the variables from .env


# 1. Setup Client
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY")) # Replace with your actual key
MODEL_ID = "gemini-2.0-flash" 

# 2. Define the Schema (Structured Output)
class Holding(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL)")
    quantity: float = Field(description="The total number of shares owned")

class Portfolio(BaseModel):
    holdings: list[Holding]

def extract_and_analyze(pdf_path):
    # --- STEP 1: PARSE PDF WITH GEMINI ---
    print("Uploading and parsing PDF with Gemini...")
    
    # Upload file to Gemini Files API (Stored for 48 hours)
    statement_file = client.files.upload(file=pdf_path)
    
    prompt = """Extract every stock and ETF holding from this monthly statement. 
    Look at BOTH 'Securities Held' and 'Loaned Securities' sections. 
    Combine quantities if the same ticker appears in both. 
    Return a clean list of Tickers and Quantities."""

    # Generate structured response
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[statement_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Portfolio
        )
    )
    
    holdings_list = response.parsed.holdings
    df_holdings = pd.DataFrame([h.dict() for h in holdings_list])
    
    # --- STEP 2: CALCULATE WEIGHTS & SHARPE ---
    print("Fetching market data and calculating Sharpe Ratios...")
    tickers = df_holdings['ticker'].tolist()
    
    # Get current prices for weighting
    current_prices = yf.download(tickers, period="1d")['Close'].iloc[-1]
    df_holdings['price'] = df_holdings['ticker'].map(current_prices)
    df_holdings['market_value'] = df_holdings['quantity'] * df_holdings['price']
    df_holdings['weight'] = df_holdings['market_value'] / df_holdings['market_value'].sum()
    
    # Fetch 10-year history for analysis
    hist_data = yf.download(tickers, period="10y", group_by='ticker')
    
    results = []
    for ticker in tickers:
        try:
            # Annualized Return & Volatility calculation
            stock_hist = hist_data[ticker].dropna()
            returns = stock_hist['Close'].pct_change()
            ann_return = returns.mean() * 252
            ann_vol = returns.std() * np.sqrt(252)
            sharpe = (ann_return - 0.04) / ann_vol # 4% Risk-free rate
            
            results.append({'Ticker': ticker, 'Sharpe': sharpe})
        except:
            results.append({'Ticker': ticker, 'Sharpe': np.nan})
            
    # --- STEP 3: RESULTS ---
    df_final = pd.merge(df_holdings, pd.DataFrame(results), left_on='ticker', right_on='Ticker')
    portfolio_sharpe = (df_final['weight'] * df_final['Sharpe'].fillna(0)).sum()
    
    print("\n--- FINAL PORTFOLIO SHARPE RATIO ---")
    print(f"Aggregated Score: {portfolio_sharpe:.2f}")
    print("\nIndividual Breakdown:")
    print(df_final[['ticker', 'weight', 'Sharpe']].sort_values('weight', ascending=False))

# Run the analyzer
extract_and_analyze("statement.pdf")