import pdfplumber
import pandas as pd
import io
import json
import os
from google import genai
from google.genai import types

def get_example_csv():
    """Returns the default example portfolio as a CSV string."""
    return """ticker, quantity
AMD, 1.090641
AMZN, 2.239950
BABA, 0.597157
BRK.B, 0.094141
COST, 0.183930
D, 0.898666
FIG, 1.193170
GLD, 1.764702
GOOGL, 1.008079
HOOD, 2.446729
INTC, 4.052924
ITA, 1.107081
LMT, 0.212240
MDLN, 1.000000
MSFT, 1.196164
NVDA, 10.861051
NVDY, 9.129890
OKLO, 1.049747
PLTR, 0.570360
QQQ, 0.465272
RGTI, 4.413720
RHHBY, 4.000000
SLV, 6.178850
SPYG, 13.379623
TJX, 0.385498
TSM, 4.472605
UBER, 2.701419
URNM, 1.224473
VGK, 0.718596
VOO, 4.060522
VST, 0.406087
VUG, 0.285199
WMT, 6.547545"""

def parse_manual_data(csv_text):
    """Parses raw CSV text into the standard list-of-dicts format."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        
        # Normalize headers
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Map common header variations
        rename_map = {'symbol': 'ticker', 'qty': 'quantity', 'shares': 'quantity'}
        df.rename(columns=rename_map, inplace=True)
        
        if 'ticker' not in df.columns or 'quantity' not in df.columns:
            return []

        return df[['ticker', 'quantity']].to_dict('records')
    except Exception:
        return []

def extract_holdings_from_pdf(file_obj, api_key=None):
    """Extracts raw holding data from a PDF file object using Gemini."""
    try:
        full_text = ""
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: full_text += text + "\n"
        
        if len(full_text) > 50 and api_key:
            client = genai.Client(api_key=api_key)
            prompt = f"""
            Extract stock holdings from this text. Ignore cash/sweeps.
            Return ONLY valid JSON: {{ "holdings": [ {{"ticker": "STR", "quantity": 1.0}} ] }}
            TEXT: {full_text[:30000]}
            """
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text)
            return data.get('holdings', [])
    except:
        return []
    return []

