import pdfplumber
import os
import json
from google import genai
from google.genai import types

def extract_holdings(file_obj, api_key=None, use_vision=False):
    """
    Extracts raw holding data from a PDF file object.
    Returns a list of dictionaries: [{'ticker': 'AAPL', 'quantity': 10}, ...]
    """
    # PATH A: TEXT EXTRACTION (Fast & Free)
    if not use_vision:
        full_text = ""
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: full_text += text + "\n"
        
        # If we successfully extracted text, use Gemini to parse it to JSON
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
            
    # PATH B: VISION EXTRACTION (For Scans) - Placeholder logic if you implement file upload
    # (Requires uploading file to Google servers first, omitted here for speed/simplicity 
    # unless specifically requested for the modular version)
    
    return []