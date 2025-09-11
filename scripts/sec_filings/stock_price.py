# scripts/get_stock_price.py

import os
import requests
from typing import Optional
from dotenv import load_dotenv

# Load the .env file from parent directory
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path)

# Correct variable name
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

def get_realtime_stock_price(ticker: str) -> Optional[float]:
    """
    Fetches the latest stock price for the given ticker from Alpha Vantage.
    Returns the price as a float, or None if something fails.
    """
    if not ALPHA_VANTAGE_API_KEY:
        raise EnvironmentError("Missing ALPHA_VANTAGE_API_KEY in environment variables.")

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    try:
        response = requests.get(ALPHA_VANTAGE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            price = float(data["Global Quote"]["05. price"])
            return price
        else:
            print(f"Could not fetch price for {ticker}. Response: {data}")
            return None

    except Exception as e:
        print(f"Error fetching stock price for {ticker}: {e}")
        return None

import json

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    price = get_realtime_stock_price(ticker)
    if price:
        print(f"{ticker} latest price: ${price}")

        print("CWD:", os.getcwd())
        
        # Save to file
        output_path = os.path.abspath( os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sec_filings', f"{ticker}_stock_price.json"))

        print(output_path)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the stock price
        with open(output_path, "w") as f:
            json.dump({"sticker": ticker, "price": price}, f)
    else:
        print("Failed to fetch price.")