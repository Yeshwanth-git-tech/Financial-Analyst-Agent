# scripts/market_data.py

import requests
import os

def get_stock_data(symbol):
    """
    Fetches daily stock data for the given symbol using Alpha Vantage API.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None