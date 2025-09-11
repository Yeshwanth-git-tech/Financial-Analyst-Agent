# scripts/sticker_lookup.py

import requests
from typing import Optional, Dict

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {
    "User-Agent": "StudentFinanceResearch/1.0 (Yeshwanth Satheesh; yeshwanthsatheesh91@gmail.com)"
}


def load_ticker_cik_map() -> Dict[str, Dict[str, str]]:
    """
    Loads the ticker-to-CIK mapping from SEC's public JSON endpoint.
    Handles both old and new JSON formats.
    """
    resp = requests.get(TICKER_CIK_URL, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}

    if isinstance(data, list):  # Newer SEC format
        for item in data:
            ticker = item["ticker"].upper()
            cik = str(item["cik"]).zfill(10)
            name = item["title"]
            mapping[ticker] = {"cik": cik, "name": name}
    elif isinstance(data, dict) and "0" in data:  # Older format
        for _, item in data.items():
            ticker = item["ticker"].upper()
            cik = str(item["cik_str"]).zfill(10)
            name = item["title"]
            mapping[ticker] = {"cik": cik, "name": name}
    else:
        raise ValueError("Unexpected SEC JSON format.")

    return mapping


def get_company_info(query: str) -> Optional[Dict[str, str]]:
    """
    Looks up company info (CIK, ticker, title) based on user query.
    Supports either ticker or company name.
    """
    mapping = load_ticker_cik_map()
    query_lower = query.strip().lower()

    for ticker, info in mapping.items():
        if query_lower == ticker.lower() or query_lower == info["name"].lower():
            return {
                "cik": info["cik"],
                "sticker": ticker,
                "title": info["name"]
            }

    return None


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """
    Shortcut to get just the CIK for a given ticker.
    """
    info = get_company_info(ticker)
    return info["cik"] if info else None


# Optional CLI usage
if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    print(get_company_info(q))