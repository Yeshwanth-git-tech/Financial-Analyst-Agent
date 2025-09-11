# scripts/sec_filings/get_latest_filing.py

import os
import sys
import requests
from typing import Optional, Tuple
from bs4 import BeautifulSoup

from scripts.sec_filings.sticker_lookup import get_company_info

SEC_HEADERS = {
    "User-Agent": "StudentFinanceResearch/1.0 (Yeshwanth Satheesh; yeshwanthsatheesh91@gmail.com)"
}

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/"

SAVE_DIR = "data/sec_filings"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_latest_filing_metadata(cik: str, form_type: str) -> Optional[Tuple[str, str]]:
    cik = cik.zfill(10)
    url = SEC_SUBMISSIONS_URL.format(cik)

    try:
        resp = requests.get(url, headers=SEC_HEADERS)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error fetching submissions for CIK {cik}: {e}")
        return None

    data = resp.json()
    filings = data.get("filings", {}).get("recent", {})
    form_types = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    primary_documents = filings.get("primaryDocument", [])

    for form, acc_no, doc in zip(form_types, accession_numbers, primary_documents):
        if form == form_type:
            acc_no_clean = acc_no.replace("-", "")
            return acc_no_clean, doc

    print(f"No {form_type} filing found for CIK {cik}")
    return None

def download_and_save_filing(cik: str, acc_no: str, doc: str, ticker: str, form_type: str):
    filing_url = f"{SEC_ARCHIVES_BASE}edgar/data/{cik}/{acc_no}/{doc}"
    try:
        resp = requests.get(filing_url, headers=SEC_HEADERS)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error downloading filing: {e}")
        return

    html = resp.text
    filename_base = f"{ticker}_{form_type}_{acc_no}"
    html_path = os.path.join(SAVE_DIR, f"{filename_base}.html")
    md_path = os.path.join(SAVE_DIR, f"{filename_base}.md")

    with open(html_path, "w") as f:
        f.write(html)

    # Also convert to Markdown (for clean RAG ingestion)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    with open(md_path, "w") as f:
        f.write(text)

    print(f"\n✅ Saved filing to:")
    print(f"→ HTML: {html_path}")
    print(f"→ Markdown: {md_path}")
    print(f"→ Original URL: {filing_url}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.sec_filings.get_latest_filing <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1]
    form_types = ["10-K", "10-Q", "8-K"]

    company_info = get_company_info(ticker)
    if not company_info:
        print(f"Could not find company info for ticker {ticker}")
        sys.exit(1)

    cik = company_info["cik"]
    title = company_info["title"]

    print(f"\n Fetching latest filings for {title} ({ticker}) [CIK: {cik}]")

    for form_type in form_types:
        print(f"\n Looking for latest {form_type}...")
        result = get_latest_filing_metadata(cik, form_type)
        if result is None:
            print(f"No recent {form_type} filing found.")
            continue

        acc_no, primary_doc = result
        download_and_save_filing(cik, acc_no, primary_doc, ticker, form_type)

if __name__ == "__main__":
    main()