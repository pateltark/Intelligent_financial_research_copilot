"""
FILE 1: ingestion/sec_scraper.py
=================================
PURPOSE  : Download 10-K, 10-Q, 8-K filings from SEC EDGAR
COST     : 100% FREE — SEC EDGAR is a US government public database
API KEY  : NOT needed. Just set your name+email as User-Agent (SEC policy)
SAVES TO : ./data/raw/  folder

HOW IT WORKS (simple explanation):
  Step 1 → Convert ticker (AAPL) to SEC company ID called CIK
  Step 2 → Ask SEC API: "give me list of recent filings for this CIK"
  Step 3 → Download the actual filing document (HTML or PDF)
  Step 4 → Save to disk with metadata

RUN IT:
  python ingestion/sec_scraper.py
"""

import os
import time
import json
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Settings — change these ─────────────────────────────────────
# SEC REQUIRES this header — put your real name and email
# They use it to contact you if your scraper causes issues
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "FinancialCopilot yourname@email.com"   # ← change this
)

# Which companies to track
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

# Which filing types to download
# 10-K  = annual report (most important, very detailed)
# 10-Q  = quarterly report (every 3 months)
# 8-K   = major event report (earnings, CEO change, lawsuits etc)
FILING_TYPES = ["10-K", "10-Q", "8-K"]

# Where to save downloaded files
SAVE_DIR = "./data/raw"

# SEC base URLs (all public, no auth needed)
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
EDGAR_ARCHIVE_URL     = "https://www.sec.gov/Archives/edgar/data"
COMPANY_TICKERS_URL   = "https://www.sec.gov/files/company_tickers.json"


# ══════════════════════════════════════════════════════════════════
# STEP 1: Get CIK number from ticker symbol
# ══════════════════════════════════════════════════════════════════
def get_cik(ticker: str) -> str | None:
    """
    What is CIK?
    ------------
    CIK = Central Index Key. Every company registered with SEC has one.
    Example: Apple = 0000320193, Microsoft = 0000789019

    How we get it:
    --------------
    SEC publishes a public JSON file that maps ALL tickers → CIK numbers.
    We download it once and search through it.

    Returns: CIK as zero-padded 10-digit string, or None if not found
    """
    log.info(f"Looking up CIK for ticker: {ticker}")

    headers = {"User-Agent": SEC_USER_AGENT}
    response = requests.get(COMPANY_TICKERS_URL, headers=headers, timeout=15)
    response.raise_for_status()

    # The JSON looks like: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    all_companies = response.json()

    ticker_upper = ticker.upper()
    for company in all_companies.values():
        if company["ticker"].upper() == ticker_upper:
            # Zero-pad to 10 digits: 320193 → "0000320193"
            cik = str(company["cik_str"]).zfill(10)
            log.info(f"  Found CIK for {ticker}: {cik} ({company['title']})")
            return cik

    log.warning(f"  Could not find CIK for ticker: {ticker}")
    return None


# ══════════════════════════════════════════════════════════════════
# STEP 2: Get list of filings for a company
# ══════════════════════════════════════════════════════════════════
def get_filings_list(cik: str, ticker: str, days_back: int = 90) -> list[dict]:
    """
    What this does:
    ---------------
    Calls SEC API: "Give me all recent filings for company CIK=0000320193"
    Filters to only the filing types we want (10-K, 10-Q, 8-K)
    Filters to only filings from the last N days

    Returns: list of filing metadata dicts
    """
    url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik}.json"
    headers = {"User-Agent": SEC_USER_AGENT}

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    # SEC API returns all filings in arrays — each index matches across arrays
    # Example: forms[0]="10-K", dates[0]="2024-01-15", accessions[0]="0000320193-24-000006"
    recent = data.get("filings", {}).get("recent", {})
    forms        = recent.get("form", [])
    dates        = recent.get("filingDate", [])
    accessions   = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    # Date cutoff
    cutoff_date = datetime.now() - timedelta(days=days_back)

    results = []
    for form, date_str, accession, doc_filename in zip(forms, dates, accessions, primary_docs):

        # Only keep filing types we want
        if form not in FILING_TYPES:
            continue

        # Only keep recent filings
        filing_date = datetime.strptime(date_str, "%Y-%m-%d")
        if filing_date < cutoff_date:
            continue

        # Build the direct URL to the document
        # SEC URL format: /Archives/edgar/data/{cik_no_padding}/{accession_no_dashes}/{filename}
        cik_no_padding    = str(int(cik))           # "0000320193" → "320193"
        accession_no_dash = accession.replace("-", "")  # "0000320193-24-000006" → "000032019324000006"
        doc_url = f"{EDGAR_ARCHIVE_URL}/{cik_no_padding}/{accession_no_dash}/{doc_filename}"

        results.append({
            "ticker":     ticker,
            "cik":        cik,
            "form":       form,
            "date":       date_str,
            "accession":  accession,
            "doc_url":    doc_url,
            "filename":   doc_filename,
        })

    log.info(f"  Found {len(results)} {FILING_TYPES} filings in last {days_back} days")
    return results


# ══════════════════════════════════════════════════════════════════
# STEP 3: Download the actual document
# ══════════════════════════════════════════════════════════════════
def download_filing(filing: dict) -> str | None:
    """
    Downloads the HTML or PDF document from SEC and saves it to ./data/raw/

    File naming convention:
        AAPL_10-K_2024-01-15_filename.htm
        ↑     ↑    ↑          ↑
      ticker form  date    original filename

    Returns: local file path if successful, None if failed
    """
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Build local filename
    safe_filename = f"{filing['ticker']}_{filing['form']}_{filing['date']}_{filing['filename']}"
    safe_filename = safe_filename.replace("/", "_")  # sanitize
    local_path    = os.path.join(SAVE_DIR, safe_filename)

    # Skip if already downloaded (avoid re-downloading on re-runs)
    if os.path.exists(local_path):
        log.info(f"  Already exists, skipping: {safe_filename}")
        return local_path

    try:
        headers  = {"User-Agent": SEC_USER_AGENT}
        response = requests.get(filing["doc_url"], headers=headers, timeout=30)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(response.content)

        size_kb = len(response.content) // 1024
        log.info(f"  Downloaded: {safe_filename} ({size_kb} KB)")

        # IMPORTANT: SEC rate limit = max 10 requests/second
        # We sleep 0.2s between downloads to stay well under that
        time.sleep(0.2)

        return local_path

    except requests.exceptions.HTTPError as e:
        log.error(f"  HTTP error downloading {filing['doc_url']}: {e}")
        return None
    except Exception as e:
        log.error(f"  Failed to download {filing['doc_url']}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# MAIN: Run scraper for all tickers
# ══════════════════════════════════════════════════════════════════
def run_sec_scraper(
    tickers:  list[str] = TICKERS,
    days_back: int      = 90
) -> list[dict]:
    """
    Main function — runs the full scraping pipeline for all tickers.

    Returns:
        List of dicts, one per successfully downloaded filing:
        {
            "ticker":     "AAPL",
            "cik":        "0000320193",
            "form":       "10-K",
            "date":       "2024-01-15",
            "local_path": "./data/raw/AAPL_10-K_2024-01-15_aapl10k.htm",
            "doc_url":    "https://www.sec.gov/Archives/...",
            "source":     "SEC_EDGAR"
        }
    """
    log.info(f"Starting SEC EDGAR scraper for {len(tickers)} tickers...")
    log.info(f"Looking back {days_back} days | Filing types: {FILING_TYPES}")
    log.info("=" * 60)

    all_downloaded = []

    for ticker in tickers:
        log.info(f"\nProcessing: {ticker}")

        # Step 1: Get CIK
        cik = get_cik(ticker)
        if not cik:
            log.warning(f"Skipping {ticker} — CIK not found")
            continue

        # Step 2: Get filings list
        filings = get_filings_list(cik, ticker, days_back=days_back)
        if not filings:
            log.info(f"  No recent filings found for {ticker}")
            continue

        # Step 3: Download each filing
        for filing in filings:
            local_path = download_filing(filing)
            if local_path:
                all_downloaded.append({
                    "ticker":     ticker,
                    "cik":        cik,
                    "form":       filing["form"],
                    "date":       filing["date"],
                    "local_path": local_path,
                    "doc_url":    filing["doc_url"],
                    "source":     "SEC_EDGAR",
                })

        # Pause 1 second between companies (polite to SEC servers)
        time.sleep(1)

    # Save manifest — this is used by parser.py in the next step
    manifest_path = os.path.join(SAVE_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(all_downloaded, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info(f"DONE. Downloaded {len(all_downloaded)} filings")
    log.info(f"Manifest saved to: {manifest_path}")
    return all_downloaded


# ══════════════════════════════════════════════════════════════════
# Run directly for testing
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Start with just 2 companies and 30 days to test quickly
    results = run_sec_scraper(
        tickers   = ["AAPL", "MSFT"],
        days_back = 30
    )

    print(f"\n{'='*50}")
    print(f"Total filings downloaded: {len(results)}")
    for r in results:
        print(f"  {r['ticker']} | {r['form']} | {r['date']} → {r['local_path']}")