import os
import json
import time
import logging
import hashlib
import requests
from datetime import datetime
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────
# Get your FREE key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")

# Companies to fetch news for
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

# How many articles per company (max 50 on free tier)
ARTICLES_PER_TICKER = 20

# Where to save news files
SAVE_DIR = "./data/raw/news"

# AlphaVantage API endpoint
AV_BASE_URL = "https://www.alphavantage.co/query"


# ══════════════════════════════════════════════════════════════════
# Fetch news for one ticker
# ══════════════════════════════════════════════════════════════════
def fetch_news_for_ticker(ticker: str, limit: int = 20) -> list[dict]:
    log.info(f"Fetching news for: {ticker}")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers":  ticker,
        "limit":    limit,
        "sort":     "LATEST",          # most recent first
        "apikey":   ALPHA_VANTAGE_KEY,
    }

    try:
        response = requests.get(AV_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Check for API limit message
        if "Information" in data:
            log.warning(f"  API limit hit: {data['Information']}")
            log.warning("  Free tier = 25 req/day. Wait 24hrs or get a new free key.")
            return []

        if "Error Message" in data:
            log.error(f"  API error: {data['Error Message']}")
            return []

        raw_articles = data.get("feed", [])
        log.info(f"  Got {len(raw_articles)} articles for {ticker}")

        # ── Normalize to our standard format ──────────────────────
        normalized = []
        for article in raw_articles:

            # Skip articles with no useful content
            if not article.get("title") or not article.get("summary"):
                continue

            # Find this ticker's specific sentiment in the article
            # (one article can mention multiple tickers)
            ticker_sentiment_score = 0.0
            ticker_sentiment_label = "Neutral"
            for ticker_data in article.get("ticker_sentiment", []):
                if ticker_data.get("ticker") == ticker:
                    ticker_sentiment_score = float(ticker_data.get("ticker_sentiment_score", 0))
                    ticker_sentiment_label = ticker_data.get("ticker_sentiment_label", "Neutral")
                    break

            # Create unique ID so we don't save duplicates
            content_hash = hashlib.md5(
                (article.get("title", "") + article.get("url", "")).encode()
            ).hexdigest()[:12]

            normalized.append({
                # ── Core content ──
                "id":           f"{ticker}_{content_hash}",
                "ticker":       ticker,
                "title":        article["title"],
                "summary":      article.get("summary", ""),  # ← we embed this into RAG
                "url":          article.get("url", ""),
                "source":       article.get("source", "Unknown"),

                # ── Time ──
                # AlphaVantage format: "20240115T143000" → we convert to readable
                "published_at": _parse_av_date(article.get("time_published", "")),
                "published_raw": article.get("time_published", ""),

                # ── Sentiment scores (FREE from AlphaVantage!) ──
                "overall_sentiment_label": article.get("overall_sentiment_label", "Neutral"),
                "overall_sentiment_score": float(article.get("overall_sentiment_score", 0)),
                "ticker_sentiment_label":  ticker_sentiment_label,
                "ticker_sentiment_score":  ticker_sentiment_score,

                # ── Metadata ──
                "topics":       [t.get("topic", "") for t in article.get("topics", [])],
                "source_type":  "NEWS",
                "fetched_at":   datetime.utcnow().isoformat(),
            })

        return normalized

    except requests.exceptions.Timeout:
        log.error(f"  Request timed out for {ticker}")
        return []
    except Exception as e:
        log.error(f"  Error fetching news for {ticker}: {e}")
        return []


def _parse_av_date(raw: str) -> str:
    """
    Convert AlphaVantage date format to readable ISO format.
    Input:  "20240115T143000"
    Output: "2024-01-15T14:30:00"
    """
    try:
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%S")
        return dt.isoformat()
    except Exception:
        return raw


# ══════════════════════════════════════════════════════════════════
# Save articles to disk
# ══════════════════════════════════════════════════════════════════
def save_articles(articles: list[dict], ticker: str) -> str:
    """
    Saves articles for one ticker to a JSON file.
    File path: ./data/raw/news/AAPL_2024-01-15.json

    Why JSON files instead of a database?
    → Simple, no setup, easy to inspect, easy to read in parser.py
    """
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    today     = datetime.utcnow().strftime("%Y-%m-%d")
    filepath  = os.path.join(SAVE_DIR, f"{ticker}_{today}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    log.info(f"  Saved {len(articles)} articles → {filepath}")
    return filepath


# ══════════════════════════════════════════════════════════════════
# MAIN: Run for all tickers
# ══════════════════════════════════════════════════════════════════
def run_news_scraper(
    tickers: list[str] = TICKERS,
    articles_per_ticker: int = ARTICLES_PER_TICKER
) -> list[dict]:
    
    if ALPHA_VANTAGE_KEY == "demo":
        log.warning("⚠  Using 'demo' API key — very limited results")
        log.warning("   Get your FREE key at: https://www.alphavantage.co/support/#api-key")

    log.info(f"Starting news scraper for {len(tickers)} tickers")
    log.info("=" * 60)

    all_articles  = []
    saved_files   = []
    rate_limited  = False

    for i, ticker in enumerate(tickers):
        if rate_limited:
            log.warning(f"Skipping {ticker} — rate limited")
            continue

        articles = fetch_news_for_ticker(ticker, limit=articles_per_ticker)

        if not articles:
            # Might be rate limited — check next ticker but warn
            log.warning(f"  No articles returned for {ticker} — may be rate limited")
            rate_limited = True
            continue

        # Save to disk
        filepath = save_articles(articles, ticker)
        saved_files.append(filepath)
        all_articles.extend(articles)

        # Rate limit: AlphaVantage free = 5 requests/min as well as 25/day
        # Sleep 15 seconds between requests to avoid hitting per-minute limit
        if i < len(tickers) - 1:
            log.info(f"  Waiting 15s before next request (rate limit protection)...")
            time.sleep(15)

    # Save combined manifest
    manifest = {
        "fetched_at":    datetime.utcnow().isoformat(),
        "total_articles": len(all_articles),
        "tickers":       tickers,
        "saved_files":   saved_files,
    }
    manifest_path = os.path.join(SAVE_DIR, "news_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info(f"DONE. Fetched {len(all_articles)} articles across {len(saved_files)} tickers")
    return all_articles


# ══════════════════════════════════════════════════════════════════
# Helper: Load previously saved news from disk
# ══════════════════════════════════════════════════════════════════
def load_all_saved_news() -> list[dict]:
    """
    Reads all saved news JSON files from ./data/raw/news/
    Used by parser.py to process already-downloaded news.
    """
    news_dir = Path(SAVE_DIR)
    if not news_dir.exists():
        return []

    all_articles = []
    for json_file in news_dir.glob("*.json"):
        if json_file.name in ("news_manifest.json",):
            continue
        try:
            with open(json_file, "r") as f:
                articles = json.load(f)
            all_articles.extend(articles)
        except Exception as e:
            log.warning(f"Could not read {json_file}: {e}")

    log.info(f"Loaded {len(all_articles)} saved news articles from disk")
    return all_articles


# ══════════════════════════════════════════════════════════════════
# Run directly for testing
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    articles = run_news_scraper(
        tickers             = ["AAPL", "MSFT"],   # start small
        articles_per_ticker = 10
    )

    print(f"\n{'='*50}")
    print(f"Total articles: {len(articles)}")
    print(f"\nSample article:")
    if articles:
        sample = articles[0]
        print(f"  Ticker:    {sample['ticker']}")
        print(f"  Title:     {sample['title'][:70]}...")
        print(f"  Source:    {sample['source']}")
        print(f"  Sentiment: {sample['ticker_sentiment_label']} ({sample['ticker_sentiment_score']:.2f})")
        print(f"  Summary:   {sample['summary'][:100]}...")