"""
FILE: run.py
=============
PURPOSE : One command to run the entire project
USAGE   :
  python run.py pipeline   → scrape + parse + embed (run once to build the data)
  python run.py ui         → launch the Streamlit UI
  python run.py all        → pipeline + then launch UI
  python run.py test       → test that everything works
  python run.py status     → show what's in the vector store

TYPICAL FIRST-TIME SETUP:
  1. Copy .env.example to .env and fill in your keys
  2. pip install -r requirements.txt
  3. python run.py pipeline   ← downloads + indexes filings (takes ~10 min)
  4. python run.py ui         ← opens the app at http://localhost:8501
"""

import sys
import os
import logging

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════
# COMMAND: pipeline
# ══════════════════════════════════════════════════════════════════
def run_pipeline():
    """
    Full data pipeline:
      1. Scrape SEC EDGAR filings
      2. Scrape news from AlphaVantage
      3. Parse all raw files
      4. Chunk + embed + store in ChromaDB
    """
    print("\n" + "="*60)
    print("  FINANCIAL COPILOT — DATA PIPELINE")
    print("="*60)

    # STEP 1: SEC EDGAR scraper
    print("\n[STEP 1/4] Scraping SEC EDGAR filings...")
    print("  → Downloading 10-K, 10-Q, 8-K for AAPL, MSFT, GOOGL, NVDA, TSLA")
    try:
        from ingestion.sec_scraper import run_sec_scraper
        results = run_sec_scraper(
            tickers   = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
            days_back = 90,
        )
        print(f"  ✅ Downloaded {len(results)} filings")
    except Exception as e:
        print(f"  ❌ SEC scraper failed: {e}")
        print("     Continuing with other steps...")

    # STEP 2: News scraper
    print("\n[STEP 2/4] Fetching financial news...")
    print("  → Using AlphaVantage free API (25 req/day)")
    try:
        from ingestion.news_scraper import run_news_scraper
        articles = run_news_scraper(
            tickers             = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
            articles_per_ticker = 15,
        )
        print(f"  ✅ Fetched {len(articles)} news articles")
    except Exception as e:
        print(f"  ⚠️  News scraper error: {e}")
        print("     Check your ALPHA_VANTAGE_API_KEY in .env")
        print("     Continuing without news data...")

    # STEP 3: Parse raw files
    print("\n[STEP 3/4] Parsing documents...")
    print("  → Extracting clean text from HTML/PDF files")
    try:
        from ingestion.parser import run_parser
        docs = run_parser()
        print(f"  ✅ Parsed {len(docs)} documents")
        total_words = sum(d.get("word_count", 0) for d in docs)
        print(f"     Total words: {total_words:,}")
    except Exception as e:
        print(f"  ❌ Parser failed: {e}")
        return

    # STEP 4: Chunk + embed + store
    print("\n[STEP 4/4] Building vector index (this takes the longest)...")
    print("  → Chunking documents, embedding with sentence-transformers, storing in ChromaDB")
    print("  → First run downloads the embedding model (~90MB)")
    try:
        from processing.vector_store import run_vector_pipeline
        run_vector_pipeline()
        print("  ✅ Vector index built successfully!")
    except Exception as e:
        print(f"  ❌ Vector pipeline failed: {e}")
        return

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE!")
    print("  Run:  python run.py ui  →  to open the app")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════
# COMMAND: ui
# ══════════════════════════════════════════════════════════════════
def run_ui():
    """Launch the Streamlit UI."""
    print("\n" + "="*60)
    print("  Launching Financial Copilot UI...")
    print("  Opening at: http://localhost:8501")
    print("  Press Ctrl+C to stop")
    print("="*60 + "\n")
    os.system("streamlit run ui/app.py")


# ══════════════════════════════════════════════════════════════════
# COMMAND: status
# ══════════════════════════════════════════════════════════════════
def run_status():
    """Show what's currently in the vector store."""
    print("\n" + "="*60)
    print("  VECTOR STORE STATUS")
    print("="*60)
    try:
        from processing.vector_store import get_stats
        stats = get_stats()
        print(f"\n  Total chunks: {stats['total_chunks']:,}")
        print(f"  Tickers:      {stats['tickers']}")
        print(f"  Filing types: {stats['filing_types']}")
        print(f"  Location:     {stats['chroma_dir']}")

        if stats["total_chunks"] == 0:
            print("\n  ⚠️  Vector store is empty!")
            print("  Run: python run.py pipeline")
        else:
            print("\n  ✅ Ready to answer questions!")
            print("  Run: python run.py ui")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
    print()


# ══════════════════════════════════════════════════════════════════
# COMMAND: test
# ══════════════════════════════════════════════════════════════════
def run_test():
    """Test all components are working."""
    print("\n" + "="*60)
    print("  SYSTEM TEST")
    print("="*60)

    tests_passed = 0
    tests_total  = 0

    # Test 1: Imports
    tests_total += 1
    print("\n[1] Testing imports...")
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from groq import Groq
        import streamlit
        print("  ✅ All packages imported")
        tests_passed += 1
    except ImportError as e:
        print(f"  ❌ Missing package: {e}")
        print("     Run: pip install -r requirements.txt")

    # Test 2: Environment variables
    tests_total += 1
    print("\n[2] Checking environment variables...")
    from dotenv import load_dotenv
    load_dotenv()

    missing_keys = []
    if not os.getenv("GROQ_API_KEY"):
        missing_keys.append("GROQ_API_KEY (get free at console.groq.com)")
    if not os.getenv("SEC_USER_AGENT"):
        missing_keys.append("SEC_USER_AGENT (set to 'YourName youremail@gmail.com')")

    if not missing_keys:
        print("  ✅ Environment variables set")
        tests_passed += 1
    else:
        print("  ⚠️  Missing .env variables:")
        for k in missing_keys:
            print(f"     - {k}")

    # Test 3: Embedding model
    tests_total += 1
    print("\n[3] Testing embedding model...")
    try:
        from processing.embedder import embed_single
        vec = embed_single("test financial text")
        assert len(vec) == 384, f"Expected 384 dims, got {len(vec)}"
        print(f"  ✅ Embedding model works (384-dim vectors)")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Embedding failed: {e}")

    # Test 4: ChromaDB
    tests_total += 1
    print("\n[4] Testing ChromaDB connection...")
    try:
        from processing.vector_store import get_stats
        stats = get_stats()
        print(f"  ✅ ChromaDB connected ({stats['total_chunks']} chunks stored)")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ ChromaDB error: {e}")

    # Test 5: Groq API (if key is set)
    tests_total += 1
    print("\n[5] Testing Groq API...")
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            resp   = client.chat.completions.create(
                model    = "llama-3.3-70b-versatile",
                messages = [{"role":"user","content":"Say 'ok' in one word."}],
                max_tokens = 5,
            )
            print(f"  ✅ Groq API works (response: {resp.choices[0].message.content.strip()})")
            tests_passed += 1
        except Exception as e:
            print(f"  ❌ Groq API error: {e}")
    else:
        print("  ⏭️  Skipped (GROQ_API_KEY not set)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Tests: {tests_passed}/{tests_total} passed")
    if tests_passed == tests_total:
        print("  ✅ Everything is working!")
        print("  Run: python run.py pipeline")
    else:
        print("  ⚠️  Fix the issues above, then try again")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "pipeline":
        run_pipeline()
    elif command == "ui":
        run_ui()
    elif command == "all":
        run_pipeline()
        run_ui()
    elif command == "status":
        run_status()
    elif command == "test":
        run_test()
    else:
        print("""
  Financial Research Copilot
  ==========================

  Commands:
    python run.py pipeline   → Download filings + build AI index (run once)
    python run.py ui         → Launch the web UI at localhost:8501
    python run.py all        → pipeline + ui in sequence
    python run.py status     → Check how many documents are indexed
    python run.py test       → Test that all components are working

  First time setup:
    1. cp .env.example .env
    2. Fill in GROQ_API_KEY and SEC_USER_AGENT in .env
    3. pip install -r requirements.txt
    4. python run.py test      ← verify everything works
    5. python run.py pipeline  ← download and index data
    6. python run.py ui        ← open the app
        """)