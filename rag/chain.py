"""
FILE: rag/chain.py
===================
PURPOSE : The single entry point for all AI queries.
          Ties retriever + generator into one clean answer() function.

THIS IS THE MOST IMPORTANT FILE IN THE PROJECT.

FULL FLOW:
  User question
      ↓
  retrieve()   — search ChromaDB for top 5 relevant chunks
      ↓
  format_context() — format chunks into a readable prompt context
      ↓
  generate_answer() — send context + question to LLM → get cited answer
      ↓
  Return {answer, citations, chunks, metadata}

SPECIAL MODES:
  1. Single company Q&A  — answer(question, ticker="AAPL")
  2. Compare companies   — compare(question, tickers=["AAPL","MSFT","GOOGL"])
  3. Sentiment timeline  — get_sentiment_timeline(ticker="TSLA")
"""

import logging
from rag.retriever import retrieve, retrieve_for_comparison, format_context
from rag.generator import generate_answer

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# MODE 1: Single question → answer with citations
# ══════════════════════════════════════════════════════════════════
def answer(
    question:    str,
    ticker:      str|None  = None,
    filing_type: str|None  = None,
    n_chunks:    int       = 5,
) -> dict:
    """
    THE main function of the entire project.
    Takes a question, returns a cited answer grounded in SEC filings.

    Args:
        question:    any natural language financial question
        ticker:      optionally restrict to one company (e.g. "AAPL")
        filing_type: optionally restrict to one type (e.g. "10-K")
        n_chunks:    how many context chunks to retrieve (5 is optimal)

    Returns:
        {
            "question":  "What are Apple's biggest risks?",
            "answer":    "Apple faces three major risks... [Source 1]...",
            "citations": [
                {
                    "source_num":  1,
                    "ticker":      "AAPL",
                    "filing_type": "10-K",
                    "date":        "2024-01-15",
                    "section":     "Risk Factors",
                    "source_url":  "https://sec.gov/...",
                    "preview":     "Apple faces significant risks..."
                },
                ...
            ],
            "chunks_used":   5,
            "model":         "llama3-70b-8192",
            "backend":       "groq",
        }

    Example usage:
        from rag.chain import answer
        result = answer("What are Apple's biggest risks?", ticker="AAPL")
        print(result["answer"])
        for cite in result["citations"]:
            print(f"  Source: {cite['ticker']} {cite['filing_type']} {cite['date']}")
    """
    log.info(f"\n{'='*60}")
    log.info(f"Q: {question}")
    if ticker:      log.info(f"   Ticker filter: {ticker}")
    if filing_type: log.info(f"   Type filter:   {filing_type}")

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(
        question    = question,
        n_results   = n_chunks,
        ticker      = ticker,
        filing_type = filing_type,
    )

    if not chunks:
        return {
            "question":   question,
            "answer":     "I couldn't find any relevant information in the available filings. Please make sure you've run the data pipeline first (run.py).",
            "citations":  [],
            "chunks_used": 0,
            "model":      "none",
            "backend":    "none",
        }

    # Step 2: Format chunks into context string for LLM
    context = format_context(chunks)

    # Step 3: Generate answer from LLM
    result = generate_answer(
        question = question,
        context  = context,
        chunks   = chunks,
    )

    return {
        "question":    question,
        "answer":      result["answer"],
        "citations":   result["citations"],
        "chunks_used": len(chunks),
        "model":       result["model"],
        "backend":     result["backend"],
    }


# ══════════════════════════════════════════════════════════════════
# MODE 2: Compare multiple companies on the same question
# ══════════════════════════════════════════════════════════════════
def compare(
    question: str,
    tickers:  list[str],
    n_per_ticker: int = 3,
) -> dict:
    """
    Answer a question for multiple companies simultaneously.
    Used for the "Compare companies" feature in the UI.

    Example:
        result = compare(
            "What is the biggest risk factor?",
            tickers=["AAPL", "MSFT", "GOOGL"]
        )

    Returns:
        {
            "question": "...",
            "comparisons": {
                "AAPL":  {"answer": "...", "citations": [...]},
                "MSFT":  {"answer": "...", "citations": [...]},
                "GOOGL": {"answer": "...", "citations": [...]},
            }
        }
    """
    log.info(f"Comparing {tickers} on: {question}")

    # Retrieve chunks for each company
    chunks_by_ticker = retrieve_for_comparison(question, tickers, n_per_ticker=n_per_ticker)

    comparisons = {}
    for ticker, chunks in chunks_by_ticker.items():
        if not chunks:
            comparisons[ticker] = {
                "answer":    f"No filings found for {ticker}.",
                "citations": [],
            }
            continue

        context = format_context(chunks)
        result  = generate_answer(
            question = f"For {ticker} specifically: {question}",
            context  = context,
            chunks   = chunks,
        )
        comparisons[ticker] = {
            "answer":    result["answer"],
            "citations": result["citations"],
        }
        log.info(f"  {ticker}: answer generated ({len(result['answer'].split())} words)")

    return {
        "question":    question,
        "tickers":     tickers,
        "comparisons": comparisons,
    }


# ══════════════════════════════════════════════════════════════════
# MODE 3: Sentiment timeline for a company
# ══════════════════════════════════════════════════════════════════
def get_sentiment_timeline(ticker: str) -> list[dict]:
    """
    Returns sentiment scores for a company across all available filings.
    Used to draw the sentiment timeline chart in the UI.

    Works by loading news articles from ChromaDB that have sentiment scores
    (added by AlphaVantage during ingestion).

    Returns:
        [
            {"date": "2024-01-15", "score": 0.72,  "label": "Bullish",  "count": 5},
            {"date": "2023-10-12", "score": -0.21, "label": "Bearish",  "count": 3},
            ...
        ]
        Sorted by date ascending (oldest first)
    """
    from processing.vector_store import get_collection

    log.info(f"Building sentiment timeline for: {ticker}")

    try:
        collection = get_collection()
        if collection.count() == 0:
            return []

        # Get all news chunks for this ticker (they have sentiment scores)
        results = collection.get(
            where   = {"$and": [{"ticker": ticker}, {"filing_type": "NEWS"}]},
            include = ["metadatas"],
        )

        metadatas = results.get("metadatas", [])
        if not metadatas:
            log.info(f"  No news sentiment data found for {ticker}")
            return []

        # Group by date and average the scores
        from collections import defaultdict
        daily_scores = defaultdict(list)

        for meta in metadatas:
            date  = meta.get("date", "")[:10]
            score = meta.get("sentiment_score")
            if date and score is not None:
                daily_scores[date].append(float(score))

        # Calculate daily average
        timeline = []
        for date, scores in daily_scores.items():
            avg_score = sum(scores) / len(scores)
            label = "Bullish" if avg_score > 0.15 else "Bearish" if avg_score < -0.15 else "Neutral"
            timeline.append({
                "date":  date,
                "score": round(avg_score, 4),
                "label": label,
                "count": len(scores),
            })

        # Sort by date ascending
        timeline.sort(key=lambda x: x["date"])
        log.info(f"  Timeline: {len(timeline)} data points for {ticker}")
        return timeline

    except Exception as e:
        log.warning(f"Could not build sentiment timeline: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("Testing RAG chain...")
    print("(Make sure you've run the full pipeline first: python run.py pipeline)\n")

    result = answer(
        question = "What are the biggest risk factors mentioned in the filings?",
        n_chunks = 3,
    )

    print(f"\n{'='*60}")
    print(f"QUESTION: {result['question']}")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nCITATIONS ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  [Source {c['source_num']}] {c['ticker']} | {c['filing_type']} | {c['date']} | {c['section']}")
    print(f"\nModel: {result['model']} via {result['backend']}")