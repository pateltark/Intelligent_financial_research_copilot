"""
FILE: rag/retriever.py
========================
PURPOSE : Given a user's question, find the most relevant document chunks
HOW     : Embed the question → search ChromaDB → return top N chunks

RETRIEVAL FLOW:
  User asks: "What are Apple's risks in China?"
       ↓
  Embed question → [0.234, -0.112, 0.891, ...]
       ↓
  Search ChromaDB (cosine similarity)
       ↓
  Returns 5 most similar chunks from Apple's 10-K filings
       ↓
  These chunks become the "context" for the LLM to answer from
"""

import logging
from processing.embedder     import embed_single
from processing.vector_store import query_similar, get_stats

log = logging.getLogger(__name__)


def retrieve(
    question:    str,
    n_results:   int       = 5,
    ticker:      str|None  = None,
    filing_type: str|None  = None,
) -> list[dict]:
    """
    Main retrieval function.

    Args:
        question:    the user's natural language question
        n_results:   how many chunks to retrieve (5 is the sweet spot)
        ticker:      optionally filter to one company (e.g. "AAPL")
        filing_type: optionally filter to one type (e.g. "10-K")

    Returns:
        List of relevant chunk dicts, sorted by similarity score (best first)
        Each chunk has: text, ticker, filing_type, date, section, source_url, similarity_score
    """
    log.info(f"Retrieving for: '{question[:80]}'")
    if ticker:       log.info(f"  Filter: ticker={ticker}")
    if filing_type:  log.info(f"  Filter: filing_type={filing_type}")

    # Step 1: Embed the question into a vector
    question_vector = embed_single(question)

    if not question_vector:
        log.error("Failed to embed question")
        return []

    # Step 2: Search ChromaDB for similar chunks
    chunks = query_similar(
        query_embedding = question_vector,
        n_results       = n_results,
        ticker          = ticker,
        filing_type     = filing_type,
    )

    if not chunks:
        log.warning("No chunks retrieved — ChromaDB may be empty")
        return []

    log.info(f"Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        log.info(f"  [{i}] {chunk['ticker']} | {chunk['filing_type']} | {chunk['date'][:10]} | score={chunk['similarity_score']:.3f} | {chunk['section'][:40]}")

    return chunks


def retrieve_for_comparison(
    question: str,
    tickers:  list[str],
    n_per_ticker: int = 3,
) -> dict[str, list[dict]]:
    """
    Retrieve chunks for MULTIPLE companies simultaneously.
    Used for the "compare companies" feature.

    Returns: dict mapping ticker → list of relevant chunks
    Example: {"AAPL": [...], "MSFT": [...], "GOOGL": [...]}
    """
    results = {}
    for ticker in tickers:
        chunks = retrieve(question, n_results=n_per_ticker, ticker=ticker)
        results[ticker] = chunks
        log.info(f"  {ticker}: {len(chunks)} chunks retrieved")
    return results


def format_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a single context string for the LLM prompt.

    Format:
        [Source 1: AAPL | 10-K | 2024-01-15 | Risk Factors]
        Apple faces significant risks in Greater China...

        [Source 2: AAPL | 10-K | 2024-01-15 | Competition]
        The smartphone market is intensely competitive...

    The LLM uses these source labels to cite its answers.
    """
    if not chunks:
        return "No relevant context found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_label = (
            f"[Source {i}: {chunk.get('ticker','')} | "
            f"{chunk.get('filing_type','')} | "
            f"{chunk.get('date','')[:10]} | "
            f"{chunk.get('section','')[:50]}]"
        )
        context_parts.append(f"{source_label}\n{chunk['text']}")

    return "\n\n".join(context_parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Check what's in the store first
    stats = get_stats()
    print(f"Vector store: {stats['total_chunks']} chunks | Tickers: {stats['tickers']}")

    if stats["total_chunks"] == 0:
        print("\nVector store is empty! Run this first:")
        print("  python ingestion/sec_scraper.py")
        print("  python ingestion/parser.py")
        print("  python processing/vector_store.py")
    else:
        # Test retrieval
        question = "What are the biggest risk factors?"
        chunks   = retrieve(question, n_results=3)

        print(f"\nQuestion: {question}")
        print(f"Retrieved {len(chunks)} chunks:\n")
        for chunk in chunks:
            print(f"  [{chunk['ticker']} | {chunk['filing_type']} | score={chunk['similarity_score']:.3f}]")
            print(f"  Section: {chunk['section']}")
            print(f"  Text: {chunk['text'][:150]}...\n")