"""
FILE: processing/chunker.py
============================
PURPOSE : Take parsed documents → split into small overlapping chunks
WHY     : LLMs have context limits. We can't feed a 200-page 10-K filing
          into a prompt. We split it into ~512-token chunks, then at
          query time we only retrieve the 5 most relevant chunks.

CHUNK SIZE CHOICE:
  - 512 tokens  ≈ 380 words ≈ 2-3 paragraphs
  - Too small  → loses context, misses meaning
  - Too large  → retrieval is less precise
  - 512 is the sweet spot for financial text

OVERLAP:
  - 64 tokens overlap between chunks
  - Why? So we don't cut a sentence in half and lose meaning
  - Example: chunk 1 ends with "revenue increased due to strong..."
             chunk 2 starts with "...due to strong iPhone sales in China"

OUTPUT: list of chunk dicts:
  {
    "chunk_id":    "AAPL_10-K_2024_abc123_0",
    "doc_id":      "abc123",
    "ticker":      "AAPL",
    "filing_type": "10-K",
    "date":        "2024-01-15",
    "source_url":  "https://...",
    "section":     "Risk Factors",
    "text":        "We face intense competition...",
    "chunk_index": 0,
    "total_chunks": 42,
    "word_count":  95,
  }
"""

import re
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk (≈512 tokens)
CHUNK_OVERLAP = 60    # words of overlap between chunks
MIN_CHUNK_LEN = 50    # skip chunks shorter than this (likely garbage)


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into overlapping chunks by word count.

    Strategy:
      1. Split text into individual words
      2. Slide a window of `chunk_size` words across the text
      3. Each new chunk starts `chunk_size - overlap` words after the last
      4. This gives us overlapping chunks that don't cut sentences randomly

    Example (chunk_size=5, overlap=2):
      Text:  "A B C D E F G H I J"
      Chunk 0: "A B C D E"
      Chunk 1: "D E F G H"     ← overlaps "D E" from chunk 0
      Chunk 2: "G H I J"       ← overlaps "G H" from chunk 1
    """
    words = text.split()
    if not words:
        return []

    chunks  = []
    step    = chunk_size - overlap   # how many words to advance each time
    start   = 0

    while start < len(words):
        end        = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)

        if end == len(words):
            break
        start += step

    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """
    Takes one parsed document and returns a list of chunk dicts.
    Preserves all metadata so we know where each chunk came from.

    We chunk section by section (not the whole document at once)
    so that chunk boundaries respect the document's own structure.
    """
    doc_id       = doc.get("doc_id", "unknown")
    ticker       = doc.get("ticker", "UNKNOWN")
    filing_type  = doc.get("filing_type", "UNKNOWN")
    date         = doc.get("date", "")
    source_url   = doc.get("source_url", "")
    sentiment_score = doc.get("sentiment_score", None)
    sentiment_label = doc.get("sentiment_label", None)

    sections     = doc.get("sections", [])
    all_chunks   = []

    # If no sections, chunk the full_text directly
    if not sections:
        sections = [{"title": "Full Document", "text": doc.get("full_text", ""), "order": 1}]

    for section in sections:
        section_title = section.get("title", "Unknown Section")
        section_text  = section.get("text", "").strip()

        if not section_text or len(section_text.split()) < MIN_CHUNK_LEN:
            continue   # skip tiny sections

        # Split this section into chunks
        raw_chunks = split_into_chunks(section_text)

        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.split()) < MIN_CHUNK_LEN:
                continue   # skip tiny chunks

            chunk_id = f"{ticker}_{filing_type}_{date}_{doc_id}_{len(all_chunks)}"

            chunk = {
                "chunk_id":      chunk_id,
                "doc_id":        doc_id,
                "ticker":        ticker,
                "filing_type":   filing_type,
                "date":          date,
                "source_url":    source_url,
                "section":       section_title,
                "text":          chunk_text,
                "chunk_index":   i,
                "word_count":    len(chunk_text.split()),
            }

            # Carry over sentiment scores for news articles
            if sentiment_score is not None:
                chunk["sentiment_score"] = sentiment_score
                chunk["sentiment_label"] = sentiment_label

            all_chunks.append(chunk)

    return all_chunks


def chunk_all_documents(parsed_dir: str = "./data/parsed") -> list[dict]:
    """
    Reads all parsed JSON files and chunks them all.
    Returns a flat list of ALL chunks across ALL documents.
    """
    parsed_path = Path(parsed_dir)
    if not parsed_path.exists():
        log.error(f"Parsed directory not found: {parsed_dir}")
        log.error("Run ingestion/parser.py first!")
        return []

    all_chunks = []
    doc_files  = [f for f in parsed_path.glob("*.json") if f.name != "parsed_manifest.json"]

    log.info(f"Chunking {len(doc_files)} parsed documents...")

    for doc_file in doc_files:
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                doc = json.load(f)

            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
            log.info(f"  {doc['ticker']} | {doc['filing_type']} | {doc['date']} → {len(chunks)} chunks")

        except Exception as e:
            log.warning(f"  Failed to chunk {doc_file.name}: {e}")

    log.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    chunks = chunk_all_documents()
    print(f"\nTotal chunks: {len(chunks)}")
    if chunks:
        print(f"\nSample chunk:")
        c = chunks[0]
        print(f"  ID:      {c['chunk_id']}")
        print(f"  Ticker:  {c['ticker']} | {c['filing_type']} | {c['date']}")
        print(f"  Section: {c['section']}")
        print(f"  Words:   {c['word_count']}")
        print(f"  Text:    {c['text'][:120]}...")