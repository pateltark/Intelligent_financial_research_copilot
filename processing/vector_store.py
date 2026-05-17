"""
FILE: processing/vector_store.py
==================================
PURPOSE : Save embeddings to ChromaDB and retrieve them at query time
COST    : 100% FREE — ChromaDB runs locally, saves to ./data/chroma_db/ on disk
INSTALL : pip install chromadb

WHAT IS A VECTOR STORE?
  Think of it as a special database that stores text + its vector.
  Instead of searching by exact keywords, it searches by meaning.

  Normal DB search:  "China" → finds documents containing the word "China"
  Vector search:     "Asia revenue exposure" → finds documents about China sales
                     even if the word "China" isn't in the query

HOW ChromaDB WORKS:
  1. You add chunks → it stores text + embedding + metadata
  2. You query with a question → it embeds the question and finds closest chunks
  3. It returns the top-N most similar chunks

PERSISTENCE:
  ChromaDB saves everything to ./data/chroma_db/  on disk.
  If you restart, everything is still there. No re-indexing needed.
  Only re-run vector_store.py when you have NEW filings to add.

COLLECTIONS:
  We use ONE collection called "financial_docs" for everything.
  We use metadata filters to narrow by ticker/date at query time.
  Example: "Search only in AAPL documents from 2024"
"""

import logging
from pathlib import Path
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

log = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────
CHROMA_DIR       = "./data/chroma_db"
COLLECTION_NAME  = "financial_docs"
BATCH_SIZE       = 100     # upsert this many chunks at a time


# ── Get/create ChromaDB client ────────────────────────────────────
def get_client() -> chromadb.PersistentClient:
    """
    Returns a ChromaDB client that saves data to disk.
    Creates the directory if it doesn't exist.
    """
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_collection() -> chromadb.Collection:
    """
    Gets (or creates) our main collection.
    All financial documents go into one collection.
    We filter by ticker/date in metadata at query time.
    """
    client     = get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # use cosine similarity (best for text)
    )
    return collection


# ══════════════════════════════════════════════════════════════════
# WRITE: Add chunks to ChromaDB
# ══════════════════════════════════════════════════════════════════
def upsert_chunks(chunks: list[dict]) -> int:
    """
    Saves chunks (with embeddings) into ChromaDB.
    "Upsert" = insert if new, update if already exists.
    This means re-running is safe — no duplicates.

    ChromaDB requires 4 things per item:
      ids:        unique string ID per chunk
      embeddings: the vector (list of floats)
      documents:  the raw text
      metadatas:  dict of filterable fields (ticker, date, etc.)

    Returns: number of chunks successfully added
    """
    if not chunks:
        log.warning("No chunks to upsert")
        return 0

    # Check that embeddings are present
    if "embedding" not in chunks[0]:
        log.error("Chunks don't have embeddings. Run embedder.py first!")
        return 0

    collection = get_collection()
    total_added = 0

    # Process in batches to avoid memory issues with large datasets
    for batch_start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Upserting to ChromaDB"):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        ids         = []
        embeddings  = []
        documents   = []
        metadatas   = []

        for chunk in batch:
            # ChromaDB metadata values must be str, int, float, or bool only
            # (no lists, no None values)
            metadata = {
                "ticker":       str(chunk.get("ticker",       "")),
                "filing_type":  str(chunk.get("filing_type",  "")),
                "date":         str(chunk.get("date",         "")),
                "section":      str(chunk.get("section",      "")),
                "source_url":   str(chunk.get("source_url",   "")),
                "doc_id":       str(chunk.get("doc_id",       "")),
                "chunk_index":  int(chunk.get("chunk_index",  0)),
                "word_count":   int(chunk.get("word_count",   0)),
            }

            # Add sentiment if present (news articles)
            if chunk.get("sentiment_score") is not None:
                metadata["sentiment_score"] = float(chunk["sentiment_score"])
                metadata["sentiment_label"] = str(chunk.get("sentiment_label", "Neutral"))

            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])
            metadatas.append(metadata)

        # Upsert this batch
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total_added += len(batch)

    log.info(f"Upserted {total_added} chunks into ChromaDB collection '{COLLECTION_NAME}'")
    log.info(f"Total chunks in collection: {collection.count()}")
    return total_added


# ══════════════════════════════════════════════════════════════════
# READ: Query ChromaDB for similar chunks
# ══════════════════════════════════════════════════════════════════
def query_similar(
    query_embedding: list[float],
    n_results:       int         = 5,
    ticker:          str | None  = None,
    filing_type:     str | None  = None,
    date_from:       str | None  = None,
) -> list[dict]:
    """
    Find the most similar chunks to a query embedding.

    Args:
        query_embedding: vector for the user's question (from embedder.embed_single)
        n_results:       how many chunks to return
        ticker:          filter to one company (e.g. "AAPL")
        filing_type:     filter to one type (e.g. "10-K")
        date_from:       filter to filings on/after this date (YYYY-MM-DD)

    Returns:
        List of chunk dicts with an added "similarity_score" field
        Sorted by relevance (most relevant first)
    """
    collection = get_collection()

    if collection.count() == 0:
        log.warning("ChromaDB is empty! Run the full pipeline first.")
        return []

    # Build metadata filter (where clause)
    where = {}
    if ticker and filing_type:
        where = {"$and": [{"ticker": ticker}, {"filing_type": filing_type}]}
    elif ticker:
        where = {"ticker": ticker}
    elif filing_type:
        where = {"filing_type": filing_type}

    # Query ChromaDB
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results":        min(n_results, collection.count()),
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # Format results
    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        # ChromaDB returns cosine distance (0=identical, 2=opposite)
        # Convert to similarity score (1=identical, 0=unrelated) for readability
        similarity = round(1 - distance / 2, 4)

        chunks.append({
            "text":             text,
            "similarity_score": similarity,
            **metadata,          # include all metadata fields (ticker, date, etc.)
        })

    return chunks


# ══════════════════════════════════════════════════════════════════
# STATS: Info about what's in the store
# ══════════════════════════════════════════════════════════════════
def get_stats() -> dict:
    """Returns info about what's stored in ChromaDB."""
    collection = get_collection()
    count      = collection.count()

    if count == 0:
        return {"total_chunks": 0, "tickers": [], "filing_types": []}

    # Sample metadata to find what tickers/types we have
    sample = collection.get(limit=min(count, 1000), include=["metadatas"])
    metas  = sample["metadatas"]

    tickers      = sorted(set(m["ticker"]      for m in metas if m.get("ticker")))
    filing_types = sorted(set(m["filing_type"] for m in metas if m.get("filing_type")))

    return {
        "total_chunks":  count,
        "tickers":       tickers,
        "filing_types":  filing_types,
        "chroma_dir":    CHROMA_DIR,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN: Full pipeline — chunk + embed + store
# ══════════════════════════════════════════════════════════════════
def run_vector_pipeline(parsed_dir: str = "./data/parsed"):
    """
    Main entry point — runs the full processing pipeline:
      1. Load all parsed documents from disk
      2. Split each into chunks
      3. Embed all chunks
      4. Save to ChromaDB

    Only needs to run once (or when new filings are added).
    """
    from processing.chunker  import chunk_all_documents
    from processing.embedder import embed_chunks

    log.info("=" * 60)
    log.info("Starting vector pipeline: chunk → embed → store")
    log.info("=" * 60)

    # Step 1: Chunk all parsed documents
    log.info("\n[1/3] Chunking documents...")
    chunks = chunk_all_documents(parsed_dir)

    if not chunks:
        log.error("No chunks created. Make sure you ran ingestion/parser.py first!")
        return

    log.info(f"Created {len(chunks)} chunks total")

    # Step 2: Embed all chunks
    log.info("\n[2/3] Embedding chunks (this may take a few minutes on CPU)...")
    chunks_with_embeddings = embed_chunks(chunks)

    # Step 3: Store in ChromaDB
    log.info("\n[3/3] Storing in ChromaDB...")
    added = upsert_chunks(chunks_with_embeddings)

    # Show final stats
    stats = get_stats()
    log.info("\n" + "=" * 60)
    log.info("DONE! Vector store ready.")
    log.info(f"  Total chunks stored: {stats['total_chunks']}")
    log.info(f"  Tickers indexed:     {stats['tickers']}")
    log.info(f"  Filing types:        {stats['filing_types']}")
    log.info(f"  Stored at:           {stats['chroma_dir']}")
    log.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_vector_pipeline()

    # Show what's in the store
    stats = get_stats()
    print(f"\nVector store stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")