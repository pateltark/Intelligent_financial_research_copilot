"""
FILE: processing/embedder.py
==============================
PURPOSE : Convert text chunks into numerical vectors (embeddings)
COST    : 100% FREE — uses sentence-transformers running locally on your CPU
MODEL   : all-MiniLM-L6-v2  (downloaded once, ~90MB, then runs offline)

WHAT IS AN EMBEDDING?
  Text: "Apple's revenue grew 8% in China"
  Vector: [0.234, -0.112, 0.891, 0.043, ...]  (384 numbers)

  Two texts about similar topics → vectors are close together in space
  Two unrelated texts → vectors are far apart
  This is how semantic search works — we search by vector distance, not keywords.

WHY sentence-transformers instead of OpenAI?
  - OpenAI charges $0.0001 per 1000 tokens
  - 100,000 chunks × 400 words = ~200M tokens = ~$20 just for embeddings
  - sentence-transformers: $0.00, runs on your laptop CPU in ~5 minutes

MODEL QUALITY:
  all-MiniLM-L6-v2 produces 384-dimensional vectors.
  OpenAI ada-002 produces 1536-dimensional vectors.
  For financial text RAG, MiniLM is ~90% as good as OpenAI. Plenty for a demo.
"""

import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── Global model instance (loaded once, reused) ───────────────────
_model = None

def get_model():
    """
    Load the sentence-transformer model.
    First call downloads ~90MB model. Subsequent calls use cached version.
    Cache location: ~/.cache/huggingface/hub/
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model: all-MiniLM-L6-v2")
        log.info("(First run downloads ~90MB — subsequent runs are instant)")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Model loaded successfully")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> list[list[float]]:
    """
    Convert a list of text strings into a list of embedding vectors.

    Args:
        texts:         list of strings to embed
        batch_size:    how many texts to embed at once (64 is good for CPU)
        show_progress: show tqdm progress bar

    Returns:
        list of vectors, one per input text
        Each vector is a list of 384 floats

    Example:
        texts = ["Apple revenue grew", "Tesla deliveries declined"]
        vectors = embed_texts(texts)
        # vectors[0] = [0.234, -0.112, ...]   ← for "Apple revenue grew"
        # vectors[1] = [0.891,  0.043, ...]   ← for "Tesla deliveries declined"
    """
    if not texts:
        return []

    model = get_model()

    log.info(f"Embedding {len(texts)} texts in batches of {batch_size}...")

    # encode() handles batching internally, returns numpy array
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # normalize to unit length for cosine similarity
    )

    # Convert numpy arrays to plain Python lists (for JSON serialization)
    return [vec.tolist() for vec in embeddings]


def embed_single(text: str) -> list[float]:
    """
    Embed a single text string.
    Used at query time to embed the user's question.

    Example:
        question = "What are Apple's biggest risks in China?"
        vector   = embed_single(question)
        # → [0.234, -0.112, ...]
        # Then we search ChromaDB for the closest vectors
    """
    vectors = embed_texts([text], show_progress=False)
    return vectors[0] if vectors else []


def embed_chunks(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    """
    Takes chunk dicts from chunker.py and adds an "embedding" field to each.

    Input:  [{"chunk_id": "...", "text": "...", ...}, ...]
    Output: [{"chunk_id": "...", "text": "...", "embedding": [0.23, ...], ...}, ...]
    """
    if not chunks:
        return []

    # Extract just the text from each chunk for batch embedding
    texts = [c["text"] for c in chunks]

    # Embed all texts at once (much faster than one at a time)
    log.info(f"Embedding {len(chunks)} chunks...")
    vectors = embed_texts(texts, batch_size=batch_size)

    # Attach embedding back to each chunk dict
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    log.info(f"Done. Each embedding has {len(vectors[0])} dimensions.")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Quick test
    test_texts = [
        "Apple's revenue in China declined due to regulatory pressure.",
        "Microsoft Azure cloud revenue grew 28% year over year.",
        "Tesla delivered 484,000 vehicles in Q4 2024.",
    ]

    print("Testing embedder...")
    vectors = embed_texts(test_texts)

    print(f"\nResults:")
    for text, vec in zip(test_texts, vectors):
        print(f"  '{text[:50]}...'")
        print(f"   → vector dims: {len(vec)}, first 5 values: {[round(v,4) for v in vec[:5]]}")