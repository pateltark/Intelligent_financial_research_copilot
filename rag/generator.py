"""
FILE: rag/generator.py
========================
PURPOSE : Call an LLM with context chunks → get a cited answer
COST    : FREE — uses Groq API (free tier) or Ollama (local)

TWO BACKEND OPTIONS:
  1. Groq  (recommended for dev) — free API, runs llama3-70b in the cloud
     Get key: https://console.groq.com  (1 min, no credit card)
     Limit: 6,000 tokens/min on free tier (enough for ~15 questions/min)

  2. Ollama (recommended for offline) — runs llama3 locally
     Install: https://ollama.com  → then: ollama pull llama3.1
     No limits, completely offline, needs ~8GB RAM

HOW TO SWITCH BACKENDS:
  Set in .env:
    LLM_BACKEND=groq      ← uses Groq API (default)
    LLM_BACKEND=ollama    ← uses local Ollama

PROMPT DESIGN (important!):
  We give the LLM a strict system prompt that:
  1. Tells it to ONLY answer from the provided context (no hallucination)
  2. Tells it to cite sources using [Source N] tags
  3. Gives it the persona of a financial analyst
  4. Instructs it to say "I don't know" if context is insufficient
"""

import os
import re
import logging
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
LLM_BACKEND  = os.getenv("LLM_BACKEND",   "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY",  "")
GROQ_MODEL   = "llama-3.3-70b-versatile"          # best free model on Groq
OLLAMA_MODEL = "llama3.1"                 # change to "mistral" if you prefer
OLLAMA_URL   = "http://localhost:11434"

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior financial analyst assistant with expertise in reading SEC filings, earnings call transcripts, and financial news.

Your job is to answer questions about publicly traded companies based ONLY on the context provided below.

RULES YOU MUST FOLLOW:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. Always cite your sources using [Source N] notation at the end of each claim.
   Example: "Apple's revenue in China declined 13% [Source 1]."
3. If the context does not contain enough information to answer, say:
   "I don't have enough information in the available filings to answer this question."
4. Be specific — include numbers, percentages, and dates when available in the context.
5. Keep answers clear and professional, like a research note.
6. If multiple sources say different things, note the discrepancy.

FORMAT YOUR ANSWER:
- Lead with the most important finding
- Use bullet points for multiple risk factors or data points
- End with a "Sources used:" section listing which [Source N] you cited
"""


# ══════════════════════════════════════════════════════════════════
# BACKEND 1: Groq (free cloud API)
# ══════════════════════════════════════════════════════════════════
def call_groq(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """
    Calls Groq API with llama3-70b model.
    Groq is extremely fast (~500 tokens/second) and free tier is generous.

    Free limits:
      - 6,000 tokens/minute
      - 500,000 tokens/day
      - No credit card needed

    Returns: the LLM's response as a string
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set in .env!\n"
            "Get your free key at: https://console.groq.com\n"
            "Then add: GROQ_API_KEY=your_key_here  to your .env file"
        )

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {"role": "system",  "content": system},
                {"role": "user",    "content": prompt},
            ],
            temperature  = 0.1,     # low temp = more factual, less creative
            max_tokens   = 1024,
        )
        return response.choices[0].message.content

    except Exception as e:
        log.error(f"Groq API error: {e}")
        raise


# ══════════════════════════════════════════════════════════════════
# BACKEND 2: Ollama (free local)
# ══════════════════════════════════════════════════════════════════
def call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """
    Calls a locally running Ollama instance.

    Setup (one time):
      1. Download Ollama from https://ollama.com
      2. Run: ollama pull llama3.1
      3. Ollama starts automatically as a background service

    Returns: the LLM's response as a string
    """
    import requests

    full_prompt = f"{system}\n\n{prompt}"

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1024},
            },
            timeout=120,    # local models can be slow on CPU
        )
        response.raise_for_status()
        return response.json()["response"]

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama!\n"
            "Make sure Ollama is running: https://ollama.com\n"
            "And the model is downloaded: ollama pull llama3.1"
        )


# ══════════════════════════════════════════════════════════════════
# CITATION EXTRACTION
# ══════════════════════════════════════════════════════════════════
def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Finds [Source N] references in the LLM's answer and maps them
    back to the actual chunk metadata (ticker, date, URL etc.)

    Example:
      answer: "Revenue grew 8% [Source 1] driven by iPhone sales [Source 2]"
      chunks: [chunk0, chunk1, chunk2, ...]

      Returns:
        [
          {"source_num": 1, "ticker": "AAPL", "filing_type": "10-K", "date": "2024-01-15", ...},
          {"source_num": 2, "ticker": "AAPL", "filing_type": "10-K", "date": "2024-01-15", ...},
        ]
    """
    # Find all [Source N] references in the answer
    cited_nums = set(int(n) for n in re.findall(r"\[Source (\d+)\]", answer))

    citations = []
    for num in sorted(cited_nums):
        chunk_index = num - 1   # [Source 1] = chunks[0]
        if 0 <= chunk_index < len(chunks):
            chunk = chunks[chunk_index]
            citations.append({
                "source_num":   num,
                "ticker":       chunk.get("ticker",      ""),
                "filing_type":  chunk.get("filing_type", ""),
                "date":         chunk.get("date",        "")[:10],
                "section":      chunk.get("section",     ""),
                "source_url":   chunk.get("source_url",  ""),
                "preview":      chunk.get("text",        "")[:200] + "...",
            })

    return citations


# ══════════════════════════════════════════════════════════════════
# MAIN GENERATE FUNCTION
# ══════════════════════════════════════════════════════════════════
def generate_answer(question: str, context: str, chunks: list[dict]) -> dict:
    """
    The main function — given a question + context chunks, generate an answer.

    Args:
        question: user's question string
        context:  formatted context string from retriever.format_context()
        chunks:   original chunk dicts (for citation mapping)

    Returns:
        {
            "answer":    "Apple's biggest risk is China...[Source 1]",
            "citations": [{"source_num": 1, "ticker": "AAPL", ...}],
            "model":     "llama3-70b-8192",
            "backend":   "groq"
        }
    """
    # Build the user prompt with context + question
    user_prompt = f"""Here is the relevant financial document context:

{context}

---

Question: {question}

Please answer based only on the context above. Cite sources as [Source N]."""

    log.info(f"Generating answer using backend: {LLM_BACKEND}")

    # Call selected backend
    if LLM_BACKEND == "groq":
        raw_answer = call_groq(user_prompt)
        model_used = GROQ_MODEL
    elif LLM_BACKEND == "ollama":
        raw_answer = call_ollama(user_prompt)
        model_used = OLLAMA_MODEL
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}. Use 'groq' or 'ollama'")

    # Extract citations from the answer
    citations = extract_citations(raw_answer, chunks)

    log.info(f"Answer generated ({len(raw_answer.split())} words, {len(citations)} citations)")

    return {
        "answer":    raw_answer,
        "citations": citations,
        "model":     model_used,
        "backend":   LLM_BACKEND,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Quick test with fake context
    test_chunks = [
        {
            "text": "Apple faces significant risks in Greater China, which accounted for 19% of net sales in fiscal 2024. Regulatory changes and US-China tensions could materially impact operations.",
            "ticker": "AAPL", "filing_type": "10-K", "date": "2024-01-15",
            "section": "Risk Factors", "source_url": "https://sec.gov/test"
        },
        {
            "text": "Competition in the smartphone market remains intense. Customers may choose competitors offering similar products at lower prices, particularly in emerging markets.",
            "ticker": "AAPL", "filing_type": "10-K", "date": "2024-01-15",
            "section": "Competition", "source_url": "https://sec.gov/test"
        },
    ]

    from rag.retriever import format_context
    context  = format_context(test_chunks)
    result   = generate_answer(
        question = "What are Apple's biggest risks?",
        context  = context,
        chunks   = test_chunks,
    )

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nCitations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  [Source {c['source_num']}] {c['ticker']} | {c['filing_type']} | {c['date']}")