import os
import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

RAW_DIR    = "./data/raw"
PARSED_DIR = "./data/parsed"


# ══════════════════════════════════════════════════════════════════
# TEXT CLEANING (applied after every parser)
# ══════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove BOM and zero-width characters
    text = text.replace("\ufeff", "").replace("\u200b", "")

    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove page number patterns
    text = re.sub(r"\-\s*\d+\s*\-",              "", text)   # "- 42 -"
    text = re.sub(r"[Pp]age\s+\d+\s+of\s+\d+",  "", text)   # "Page 5 of 198"
    text = re.sub(r"^\s*\d+\s*$",               "", text, flags=re.MULTILINE)  # lone numbers

    # Remove SEC boilerplate that appears on every page
    text = re.sub(r"Table of Contents\s*\n?",    "", text, flags=re.IGNORECASE)
    text = re.sub(r"EDGAR.*?Filing\s*\n",        "", text, flags=re.IGNORECASE)

    # Remove lines that are only dashes, underscores, or equals signs
    text = re.sub(r"^[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ blank lines to 2 blank lines max
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ══════════════════════════════════════════════════════════════════
# PARSER 1: HTML files (most SEC filings are HTML)
# ══════════════════════════════════════════════════════════════════
def parse_html(filepath: str) -> tuple[str, list[dict]]:
    from bs4 import BeautifulSoup

    log.info(f"  Parsing HTML: {Path(filepath).name}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # ── Remove all noise tags ──────────────────────────────────────
    noise_tags = [
        "script", "style", "nav", "header", "footer",
        "meta", "link", "noscript", "iframe", "form",
        "button", "input", "select"
    ]
    for tag in soup(noise_tags):
        tag.decompose()

    # ── Extract sections ───────────────────────────────────────────
    sections     = []
    current      = {"title": "Introduction", "text": "", "order": 1}
    section_num  = 1
    HEADING_TAGS = {"h1", "h2", "h3", "h4"}

    for element in soup.find_all(["h1","h2","h3","h4","p","li","td","th","span","div"]):
        tag  = element.name
        text = element.get_text(separator=" ", strip=True)

        if not text or len(text) < 3:
            continue

        if tag in HEADING_TAGS and len(text) < 300:
            # Save what we have so far as a section
            if current["text"].strip():
                current["text"] = clean_text(current["text"])
                sections.append(dict(current))

            # Start a new section
            section_num += 1
            current = {
                "title": text[:200],
                "text":  "",
                "order": section_num
            }
        else:
            # Add text to current section
            cleaned_line = re.sub(r"\s+", " ", text).strip()
            current["text"] += cleaned_line + "\n"

    # Don't forget the last section
    if current["text"].strip():
        current["text"] = clean_text(current["text"])
        sections.append(current)

    # Build full text by joining all sections
    full_text = clean_text("\n\n".join(
        f"## {s['title']}\n{s['text']}" for s in sections
    ))

    log.info(f"  Extracted {len(sections)} sections, {len(full_text.split())} words")
    return full_text, sections


# ══════════════════════════════════════════════════════════════════
# PARSER 2: PDF files
# ══════════════════════════════════════════════════════════════════
def parse_pdf(filepath: str) -> tuple[str, list[dict]]:
    
    try:
        import pdfplumber
    except ImportError:
        log.error("pdfplumber not installed. Run: pip install pdfplumber")
        return "", []

    log.info(f"  Parsing PDF: {Path(filepath).name}")

    sections  = []
    all_pages = []

    with pdfplumber.open(filepath) as pdf:
        total_pages = len(pdf.pages)
        log.info(f"  PDF has {total_pages} pages")

        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract text from this page
            page_text = page.extract_text()

            if not page_text or len(page_text.strip()) < 20:
                continue   # skip blank pages

            cleaned = clean_text(page_text)
            all_pages.append(cleaned)

            sections.append({
                "title": f"Page {page_num}",
                "text":  cleaned,
                "order": page_num,
            })

    full_text = clean_text("\n\n".join(all_pages))
    log.info(f"  Extracted {len(sections)} pages, {len(full_text.split())} words")
    return full_text, sections


# ══════════════════════════════════════════════════════════════════
# PARSER 3: News JSON files (already structured, just normalize)
# ══════════════════════════════════════════════════════════════════
def parse_news_json(filepath: str) -> list[dict]:
    """
    News articles from AlphaVantage are already JSON with clean text.
    We just read them and format consistently for the next pipeline step.

    Each article becomes one "document" (not split into sections)
    because articles are already short enough to embed whole.
    """
    log.info(f"  Parsing news: {Path(filepath).name}")

    with open(filepath, "r", encoding="utf-8") as f:
        articles = json.load(f)

    parsed_docs = []
    for article in articles:
        # Combine title + summary as the text to embed
        text = f"{article.get('title', '')}\n\n{article.get('summary', '')}"
        text = clean_text(text)

        if len(text) < 50:
            continue

        doc_id = hashlib.md5(article.get("url", text).encode()).hexdigest()[:12]

        parsed_docs.append({
            "doc_id":       doc_id,
            "ticker":       article.get("ticker", ""),
            "filing_type":  "NEWS",
            "date":         article.get("published_at", "")[:10],
            "source_url":   article.get("url", ""),
            "sections": [{
                "title": "Article",
                "text":  text,
                "order": 1,
            }],
            "full_text":    text,
            "word_count":   len(text.split()),
            "parse_method": "news_json",
            "sentiment_score": article.get("ticker_sentiment_score", 0.0),
            "sentiment_label": article.get("ticker_sentiment_label", "Neutral"),
            "source_name":  article.get("source", ""),
        })

    log.info(f"  Parsed {len(parsed_docs)} news articles")
    return parsed_docs


# ══════════════════════════════════════════════════════════════════
# MASTER PARSER: Auto-detect file type and parse
# ══════════════════════════════════════════════════════════════════
def parse_filing(filing_meta: dict) -> dict | None:
    """
    Takes a filing metadata dict (from sec_scraper manifest) and
    returns a clean parsed document dict ready for chunking.

    Auto-detects file type from extension:
      .htm / .html → parse_html()
      .pdf         → parse_pdf()
    """
    local_path = filing_meta.get("local_path", "")

    if not local_path or not Path(local_path).exists():
        log.warning(f"  File not found: {local_path}")
        return None

    file_ext = Path(local_path).suffix.lower()
    ticker   = filing_meta.get("ticker", "UNKNOWN")
    form     = filing_meta.get("form",   "UNKNOWN")
    date     = filing_meta.get("date",   "")
    url      = filing_meta.get("doc_url", local_path)

    # Parse based on file type
    try:
        if file_ext in (".htm", ".html"):
            full_text, sections = parse_html(local_path)
            method = "html_beautifulsoup"

        elif file_ext == ".pdf":
            full_text, sections = parse_pdf(local_path)
            method = "pdfplumber"

        else:
            # Try HTML parser as fallback
            log.warning(f"  Unknown extension {file_ext}, trying HTML parser")
            full_text, sections = parse_html(local_path)
            method = "html_beautifulsoup_fallback"

    except Exception as e:
        log.error(f"  Failed to parse {local_path}: {e}")
        return None

    # Skip empty documents
    if not full_text or len(full_text.split()) < 50:
        log.warning(f"  Skipping — too little text extracted from {local_path}")
        return None

    # Generate stable unique ID for this document
    doc_id = hashlib.md5(url.encode()).hexdigest()[:12]

    return {
        "doc_id":       doc_id,
        "ticker":       ticker,
        "filing_type":  form,
        "date":         date,
        "source_url":   url,
        "local_path":   local_path,
        "sections":     sections,
        "full_text":    full_text,
        "word_count":   len(full_text.split()),
        "section_count": len(sections),
        "parse_method": method,
        "parsed_at":    datetime.utcnow().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════
# Save parsed document to disk
# ══════════════════════════════════════════════════════════════════
def save_parsed_doc(doc: dict) -> str:
    """
    Saves one parsed document as JSON to ./data/parsed/
    Filename: AAPL_10-K_2024-01-15_abc123.json
    """
    Path(PARSED_DIR).mkdir(parents=True, exist_ok=True)

    filename = f"{doc['ticker']}_{doc['filing_type']}_{doc['date']}_{doc['doc_id']}.json"
    filepath = os.path.join(PARSED_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    return filepath


# ══════════════════════════════════════════════════════════════════
# MAIN: Parse all files from manifest
# ══════════════════════════════════════════════════════════════════
def run_parser() -> list[dict]:
    """
    Main function:
      1. Reads the manifest created by sec_scraper.py
      2. Parses every downloaded SEC filing
      3. Also parses all saved news JSON files
      4. Saves everything to ./data/parsed/

    Returns: list of all parsed documents
    """
    log.info("Starting parser...")
    log.info("=" * 60)
    all_parsed = []

    # ── Parse SEC filings ──────────────────────────────────────────
    manifest_path = os.path.join(RAW_DIR, "manifest.json")

    if Path(manifest_path).exists():
        with open(manifest_path, "r") as f:
            filings = json.load(f)

        log.info(f"Found {len(filings)} SEC filings to parse")

        for i, filing in enumerate(filings, 1):
            log.info(f"\n[{i}/{len(filings)}] {filing['ticker']} | {filing['form']} | {filing['date']}")
            doc = parse_filing(filing)

            if doc:
                filepath = save_parsed_doc(doc)
                all_parsed.append(doc)
                log.info(f"  Saved → {filepath}")
                log.info(f"  Stats: {doc['word_count']} words | {doc['section_count']} sections")
    else:
        log.warning(f"No manifest found at {manifest_path}")
        log.warning("Run sec_scraper.py first!")

    # ── Parse news files ───────────────────────────────────────────
    news_dir = Path(RAW_DIR) / "news"
    if news_dir.exists():
        news_files = [f for f in news_dir.glob("*.json")
                      if f.name != "news_manifest.json"]

        log.info(f"\nFound {len(news_files)} news files to parse")

        for news_file in news_files:
            news_docs = parse_news_json(str(news_file))
            for doc in news_docs:
                filepath = save_parsed_doc(doc)
                all_parsed.append(doc)

    # ── Save combined output manifest ──────────────────────────────
    output_manifest = {
        "parsed_at":    datetime.utcnow().isoformat(),
        "total_docs":   len(all_parsed),
        "total_words":  sum(d["word_count"] for d in all_parsed),
        "by_ticker":    {},
    }

    # Count docs per ticker
    for doc in all_parsed:
        t = doc["ticker"]
        if t not in output_manifest["by_ticker"]:
            output_manifest["by_ticker"][t] = 0
        output_manifest["by_ticker"][t] += 1

    manifest_out = os.path.join(PARSED_DIR, "parsed_manifest.json")
    Path(PARSED_DIR).mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump(output_manifest, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info(f"DONE. Parsed {len(all_parsed)} documents")
    log.info(f"Total words across all docs: {output_manifest['total_words']:,}")
    log.info(f"Manifest saved → {manifest_out}")

    return all_parsed


# ══════════════════════════════════════════════════════════════════
# Helper: Load all parsed docs (used by chunker.py next)
# ══════════════════════════════════════════════════════════════════
def load_all_parsed_docs() -> list[dict]:
    """
    Reads all parsed JSON files from ./data/parsed/
    Called by processing/chunker.py in the next step.
    """
    parsed_dir = Path(PARSED_DIR)
    if not parsed_dir.exists():
        log.warning(f"Parsed dir not found: {PARSED_DIR}. Run parser first.")
        return []

    docs = []
    for json_file in parsed_dir.glob("*.json"):
        if json_file.name == "parsed_manifest.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                docs.append(json.load(f))
        except Exception as e:
            log.warning(f"Could not read {json_file.name}: {e}")

    log.info(f"Loaded {len(docs)} parsed documents from disk")
    return docs


# ══════════════════════════════════════════════════════════════════
# Run directly for testing
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    docs = run_parser()

    print(f"\n{'='*50}")
    print(f"Total documents parsed: {len(docs)}")

    for doc in docs[:3]:   # show first 3
        print(f"\n  [{doc['ticker']}] {doc['filing_type']} | {doc['date']}")
        print(f"  Words: {doc['word_count']:,} | Sections: {doc['section_count']}")
        print(f"  Method: {doc['parse_method']}")
        if doc["sections"]:
            first_section = doc["sections"][0]
            preview = first_section["text"][:120].replace("\n", " ")
            print(f"  Preview: {preview}...")