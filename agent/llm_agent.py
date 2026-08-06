# llm_agent.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import os
from groq import Groq, BadRequestError
from dotenv import load_dotenv

from rag.db import load_chat, save_chat,get_sec_document
from agent.tools import TOOLS
# from agent.tools import fetch_sec_document, retrieve_documents, extract_document
from agent.executor import run_tool
from agent.session import set_active_doc,get_active_doc, set_active_set

from sec.edgar import (
    fetch_sec_filings,
    download_doc,
    extract_text
)

from sentence_transformers import CrossEncoder

from agent.gaurd_before_fetch import check_avail_sec_doc, planner
from rag.db import save_sec_document,get_sec_document,related_sec_chunks, save_sec_vector, related_chunks,save_doc_info,related_chunks_per_doc  
from rag.emb_chunks import ingest_text,ingest_sec_text, create_vectorstore
from rag.retrieve import get_retrieve
from collections import defaultdict


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Downloads once automatically (~1.1 GB) and stays in memory
reranker = CrossEncoder("BAAI/bge-reranker-large")  # or "BAAI/bge-reranker-base"

def rerank_chunks(query: str, chunks: list, top_k: int = 4) -> list:
    """
    Reranks candidate chunks retrieved from PostgreSQL using a Cross-Encoder.

    :param query: The original user question string.
    :param chunks: List of chunk tuples or list of lists of chunk tuples from DB search.
                   Assumes chunk[0] contains the chunk text content.
    :param top_k: Number of highest-scoring chunks to keep for the LLM context.
    :return: Top k original chunk tuples sorted by cross-encoder relevance score.
    """
    if not chunks:
        return []

    # 1. Flatten nested lists (e.g. results from related_chunks_per_doc)
    flat_chunks = []
    if isinstance(chunks, list) and len(chunks) > 0 and isinstance(chunks[0], list):
        for group in chunks:
            flat_chunks.extend(group)
    else:
        flat_chunks = chunks

    if not flat_chunks:
        return []

    # 2. Prepare (query, passage) pairs for Cross-Encoder scoring
    # chunk[0] is always 'content' across both PDF and SEC tuple schemas
    pairs = [[query, chunk[0]] for chunk in flat_chunks]

    # 3. Compute cross-attention relevance scores
    scores = reranker.predict(pairs)

    # 4. Pair scores with original chunk tuples and sort descending
    scored_chunks = list(zip(scores, flat_chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # 5. Extract and return top_k original chunk tuples
    return [chunk for score, chunk in scored_chunks[:top_k]]


# Cosine distance from pgvector's `<=>` operator ranges 0 (identical) to
# 2 (opposite). Anything above this is treated as "not actually related
# to the question" and short-circuits before an LLM call is even made —
# cheaper than a Groq call, and doesn't rely on the model policing itself.
#
# 0.6 turned out to be too strict — open-ended questions like "what is
# this doc about?" don't share vocabulary with any single chunk even in
# a genuinely relevant document, so their best-match distance runs
# higher than a specific factual question's would. Loosened to 1.0
# (roughly "not orthogonal") as a safer default. DEBUG_RELEVANCE=1
# prints the actual distance for every query so you can tune this
# against your real data instead of guessing again.
RELEVANCE_THRESHOLD = 1.0
DEBUG_RELEVANCE = os.getenv("DEBUG_RELEVANCE") == "1"




from collections import defaultdict

def format_citations(context_chunks, mode: str = "doc") -> str:
    """
    Generates structured, deduplicated sources at the end of LLM responses.
    
    :param context_chunks: The raw context retrieved from PostgreSQL.
    :param mode: 'doc' for uploaded PDFs, 'sec' for SEC filings.
    :return: Markdown-formatted sources string.
    """
    if not context_chunks or isinstance(context_chunks, str):
        return ""

    # ==========================================
    # 1. SEC AGENT MODE (No page numbers)
    # ==========================================
    if mode == "sec":
        sec_sources = set()
        
        # Flatten if context happens to be nested
        flat_sec_rows = []
        if isinstance(context_chunks, list) and len(context_chunks) > 0 and isinstance(context_chunks[0], list):
            for group in context_chunks:
                flat_sec_rows.extend(group)
        else:
            flat_sec_rows = context_chunks

        for row in flat_sec_rows:
            # Expected tuple from related_sec_chunks:
            # (content, ticker, form_type, filename, distance)
            if len(row) >= 4:
                ticker = row[1] if row[1] else "SEC"
                form_type = row[2] if row[2] else "Filing"
                filename = row[3] if row[3] else "document.htm"
                sec_sources.add(f"**{ticker} {form_type}** ({filename})")
            elif len(row) == 3:
                # Fallback: (content, filename, distance)
                filename = row[1] if row[1] else "SEC Filing"
                sec_sources.add(f"**{filename}**")

        if sec_sources:
            return "\n\n**Sources:** " + " | ".join(sorted(list(sec_sources)))
        return ""

    # ==========================================
    # 2. PDF RAG MODE (With page numbers)
    # ==========================================
    doc_pages = defaultdict(set)

    # Flatten nested list of lists (multi-document search results)
    flat_pdf_rows = []
    if isinstance(context_chunks, list) and len(context_chunks) > 0 and isinstance(context_chunks[0], list):
        for group in context_chunks:
            flat_pdf_rows.extend(group)
    else:
        flat_pdf_rows = context_chunks

    for row in flat_pdf_rows:
        filename = "Document"
        page_num = None

        # Determine tuple layout dynamically:
        # 4-element (related_chunks): (content, filename, page_number, distance)
        # 6-element (related_chunks_per_doc): (content, document_id, filename, page_number, distance, rrf_score)
        if len(row) == 4:
            filename = row[1] if row[1] else "Document"
            page_num = row[2]
        elif len(row) >= 5:
            filename = row[2] if row[2] else "Document"
            page_num = row[3]

        if page_num is not None:
            try:
                doc_pages[filename].add(int(page_num))
            except (ValueError, TypeError):
                doc_pages[filename].add(None)
        else:
            doc_pages[filename].add(None)

    if not doc_pages:
        return ""

    citation_parts = []
    for doc_name, pages in doc_pages.items():
        valid_pages = sorted([p for p in pages if p is not None])
        if valid_pages:
            page_str = ", ".join(str(p) for p in valid_pages)
            label = "Page" if len(valid_pages) == 1 else "Pages"
            citation_parts.append(f"**{doc_name}** ({label}: {page_str})")
        else:
            citation_parts.append(f"**{doc_name}**")

    return "\n\n**Sources:** " + " | ".join(citation_parts)


# for intent based que..

def contextualize_question(question: str, user_id: str, mode: str = "sec") -> str:
    """
    Rewrites vague follow-up questions (e.g., "tell me more about it") 
    into standalone search queries using conversation history.
    """
    history = load_chat(user_id, mode=mode)[-4:]
    if not history:
        return question

    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    rewrite_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone search query that contains all necessary context (names, topics, document references). Do NOT answer the question, just rewrite it.

Chat History:
{history_text}

Follow-up Question: {question}

Standalone Search Query:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"[Query Rewriter] Originally: '{question}' -> Rewritten: '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[Query Rewriter Error] {e}")
        return question



def _best_distance(chunks):
    """chunks: either a flat list of rows (each row's last element is the
    distance), or a list of per-document groups (list of lists). Returns
    the smallest distance found — i.e. the single closest match — or
    infinity if there's nothing to compare."""
    if not chunks:
        return float("inf")

    if isinstance(chunks[0], list):
        distances = [row[-1] for group in chunks for row in group]
    else:
        distances = [row[-1] for row in chunks]

    return min(distances) if distances else float("inf")



def _is_relevant(chunks, threshold: float = RELEVANCE_THRESHOLD) -> bool:
    distance = _best_distance(chunks)
    if DEBUG_RELEVANCE:
        print(f"[relevance] best_distance={distance:.4f} threshold={threshold}")
    return distance <= threshold



def _format_chunk_row(row) -> str:
    """Prefixes chunk text with a page marker when page data is
    available, so the model can cite it. Row shape varies by source:
      - SEC (no page data yet):        (content, distance)
      - PDF, single document:          (content, page_number, distance)
      - PDF, multi-document compare:   (content, document_id, filename, page_number, distance)
    """
    content = row[0]
    page_number = None
    if len(row) == 3:
        page_number = row[1]
    elif len(row) == 5:
        page_number = row[3]

    if page_number:
        return f"[p. {page_number}] {content}"
    return content




def generate_answer(
    question: str,
    context_chunks,
    user_id: str,
    mode: str = "doc",
    labels: list[str] | None = None
) -> str:
    """
    Formats DB context chunks, queries Groq LLM, and appends clean citations.

    :param question: The user's query.
    :param context_chunks: Retrieved DB chunks (list of tuples or list of lists).
    :param user_id: User identifier.
    :param mode: 'doc' for uploaded PDFs, 'sec' for SEC filings.
    :param labels: Optional labels/filenames for explicit document identification.
    :return: Completed answer string with sources appended.
    """
    NOT_ENOUGH_INFO = "The provided document context does not contain enough information to answer this question."

    if not context_chunks:
        return NOT_ENOUGH_INFO

    # -------------------------------------------------------------
    # 1. Flatten Chunks (Handles both Single-doc and Multi-doc lists)
    # -------------------------------------------------------------
    flat_rows = []
    if isinstance(context_chunks, list) and len(context_chunks) > 0 and isinstance(context_chunks[0], list):
        for group in context_chunks:
            flat_rows.extend(group)
    else:
        flat_rows = context_chunks

    if not flat_rows:
        return NOT_ENOUGH_INFO

    # -------------------------------------------------------------
    # 2. Format Context Blocks for Prompt
    # -------------------------------------------------------------
    context_blocks = []
    for idx, row in enumerate(flat_rows, start=1):
        content = row[0]
        
        if mode == "sec":
            # Expected SEC tuple: (content, ticker, form_type, filename, distance)
            ticker = row[1] if len(row) > 1 and row[1] else "SEC"
            form_type = row[2] if len(row) > 2 and row[2] else "Filing"
            filename = row[3] if len(row) > 3 and row[3] else ""
            header = f"[Source {idx}: {ticker} {form_type} - {filename}]"
            
        else:
            # Expected PDF tuple: 
            # 4-element: (content, filename, page_number, distance)
            # 6-element: (content, doc_id, filename, page_number, distance, rrf)
            if len(row) == 4:
                fname = row[1] or "Document"
                pnum = f", Page {row[2]}" if row[2] is not None else ""
            elif len(row) >= 5:
                fname = row[2] or "Document"
                pnum = f", Page {row[3]}" if row[3] is not None else ""
            else:
                fname = "Document"
                pnum = ""
            header = f"[Source {idx}: {fname}{pnum}]"

        context_blocks.append(f"{header}\n{content}")

    formatted_context = "\n\n".join(context_blocks)

    # -------------------------------------------------------------
    # 3. Construct System and User Prompts
    # -------------------------------------------------------------
    system_prompt = (
        "You are an expert research copilot. Answer the user's question accurately "
        "and concisely using STRICTLY the provided document context below.\n\n"
        "Rules:\n"
        "1. Base your answer ONLY on the provided context.\n"
        "2. If the context does not contain enough information to answer the question, state EXACTLY:\n"
        f"   '{NOT_ENOUGH_INFO}'\n"
        "3. Do NOT make up citations or page numbers in your text body; sources will be automatically appended."
    )

    user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {question}"

    # -------------------------------------------------------------
    # 4. Call Groq LLM API
    # -------------------------------------------------------------
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[generate_answer LLM Error]: {e}")
        return "An error occurred while generating the response. Please try again."

    # -------------------------------------------------------------
    # 5. Handle Fallback (Do NOT append citations if context was insufficient)
    # -------------------------------------------------------------
    if NOT_ENOUGH_INFO.lower() in answer_text.lower():
        return NOT_ENOUGH_INFO

    # -------------------------------------------------------------
    # 6. Append Aggregated Citations
    # -------------------------------------------------------------
    citations = format_citations(context_chunks, mode=mode)
    return f"{answer_text}{citations}"



def ask_sec(question: str, user_id: str):
    NOT_ENOUGH_INFO = "The provided document context does not contain enough information to answer this question."

    # ----------------------------------------
    # 1. Ask planner what to do
    # ----------------------------------------
    plan = planner(question)

    # ========================================
    # CASE 1 : Continue chatting with current SEC document
    # ========================================
    if plan.action == "CURRENT_DOC":

        active_doc = get_active_doc(user_id)

        if active_doc is None:
            return (
                "There is no active SEC document. "
                "Please ask for a filing first, e.g. "
                "'Show Tesla's latest 10-K'."
            )

        doc_id = active_doc["document_id"]

        # Stage 1: Fetch candidate chunks (Broad Net: k=15)
        chunks = related_sec_chunks(
            document_id=doc_id,
            question=question,
            k=15
        )

        # Fallback contextual re-query if initial search returned insufficient context
        if not chunks or not _is_relevant(chunks):
            context_query = contextualize_question(question, user_id)

            chunks = related_sec_chunks(
                document_id=doc_id,
                question=context_query,
                k=15
            )

        if not chunks:
            return NOT_ENOUGH_INFO

        # Stage 2: Rerank candidates down to top 4
        reranked_chunks = rerank_chunks(query=question, chunks=chunks, top_k=4)

        if not reranked_chunks:
            return NOT_ENOUGH_INFO

        # Stage 3: Generate Answer
        return generate_answer(question, reranked_chunks, user_id, mode="sec")


    # ========================================
    # CASE 2 : Fetch a new SEC filing
    # ========================================

    docs = check_avail_sec_doc(plan)

    sec_chunks = []
    touched_docs = []

    for doc in docs:

        request = doc["request"]
        db_row = doc["db_row"]

        # ------------------------------------
        # Already exists in DB
        # ------------------------------------
        if db_row:

            document_id = db_row["id"]

            # Make this the current document
            set_active_doc(
                user_id=user_id,
                document_id=document_id,
                ticker=request.ticker,
                form_type=request.form_type,
            )

        # ------------------------------------
        # Need to fetch
        # ------------------------------------
        else:

            print(f"Fetching {request.ticker} {request.form_type}...")

            filings = fetch_sec_filings(
                ticker=request.ticker,
                form_type=request.form_type
            )

            if not filings:
                continue

            filing = filings[0]

            path = download_doc(
                filing["url"],
                filing["filename"]
            )

            text = extract_text(path)

            document_id = save_sec_document(
                ticker=request.ticker,
                form_type=request.form_type,
                filed_at=filing["filed_at"],
                filename=filing["filename"],
                url=filing["url"],
                path=path
            )

            ingest_sec_text(
                text=text,
                document_id=document_id,
                ticker=request.ticker,
                form_type=request.form_type,
                filename=filing["filename"]
            )

            # Make fetched document active
            set_active_doc(
                user_id=user_id,
                document_id=document_id,
                ticker=request.ticker,
                form_type=request.form_type,
            )

        touched_docs.append({
            "document_id": document_id, 
            "ticker": request.ticker, 
            "form_type": request.form_type
        })

        # Stage 1: Retrieve candidate chunks per document (k=15)
        doc_chunks = related_sec_chunks(
            document_id=document_id,
            question=question,
            k=15
        )

        if doc_chunks:
            sec_chunks.extend(doc_chunks)

    # ------------------------------------
    # Contextual Fallback Search across all touched docs
    # ------------------------------------
    if not sec_chunks or not _is_relevant(sec_chunks):
        context_query = contextualize_question(question, user_id)
        sec_chunks = []

        for t_doc in touched_docs:
            doc_chunks = related_sec_chunks(
                document_id=t_doc["document_id"],
                question=context_query,
                k=15
            )
            if doc_chunks:
                sec_chunks.extend(doc_chunks)

    if not sec_chunks:
        return NOT_ENOUGH_INFO

    # ------------------------------------
    # Stage 2: Cross-Encoder Reranking
    # ------------------------------------
    reranked_chunks = rerank_chunks(query=question, chunks=sec_chunks, top_k=4)

    if not reranked_chunks:
        return NOT_ENOUGH_INFO

    set_active_set(user_id, touched_docs)

    # Stage 3: LLM Generation
    return generate_answer(question, reranked_chunks, user_id, mode="sec")


def ask_upload(question: str, user_id: str, document_ids: list[str] | None = None):
    NOT_ENOUGH_INFO = "The provided document context does not contain enough information to answer this question."

    if not document_ids:
        return "Please select at least one document to chat with."

    # Stage 1: Fetch Candidate Chunks from DB (10 candidates per document)
    raw_chunks = related_chunks_per_doc(
        user_id=user_id, 
        question=question, 
        document_ids=document_ids,
        k_per_doc=10  # Retrieve a broad net of candidates for reranking
    )

    # Filter out empty lists
    raw_chunks = [group for group in raw_chunks if group]

    if not raw_chunks:
        return NOT_ENOUGH_INFO

    # Stage 2: Rerank candidates down to the top 4 most relevant chunks
    reranked_chunks = rerank_chunks(query=question, chunks=raw_chunks, top_k=4)

    if not reranked_chunks:
        return NOT_ENOUGH_INFO

    # Stage 3: Extract filenames from flat reranked tuples
    # Tuple format: (content, document_id, filename, page_number, distance, rrf_score)
    labels = list({row[2] for row in reranked_chunks if len(row) > 2 and row[2]})

    # Stage 4: LLM Answer Generation
    return generate_answer(
        question=question, 
        context_chunks=reranked_chunks, 
        user_id=user_id, 
        mode="doc", 
        labels=labels
    )