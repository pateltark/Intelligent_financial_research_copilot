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

from agent.gaurd_before_fetch import check_avail_sec_doc, planner
from rag.db import save_sec_document,get_sec_document,related_sec_chunks, save_sec_vector, related_chunks,save_doc_info,related_chunks_per_doc  
from rag.emb_chunks import ingest_text,ingest_sec_text, create_vectorstore
from rag.retrieve import get_retrieve

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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




def generate_answer(question: str, context_chunks, user_id: str, mode: str = "sec", labels: list[str] | None = None) -> str:
    if not context_chunks or context_chunks == "No relevant SEC context found.":
        return "I couldn't find relevant information in the database to answer your question."

    is_multi_doc = (
        isinstance(context_chunks, list)
        and len(context_chunks) > 0
        and isinstance(context_chunks[0], list)
    )

    if is_multi_doc:
        sections = []
        for i, doc_group in enumerate(context_chunks):
            if not doc_group:
                continue
            label = labels[i] if labels and i < len(labels) else f"Document {i+1}"
            body = "\n".join(_format_chunk_row(row) for row in doc_group)
            sections.append(f"=== DOCUMENT: {label} ===\n{body}")
        formatted_context = "\n\n".join(sections)
    else:
        flat_chunks = [_format_chunk_row(chunk) for chunk in context_chunks]
        formatted_context = "\n\n---\n\n".join(flat_chunks)


    # 3. Load recent chat history specifically for this mode (last 6 messages / 3 turns)
    history = load_chat(user_id, mode=mode)[-6:]
    formatted_history = ""
    if history:
        history_lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]
        formatted_history = "RECENT CHAT HISTORY:\n" + "\n".join(history_lines) + "\n\n---\n\n"

    # 4. Construct the RAG prompt with History
    prompt = f"""You are an expert financial research assistant. Answer the question using ONLY the provided context and conversation history below.

    {formatted_history}DOCUMENT CONTEXT:

    {formatted_context}

    USER QUESTION:
    {question}

    INSTRUCTIONS:
    - Answer using ONLY the information in DOCUMENT CONTEXT above. Do not use any outside or general knowledge, even if you happen to know the answer.
    - If the context does not address the question, respond with EXACTLY this sentence and nothing else: "The provided document context does not contain enough information to answer this question." Do not follow it with an answer from general knowledge.
    - Otherwise, give a concise, clear answer based strictly on the provided context and previous history.
    - Some context lines are prefixed with a page marker like "[p. 12]". When you use information from a marked line, cite the page inline, e.g. "Revenue grew 12% (p. 12)." Do not invent a page number for unmarked content.
    - If the context contains multiple documents (marked with "=== DOCUMENT: ... ==="), treat each as a separate source and refer to them by name when comparing.
    """

    # 5. Query Groq API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Recommended model on Groq
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content



def ask_sec(question: str, user_id: str):

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

        chunks = related_sec_chunks(
            document_id=active_doc["document_id"],
            question=question
        )

        if not chunks or not _is_relevant(chunks):
            context_query = contextualize_question(question, user_id)

            chunks = related_sec_chunks(
                document_id=active_doc["document_id"],
                question=context_query
            )

        return generate_answer(question, chunks, user_id, mode="sec")


    # ========================================
    # CASE 2 : Fetch a new SEC filing
    # ========================================

    docs = check_avail_sec_doc(plan)

    sec_chunks = []
    sec_labels = []
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

        touched_docs.append({"document_id": document_id, "ticker": request.ticker, "form_type": request.form_type})

        # ------------------------------------
        # Retrieve using document_id
        # ------------------------------------

        doc_chunks = related_sec_chunks(
            document_id=document_id,
            question=question
        )

        if doc_chunks:
            sec_chunks.append(doc_chunks)
            sec_labels.append(f"{request.ticker} {request.form_type}")

    if not sec_chunks or not _is_relevant(sec_chunks):
        context_query = contextualize_question(question, user_id)

        doc_chunks = related_sec_chunks(
            document_id=document_id,
            question=context_query
        )   
        if doc_chunks:
            sec_chunks.append(doc_chunks)
            sec_labels.append(f"{request.ticker} {request.form_type}")

        # return "No relevant SEC context found."

    set_active_set(user_id, touched_docs)

    return generate_answer(question, sec_chunks, user_id, mode="sec", labels=sec_labels)



def ask_upload(question: str, user_id: str, document_ids: list[str] | None = None):
    NOT_ENOUGH_INFO = "The provided document context does not contain enough information to answer this question."

    if not document_ids:
        return "Please select at least one document to chat with."

    if len(document_ids) > 1:
        chunks = related_chunks_per_doc(user_id=user_id, question=question, document_ids=document_ids)
        chunks = [group for group in chunks if group]  # drop empty per-doc groups
        if not chunks or not _is_relevant(chunks):
            return NOT_ENOUGH_INFO
        labels = [group[0][2] if group else "Unknown" for group in chunks]  # row = (content, document_id, filename, distance)
        return generate_answer(question, chunks, user_id=user_id, mode="doc", labels=labels)
    else:
        chunks = related_chunks(user_id=user_id, question=question, document_ids=document_ids)
        if not chunks or not _is_relevant(chunks):
            return NOT_ENOUGH_INFO
        return generate_answer(question, chunks, user_id=user_id, mode="doc")