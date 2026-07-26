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
from agent.session import set_active_doc,get_active_doc

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




def generate_answer(question: str, context_chunks, user_id: str, mode: str = "sec") -> str:
    
    if not context_chunks or context_chunks == "No relevant SEC context found.":
        return "I couldn't find relevant information in the database to answer your question."

    is_multi_doc = (
        isinstance(context_chunks, list)
        and len(context_chunks) > 0
        and isinstance(context_chunks[0], list)
    )

    if is_multi_doc:
        sections = []
        for doc_group in context_chunks:
            if not doc_group:
                continue
            # row shape: (content, document_id, filename, distance)
            label = doc_group[0][2] if len(doc_group[0]) > 2 else doc_group[0][1]
            body = "\n".join(row[0] for row in doc_group)
            sections.append(f"=== DOCUMENT: {label} ===\n{body}")
        formatted_context = "\n\n".join(sections)
    else:
        flat_chunks = [chunk[0] for chunk in context_chunks]
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
    - Give a concise, clear answer based strictly on the provided context and previous history.
    - If the context contains multiple documents (marked with "=== DOCUMENT: ... ==="), treat each as a separate source and refer to them by name when comparing.
    - If the answer is not present in the context, explicitly state "The provided document context does not contain enough information to answer this question."
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

        if not chunks:
            return "No relevant SEC context found."

        return generate_answer(question, chunks, user_id)


    # ========================================
    # CASE 2 : Fetch a new SEC filing
    # ========================================

    docs = check_avail_sec_doc(plan)

    sec_chunks = []

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

        # ------------------------------------
        # Retrieve using document_id
        # ------------------------------------

        doc_chunks = related_sec_chunks(
            document_id=document_id,
            question=question
        )

        if doc_chunks:
            sec_chunks.append(doc_chunks)

    if not sec_chunks:
        return "No relevant SEC context found."

    return generate_answer(
        question,
        sec_chunks,
        user_id
    )



def ask_upload(question, user_id, document_ids=None):
    if document_ids and len(document_ids) > 1:
        chunks = related_chunks_per_doc(user_id=user_id, question=question, document_ids=document_ids)
        print(f"[per_doc] docs={document_ids} -> groups={len(chunks)}, sizes={[len(g) for g in chunks]}")
    else:
        chunks = related_chunks(user_id=user_id, question=question, document_ids=document_ids)
        print(f"[single] docs={document_ids} -> rows={len(chunks)}")

    return generate_answer(question, chunks, user_id=user_id)