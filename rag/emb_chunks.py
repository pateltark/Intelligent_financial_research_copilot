from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

import json
from groq import Groq
import os
from dotenv import load_dotenv

from rag.db import (
    save_emb, related_chunks, load_chat, save_sec_vector, related_sec_chunks,
    update_document_status,
)

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Docling is paused for now ────────────────────────────────
# Its default OCR pipeline was taking well over an hour on large,
# digitally-generated PDFs (running full page-by-page OCR on text that
# was already extractable, and do_ocr=False didn't fully suppress it in
# our version). Reverted to pypdf here so uploads work reliably again.
# When Docling comes back: re-add its imports, restore
# ingest_docling_document + the page-aware chunking, and make sure the
# OCR fix (do_ocr=False + PyPdfiumDocumentBackend) actually holds before
# relying on it again. Citations (page_number) will be NULL for
# anything ingested through this pypdf path — that's expected and
# handled gracefully by _format_chunk_row in llm_agent.py (no page
# marker gets added when page_number is None).

# Docling + embedding is CPU-heavy. This caps how many ingestion jobs run
# truly in parallel, no matter how many upload requests arrive at once —
# so 100 uploads queue up and process a couple at a time instead of all
# fighting for CPU simultaneously and slowing down every other request
# on the server (SEC chats, other users' PDF chats, everything).
# Tune based on your container's CPU count. Kept even with pypdf, since
# embedding 100+ PDFs concurrently is still worth bounding.
_ingest_executor = ThreadPoolExecutor(max_workers=2)



def submit_ingest_job(pdf_path: str, user_id: str, document_id: str, filename: str):
    """Queue a PDF for background ingestion. Returns immediately — the
    caller (the /upload route) doesn't wait for extraction to finish."""
    _ingest_executor.submit(_process_upload, pdf_path, user_id, document_id, filename)


def _process_upload(pdf_path: str, user_id: str, document_id: str, filename: str):
    try:
        create_vectorstore(
            pdf_path=pdf_path,
            user_id=user_id,
            source=filename,
            document_id=document_id,
        )
        update_document_status(document_id, "ready")
    except Exception as e:
        print(f"[ingest] failed for document_id={document_id}: {e}")
        update_document_status(document_id, "failed")


def ingest_text(
    text: str,
    user_id: str,
    source: str = None,
    document_id: str = None,
):
    def clean_string(value):
        if value is None:
            return None
        if isinstance(value, str):
            return value.replace("\x00", "")
        return value

    # Clean input strings
    text = clean_string(text)
    user_id = clean_string(user_id)
    source = clean_string(source)
    document_id = clean_string(document_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
    )

    docs = [
        Document(
            page_content=text,
            metadata={"source": source} if source else {},
        )
    ]

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):

        clean_content = clean_string(chunk.page_content)
        clean_source = clean_string(chunk.metadata.get("source"))

        emb = model.encode(clean_content).tolist()


        emb_json = json.dumps(emb)
        # print("emb_json null:", "\x00" in emb_json)

        try:
            save_emb(
                content=clean_content,
                user_id=user_id,
                embedding=emb,
                source=clean_source,
                document_id=document_id,
            )
        except Exception as e:
            print("\n===== SAVE_EMB ERROR =====")
            print(type(e).__name__, e)
            print("content type :", type(clean_content))
            print("user_id type :", type(user_id))
            print("source type  :", type(clean_source))
            print("doc_id type  :", type(document_id))
            print("embedding type:", type(emb))
            print("embedding len :", len(emb))
            raise

    return True



def ingest_sec_text(
    text,
    document_id,
    ticker,
    form_type,
    filename,
):
    
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
)
    chunks = splitter.split_text(text)

    embeddings = model.encode(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        save_sec_vector(
                document_id=document_id,
                ticker=ticker,
                form_type=form_type,
                filename=filename,
                chunk_index=i,
                content=chunk,
                embedding=embedding.tolist()
            )


def create_vectorstore(
    pdf_path,
    user_id,
    source=None,
    document_id=None
):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text = "\n".join(page.page_content for page in pages)
    print("Contains NUL:", "\x00" in text)

    if "\x00" in text:
        print("NUL found! Cleaning text...")
        text = text.replace("\x00", "")

    if not text or len(text.strip()) < 50:
        # Without this check, an empty/near-empty extraction still
        # "succeeds" silently — ingest_text just produces zero chunks,
        # and the document gets marked "ready" with nothing actually
        # embedded. Every question then falls through to the relevance
        # guard's "not enough information" response with no indication
        # of what actually went wrong. Raise here so _process_upload's
        # except clause marks it "failed" instead.
        raise ValueError(
            f"pypdf extracted no usable text from {pdf_path} "
            f"(got {len(text.strip()) if text else 0} characters) — "
            f"likely a scanned/image-only PDF, which pypdf can't read."
        )

    return ingest_text(
        text=text,
        user_id=user_id,
        source=source or pdf_path,
        document_id=document_id
    )