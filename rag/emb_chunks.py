from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rag.embeddings import get_embedding_model
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

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = get_embedding_model()
_ingest_executor = ThreadPoolExecutor(max_workers=2)


def submit_ingest_job(pdf_path: str, user_id: str, document_id: str, filename: str):
    """Queue a PDF for background ingestion."""
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


def clean_string(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def ingest_text(
    text: str,
    user_id: str,
    source: str = None,
    document_id: str = None,
    page_number: int = None,
):
    """Ingest raw string text (for non-PDF text ingestion)."""
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

    for chunk in chunks:
        clean_content = clean_string(chunk.page_content)
        clean_source = clean_string(chunk.metadata.get("source"))

        emb = model.encode(clean_content).tolist()

        try:
            save_emb(
                content=clean_content,
                user_id=user_id,
                embedding=emb,
                source=clean_source,
                document_id=document_id,
                page_number=page_number,  # Cleanly handles optional page_number
            )
        except Exception as e:
            print("\n===== SAVE_EMB ERROR =====")
            print(type(e).__name__, e)
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
            embedding=embedding.tolist(),
        )


def create_vectorstore(
    pdf_path: str,
    user_id: str,
    source: str = None,
    document_id: str = None,
):
    """Preserves PDF page numbers while chunking and saving embeddings."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    if not pages:
        raise ValueError(f"pypdf extracted no pages from {pdf_path}")

    # Clean NUL bytes on page content and validate text presence
    total_length = 0
    for page in pages:
        if page.page_content:
            page.page_content = page.page_content.replace("\x00", "")
            total_length += len(page.page_content.strip())

    if total_length < 50:
        raise ValueError(
            f"pypdf extracted no usable text from {pdf_path} "
            f"(got {total_length} characters) — likely a scanned/image-only PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
    )

    # split_documents propagates page metadata (0-indexed) to every split chunk
    chunks = splitter.split_documents(pages)

    clean_user_id = clean_string(user_id)
    clean_source = clean_string(source or pdf_path)
    clean_doc_id = clean_string(document_id)

    for chunk in chunks:
        clean_content = clean_string(chunk.page_content)
        if not clean_content or not clean_content.strip():
            continue

        # PyPDFLoader uses 0-based indexing for pages (0 -> Page 1)
        raw_page = chunk.metadata.get("page")
        page_num = (raw_page + 1) if raw_page is not None else None

        emb = model.encode(clean_content).tolist()

        save_emb(
            content=clean_content,
            user_id=clean_user_id,
            embedding=emb,
            source=clean_source,
            document_id=clean_doc_id,
            page_number=page_num,  # Correct page number saved to database
        )

    return True