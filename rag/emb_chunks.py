from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from concurrent.futures import ThreadPoolExecutor

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

# Created once at module load — Docling loads layout/OCR models on first
# use internally, so reusing this converter across uploads avoids paying
# that cost on every single request.
_docling_converter = DocumentConverter()

# Docling + embedding is CPU-heavy. This caps how many ingestion jobs run
# truly in parallel, no matter how many upload requests arrive at once —
# so 100 uploads queue up and process a couple at a time instead of all
# fighting for CPU simultaneously and slowing down every other request
# on the server (SEC chats, other users' PDF chats, everything).
# Tune based on your container's CPU count.
_ingest_executor = ThreadPoolExecutor(max_workers=2)


def submit_ingest_job(pdf_path: str, user_id: str, document_id: str, filename: str):
    """Queue a PDF for background ingestion. Returns immediately — the
    caller (the /upload route) doesn't wait for Docling to finish."""
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
    document_id: str = None
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60
    )

    docs = [
        Document(
            page_content=text,
            metadata={"source": source} if source else {}
        )
    ]

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        emb = model.encode(chunk.page_content).tolist()

        save_emb(
            content=chunk.page_content,
            user_id=user_id,
            embedding=emb,
            source=chunk.metadata.get("source"),
            document_id=document_id      # <-- NEW
        )

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
    # Docling does layout-aware parsing — tables come out as proper
    # Markdown tables instead of pypdf's flattened, column-scrambled
    # text, and it can OCR scanned pages that pypdf would return empty
    # or garbled text for.
    result = _docling_converter.convert(pdf_path)
    text = result.document.export_to_markdown()

    return ingest_text(
        text=text,
        user_id=user_id,
        source=source or pdf_path,
        document_id=document_id
    )