from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from groq import Groq
import os
from dotenv import load_dotenv

from rag.db import save_emb, related_chunks, load_chat, save_sec_vector, related_sec_chunks

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def ingest_text(text: str, user_id: str, source: str = None):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60
    )

    docs = [Document(page_content=text, metadata={"source": source} if source else {})]

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        emb = model.encode(chunk.page_content).tolist()
        save_emb(
            chunk.page_content,
            user_id,
            emb,
            source=chunk.metadata.get("source")
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


def create_vectorstore(pdf_path, user_id, source=None):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text = "\n".join(page.page_content for page in pages)

    return ingest_text(text, user_id, source=source or pdf_path)