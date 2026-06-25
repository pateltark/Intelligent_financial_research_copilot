from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_postgres.vectorstores import PGVector


from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


from db import save_emb

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
CONNECTION_STRING = "postgresql+psycopg2://postgres:Login@100@localhost:5432/ai"


# pdf_path = "D:\intelligent_financial_research_copilot\_10-K-2025-As-Filed.pdf"

# que = "What is about this document is ?"

def create_vectorstore(pdf_path):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=CONNECTION_STRING,
        collection_name="chat_emb",  
    )

    return vectorstore


#LLM 
def ask_llm(que, vector_store):


    context = vector_store.similarity_search(que, k=3)

    prompt = f"""
    You are a helpful assistant.

    Answer using the provided context.

    Context:
    {context}

    Question:
    {que}
    """


    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages= [{
            "role" : "user",
            "content" : prompt
        }], temperature= 0
    )

    return response.choices[0].message.content



def start(que, pdf_path):

    similar_vectors = create_vectorstore(pdf_path)

    ans = ask_llm(que, similar_vectors)
   
    return ans