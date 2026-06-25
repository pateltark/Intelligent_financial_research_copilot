from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from db import save_emb, related_chunks
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def create_vectorstore(pdf_path):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        content = chunk.page_content
        embedding = model.encode(content).tolist()
        save_emb(content, embedding)

    print(f"{len(chunks)} chunks stored.")
    return True  



def ask_llm(question, vectorstore=None):

    results = related_chunks(question, k=3)

    context = "\n".join([row[0] for row in results])

    prompt = f"""You are a helpful assistant.
                Answer using the provided context only.

                Context:
                {context}

                Question:
                {question}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content