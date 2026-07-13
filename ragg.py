from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from groq import Groq
import os
from dotenv import load_dotenv

from db import save_emb, related_chunks, load_chat

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def create_vectorstore(pdf_path, user_id):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        content = chunk.page_content
        embedding = model.encode(content).tolist()
        save_emb(content, user_id, embedding)

    print(f"{len(chunks)} chunks stored.")
    return True  



def ask_llm(question, user_id,vectorstore=None):

    results = related_chunks(user_id, question, k=3)

    context = "\n".join([row[0] for row in results])

    chat_history = load_chat(user_id)[-10:]

    history_chat = ""

    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "AI"
        history_chat += f"{role} : {msg['content']}\n"

    prompt = f"""You are a helpful financial research assistant.
                    Answer using the provided context only.

                    Previous conversation:
                    {history_chat}

                    Context:
                    {context}

                    Current Question:
                    {question}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
