from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# pdf_path = "D:\intelligent_financial_research_copilot\_10-K-2025-As-Filed.pdf"

# que = "What is about this document is ?"

def create_vectorstore(pdf_path):

#PDF load

    loader = PyPDFLoader(pdf_path)

    pages = loader.load()

    #print(len(pages))

    # print (full_text[:100])


    # chunking

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)

    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
    )

    persist_directory = "./chroma_db"

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vector_store



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