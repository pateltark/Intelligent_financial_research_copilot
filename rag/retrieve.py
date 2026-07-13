# for fetch relted chunks from db according to quetion


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from groq import Groq
import os
from dotenv import load_dotenv

from db import save_emb, related_chunks, load_chat

from rag.db import related_chunks



def get_retrieve(user_id, quetion, k=1):

    result = related_chunks(user_id, quetion, k=1)

    releted_context = "\n".join([row[0] for row in result])

    return releted_context