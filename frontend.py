import streamlit as st
from app import start, create_vectorstore
import tempfile

pdf = st.file_uploader("Upload File..")

question = st.chat_input("Ask que..")

pdf = st.file_uploader("Upload PDF")

if pdf and "vectorstore" not in st.session_state:

    pdf_path = save_pdf(pdf)

    st.session_state.vectorstore = create_vectorstore(pdf_path)

    st.success("PDF processed!")