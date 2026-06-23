import streamlit as st
from app import start, create_vectorstore, ask_llm
import tempfile

pdf = st.file_uploader("Upload File..")



import tempfile

def save_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.getvalue())

        return tmp.name
    

if pdf and "vectorstore" not in st.session_state:

    pdf_path = save_pdf(pdf)

    st.session_state.vectorstore = create_vectorstore(pdf_path)

    st.success("PDF processed!")


question = st.chat_input("Ask a question")


if question:

    answer = ask_llm(
        question,
        st.session_state.vectorstore
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        st.write(answer)