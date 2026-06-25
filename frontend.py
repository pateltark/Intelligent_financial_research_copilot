import streamlit as st
from app import create_vectorstore, ask_llm
from db import save_chat, load_chat
import tempfile



session_id = "user_1"


if "messages" not in st.session_state:
    st.session_state.messages = load_chat(session_id)


if "vectorstores" not in st.session_state:
    st.session_state.vectorstore = None


# For PDF path
import tempfile

def save_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name
    

pdf = st.file_uploader("Upload File..")

if pdf and st.session_state.vectorstore is None:
    pdf_path = save_pdf(pdf)
    st.session_state.vectorstore = create_vectorstore(pdf_path)
    st.success("PDF processed!")    
    

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


question = st.chat_input("Ask a question...")

if question:

    
    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first before asking a question.")
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({"role": "user", "content": question})
    save_chat(session_id, "human", question)

    # Get answer and show it
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_llm(question, st.session_state.vectorstore)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_chat(session_id, "ai", answer)

    st.rerun()
