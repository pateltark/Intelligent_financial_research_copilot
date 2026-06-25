import streamlit as st
from app import start, create_vectorstore, ask_llm
import tempfile

from db import save_chat, load_chat


pdf = st.file_uploader("Upload File..")


session_id = "user_1"

# For PDF path
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
    
    
if "messages" not in st.session_state:
    st.session_state.messages = load_chat(session_id)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    save_chat(session_id, "human", question)

    answer = ask_llm(
        question,
        st.session_state.vectorstore
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    save_chat(session_id, "ai", answer)

    st.rerun()
    




# question = st.chat_input("Ask a question")


# if question:

#     answer = ask_llm(
#         question,
#         st.session_state.vectorstore
#     )

#     with st.chat_message("user"):
#         st.write(question)

#     with st.chat_message("assistant"):
#         st.write(answer)