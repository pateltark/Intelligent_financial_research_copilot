# llm_agent.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import os
from groq import Groq, BadRequestError
from dotenv import load_dotenv

from rag.db import load_chat, save_chat,get_sec_document
from agent.tools import TOOLS
# from agent.tools import fetch_sec_document, retrieve_documents, extract_document
from agent.executor import run_tool


from sec.edgar import (
    fetch_sec_filings,
    download_doc,
    extract_text
)

from agent.gaurd_before_fetch import check_avail_sec_doc, planner

from rag.db import save_sec_document,get_sec_document,related_sec_chunks, save_sec_vector, related_chunks,save_doc_info

from rag.emb_chunks import ingest_text,ingest_sec_text, create_vectorstore
from rag.retrieve import get_retrieve

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))





def generate_answer(question: str, context_chunks) -> str:
    # 1. Guard against empty context
    if not context_chunks or context_chunks == "No relevant SEC context found.":
        return "I couldn't find relevant information in the database to answer your question."

    # 2. Extract text from PostgreSQL output tuple: (content, distance)
    flat_chunks = []
    
    # Handles nested lists from ask_sec OR flat list from ask_upload
    if isinstance(context_chunks, list) and len(context_chunks) > 0 and isinstance(context_chunks[0], list):
        for doc_group in context_chunks:
            for chunk in doc_group:
                flat_chunks.append(chunk[0])  # chunk[0] is the text content
    else:
        for chunk in context_chunks:
            flat_chunks.append(chunk[0])

    formatted_context = "\n\n---\n\n".join(flat_chunks)

    # 3. Construct the RAG prompt
    prompt = f"""You are an expert financial research assistant. Answer the question using ONLY the provided context below.

DOCUMENT CONTEXT:
{formatted_context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Give a concise, clear answer based strictly on the provided context.
- If the answer is not present in the context, explicitly state "The provided document context does not contain enough information to answer this question."
"""

    # 4. Query Groq API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Recommended model on Groq
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content



def ask_sec(question: str, user_id: str):
    # 1. Parse required documents from the question
    planner_output = planner(question)
    docs = check_avail_sec_doc(planner_output)
    sec_chunks = []

    for doc in docs:
        request = doc["request"]
        db_row = doc["db_row"]

        # 2. IF NOT IN DB -> Fetch & Index directly using Python
        if not db_row:
            print(f"Fetching {request.ticker} {request.form_type} from SEC...")
            filings = fetch_sec_filings(
                ticker=request.ticker, 
                form_type=request.form_type
            )

            # Handle case where SEC API returns no results for this specific doc
            if not filings:
                print(f"No {request.form_type} filing found for {request.ticker}.")
                continue  # Skip to next doc instead of exiting the whole function

            filing = filings[0]
            path = download_doc(filing["url"], filing["filename"])
            text = extract_text(path)
            
            # Save document in relational DB
            document_id = save_sec_document(
                ticker=request.ticker,          # Fixed: use request.ticker
                form_type=request.form_type,    # Fixed: use request.form_type
                filed_at=filing["filed_at"],
                filename=filing["filename"],
                url=filing["url"],
                path=path
            )

            # Ingest/embed text in Vector DB
            ingest_sec_text(
                text=text,
                document_id=document_id,
                ticker=request.ticker,          # Fixed: use request.ticker
                form_type=request.form_type,    # Fixed: use request.form_type
                filename=filing["filename"]
            )    

        # 3. IF IN DB (or freshly fetched) -> Retrieve chunks
        print(f"Retrieving context for {request.ticker}...")
        doc_chunks = related_sec_chunks(
            ticker=request.ticker,
            form_type=request.form_type,
            question=question
        )
        
        if doc_chunks:
            sec_chunks.append(doc_chunks)

    llm_ans = generate_answer(question, sec_chunks)

    return llm_ans if sec_chunks else "No relevant SEC context found."


# test = ask_sec("Summarize Microsoft's last 10-Q filing.","D:\intelligent_financial_research_copilot\edgar_docs\TSLA_8-K_2026-07-02.htm",user_id="u101" )

# print(test)

def ask_upload(question, pdf_path:str, user_id: str):

    create_vectorstore(pdf_path, user_id) # -> for extracting Texts from PDF and devide into chunks
    save_doc_info(user_id, pdf_path)              # -> saving doc into DB
    rtld_chunks = related_chunks(user_id, question) # -> search and return similar chunks

    llm_ans = generate_answer(question, rtld_chunks)
    return llm_ans



test = ask_upload("give me brief Summary of this doc.","D:\intelligent_financial_research_copilot\edgar_docs\MSFT_10-Q_2026-04-29.pdf",user_id="u101" )

print(test)

# SYSTEM_PROMPT = """
# You are a financial research assistant.

# You have access to two tools.

# 1. fetch_sec_document
#    Download and index an SEC filing when it is not already available.

# 2. retrieve_documents
#    Search indexed documents to answer the user's question.

# Never invent financial numbers.

# Always use retrieved context when answering.

# If retrieved context does not contain the answer after one search,
# tell the user what information is missing instead of calling tools again.
# Do not repeat the same tool call with the same or similar arguments.
# """


# def ask(question: str, user_id: str):

#     chat = load_chat(user_id)[-10:]

#     messages = [
#         {
#             "role": "system",
#             "content": SYSTEM_PROMPT
#         }
#     ]

#     # Previous conversation
#     for msg in chat:

#         role = msg["role"].lower()

#         if role not in ("system", "user", "assistant", "tool"):
#             continue

#         messages.append({
#             "role": role,
#             "content": msg["content"]
#         })

#     # Current user message
#     messages.append({
#         "role": "user",
#         "content": question
#     })

#     save_chat(user_id, "user", question)

#     MAX_ITERATIONS = 5
#     MAX_MALFORMED_RETRIES = 2
#     malformed_retries = 0

#     # Track repeated tool calls to avoid infinite retry loops
#     seen_calls = set()

#     for _ in range(MAX_ITERATIONS):

#         try:
#             response = client.chat.completions.create(
#                 model="llama-3.3-70b-versatile",
#                 messages=messages,
#                 tools=TOOLS,
#                 tool_choice="auto",
#                 temperature=0,
#                 max_tokens=1024
#             )
#         except BadRequestError:

#             malformed_retries += 1

#             if malformed_retries > MAX_MALFORMED_RETRIES:
#                 answer = (
#                     "I ran into trouble generating a valid tool call. "
#                     "Could you rephrase your question?"
#                 )
#                 save_chat(
#                     user_id=user_id,
#                     role="assistant",
#                     content=answer
#                 )
#                 return answer

#             messages.append({
#                 "role": "user",
#                 "content": (
#                     "Your previous tool call was malformed. "
#                     "Please try again with valid JSON arguments."
#                 )
#             })
#             continue

#         assistant_msg = response.choices[0].message

#         # Final answer
#         if not assistant_msg.tool_calls:

#             answer = assistant_msg.content or ""

#             save_chat(
#                 user_id=user_id,
#                 role="assistant",
#                 content=answer
#             )

#             return answer

#         # Add assistant message containing tool calls
#         messages.append(assistant_msg)

#         # Execute tools
#         for tool_call in assistant_msg.tool_calls:

#             tool_name = tool_call.function.name
#             tool_args = json.loads(tool_call.function.arguments)

#             call_signature = (
#                 tool_name,
#                 json.dumps(tool_args, sort_keys=True)
#             )

#             if call_signature in seen_calls:
#                 tool_result = (
#                     "This exact search was already tried and did not "
#                     "contain the answer. Do not repeat it — answer with "
#                     "what is available or tell the user it's missing."
#                 )
#             else:
#                 seen_calls.add(call_signature)
#                 tool_result = run_tool(
#                     tool_name=tool_name,
#                     tool_args=tool_args,
#                     user_id=user_id,
#                     question=question
#                 )

#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tool_call.id,
#                 "content": str(tool_result)
#             })

#     return "Maximum tool iterations reached."