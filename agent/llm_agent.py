# llm_agent.py

import json
import os
from groq import Groq, BadRequestError
from dotenv import load_dotenv

from rag.db import load_chat, save_chat,get_sec_document
from agent.tools import TOOLS
from agent.executor import run_tool

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


SYSTEM_PROMPT = """
You are a financial research assistant.

You have access to two tools.

1. fetch_sec_document
   Download and index an SEC filing when it is not already available.

2. retrieve_documents
   Search indexed documents to answer the user's question.

Never invent financial numbers.

Always use retrieved context when answering.

If retrieved context does not contain the answer after one search,
tell the user what information is missing instead of calling tools again.
Do not repeat the same tool call with the same or similar arguments.
"""


def ask(question: str, user_id: str):

    chat = load_chat(user_id)[-10:]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Previous conversation
    for msg in chat:

        role = msg["role"].lower()

        if role not in ("system", "user", "assistant", "tool"):
            continue

        messages.append({
            "role": role,
            "content": msg["content"]
        })

    # Current user message
    messages.append({
        "role": "user",
        "content": question
    })

    save_chat(user_id, "user", question)

    MAX_ITERATIONS = 5
    MAX_MALFORMED_RETRIES = 2
    malformed_retries = 0

    # Track repeated tool calls to avoid infinite retry loops
    seen_calls = set()

    for _ in range(MAX_ITERATIONS):

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
                max_tokens=1024
            )
        except BadRequestError:

            malformed_retries += 1

            if malformed_retries > MAX_MALFORMED_RETRIES:
                answer = (
                    "I ran into trouble generating a valid tool call. "
                    "Could you rephrase your question?"
                )
                save_chat(
                    user_id=user_id,
                    role="assistant",
                    content=answer
                )
                return answer

            messages.append({
                "role": "user",
                "content": (
                    "Your previous tool call was malformed. "
                    "Please try again with valid JSON arguments."
                )
            })
            continue

        assistant_msg = response.choices[0].message

        # Final answer
        if not assistant_msg.tool_calls:

            answer = assistant_msg.content or ""

            save_chat(
                user_id=user_id,
                role="assistant",
                content=answer
            )

            return answer

        # Add assistant message containing tool calls
        messages.append(assistant_msg)

        # Execute tools
        for tool_call in assistant_msg.tool_calls:

            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            call_signature = (
                tool_name,
                json.dumps(tool_args, sort_keys=True)
            )

            if call_signature in seen_calls:
                tool_result = (
                    "This exact search was already tried and did not "
                    "contain the answer. Do not repeat it — answer with "
                    "what is available or tell the user it's missing."
                )
            else:
                seen_calls.add(call_signature)
                tool_result = run_tool(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    user_id=user_id,
                    question=question
                )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

    return "Maximum tool iterations reached."

# # llm_agent.py

# import json
# import os
# from groq import Groq
# from dotenv import load_dotenv

# from rag.db import load_chat, save_chat
# from agent.tools import TOOLS
# from agent.executor import run_tool

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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




# import json

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

#         # import json
#         # print(json.dumps(messages, indent=2))

#     # Current user message
#     messages.append({
#         "role": "user",
#         "content": question
#     })

#     save_chat(user_id, "user", question)

#     MAX_ITERATIONS = 5

#     for _ in range(MAX_ITERATIONS):

#         response = client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=messages,
#             tools=TOOLS,
#             tool_choice="auto",
#             temperature=0,
#             max_tokens=1024
#         )

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

#             tool_result = run_tool(
#                 tool_name=tool_name,
#                 tool_args=tool_args,
#                 user_id=user_id
#             )

#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tool_call.id,
#                 "content": str(tool_result)
#             })

#     return "Maximum tool iterations reached."