# financial_agent.py

import json
import os
from groq import Groq
from dotenv import load_dotenv

from rag.db import load_chat, save_chat
from agent.tools import TOOLS
from agent.executor import run_tool

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


SYSTEM_PROMPT = """
You are a financial research assistant.

You have access to three tools.

1. extract_document
   Use this tool ONLY to identify which SEC filing the user is referring to.

   Always return:

   {
     "documents": [
       {
         "company": "",
         "ticker": "",
         "form_type": "",
         "period": ""
       }
     ]
   }

   Even if there is only one company,
   documents MUST be an array.

   If the user is not requesting SEC filings,
   return:

   {
     "documents": []
   }

2. fetch_sec_document
   Download and index an SEC filing when it is not already available.

3. retrieve_documents
   Search indexed documents to answer the user's question.

Never invent financial numbers.

Always use retrieved context when answering.
"""


import json

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

    for _ in range(MAX_ITERATIONS):

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            max_tokens=1024
        )

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

            tool_result = run_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                user_id=user_id
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

    return "Maximum tool iterations reached."