#planner.py

import sys
from pathlib import Path

# Add project root (intelligent_financial_research_copilot) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Literal

from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from rag.db import get_sec_document, save_sec_vector
from agent.tools import TOOLS
import json
import os


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ======================================================
# Models
# ======================================================

class DocumentRequest(BaseModel):

    company: str

    ticker: str | None = None

    form_type: Literal[
        "10-K",
        "10-Q",
        "8-K",
        "DEF 14A",
        "S-1"
    ] | None = None

    period: str | None = None


class PlannerOutput(BaseModel):

    documents: list[DocumentRequest]


# ======================================================
# Prompt
# ======================================================

SYSTEM_PROMPT = """
You are an SEC document extraction agent.

Never answer the question.

Always call extract_document.

Extract every SEC filing the user needs.

Return one document for each company.

Examples

Tesla revenue

↓

Tesla
10-K

------------------------------------

Apple latest SEC news

↓

Apple
8-K

------------------------------------

Microsoft quarterly report

↓

Microsoft
10-Q

------------------------------------

Compare Tesla and Apple revenue

↓

Tesla 10-K

Apple 10-K

------------------------------------

If no SEC filing is required

return

documents=[]

Never use empty strings.

Unknown values should be omitted.
"""


# ======================================================
# Planner
# ======================================================

def planner(question: str) -> PlannerOutput:

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        temperature=0,

        tool_choice={
            "type": "function",
            "function": {
                "name": "extract_document"
            }
        },

        tools=TOOLS,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    tool_call = response.choices[0].message.tool_calls[0]

    args = json.loads(
        tool_call.function.arguments
    )

    print("\nTool Output\n")
    print(json.dumps(args, indent=4))

    return PlannerOutput.model_validate(args)



#planner LLM


# Checks if que relted doc available or not in db.

# def check_avail_sec_doc(extract_sec_text):
#     extrat_sec_text = planner(que)
#     first_doc = extrat_sec_text.documents[0]

#     ticker = first_doc.ticker
#     form_type = first_doc.form_type
#     print (ticker,form_type)
#     downloaded_row = get_sec_document(ticker, form_type)
    
#     if downloaded_row is None:
#         print("none")
#         return None
#     else:
#         print (downloaded_row)
#         return downloaded_row


def check_avail_sec_doc(planner_output):

    results = []

    for doc in planner_output.documents:  # doc = ticker, form_type.. 

        db_row = get_sec_document(
            ticker=doc.ticker,
            form_type=doc.form_type
        )

        results.append({
            "request": doc,
            "db_row": db_row
        })

    return results

# ======================================================
# Test
# ======================================================

if __name__ == "__main__":
    test_query = "can u give me latest TESLA 10-k news ?"

    has_doc = check_avail_sec_doc(test_query)

    print(has_doc)