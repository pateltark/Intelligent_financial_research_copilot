from typing import Literal

from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

import json
import os

from tools import TOOLS

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


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    tests = [

        "Tesla revenue",

        "Latest SEC news about Apple",

        "What did Microsoft say about AI in its last quarterly report?",

        "Show NVIDIA annual report",

        "Compare Tesla and Apple revenue",

        "Hello",

        "What is AI?"

    ]

    for q in tests:

        print("\n" + "=" * 80)

        print(q)

        try:

            result = planner(q)

            print("\nParsed\n")

            print(
                json.dumps(
                    result.model_dump(),
                    indent=4
                )
            )

        except Exception as e:

            print(e)