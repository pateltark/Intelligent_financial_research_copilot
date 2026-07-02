# sec_agent.py
import os, json, requests
from dotenv import load_dotenv
from groq import Groq
from db import load_chat

load_dotenv()

client      = Groq(api_key=os.getenv("GROQ_API_KEY"))
SEC_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT")}

# ── SEC Tool Functions ────────────────────────────────────

def ticker_to_cik(ticker: str) -> str:
    data = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS
    ).json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"])
    raise ValueError(f"Ticker {ticker} not found")

def fetch_sec_filings(ticker: str, form_type: str = "10-K", n: int = 3) -> list[dict]:
    cik    = ticker_to_cik(ticker)
    padded = cik.zfill(10)
    data   = requests.get(
        f"https://data.sec.gov/submissions/CIK{padded}.json",
        headers=SEC_HEADERS
    ).json()

    filings = data["filings"]["recent"]
    results = []

    for i, form in enumerate(filings["form"]):
        if form == form_type and len(results) < n:
            accession = filings["accessionNumber"][i].replace("-", "")
            doc_name  = filings["primaryDocument"][i]
            filed_at  = filings["filingDate"][i]
            cik_raw   = cik.lstrip("0") or cik

            results.append({
                "filed_at":  filed_at,
                "form_type": form_type,
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{accession}/{doc_name}",
                "filename": f"{ticker}_{form_type}_{filed_at}.htm",
            })

    return results

def download_and_read(url: str, filename: str, save_dir: str = "edgar_docs") -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    if not os.path.exists(path):
        r = requests.get(url, headers=SEC_HEADERS)
        with open(path, "wb") as f:
            f.write(r.content)

    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
        def handle_data(self, data):
            stripped = data.strip()
            if stripped:
                self.text.append(stripped)

    with open(path, "r", errors="ignore") as f:
        raw = f.read()

    parser = TextExtractor()
    parser.feed(raw)
    full_text = " ".join(parser.text)

    keywords = ["revenue", "net income", "earnings", "total revenue",
                "gross profit", "operating income", "cash flow"]

    words     = full_text.split()
    extracted = []

    for i, word in enumerate(words):
        if any(kw in word.lower() for kw in keywords):
            start = max(0, i - 50)
            end   = min(len(words), i + 50)
            chunk = " ".join(words[start:end])
            if chunk not in extracted:
                extracted.append(chunk)

        if len(extracted) >= 10:
            break

    result = "\n---\n".join(extracted)
    return result[:3000]


# ── Tool Definition ───────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_sec_document",
            "description": "Fetch latest SEC EDGAR filings for a company. Call this ONCE per question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker e.g. TSLA, AAPL"
                    },
                    "form_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
                        "description": "10-K=annual, 10-Q=quarterly, 8-K=events"
                    },
                    "n_docs": {
                        "type": "integer",       # ← integer only, no "string"
                        "description": "Number of filings to fetch",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 3
                    }
                },
                "required": ["ticker", "form_type"]
            }
        }
    }
]


# ── Tool Executor ─────────────────────────────────────────

def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "fetch_sec_document":
        ticker    = tool_input["ticker"]
        form_type = tool_input.get("form_type", "10-K")
        n         = int(tool_input.get("n_docs", 1))

        print(f"\n🔧 Fetching: {ticker} | {form_type} | n={n}")

        try:
            filings = fetch_sec_filings(ticker, form_type, n)
        except ValueError as e:
            return str(e)

        if not filings:
            return f"No {form_type} filings found for {ticker}."

        all_text = ""
        for f in filings:
            print(f"  ↓ {f['filed_at']} — {f['filename']}")
            text = download_and_read(f["url"], f["filename"])
            all_text += f"\n\n--- {ticker} {form_type} filed {f['filed_at']} ---\n{text}"

        return all_text

    return "Unknown tool."


# ── Agent Loop ────────────────────────────────────────────

def ask(question, user_id):

    chat_history = load_chat(user_id)[-10:]

    history_text = ""
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        history_text += f"{role}: {msg['content']}\n"

    messages = [
        {
            "role": "system",
            "content": "You are a financial research assistant. Call fetch_sec_document ONCE to retrieve filings, then answer based on the result. Do not call the tool more than once per question."
        },
        {
            "role": "user",
            "content": f"""Previous conversation:
{history_text}

Current Question:
{question}"""
        }
    ]

    max_iterations = 3   # ← loop guard
    iterations     = 0

    while iterations < max_iterations:
        iterations += 1

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )

        msg         = res.choices[0].message
        stop_reason = res.choices[0].finish_reason

        if stop_reason == "tool_calls" and msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                tool_input = json.loads(tc.function.arguments)
                result     = run_tool(tc.function.name, tool_input)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

        elif stop_reason == "stop":
            print(f"\n🤖 Answer:\n{msg.content}")
            return msg.content

    return "Max iterations reached without a final answer."


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    ask("What is Tesla's latest annual revenue and net income?")
    ask("Any recent 8-K events from Apple?")
    ask("What did Microsoft say about AI in their last quarterly report?")