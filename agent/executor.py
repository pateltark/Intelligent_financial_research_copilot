# executor.py

from sec.edgar import (
    fetch_sec_filings,
    download_doc,
    extract_text
)

from agent.gaurd_before_fetch import check_avail_sec_doc, planner

from rag.db import save_sec_document,get_sec_document,related_sec_chunks

from rag.emb_chunks import ingest_text,ingest_sec_text
from rag.retrieve import get_retrieve


def run_tool(tool_name, tool_args, user_id, question):

    if tool_name == "fetch_sec_document":

        planner_output = planner(question)
        docs = check_avail_sec_doc(planner_output)

        already_indexed = []

        for doc in docs:
            request = doc["request"]
            db_row = doc["db_row"]

            if db_row:
                already_indexed.append(f"{request.ticker} {request.form_type}")

        if already_indexed:
            return f"Already indexed: {', '.join(already_indexed)}. Use retrieve_documents to get context."

        # --- not available yet, fetch fresh ---
        filings = fetch_sec_filings(
            ticker=tool_args["ticker"],
            form_type=tool_args["form_type"],
            n=1
        )

        if not filings:
            return (
                f"No {tool_args['form_type']} filing found "
                f"for {tool_args['ticker']}."
            )

        filing = filings[0]

        path = download_doc(filing["url"], filing["filename"])
        text = extract_text(path)

        document_id = save_sec_document(
            ticker=tool_args["ticker"],
            form_type=tool_args["form_type"],
            filed_at=filing["filed_at"],
            filename=filing["filename"],
            url=filing["url"],
            path=path
        )

        ingest_sec_text(
            text=text,
            document_id=document_id,
            ticker=tool_args["ticker"],
            form_type=tool_args["form_type"],
            filename=filing["filename"]
        )

        return f"Indexed {filing['filename']} successfully. Use retrieve_documents to get context."

    elif tool_name == "retrieve_documents":
        return get_retrieve(
            question=tool_args["question"],
            user_id=user_id
        )

    return f"Unknown tool: {tool_name}"