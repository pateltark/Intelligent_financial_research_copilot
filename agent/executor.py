import json

from sec.edgar import (
    fetch_sec_filings,
    download_doc,
    extract_text
)

from rag.emb_chunks import ingest_text
from rag.retrieve import get_retrieve


def run_tool(name, args, user_id):

    if name == "fetch_sec_document":

        filings = fetch_sec_filings(
            ticker=args["ticker"],
            form_type=args["form_type"],
            n=1
        )

        filing = filings[0]

        path = download_doc(
            filing["url"],
            filing["filename"]
        )

        text = extract_text(path)

        ingest_text(
            text=text,
            user_id=user_id,
            source=filing["filename"]
        )

        return "Document indexed successfully."



    elif name == "retrieve_documents":

        context = get_retrieve(
            question=args["question"],
            user_id=user_id
        )

        return context



    return "Unknown tool."