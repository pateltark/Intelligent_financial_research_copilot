from sec.edgar import (
    fetch_sec_filings,
    download_doc,
    extract_text
)

from rag.emb_chunks import ingest_text
from rag.retrieve import get_retrieve


def run_tool(tool_name, tool_args, user_id):

    if tool_name == "fetch_sec_document":

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

        return (
            f"Indexed {filing['filename']} successfully."
        )



    elif tool_name == "retrieve_documents":

        context = get_retrieve(
            question=tool_args["question"],
            user_id=user_id
        )

        return context


    return f"Unknown tool: {tool_name}"