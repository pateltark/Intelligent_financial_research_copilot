ACTIVE_SEC_DOC = {}   # user_id -> single doc (last touched) — used for CURRENT_DOC follow-ups
ACTIVE_SEC_SET = {}   # user_id -> list of docs from the most recent fetch/compare turn

# NOTE: both dicts live in process memory. Fine for local dev / a single
# uvicorn worker. If this ever runs with --workers > 1 or behind multiple
# server instances, this state won't be shared across them — move to a
# DB table or Redis at that point.


def set_active_doc(user_id, document_id, ticker, form_type):
    ACTIVE_SEC_DOC[user_id] = {
        "document_id": document_id,
        "ticker": ticker,
        "form_type": form_type,
    }


def get_active_doc(user_id):
    return ACTIVE_SEC_DOC.get(user_id)


def clear_active_doc(user_id):
    ACTIVE_SEC_DOC.pop(user_id, None)


def set_active_set(user_id, docs):
    """docs: list of {document_id, ticker, form_type} — the full set of
    documents touched on the most recent fetch/compare turn."""
    ACTIVE_SEC_SET[user_id] = docs


def get_active_set(user_id):
    return ACTIVE_SEC_SET.get(user_id, [])


def clear_active_set(user_id):
    ACTIVE_SEC_SET.pop(user_id, None)