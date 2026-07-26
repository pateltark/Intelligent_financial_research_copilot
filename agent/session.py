ACTIVE_SEC_DOC = {}

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