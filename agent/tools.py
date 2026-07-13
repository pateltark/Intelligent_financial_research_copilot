TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_sec_document",
            "description": (
                "Download and index an SEC filing. "
                "Use when the required SEC filing is not already available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string"
                    },
                    "form_type": {
                        "type": "string",
                        "enum": ["10-K","10-Q","8-K","DEF 14A","S-1"]
                    }
                },
                "required": ["ticker","form_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "Retrieve relevant document chunks from indexed documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    }
                },
                "required": ["question"]
            }
        }
    }
]