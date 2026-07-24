import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import uuid

load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

conn = psycopg2.connect(
    database="ai",
    user="postgres",
    password="Login@100",
    host="localhost",
    port=5432
)

conn.autocommit = True
cursor = conn.cursor()

# Enable pgvector extension if not present
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

table_query = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        role VARCHAR(20) NOT NULL,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""

create_vector_table = """
    CREATE TABLE IF NOT EXISTS chat_emb (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding VECTOR(384),
        source TEXT,
        document_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""

sec_documents_table = """
    CREATE TABLE IF NOT EXISTS sec_documents_table (
        id SERIAL PRIMARY KEY,
        ticker TEXT NOT NULL,
        form_type TEXT NOT NULL,
        filed_at TEXT NOT NULL,
        filename TEXT NOT NULL,
        url TEXT NOT NULL,
        path TEXT NOT NULL,
        downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (ticker, form_type, filed_at)
    );
"""

sec_vector = """
CREATE TABLE IF NOT EXISTS sec_vectors (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES sec_documents_table(id),
    ticker TEXT NOT NULL,
    form_type TEXT NOT NULL,
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);
"""

user_login = """
    CREATE TABLE IF NOT EXISTS user_login (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        email VARCHAR (254) NOT NULL,
        pass_word TEXT NOT NULL,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""

doc_table = """CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);"""

# Run migrations/creations
cursor.execute(table_query)
cursor.execute(doc_table)
cursor.execute(user_login)
cursor.execute(sec_documents_table)
cursor.execute(sec_vector)
cursor.execute(create_vector_table)

# Run ALTER in case the table already existed without document_id
cursor.execute("ALTER TABLE chat_emb ADD COLUMN IF NOT EXISTS document_id TEXT;")


def save_sec_vector(
    document_id: int,
    ticker: str,
    form_type: str,
    filename: str,
    chunk_index: int,
    content: str,
    embedding,
):
    cursor.execute(
        """
        INSERT INTO sec_vectors (
            document_id, ticker, form_type, filename, chunk_index, content, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (document_id, ticker, form_type, filename, chunk_index, content, json.dumps(embedding)),
    )


def related_sec_chunks(ticker: str, form_type: str, question: str, k: int = 5):
    query_embedding = model.encode(question).tolist()
    cursor.execute(
        """
        SELECT content, embedding <=> %s::vector AS distance
        FROM sec_vectors
        WHERE ticker = %s AND form_type = %s
        ORDER BY distance
        LIMIT %s
        """,
        (json.dumps(query_embedding), ticker, form_type, k)
    )
    return cursor.fetchall()


def save_user_info(user_id, email, pass_word, name):
    cursor.execute(
        """
        INSERT INTO user_login (user_id, email, pass_word, name)
        VALUES (%s, %s, %s, %s)
        """,
        (user_id, email, pass_word, name)
    )


def save_doc_info(user_id, pdf_name):
    doc_id = str(uuid.uuid4())
    cursor.execute(
        """
        INSERT INTO documents (id, user_id, filename)
        VALUES (%s, %s, %s)
        """,
        (doc_id, user_id, pdf_name)
    )
    return doc_id


def save_emb(content, user_id, embedding, source=None, document_id=None):
    emb_json = json.dumps(embedding)
    cursor.execute(
        """
        INSERT INTO chat_emb (content, user_id, embedding, source, document_id)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (content, user_id, emb_json, source, document_id)
    )


def save_chat(user_id, role, content):
    cursor.execute(
        """
        INSERT INTO chat_history (user_id, role, content)
        VALUES (%s, %s, %s)
        """,
        (user_id, role, content)
    )


def has_pdf(user_id):
    cursor.execute(
        "SELECT 1 FROM chat_emb WHERE user_id = %s LIMIT 1",
        (user_id,)
    )
    return cursor.fetchone() is not None


def load_chat(user_id):
    cursor.execute(
        """
        SELECT role, content
        FROM chat_history
        WHERE user_id = %s
        ORDER BY id
        """,
        (user_id,)
    )
    rows = cursor.fetchall()
    return [{"role": "user" if role == "user" else "assistant", "content": content} for role, content in rows]


def related_chunks(user_id, question, k=3, document_id=None):
    query_embedding = model.encode(question).tolist()
    if document_id:
        cursor.execute(
            """
            SELECT content, embedding <=> %s::vector AS distance
            FROM chat_emb
            WHERE user_id = %s AND document_id = %s
            ORDER BY distance
            LIMIT %s
            """,
            (json.dumps(query_embedding), user_id, document_id, k)
        )
    else:
        cursor.execute(
            """
            SELECT content, embedding <=> %s::vector AS distance
            FROM chat_emb
            WHERE user_id = %s
            ORDER BY distance
            LIMIT %s
            """,
            (json.dumps(query_embedding), user_id, k)
        )
    return cursor.fetchall()


def get_user_by_email(email):
    cursor.execute(
        "SELECT user_id, email, pass_word, name FROM user_login WHERE email = %s",
        (email,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {"user_id": row[0], "email": row[1], "pass_word": row[2], "name": row[3]}


def get_sec_document(ticker, form_type):
    cursor.execute(
        """
        SELECT id, path
        FROM sec_documents_table
        WHERE ticker = %s AND form_type = %s
        ORDER BY filed_at DESC
        LIMIT 1
        """,
        (ticker, form_type)
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {"id": row[0], "path": row[1]}


def save_sec_document(ticker, form_type, filed_at, filename, url, path):
    cursor.execute(
        """
        INSERT INTO sec_documents_table (ticker, form_type, filed_at, filename, url, path)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, form_type, filed_at) DO NOTHING
        RETURNING id
        """,
        (ticker, form_type, filed_at, filename, url, path)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    cursor.execute(
        """
        SELECT id
        FROM sec_documents_table
        WHERE ticker = %s AND form_type = %s AND filed_at = %s
        """,
        (ticker, form_type, filed_at)
    )
    return cursor.fetchone()[0]