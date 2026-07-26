import os
import json
import uuid
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Connection pool ─────────────────────────────────────────
# Replaces the old single shared `conn`/`cursor` globals. Each request now
# borrows its own connection from the pool and returns it when done, so
# concurrent requests (FastAPI's threadpool, multiple uvicorn workers, or
# multiple ECS/EC2 tasks all pointed at the same DB) don't share a cursor.
#
# All credentials now come from environment variables — set these in your
# docker-compose.yml / ECS task definition / .env (gitignored), never
# hardcoded. If "Login@100" was ever a real password, rotate it — it was
# committed to the public repo.
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=int(os.getenv("DB_POOL_MAX", "10")),
    database=os.getenv("DB_NAME", "ai"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
)


@contextmanager
def get_db():
    """Borrow a connection+cursor from the pool for the duration of one
    call, then return it. Use as: `with get_db() as cursor: ...`"""
    conn = connection_pool.getconn()
    conn.autocommit = True
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        connection_pool.putconn(conn)


# ── Schema setup (runs once at import, using its own connection) ──────────
def _run_migrations():
    with get_db() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_login (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                email VARCHAR(254) NOT NULL,
                pass_word TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
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
        """)

        cursor.execute("""
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
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_emb (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(384),
                source TEXT,
                document_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Migrations for columns added after initial table creation
        cursor.execute("ALTER TABLE chat_emb ADD COLUMN IF NOT EXISTS document_id TEXT;")
        cursor.execute("ALTER TABLE chat_history ADD COLUMN IF NOT EXISTS mode VARCHAR(20) DEFAULT 'sec';")


_run_migrations()


# ── SEC vectors ─────────────────────────────────────────────
def save_sec_vector(document_id: int, ticker: str, form_type: str, filename: str,
                     chunk_index: int, content: str, embedding):
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO sec_vectors (
                document_id, ticker, form_type, filename, chunk_index, content, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (document_id, ticker, form_type, filename, chunk_index, content, json.dumps(embedding)),
        )


def related_sec_chunks(document_id: int, question: str, k: int = 5):
    query_embedding = model.encode(question).tolist()
    with get_db() as cursor:
        cursor.execute(
            """
            SELECT content, embedding <=> %s::vector AS distance
            FROM sec_vectors
            WHERE document_id = %s
            ORDER BY distance
            LIMIT %s
            """,
            (json.dumps(query_embedding), document_id, k),
        )
        return cursor.fetchall()


def get_sec_document(ticker, form_type):
    with get_db() as cursor:
        cursor.execute(
            """
            SELECT id, path
            FROM sec_documents_table
            WHERE ticker = %s AND form_type = %s
            ORDER BY filed_at DESC
            LIMIT 1
            """,
            (ticker, form_type),
        )
        row = cursor.fetchone()
    if not row:
        return None
    return {"id": row[0], "path": row[1]}


def save_sec_document(ticker, form_type, filed_at, filename, url, path):
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO sec_documents_table (ticker, form_type, filed_at, filename, url, path)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, form_type, filed_at) DO NOTHING
            RETURNING id
            """,
            (ticker, form_type, filed_at, filename, url, path),
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute(
            """
            SELECT id FROM sec_documents_table
            WHERE ticker = %s AND form_type = %s AND filed_at = %s
            """,
            (ticker, form_type, filed_at),
        )
        return cursor.fetchone()[0]


def list_sec_documents(limit=50):
    with get_db() as cursor:
        cursor.execute(
            """
            SELECT id, ticker, form_type, filed_at, filename
            FROM sec_documents_table
            ORDER BY filed_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()
    return [
        {"id": r[0], "ticker": r[1], "form_type": r[2], "filed_at": str(r[3]), "filename": r[4]}
        for r in rows
    ]


# ── Users ───────────────────────────────────────────────────
def save_user_info(user_id, email, pass_word, name):
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO user_login (user_id, email, pass_word, name)
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, email, pass_word, name),
        )


def get_user_by_email(email):
    with get_db() as cursor:
        cursor.execute(
            "SELECT user_id, email, pass_word, name FROM user_login WHERE email = %s",
            (email,),
        )
        row = cursor.fetchone()
    if not row:
        return None
    return {"user_id": row[0], "email": row[1], "pass_word": row[2], "name": row[3]}


# ── Uploaded PDF documents ──────────────────────────────────
def save_doc_info(user_id, pdf_name):
    doc_id = str(uuid.uuid4())
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (id, user_id, filename)
            VALUES (%s, %s, %s)
            """,
            (doc_id, user_id, pdf_name),
        )
    return doc_id


def get_user_documents(user_id):
    with get_db() as cursor:
        cursor.execute(
            """
            SELECT id, filename
            FROM documents
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
    return [{"id": row[0], "filename": row[1]} for row in rows]


def delete_document(user_id, document_id):
    # Scoped to user_id so a user can only delete their own documents,
    # even if they somehow got hold of another user's document_id.
    with get_db() as cursor:
        cursor.execute(
            "DELETE FROM chat_emb WHERE user_id = %s AND document_id = %s",
            (user_id, document_id),
        )
        cursor.execute(
            "DELETE FROM documents WHERE user_id = %s AND id = %s",
            (user_id, document_id),
        )


# ── PDF RAG embeddings ──────────────────────────────────────
def save_emb(content, user_id, embedding, source=None, document_id=None):
    emb_json = json.dumps(embedding)
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO chat_emb (content, user_id, embedding, source, document_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (content, user_id, emb_json, source, document_id),
        )


def related_chunks(user_id, question, k=3, document_ids=None):
    query_embedding = model.encode(question).tolist()
    with get_db() as cursor:
        if document_ids:
            cursor.execute(
                """
                SELECT content, embedding <=> %s::vector AS distance
                FROM chat_emb
                WHERE user_id = %s AND document_id = ANY(%s)
                ORDER BY distance
                LIMIT %s
                """,
                (json.dumps(query_embedding), user_id, document_ids, k),
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
                (json.dumps(query_embedding), user_id, k),
            )
        return cursor.fetchall()


def related_chunks_per_doc(user_id, question, document_ids, k_per_doc=4):
    query_embedding = model.encode(question).tolist()
    results = []
    with get_db() as cursor:
        for doc_id in document_ids:
            cursor.execute(
                """
                SELECT ce.content, ce.document_id, d.filename,
                       ce.embedding <=> %s::vector AS distance
                FROM chat_emb ce
                JOIN documents d ON d.id = ce.document_id
                WHERE ce.user_id = %s AND ce.document_id = %s
                ORDER BY distance
                LIMIT %s
                """,
                (json.dumps(query_embedding), user_id, doc_id, k_per_doc),
            )
            results.append(cursor.fetchall())
    return results


# ── Chat history ────────────────────────────────────────────
def save_chat(user_id, role, content, mode='sec'):
    with get_db() as cursor:
        cursor.execute(
            """
            INSERT INTO chat_history (user_id, role, content, mode)
            VALUES (%s, %s, %s, %s)
            """,
            (user_id, role, content, mode),
        )


def load_chat(user_id, mode=None):
    with get_db() as cursor:
        if mode:
            cursor.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE user_id = %s AND mode = %s
                ORDER BY id
                """,
                (user_id, mode),
            )
        else:
            cursor.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE user_id = %s
                ORDER BY id
                """,
                (user_id,),
            )
        rows = cursor.fetchall()
    return [{"role": "user" if role == "user" else "assistant", "content": content} for role, content in rows]