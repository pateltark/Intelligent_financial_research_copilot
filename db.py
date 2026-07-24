import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json

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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def save_user_info(user_id, email, pass_word, name):
    cursor.execute(
        """
        INSERT INTO user_login (user_id, email, pass_word, name)
        VALUES (%s, %s, %s, %s)
        """,
        (user_id, email, pass_word, name)
    )


def save_doc_info(user_id, pdf_name):
    cursor.execute(
        """
        INSERT INTO documents (user_id, filename)
        VALUES (%s, %s)
        """,
        (user_id, pdf_name)
    )




def save_emb(content, user_id, embedding):
    emb_json = json.dumps(embedding)
    cursor.execute(
        "INSERT INTO chat_emb (content, user_id, embedding) VALUES (%s, %s, %s)",
        (content, user_id, emb_json)
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
    messages = []

    for role, content in rows:
        messages.append({
            "role": "user" if role == "user" else "ai",
            "content": content
        })

    return messages



def related_chunks(user_id, question, k=3):
    query_embedding = model.encode(question).tolist()

    cursor.execute(
        """
        SELECT content,
               embedding <=> %s::vector AS distance
        FROM chat_emb
        WHERE user_id = %s
        ORDER BY distance
        LIMIT %s
        """,
        (
            json.dumps(query_embedding),
            user_id,
            k
        )
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




cursor.execute(table_query)
cursor.execute(doc_table)
cursor.execute(create_vector_table)
cursor.execute(user_login)