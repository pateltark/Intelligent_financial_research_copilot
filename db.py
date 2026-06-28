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
        session_id VARCHAR(50) NOT NULL,
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


def save_user_info(session_id, user_id, email, pass_word, name):
    cursor.execute(
        """
        INSERT INTO user_login (session_id, user_id, email, pass_word, name)
        VALUES (%s, %s, %s)
        """,
        (session_id, user_id, email, pass_word, name)
    )


def save_chat(session_id, user_id, role, content):
    cursor.execute(
        """
        INSERT INTO chat_history (session_id, user_id,  role, content)
        VALUES (%s, %s, %s)
        """,
        (session_id, user_id,  role, content)
    )



def save_emb(content, user_id, embedding):
    cursor.execute(
        """
        INSERT INTO chat_emb (content, user_id, embedding)
        VALUES (%s, %s)
        """,
        (
            content,
            json.dumps(embedding)   
        )
    )


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
            "role": "user" if role == "human" else "assistant",
            "content": content
        })

    return messages



def related_chunks(question, k=3):
    query_embedding = model.encode(question).tolist()

    cursor.execute(
        """
        SELECT content,
               embedding <=> %s::vector AS distance
        FROM chat_emb
        ORDER BY distance
        LIMIT %s
        """,
        (
            json.dumps(query_embedding), 
            k
        )
    )

    return cursor.fetchall()


cursor.execute(table_query)
cursor.execute(create_vector_table)