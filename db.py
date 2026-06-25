import psycopg2


conn = psycopg2.connect(
    database = "ai",
    user = "postgres",
    password = "Login@100",
    host = "localhost",
    port = 5432
)

conn.autocommit = True

cursor = conn.cursor()  #creating objects

table_query = """
        CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(50) NOT NULL,
        role VARCHAR(20) NOT NULL,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

         """
           
create_vector_table = """
        CREATE TABLE IF NOT EXISTS chat_emb(
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        role VARCHAR(20),
        content TEXT NOT NULL,
        embedding VECTOR(768),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )

      """



def save_chat(session_id, role, content):
    query = """
            INSERT INTO chat_history (session_id, role, content)
            VALUES (%s, %s, %s)
            """

    cursor.execute(
        query,
        (session_id, role, content)
    )


def save_emb(session_id, role, content, embedding):

    cursor.execute(
        """
        INSERT INTO chat_emb
        (session_id, role, content, embedding)
        VALUES (%s, %s, %s, %s)
        """,
        (
            session_id,
            role,
            content,
            embedding
        )
    )



def load_chat(session_id):

    cursor.execute ("""
            SELECT role, content
            FROM chat_history
            WHERE session_id = %s
            ORDER by id
         """,
         (session_id,))
    
    rows = cursor.fetchall()

    messages = []

    for role, content in rows:

        messages.append({
            "role": "user" if role == "human" else "assistant",
            "content": content
        })

    return messages


cursor.execute(table_query)
cursor.execute(create_vector_table)



