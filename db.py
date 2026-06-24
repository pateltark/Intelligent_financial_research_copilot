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

cursor.execute(table_query)
cursor.execute(create_vector_table)



