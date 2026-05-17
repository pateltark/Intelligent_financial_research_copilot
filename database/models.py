
import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, Boolean, Float, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ── Database setup ─────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./financial_copilot.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + threading
    echo=False,  # set True to see SQL queries in logs
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ══════════════════════════════════════════════════════════════════
# TABLE 1: users
# ══════════════════════════════════════════════════════════════════
class User(Base):
    __tablename__ = "users"

    id             = Column(Integer, primary_key=True, index=True)
    email          = Column(String(255), unique=True, index=True, nullable=False)
    username       = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password= Column(String(255), nullable=False)
    is_active      = Column(Boolean, default=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    last_login     = Column(DateTime, nullable=True)

    # Relationships — one user has many conversations, watchlist items
    conversations  = relationship("Conversation", back_populates="user", cascade="all, delete")
    watchlist      = relationship("Watchlist",     back_populates="user", cascade="all, delete")

    def __repr__(self):
        return f"<User id={self.id} email={self.email}>"


# ══════════════════════════════════════════════════════════════════
# TABLE 2: conversations
# Each "conversation" is one chat session — like a thread in ChatGPT
# ══════════════════════════════════════════════════════════════════
class Conversation(Base):
    __tablename__ = "conversations"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title       = Column(String(255), default="New Conversation")  # auto-generated from first question
    ticker      = Column(String(20),  nullable=True)   # optional company filter used in this convo
    filing_type = Column(String(20),  nullable=True)   # optional filing type filter
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user     = relationship("User",    back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete",
                            order_by="Message.created_at")

    def __repr__(self):
        return f"<Conversation id={self.id} user={self.user_id} title='{self.title}'>"


# ══════════════════════════════════════════════════════════════════
# TABLE 3: messages
# Individual messages inside a conversation
# ══════════════════════════════════════════════════════════════════
class Message(Base):
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role            = Column(String(20), nullable=False)   # "user" or "assistant"
    content         = Column(Text, nullable=False)         # the message text
    citations       = Column(JSON, nullable=True)          # list of citation dicts
    chunks_used     = Column(Integer, nullable=True)       # how many RAG chunks were used
    model_used      = Column(String(100), nullable=True)   # e.g. "llama-3.3-70b-versatile"
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Relationship
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message id={self.id} role={self.role} convo={self.conversation_id}>"


# ══════════════════════════════════════════════════════════════════
# TABLE 4: watchlist
# Companies a user is monitoring
# ══════════════════════════════════════════════════════════════════
class Watchlist(Base):
    __tablename__ = "watchlist"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ticker     = Column(String(20), nullable=False)
    added_at   = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="watchlist")

    def __repr__(self):
        return f"<Watchlist user={self.user_id} ticker={self.ticker}>"


# ══════════════════════════════════════════════════════════════════
# Create all tables
# ══════════════════════════════════════════════════════════════════
def init_db():
    """
    Creates all tables in the database.
    Safe to call multiple times — only creates tables that don't exist yet.
    """
    Base.metadata.create_all(bind=engine)
    print(" Database tables created (financial_copilot.db)")


def get_db():
    """
    Dependency function — yields a DB session and closes it after use.
    Used in auth.py and db_operations.py.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("Tables created:")
    for table in Base.metadata.tables:
        print(f"  → {table}")