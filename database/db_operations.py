import logging
from datetime import datetime
from sqlalchemy.orm import Session
from database.models import User, Conversation, Message, Watchlist, SessionLocal
from auth.auth import hash_password, verify_password

log = logging.getLogger(__name__)


def get_session() -> Session:
    """Returns a new database session. Remember to close it after use."""
    return SessionLocal()


def _user_to_dict(user: User) -> dict:
    """
    Convert a User ORM object to a plain dict BEFORE the session closes.
    This prevents DetachedInstanceError when accessing fields after db.close().
    """
    return {
        "id":       user.id,
        "email":    user.email,
        "username": user.username,
    }


def create_user(email: str, username: str, password: str) -> dict:
    db = get_session()
    try:
        # Check for existing email
        if db.query(User).filter(User.email == email.lower()).first():
            return {"success": False, "error": "Email already registered"}

        # Check for existing username
        if db.query(User).filter(User.username == username.lower()).first():
            return {"success": False, "error": "Username already taken"}

        # Create user
        user = User(
            email           = email.lower().strip(),
            username        = username.lower().strip(),
            hashed_password = hash_password(password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # Auto-add default watchlist for new users
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            db.add(Watchlist(user_id=user.id, ticker=ticker))
        db.commit()

        # Convert to plain dict BEFORE session closes
        user_dict = _user_to_dict(user)
        log.info(f"New user created: {email}")
        return {"success": True, "user": user_dict}

    except Exception as e:
        db.rollback()
        log.error(f"create_user error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        db.close()


def login_user(email: str, password: str) -> dict:
    db = get_session()
    try:
        user = db.query(User).filter(User.email == email.lower()).first()

        if not user:
            return {"success": False, "error": "No account found with this email"}

        if not user.is_active:
            return {"success": False, "error": "Account is deactivated"}

        if not verify_password(password, user.hashed_password):
            return {"success": False, "error": "Incorrect password"}

        # Update last login timestamp
        user.last_login = datetime.utcnow()
        db.commit()

        # Convert to plain dict BEFORE session closes
        user_dict = _user_to_dict(user)
        log.info(f"User logged in: {email}")
        return {"success": True, "user": user_dict}

    except Exception as e:
        log.error(f"login_user error: {e}")
        return {"success": False, "error": "Login failed — please try again"}
    finally:
        db.close()


def get_user_by_id(user_id: int) -> dict | None:
    db = get_session()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        return _user_to_dict(user) if user else None
    finally:
        db.close()


def create_conversation(
    user_id:     int,
    title:       str       = "New Conversation",
    ticker:      str|None  = None,
    filing_type: str|None  = None,
) -> Conversation:
    
    db = get_session()
    try:
        conv = Conversation(
            user_id     = user_id,
            title       = title[:255],
            ticker      = ticker,
            filing_type = filing_type,
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        log.info(f"Conversation created: id={conv.id} user={user_id}")
        return conv
    finally:
        db.close()


def get_user_conversations(user_id: int, limit: int = 20) -> list[dict]:
    db = get_session()
    try:
        convos = (
            db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .all()
        )

        result = []
        for c in convos:
            msg_count = db.query(Message).filter(Message.conversation_id == c.id).count()
            result.append({
                "id":            c.id,
                "title":         c.title,
                "ticker":        c.ticker,
                "filing_type":   c.filing_type,
                "created_at":    c.created_at.strftime("%Y-%m-%d %H:%M") if c.created_at else "",
                "updated_at":    c.updated_at.strftime("%Y-%m-%d %H:%M") if c.updated_at else "",
                "message_count": msg_count,
            })
        return result
    finally:
        db.close()


def update_conversation_title(conversation_id: int, title: str):
    """Update the title of a conversation (called after first message)."""
    db = get_session()
    try:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            conv.title      = title[:255]
            conv.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


def delete_conversation(conversation_id: int, user_id: int) -> bool:
    db = get_session()
    try:
        conv = db.query(Conversation).filter(
            Conversation.id      == conversation_id,
            Conversation.user_id == user_id,
        ).first()

        if not conv:
            return False

        db.delete(conv)   # cascades to messages automatically
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        log.error(f"delete_conversation error: {e}")
        return False
    finally:
        db.close()


def save_message(
    conversation_id: int,
    role:            str,        # "user" or "assistant"
    content:         str,
    citations:       list  = None,
    chunks_used:     int   = None,
    model_used:      str   = None,
) -> Message:

    db = get_session()
    try:
        msg = Message(
            conversation_id = conversation_id,
            role            = role,
            content         = content,
            citations       = citations,
            chunks_used     = chunks_used,
            model_used      = model_used,
        )
        db.add(msg)

        # Update conversation's updated_at timestamp
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv:
            conv.updated_at = datetime.utcnow()

            # Auto-generate title from first user message
            if role == "user" and conv.title == "New Conversation":
                # Use first 60 chars of first question as title
                conv.title = content[:60] + ("..." if len(content) > 60 else "")

        db.commit()
        db.refresh(msg)
        return msg
    finally:
        db.close()


def get_conversation_messages(conversation_id: int) -> list[dict]:
    db = get_session()
    try:
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )

        return [
            {
                "id":             m.id,
                "role":           m.role,
                "content":        m.content,
                "citations":      m.citations or [],
                "chunks_used":    m.chunks_used,
                "model_used":     m.model_used,
                "created_at":     m.created_at.strftime("%H:%M") if m.created_at else "",
            }
            for m in messages
        ]
    finally:
        db.close()


def get_watchlist(user_id: int) -> list[str]:
    db = get_session()
    try:
        items = db.query(Watchlist).filter(Watchlist.user_id == user_id).all()
        return [item.ticker for item in items]
    finally:
        db.close()


def add_to_watchlist(user_id: int, ticker: str) -> bool:
    db = get_session()
    try:
        existing = db.query(Watchlist).filter(
            Watchlist.user_id == user_id,
            Watchlist.ticker  == ticker.upper(),
        ).first()

        if existing:
            return False   # already in watchlist

        db.add(Watchlist(user_id=user_id, ticker=ticker.upper()))
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        log.error(f"add_to_watchlist error: {e}")
        return False
    finally:
        db.close()


def remove_from_watchlist(user_id: int, ticker: str) -> bool:
    """Remove a ticker from user's watchlist."""
    db = get_session()
    try:
        item = db.query(Watchlist).filter(
            Watchlist.user_id == user_id,
            Watchlist.ticker  == ticker.upper(),
        ).first()

        if not item:
            return False

        db.delete(item)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        return False
    finally:
        db.close()