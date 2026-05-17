import os
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()


SECRET_KEY      = os.getenv("SECRET_KEY", "change-this-to-a-random-secret-in-production")
ALGORITHM       = "HS256"
TOKEN_EXPIRE_DAYS = 7

# bcrypt context for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



def hash_password(plain_password: str) -> str:
    # bcrypt max length = 72 bytes
    plain_password = plain_password[:72]
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    plain_password = plain_password[:72]
    return pwd_context.verify(plain_password, hashed_password)


def create_token(user_id: int, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email":   email,
        "exp":     datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS),
        "iat":     datetime.utcnow(),   # issued at
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "user_id": payload["user_id"],
            "email":   payload["email"],
        }
    except jwt.ExpiredSignatureError:
        return None   # token has expired
    except jwt.InvalidTokenError:
        return None   # token is invalid/tampered


def is_token_valid(token: str) -> bool:
    """Quick check — returns True if token is valid."""
    return verify_token(token) is not None