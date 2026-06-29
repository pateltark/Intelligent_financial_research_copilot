import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import JWTError

from sec_agent import ask as sec_ask
from rag import create_vectorstore, ask_llm
from auth import hash_password, verify_password, create_access_token, decode_token
from db import save_user_info, get_user_by_email

app = FastAPI(title="SEC Edgar Research API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


pdf_ready = False


class QueryRequest(BaseModel):
    query: str

class RegisterRequest(BaseModel):
    user_id: str
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)
        return payload  
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")



@app.post("/auth/register")
def register(body: RegisterRequest):
    existing = get_user_by_email(body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    hashed = hash_password(body.password)
    save_user_info(body.user_id, body.email, hashed, body.name)
    return {"message": "User registered successfully."}


@app.post("/auth/login")
def login(body: LoginRequest):
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["pass_word"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token(user["user_id"], user["email"])
    return {"access_token": token, "token_type": "bearer"}



@app.get("/health")
def health():
    return {"status": "ok", "pdf_ready": pdf_ready}



@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user=Depends(get_current_user)):
    global pdf_ready
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        create_vectorstore(tmp_path)
        pdf_ready = True
    finally:
        os.unlink(tmp_path)
    return {"message": "PDF indexed successfully."}


@app.delete("/upload")
def clear_pdf(user=Depends(get_current_user)):
    global pdf_ready
    pdf_ready = False
    return {"message": "PDF cleared."}


@app.post("/ask/sec")
def ask_sec(body: QueryRequest, user=Depends(get_current_user)):
    try:
        answer = sec_ask(body.query)
        return {"mode": "sec", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/rag")
def ask_rag(body: QueryRequest, user=Depends(get_current_user)):
    if not pdf_ready:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet.")
    try:
        answer = ask_llm(body.query)
        return {"mode": "rag", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_auto(body: QueryRequest, user=Depends(get_current_user)):
    q = body.query.lower()
    doc_hints = ["this document", "this report", "this pdf", "uploaded", "according to", "in the file"]
    sec_keywords = [
        "10-k", "10-q", "8-k", "sec", "edgar", "filing",
        "revenue", "net income", "earnings", "annual report",
        "quarterly", "ticker", "stock", "ipo", "s-1", "proxy",
    ]
    if pdf_ready and any(h in q for h in doc_hints):
        mode = "rag"
    elif not pdf_ready:
        mode = "sec"
    elif any(kw in q for kw in sec_keywords):
        mode = "sec"
    else:
        mode = "rag" if pdf_ready else "sec"
    try:
        if mode == "sec":
            answer = sec_ask(body.query)
        else:
            answer = ask_llm(body.query)
        return {"mode": mode, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))