import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import JWTError

from agent.llm_agent import ask
from rag.emb_chunks import create_vectorstore
from auth import hash_password, verify_password, create_access_token, decode_token
from rag.db import save_user_info, get_user_by_email, save_chat, save_emb, has_pdf, save_doc_info

app = FastAPI(title="SEC Edgar Research API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


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
    return {"status": "ok"}



@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user=Depends(get_current_user)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        create_vectorstore(tmp_path, user["sub"])
        save_doc_info(user["sub"], tmp_path)
        pdf_ready = True
    finally:
        os.unlink(tmp_path)
    return {"message": "PDF indexed successfully."}


@app.delete("/upload")
def clear_pdf(user=Depends(get_current_user)):

    pdf_ready = False
    return {"message": "PDF cleared."}



@app.post("/chat")
async def chat(request: QueryRequest, user=Depends(get_current_user)):

    answer = ask(
    question=request.query,
    user_id=user["sub"]   # see next problem
    )

    return {
        "answer": answer
    }