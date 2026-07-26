import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import JWTError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from agent.llm_agent import ask_upload,ask_sec
from rag.emb_chunks import create_vectorstore
from auth import hash_password, verify_password, create_access_token, decode_token
from rag.db import save_user_info, get_user_by_email, save_chat, save_emb, save_doc_info

app = FastAPI(title="SEC Edgar Research API")

security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] | None = None

class RegisterRequest(BaseModel):
    user_id: str
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str


# def get_current_user(token: str = Depends(oauth2_scheme)):
#     try:
#         payload = decode_token(token)
#         return payload  
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Invalid or expired token.")

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        token = credentials.credentials
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
        doc_id = save_doc_info(user["sub"], file.filename)  # real filename, get id back FIRST
        create_vectorstore(tmp_path, user["sub"], document_id=doc_id)  # then embed, tagged with doc_id
    finally:
        os.unlink(tmp_path)

    return {"message": "PDF indexed successfully.", "document_id": doc_id}


@app.delete("/upload")
def clear_pdf(user=Depends(get_current_user)):

    pdf_ready = False
    return {"message": "PDF cleared."}



@app.post("/chat/doc")
async def chat_with_doc(req: QueryRequest, user=Depends(get_current_user)):
    user_id = user["sub"]
    
    # Save user message to history under 'doc' mode
    save_chat(user_id=user_id, role="user", content=req.question, mode="doc")
    
    # Generate Answer
    answer = ask_upload(
        question=req.question, 
        user_id=user_id, 
        ocument_ids=req.document_ids
    )
    
    # Save bot answer to history under 'doc' mode
    save_chat(user_id=user_id, role="assistant", content=answer, mode="doc")
    
    return {"answer": answer}



@app.post("/chat/sec")
async def chat_with_sec(req: QueryRequest, user=Depends(get_current_user)):
    user_id = user["sub"]
    
    # Save user message to history under 'sec' mode
    save_chat(user_id=user_id, role="user", content=req.question, mode="sec")
    answer = ask_sec(question=req.question, user_id=user_id)
    # Save bot answer to history under 'sec' mode
    save_chat(user_id=user_id, role="assistant", content=answer, mode="sec")
    
    return {"answer": answer}