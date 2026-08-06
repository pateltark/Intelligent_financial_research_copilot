const BASE = "http://localhost:8000";

function getToken() {
  return localStorage.getItem("token");
}

function authHeaders() {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${getToken()}`,
  };
}

export async function register({ user_id, email, password, name }) {
  const res = await fetch(`${BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id, email, password, name }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Registration failed.");
  return data;
}

export async function login({ email, password }) {
  const res = await fetch(`${BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Login failed.");
  localStorage.setItem("token", data.access_token);
  localStorage.setItem(
    "user",
    JSON.stringify({
      name: data.name || data.user?.name || email.split("@")[0],
      email: data.email || data.user?.email || email,
    })
  );
  return data;
}

export function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("user");
}

export async function uploadPDF(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` },
    body: form,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Upload failed.");
  return data;
}

export async function clearPDF() {
  const res = await fetch(`${BASE}/upload`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Clear failed.");
  return data;
}

export async function getSecActive() {
  const res = await fetch(`${BASE}/sec/active`, {
    method: "GET",
    headers: authHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to load active SEC document.");
  return data; // { active_doc, active_set }
}

export async function listSecDocuments() {
  const res = await fetch(`${BASE}/sec/documents`, {
    method: "GET",
    headers: authHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to load SEC filings.");
  return data; // list of {id, ticker, form_type, filed_at, filename}
}

export async function chatSEC(question) {
  const res = await fetch(`${BASE}/chat/sec`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ question }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "SEC query failed.");
  return data; // { answer, active_doc, active_set }
}

export async function listDocuments() {
  const res = await fetch(`${BASE}/documents`, {
    method: "GET",
    headers: authHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to load documents.");
  return data;
}

export async function chatDoc(question, documentIds) {
  const res = await fetch(`${BASE}/chat/doc`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ question, document_ids: documentIds }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Document query failed.");
  return data;
}

export async function deleteDocument(documentId) {
  const res = await fetch(`${BASE}/documents/${documentId}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Delete failed.");
  return data;
}

export async function getChatHistory(mode) {
  const res = await fetch(`${BASE}/chat/history?mode=${mode}`, {
    method: "GET",
    headers: authHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to load chat history.");
  return data; // { messages }
}

export async function clearChatHistory(mode) {
  const res = await fetch(`${BASE}/chat/history?mode=${mode}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to clear chat history.");
  return data;
}