import { useState, useEffect, useRef } from "react";
import { register, login, logout, chatSEC, chatDoc, uploadPDF, listDocuments } from "./api.js";
import "./App.css";

// ── Auth Guard ────────────────────────────────────────────
function isLoggedIn() {
  return !!localStorage.getItem("token");
}

// ── PDF Upload Component ──────────────────────────────────
function PDFUpload({ onUploaded, setError }) {
  const inputRef = useRef(null);

  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await uploadPDF(file);
      onUploaded();
      if (inputRef.current) inputRef.current.value = "";
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="pdf-upload">
      <input ref={inputRef} type="file" accept=".pdf" id="pdf-input" onChange={handleFile} />
      <label htmlFor="pdf-input" className="btn-upload">↑ Upload PDF</label>
    </div>
  );
}

// ── Document Sidebar Component ─────────────────────────────
function DocSidebar({ documents, selectedIds, onToggle, onSelectAll, onClearAll }) {
  if (documents.length === 0) {
    return (
      <aside className="doc-sidebar">
        <div className="doc-sidebar-empty">No PDFs uploaded yet.</div>
      </aside>
    );
  }

  return (
    <aside className="doc-sidebar">
      <div className="doc-sidebar-header">
        <span>Your PDFs</span>
        <div className="doc-sidebar-actions">
          <button onClick={onSelectAll}>All</button>
          <button onClick={onClearAll}>None</button>
        </div>
      </div>
      <ul className="doc-list">
        {documents.map((doc) => (
          <li key={doc.id} className="doc-item">
            <label>
              <input
                type="checkbox"
                checked={selectedIds.includes(doc.id)}
                onChange={() => onToggle(doc.id)}
              />
              <span className="doc-filename" title={doc.filename}>{doc.filename}</span>
            </label>
          </li>
        ))}
      </ul>
    </aside>
  );
}

// ── Message Bubble Component ──────────────────────────────
function MessageBubble({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`msg-wrapper ${isUser ? "user" : "assistant"}`}>
      {!isUser && message.mode && (
        <span className={`badge ${message.mode}`}>
          {message.mode === "sec" ? "SEC Agent" : "PDF RAG"}
        </span>
      )}
      <div className={`bubble ${isUser ? "user" : "assistant"}`}>
        {message.content}
      </div>
    </div>
  );
}

// ── Chat Window Component ─────────────────────────────────
function ChatWindow({ messages, loading, chatMode }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="chat-window chat-empty">
        <span className="chat-empty-icon">◈</span>
        {chatMode === "sec" ? (
          <>
            <p>Ask about SEC filings for public companies.</p>
            <p className="chat-empty-hint">Try: "What was Tesla's revenue in their latest 10-K?"</p>
          </>
        ) : (
          <>
            <p>Upload one or more PDFs and ask questions about their content.</p>
            <p className="chat-empty-hint">Try: "Compare the risk factors in these two documents."</p>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="chat-window">
      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}
      {loading && (
        <div className="thinking">
          <span /><span /><span />
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}

// ── Chat Input Component ──────────────────────────────────
function ChatInput({ onSend, disabled, placeholder }) {
  const [value, setValue] = useState("");

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const q = value.trim();
    if (!q || disabled) return;
    onSend(q);
    setValue("");
  }

  return (
    <div className="chat-input-bar">
      <textarea
        className="chat-textarea"
        placeholder={placeholder || "Ask a question..."}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
        disabled={disabled}
      />
      <button className="btn-send" onClick={submit} disabled={disabled || !value.trim()}>
        ↑
      </button>
    </div>
  );
}

// ── Login Page ────────────────────────────────────────────
function LoginPage({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ user_id: "", name: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function handleChange(e) {
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }));
  }

  async function handleSubmit() {
    setError("");
    setLoading(true);
    try {
      if (mode === "register") {
        await register(form);
        setMode("login");
        alert("Registered! Please log in.");
      } else {
        await login(form);
        onLogin();
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-header">
          <span className="login-logo">◈</span>
          <h1 className="login-title">Financial Research Copilot</h1>
          <p className="login-subtitle">SEC filings + document analysis</p>
        </div>

        <div className="tabs">
          <button className={`tab ${mode === "login" ? "active" : ""}`} onClick={() => { setMode("login"); setError(""); }}>
            Sign in
          </button>
          <button className={`tab ${mode === "register" ? "active" : ""}`} onClick={() => { setMode("register"); setError(""); }}>
            Register
          </button>
        </div>

        <div className="login-form">
          {mode === "register" && (
            <>
              <input className="input" name="user_id" placeholder="User ID" value={form.user_id} onChange={handleChange} />
              <input className="input" name="name" placeholder="Full name" value={form.name} onChange={handleChange} />
            </>
          )}
          <input className="input" name="email" type="email" placeholder="Email" value={form.email} onChange={handleChange} />
          <input className="input" name="password" type="password" placeholder="Password" value={form.password} onChange={handleChange} />

          {error && <p className="error-msg">{error}</p>}

          <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
            {loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Chat Page ─────────────────────────────────────────────
function ChatPage({ onLogout }) {
  const [secMessages, setSecMessages] = useState([]);
  const [docMessages, setDocMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [chatMode, setChatMode] = useState("sec"); // "sec" | "doc"
  const [error, setError] = useState("");

  const messages = chatMode === "sec" ? secMessages : docMessages;

  async function refreshDocuments() {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      setError(err.message);
    }
  }

  // Load the user's PDF list whenever they switch into PDF RAG mode
  useEffect(() => {
    if (chatMode === "doc") {
      refreshDocuments();
    }
  }, [chatMode]);

  function toggleDoc(id) {
    setSelectedDocIds((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
    );
  }

  function selectAllDocs() {
    setSelectedDocIds(documents.map((d) => d.id));
  }

  function clearDocSelection() {
    setSelectedDocIds([]);
  }

  async function handleSend(question) {
    setError("");
    const userMessage = { role: "user", content: question };
    if (chatMode === "sec") {
      setSecMessages((msgs) => [...msgs, userMessage]);
    } else {
      setDocMessages((msgs) => [...msgs, userMessage]);
    }

    setLoading(true);
    try {
      let data;
      if (chatMode === "sec") {
        data = await chatSEC(question);
      } else {
        // If the user hasn't checked anything, default to the most
        // recently uploaded PDF (documents[0], since the API returns
        // them sorted newest-first).
        const effectiveIds =
          selectedDocIds.length > 0
            ? selectedDocIds
            : documents[0]
              ? [documents[0].id]
              : [];
        data = await chatDoc(question, effectiveIds);
      }

      const assistantMessage = {
        role: "assistant",
        content: data.answer,
        mode: chatMode,
      };

      if (chatMode === "sec") {
        setSecMessages((msgs) => [...msgs, assistantMessage]);
      } else {
        setDocMessages((msgs) => [...msgs, assistantMessage]);
      }
    } catch (err) {
      if (err.message.includes("401")) {
        onLogout();
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  function docPlaceholder() {
    if (documents.length === 0) return "Upload a PDF first to ask questions...";
    if (selectedDocIds.length > 0) {
      return `Asking about ${selectedDocIds.length} selected document${selectedDocIds.length > 1 ? "s" : ""}...`;
    }
    return `Asking about most recently uploaded PDF (${documents[0].filename})...`;
  }

  return (
    <div className="chat-page-wrapper">
      <header className="chat-header">
        <div className="header-left">
          <span className="header-logo">◈</span>
          <span className="header-brand">Research Copilot</span>
        </div>

        <div className="mode-selector">
          <button
            className={`mode-btn ${chatMode === "sec" ? "active" : ""}`}
            onClick={() => setChatMode("sec")}
          >
            🏛️ SEC Agent
          </button>
          <button
            className={`mode-btn ${chatMode === "doc" ? "active" : ""}`}
            onClick={() => setChatMode("doc")}
          >
            📄 PDF RAG
          </button>
        </div>

        <div className="header-right">
          {chatMode === "doc" && (
            <PDFUpload onUploaded={refreshDocuments} setError={setError} />
          )}
          <button className="btn-logout" onClick={onLogout}>Sign out</button>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError("")}>✕</button>
        </div>
      )}

      <div className="chat-body">
        {chatMode === "doc" && (
          <DocSidebar
            documents={documents}
            selectedIds={selectedDocIds}
            onToggle={toggleDoc}
            onSelectAll={selectAllDocs}
            onClearAll={clearDocSelection}
          />
        )}

        <div className="chat-main">
          <ChatWindow messages={messages} loading={loading} chatMode={chatMode} />
          <ChatInput
            onSend={handleSend}
            disabled={loading || (chatMode === "doc" && documents.length === 0)}
            placeholder={
              chatMode === "sec"
                ? "Ask SEC Agent about filings (e.g. 10-K, 10-Q)..."
                : docPlaceholder()
            }
          />
        </div>
      </div>
    </div>
  );
}

// ── Root App ──────────────────────────────────────────────
export default function App() {
  const [loggedIn, setLoggedIn] = useState(isLoggedIn());

  function handleLogin() {
    setLoggedIn(true);
  }

  function handleLogout() {
    logout();
    setLoggedIn(false);
  }

  return loggedIn ? (
    <ChatPage onLogout={handleLogout} />
  ) : (
    <LoginPage onLogin={handleLogin} />
  );
}