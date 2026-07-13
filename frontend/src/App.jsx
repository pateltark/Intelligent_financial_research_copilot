import { useState, useEffect, useRef } from "react";
import { register, login, logout, chat, uploadPDF, clearPDF } from "./api.js";
import "./App.css";

// ── Auth Guard ────────────────────────────────────────────
function isLoggedIn() {
  return !!localStorage.getItem("token");
}

// ── PDF Upload Component ──────────────────────────────────
function PDFUpload({ pdfReady, setPdfReady, setError }) {
  const inputRef = useRef(null);

  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await uploadPDF(file);
      setPdfReady(true);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleClear() {
    try {
      await clearPDF();
      setPdfReady(false);
      if (inputRef.current) inputRef.current.value = "";
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="pdf-upload">
      {pdfReady ? (
        <div className="pdf-loaded">
          <span className="pdf-dot" />
          <span>PDF loaded</span>
          <button className="btn-clear" onClick={handleClear}>✕ Clear</button>
        </div>
      ) : (
        <>
          <input ref={inputRef} type="file" accept=".pdf" id="pdf-input" onChange={handleFile} />
          <label htmlFor="pdf-input" className="btn-upload">↑ Upload PDF</label>
        </>
      )}
    </div>
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
function ChatWindow({ messages, loading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  if (messages.length === 0 && !loading) {
    return (
      <div className="chat-window chat-empty">
        <span className="chat-empty-icon">◈</span>
        <p>Ask about SEC filings, or upload a PDF and query it.</p>
        <p className="chat-empty-hint">Try: "What was Apple's revenue in the latest 10-K?"</p>
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
function ChatInput({ onSend, disabled }) {
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
        placeholder="Ask about filings, or query your PDF…"
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
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [pdfReady, setPdfReady] = useState(false);
  const [error, setError] = useState("");

  async function handleSend(query) {
    setError("");
    setMessages((m) => [...m, { role: "user", content: query }]);
    setLoading(true);
    try {
      const data = await chat(query);
      setMessages((m) => [...m, { role: "assistant", content: data.answer, mode: data.mode }]);
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

  return (
    <div className="chat-page">
      <header className="chat-header">
        <div className="header-left">
          <span className="header-logo">◈</span>
          <span className="header-brand">Research Copilot</span>
        </div>
        <div className="header-right">
          <PDFUpload pdfReady={pdfReady} setPdfReady={setPdfReady} setError={setError} />
          <button className="btn-logout" onClick={onLogout}>Sign out</button>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError("")}>✕</button>
        </div>
      )}

      <ChatWindow messages={messages} loading={loading} />
      <ChatInput onSend={handleSend} disabled={loading} />
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