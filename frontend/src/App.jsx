import { useState, useEffect, useRef } from "react";
import {
  register,
  login,
  logout,
  chatSEC,
  chatDoc,
  uploadPDF,
  listDocuments,
  deleteDocument,
  getSecActive,
  listSecDocuments,
  getChatHistory,
  clearChatHistory,
} from "./api.js";
import "./App.css";

// ── Auth Guard ────────────────────────────────────────────
function isLoggedIn() {
  return !!localStorage.getItem("token");
}

// ── PDF Upload Component ──────────────────────────────────
function PDFUpload({ onUploadStart, onUploaded, onUploadError }) {
  const inputRef = useRef(null);

  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    const tempId = `pending-${Date.now()}`;
    onUploadStart({ id: tempId, filename: file.name });
    if (inputRef.current) inputRef.current.value = "";

    try {
      await uploadPDF(file);
      onUploaded(tempId);
    } catch (err) {
      onUploadError(tempId, err.message);
    }
  }

  return (
    <div className="pdf-upload">
      <input ref={inputRef} type="file" accept=".pdf" id="pdf-input" onChange={handleFile} />
      <label htmlFor="pdf-input" className="btn-upload">↑ Upload PDF</label>
    </div>
  );
}

// ── PDF Document Sidebar Component ─────────────────────────
function DocSidebar({
  documents,
  selectedIds,
  onToggle,
  onSelectAll,
  onClearAll,
  onDelete,
  maxSelect,
  onUploadStart,
  onUploaded,
  onUploadError,
  pendingUploads,
  onDismissPending,
}) {
  const atLimit = selectedIds.length >= maxSelect;

  function handleDeleteClick(e, doc) {
    e.preventDefault(); // don't toggle the checkbox underneath
    e.stopPropagation();
    if (window.confirm(`Delete "${doc.filename}"? This can't be undone.`)) {
      onDelete(doc.id);
    }
  }

  const isEmpty = documents.length === 0 && pendingUploads.length === 0;

  return (
    <aside className="doc-sidebar">
      <div className="doc-sidebar-header">
        <span>
          Your PDFs {selectedIds.length > 0 && `(${selectedIds.length}/${maxSelect})`}
        </span>
        <div className="doc-sidebar-actions">
          <button onClick={onSelectAll}>All</button>
          <button onClick={onClearAll}>None</button>
        </div>
      </div>

      <div className="doc-list-scroll">
        {isEmpty ? (
          <div className="doc-sidebar-empty">No PDFs uploaded yet.</div>
        ) : (
          <ul className="doc-list">
            {pendingUploads.map((p) => (
              <li key={p.id} className="doc-item">
                <label className="doc-label-disabled" title={p.status === "failed" ? p.error : "Processing..."}>
                  {p.status === "processing" && <span className="upload-spinner" />}
                  <span className="doc-filename">{p.filename}</span>
                </label>
                {p.status === "processing" ? (
                  <span className="doc-status-pill processing">Processing…</span>
                ) : (
                  <>
                    <span className="doc-status-pill failed">Failed</span>
                    <button
                      className="doc-delete-btn"
                      title="Dismiss"
                      onClick={() => onDismissPending(p.id)}
                    >
                      ✕
                    </button>
                  </>
                )}
              </li>
            ))}
            {documents.map((doc) => {
              const isSelected = selectedIds.includes(doc.id);
              const disableCheckbox = atLimit && !isSelected;
              return (
                <li key={doc.id} className="doc-item">
                  <label
                    className={disableCheckbox ? "doc-label-disabled" : ""}
                    title={disableCheckbox ? `Limit is ${maxSelect} documents — deselect one first` : doc.filename}
                  >
                    <input
                      type="checkbox"
                      checked={isSelected}
                      disabled={disableCheckbox}
                      onChange={() => onToggle(doc.id)}
                    />
                    <span className="doc-filename">{doc.filename}</span>
                  </label>
                  <button
                    className="doc-delete-btn"
                    title="Delete document"
                    onClick={(e) => handleDeleteClick(e, doc)}
                  >
                    🗑
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="doc-sidebar-footer">
        <PDFUpload onUploadStart={onUploadStart} onUploaded={onUploaded} onUploadError={onUploadError} />
      </div>
    </aside>
  );
}

// ── SEC Sidebar Component ──────────────────────────────────
// Informational only (no manual selection) — the planner decides which
// filing(s) become active based on what the user asks for. This just
// surfaces that state so the user can see what's currently in context.
function SecSidebar({ activeDoc, activeSet, recentFilings }) {
  const isComparing = activeSet && activeSet.length > 1;

  return (
    <aside className="doc-sidebar">
      <div className="doc-sidebar-header">
        <span>Active</span>
      </div>

      {activeDoc ? (
        <div className="sec-active-block">
          <div className="sec-active-pill">
            <span className="sec-dot" />
            {activeDoc.ticker} · {activeDoc.form_type}
          </div>
        </div>
      ) : (
        <div className="doc-sidebar-empty">
          No active filing. Try "Show Tesla's latest 10-K".
        </div>
      )}

      {isComparing && (
        <>
          <div className="doc-sidebar-header">
            <span>Comparing</span>
          </div>
          <ul className="doc-list">
            {activeSet.map((doc, i) => (
              <li key={`${doc.document_id}-${i}`} className="doc-item sec-compare-item">
                <span className="sec-dot" />
                <span className="doc-filename">{doc.ticker} · {doc.form_type}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      <div className="doc-sidebar-header">
        <span>Recently Fetched</span>
      </div>
      {recentFilings.length === 0 ? (
        <div className="doc-sidebar-empty">Nothing fetched yet.</div>
      ) : (
        <ul className="doc-list">
          {recentFilings.map((doc) => (
            <li key={doc.id} className="doc-item sec-recent-item">
              <span className="doc-filename" title={doc.filename}>
                {doc.ticker} · {doc.form_type}
              </span>
            </li>
          ))}
        </ul>
      )}

      <p className="sec-sidebar-note">Shared across all users — not personal to your account.</p>
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
            <p className="chat-empty-hint">Try: "Compare Tesla and Apple's latest 10-K revenue."</p>
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

// ── User Menu (avatar + dropdown with account info & logout) ──
function UserMenu({ onLogout }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  let user = { name: "User", email: "" };
  try {
    const stored = JSON.parse(localStorage.getItem("user") || "null");
    if (stored) user = stored;
  } catch {
    // ignore malformed localStorage value
  }

  const initial = (user.name || user.email || "U").charAt(0).toUpperCase();

  useEffect(() => {
    function handleClickOutside(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="user-menu" ref={ref}>
      <button
        className="user-avatar-btn"
        onClick={() => setOpen((o) => !o)}
        title={user.name || user.email}
      >
        {initial}
      </button>
      {open && (
        <div className="user-dropdown">
          <div className="user-dropdown-name">{user.name}</div>
          {user.email && <div className="user-dropdown-email">{user.email}</div>}
          <button className="user-dropdown-logout" onClick={onLogout}>
            Sign out
          </button>
        </div>
      )}
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

  // PDF RAG state
  const [documents, setDocuments] = useState([]);
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [pendingUploads, setPendingUploads] = useState([]);

  function handleUploadStart(pending) {
    setPendingUploads((p) => [...p, { ...pending, status: "processing" }]);
  }

  async function handleUploadDone(tempId) {
    setPendingUploads((p) => p.filter((u) => u.id !== tempId));
    await refreshDocuments();
  }

  function handleUploadError(tempId, message) {
    setPendingUploads((p) =>
      p.map((u) => (u.id === tempId ? { ...u, status: "failed", error: message } : u))
    );
  }

  function dismissPendingUpload(tempId) {
    setPendingUploads((p) => p.filter((u) => u.id !== tempId));
  }

  // SEC Agent state
  const [secActiveDoc, setSecActiveDoc] = useState(null);
  const [secActiveSet, setSecActiveSet] = useState([]);
  const [secRecentFilings, setSecRecentFilings] = useState([]);

  const [chatMode, setChatMode] = useState("sec"); // "sec" | "doc"
  const [error, setError] = useState("");
  const [historyLoaded, setHistoryLoaded] = useState({ sec: false, doc: false });

  const messages = chatMode === "sec" ? secMessages : docMessages;

  async function loadHistory(mode) {
    try {
      const { messages: past } = await getChatHistory(mode);
      const mapped = (past || []).map((m) => ({
        role: m.role,
        content: m.content,
        ...(m.role === "assistant" ? { mode } : {}),
      }));
      if (mode === "doc") setDocMessages(mapped);
      else setSecMessages(mapped);
    } catch (err) {
      setError(err.message);
    } finally {
      setHistoryLoaded((h) => ({ ...h, [mode]: true }));
    }
  }

  async function handleClearHistory() {
    if (!window.confirm("Clear this conversation? This can't be undone.")) return;
    try {
      await clearChatHistory(chatMode);
      if (chatMode === "doc") setDocMessages([]);
      else setSecMessages([]);
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshDocuments() {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshSecSidebar() {
    try {
      const [active, filings] = await Promise.all([getSecActive(), listSecDocuments()]);
      setSecActiveDoc(active.active_doc);
      setSecActiveSet(active.active_set || []);
      setSecRecentFilings(filings);
    } catch (err) {
      setError(err.message);
    }
  }

  // Load the relevant sidebar data whenever the user switches modes,
  // and hydrate that mode's thread from the server the first time it's opened.
  useEffect(() => {
    if (chatMode === "doc") {
      refreshDocuments();
      if (!historyLoaded.doc) loadHistory("doc");
    } else if (chatMode === "sec") {
      refreshSecSidebar();
      if (!historyLoaded.sec) loadHistory("sec");
    }
  }, [chatMode]);

  const MAX_COMPARE_DOCS = 5;

  function toggleDoc(id) {
    setSelectedDocIds((ids) => {
      if (ids.includes(id)) {
        return ids.filter((x) => x !== id);
      }
      if (ids.length >= MAX_COMPARE_DOCS) {
        setError(`You can compare up to ${MAX_COMPARE_DOCS} documents at once.`);
        return ids;
      }
      return [...ids, id];
    });
  }

  function selectAllDocs() {
    setSelectedDocIds(documents.slice(0, MAX_COMPARE_DOCS).map((d) => d.id));
    if (documents.length > MAX_COMPARE_DOCS) {
      setError(`Only the first ${MAX_COMPARE_DOCS} documents were selected — that's the compare limit.`);
    }
  }

  function clearDocSelection() {
    setSelectedDocIds([]);
  }

  async function handleDeleteDoc(id) {
    setError("");
    try {
      await deleteDocument(id);
      setSelectedDocIds((ids) => ids.filter((x) => x !== id));
      await refreshDocuments();
    } catch (err) {
      setError(err.message);
    }
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
        // Live-update the sidebar from this response instead of refetching
        setSecActiveDoc(data.active_doc || null);
        setSecActiveSet(data.active_set || []);
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
        // A newly fetched filing won't be in secRecentFilings yet — refresh
        // that list in the background so it shows up without a manual switch.
        listSecDocuments().then(setSecRecentFilings).catch(() => { });
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
          {messages.length > 0 && (
            <button className="btn-clear" onClick={handleClearHistory} title="Clear this conversation">
              Clear
            </button>
          )}
          <UserMenu onLogout={onLogout} />
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
            onDelete={handleDeleteDoc}
            maxSelect={MAX_COMPARE_DOCS}
            onUploadStart={handleUploadStart}
            onUploaded={handleUploadDone}
            onUploadError={handleUploadError}
            pendingUploads={pendingUploads}
            onDismissPending={dismissPendingUpload}
          />
        )}
        {chatMode === "sec" && (
          <SecSidebar
            activeDoc={secActiveDoc}
            activeSet={secActiveSet}
            recentFilings={secRecentFilings}
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