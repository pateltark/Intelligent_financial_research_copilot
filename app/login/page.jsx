"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { login, register } from "@/lib/api";
import styles from "./login.module.css";

export default function LoginPage() {
    const router = useRouter();
    const [mode, setMode] = useState("login"); // "login" | "register"
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
                setError("");
                alert("Registered! Please log in.");
            } else {
                const data = await login(form);
                // store token in cookie too (for middleware)
                document.cookie = `token=${data.access_token}; path=/`;
                router.push("/chat");
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className={styles.page}>
            <div className={styles.card}>
                {/* Header */}
                <div className={styles.header}>
                    <span className={styles.logo}>◈</span>
                    <h1 className={styles.title}>Financial Research Copilot</h1>
                    <p className={styles.subtitle}>SEC filings + document analysis</p>
                </div>

                {/* Tab toggle */}
                <div className={styles.tabs}>
                    <button
                        className={`${styles.tab} ${mode === "login" ? styles.tabActive : ""}`}
                        onClick={() => { setMode("login"); setError(""); }}
                    >
                        Sign in
                    </button>
                    <button
                        className={`${styles.tab} ${mode === "register" ? styles.tabActive : ""}`}
                        onClick={() => { setMode("register"); setError(""); }}
                    >
                        Register
                    </button>
                </div>

                {/* Form */}
                <div className={styles.form}>
                    {mode === "register" && (
                        <>
                            <input
                                className={styles.input}
                                name="user_id"
                                placeholder="User ID"
                                value={form.user_id}
                                onChange={handleChange}
                            />
                            <input
                                className={styles.input}
                                name="name"
                                placeholder="Full name"
                                value={form.name}
                                onChange={handleChange}
                            />
                        </>
                    )}
                    <input
                        className={styles.input}
                        name="email"
                        type="email"
                        placeholder="Email"
                        value={form.email}
                        onChange={handleChange}
                    />
                    <input
                        className={styles.input}
                        name="password"
                        type="password"
                        placeholder="Password"
                        value={form.password}
                        onChange={handleChange}
                    />

                    {error && <p className={styles.error}>{error}</p>}

                    <button
                        className={styles.submit}
                        onClick={handleSubmit}
                        disabled={loading}
                    >
                        {loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
                    </button>
                </div>
            </div>
        </div>
    );
}