import "./globals.css";

export const metadata = {
    title: "Financial Research Copilot",
    description: "SEC EDGAR + RAG powered research assistant",
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}