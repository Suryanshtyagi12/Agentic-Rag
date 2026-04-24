# AgenticRag

> **Agentic RAG with Groq + Streamlit**

An intelligent Retrieval-Augmented Generation (RAG) system powered by Groq's ultra-fast inference and an interactive Streamlit frontend. This project uses an agentic pipeline to process, embed, retrieve, and reason over custom PDF documents.

---
🔗 Live Demo
👉 Demo Link: https://agentic-rag-l34fp6ukcntxfxanq4jzdc.streamlit.app/
## 🚀 Features

- 📄 **PDF Ingestion** — text, tables (pdfplumber), and images (PyMuPDF) extracted and chunked
- 🧠 **Semantic Search** — `all-MiniLM-L6-v2` embeddings stored in FAISS
- ⚡ **Groq LLM** — `llama3-70b-8192` for ultra-fast responses
- 🔗 **Agentic Loop** — Think → Retrieve → Evaluate → Answer (up to 3 iterations)
- 🖥️ **Streamlit UI** — Upload PDF, ask questions, inspect retrieved chunks + reasoning trace
- 💾 **Persistent Index** — FAISS index saved to `vector_db/` and reloaded between sessions

---

## 📁 Project Structure

```
AgenticRag/
├── app/
│   └── main.py                  # Streamlit UI
├── src/
│   ├── agent/
│   │   ├── agent.py             # Agentic loop (Think→Retrieve→Evaluate→Answer)
│   │   ├── tools.py             # Retrieval tool
│   │   └── prompts.py           # System + structured prompts
│   ├── embeddings/
│   │   └── embedder.py          # all-MiniLM-L6-v2 sentence embeddings
│   ├── ingestion/
│   │   ├── loader.py            # PDF loader & validator
│   │   ├── parser.py            # Text + table + image extractor
│   │   ├── chunking.py          # Overlapping chunk splitter
│   │   └── run_ingestion.py     # CLI ingestion runner
│   ├── llm/
│   │   ├── groq_client.py       # Groq API client (llama3-70b-8192)
│   │   └── test_groq.py         # LLM smoke test
│   ├── retriever/
│   │   ├── retriever.py         # Build + query FAISS index
│   │   └── test_retrieval.py    # Retrieval smoke test
│   └── vectorstore/
│       └── vectordb.py          # FAISS index + JSON metadata store
├── data/
│   ├── raw_pdfs/                # Drop PDFs here
│   └── processed/               # Auto-generated chunks JSON (git-ignored)
├── vector_db/                   # FAISS index files (git-ignored)
├── requirements.txt
├── .env                         # GROQ_API_KEY (never committed)
├── .gitignore
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Suryanshtyagi12/Agentic-RAG.git
cd AgenticRag
```

### 2. Create & Activate Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Open the `.env` file at the project root and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Never commit `.env`** — it is listed in `.gitignore` and will never be tracked by Git.

### 5. Test the LLM Connection

Verify that your Groq API key works before running the full app:
```bash
python src/llm/test_groq.py
```

Expected output:
```
============================================================
  Groq LLM Smoke Test
  Model : llama3-70b-8192
============================================================

[*] Sending prompt: 'What is RAG?'

[✓] Response received:
...
============================================================
  Test PASSED
============================================================
```

### 6. Run the App

```bash
streamlit run app/main.py
```

---

## 🔑 Getting a Groq API Key

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up / Log in
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key and paste it into your `.env` file:
   ```env
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
5. The key is loaded automatically via `python-dotenv` — no manual export needed

> 💡 The model used is `llama3-70b-8192` — Groq's fastest large-context LLaMA 3 endpoint.

---

## 📦 Tech Stack

| Component       | Technology                     |
|-----------------|--------------------------------|
| LLM Backend     | Groq — `llama3-70b-8192`       |
| UI Framework    | Streamlit                      |
| Vector Store    | FAISS (`faiss-cpu`)            |
| Embeddings      | `sentence-transformers`        |
| PDF Parsing     | PyMuPDF + pdfplumber           |
| OCR             | pytesseract + Pillow           |
| Env Management  | `python-dotenv`                |
| Data Processing | NumPy + Pandas                 |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT
