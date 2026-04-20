# AgenticRag

> **Agentic RAG with Groq + Streamlit**

An intelligent Retrieval-Augmented Generation (RAG) system powered by Groq's ultra-fast inference and an interactive Streamlit frontend. This project uses an agentic pipeline to process, embed, retrieve, and reason over custom PDF documents.

---

## 🚀 Features

- 📄 PDF ingestion and chunking
- 🧠 Vector-based semantic search
- ⚡ Groq LLM for lightning-fast responses
- 🖥️ Streamlit interactive UI
- 🔗 Agentic multi-step reasoning pipeline

---

## 📁 Project Structure

```
AgenticRag/
├── app/                         # Streamlit application
├── src/
│   ├── __init__.py
│   └── llm/
│       ├── __init__.py
│       ├── groq_client.py       # Groq LLM client
│       └── test_groq.py         # Smoke test
├── data/
│   ├── raw_pdfs/                # Upload your source PDFs here
│   └── processed/               # Auto-generated (git-ignored)
├── vector_db/                   # Persistent vector store (git-ignored)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (NOT committed)
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
