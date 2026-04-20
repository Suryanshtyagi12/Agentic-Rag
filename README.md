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
├── app/                  # Streamlit application
├── src/                  # Core source modules
├── data/
│   ├── raw_pdfs/         # Upload your source PDFs here
│   └── processed/        # Auto-generated processed chunks
├── vector_db/            # Persistent vector store
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (NOT committed)
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

Copy the `.env` template and add your keys:
```bash
# Edit .env and add your GROQ_API_KEY
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the App

```bash
streamlit run app/main.py
```

---

## 🔑 Getting a Groq API Key

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up / Log in
3. Navigate to **API Keys** and generate a new key
4. Paste it in your `.env` file

---

## 📦 Tech Stack

| Component     | Technology        |
|---------------|-------------------|
| LLM Backend   | Groq (LLaMA 3)    |
| UI Framework  | Streamlit         |
| Vector Store  | ChromaDB / FAISS  |
| Embeddings    | HuggingFace / OpenAI |
| PDF Parsing   | PyMuPDF / pdfplumber |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT
