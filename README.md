# Detailed Diagnostic Report (DDR) Generator

An AI-powered system designed to ingest, analyze, and synthesize building Site Inspection and Thermal Imaging Reports into a structured, client-ready HTML Detailed Diagnostic Report. 

Built as part of the `Agentic-Rag` exploration.

## 🚀 Features

- **Automated PDF Parsing**: Extracts text and images (with page and coordinate tagging) from raw PDF reports using `PyMuPDF`.
- **Intelligent Synthesis**: Leverages the **Groq API** and the `llama-3.3-70b-versatile` model as an expert building inspector to merge observations, remove duplicates, and definitively flag structural conflicts.
- **Client-Ready HTML Generation**: Uses `Jinja2` templates to beautifully compile the synthesized data and automatically place associated layout images into a responsive HTML report.
- **Structured JSON Output**: Guarantees output featuring 7 critical diagnostic sections: Property Summary, Area Observations, Root Cause, Severity Assessment, Recommended Actions, Additional Notes, and Missing Information.

## 📁 Project Structure

```text
ddr_generator/
│
├── requirements.txt            # Python dependencies
├── main.py                     # Entry point pipeline for the system
├── create_mock_pdfs.py         # Testing script that generates dummy PDF reports
├── test_system.py              # Full 9-point system verification testing script
│
├── templates/
│   └── report_template.html    # Jinja2 HTML layout file
│
├── src/
│   ├── agent.py                # LLM interaction using Groq SDK
│   ├── parser.py               # PyMuPDF processing logic
│   └── report_generator.py     # HTML orchestration module
│
└── data/                       
    ├── input/                  # Folder to place your client PDFs (inspection + thermal)
    └── output/                 # Folder where DDR_Report.html is compiled
```

## 🛠️ Setup Instructions

**1. Clone the repository:**
```bash
git clone https://github.com/Suryanshtyagi12/Agentic-Rag.git
cd Agentic-Rag/ddr_generator
```

**2. Set up the Python Virtual Environment:**
```bash
python -m venv venv
```
Activate on Windows:
```powershell
.\venv\Scripts\activate
```
Activate on macOS/Linux:
```bash
source venv/bin/activate
```

**3. Install Requirements:**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables:**
Create a `.env` file in the `ddr_generator` directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_api_key_here
```

## 🧪 Demo the System

You can run the full system immediately without real client data by using the included mock generator.

**1. Generate Mock PDFs:**
```bash
python create_mock_pdfs.py
```
*This creates `mock_inspection_report.pdf` and `mock_thermal_report.pdf` inside `sample_data/`.*

**2. Run the Diagnostic Tests:**
```bash
python test_system.py
```
*Ensures all modules, folders, parsers, templates, and API keys are functioning together.*

**3. Run the Full DDR Generation Pipeline:**
```bash
python main.py --inspection sample_data/mock_inspection_report.pdf --thermal sample_data/mock_thermal_report.pdf
```
*This will parse the two mocks, process the Groq inference, and dump your final rendered `DDR_Report.html` inside `data/output/`!*
