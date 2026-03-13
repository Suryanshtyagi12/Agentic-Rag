import streamlit as st
import os
import json
import logging
import streamlit.components.v1 as components

# Configure logging to go to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Adjust PYTHONPATH so src resolves correctly
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.parser import PDFParser
from src.agent import DDRAgent
from src.report_generator import ReportGenerator
from dotenv import load_dotenv

# Load environment variables (fallback if not using secrets, but secrets preferred in Streamlit)
load_dotenv()

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    groq_api_key = None

if not groq_api_key:
    st.error("API key not configured. Contact the app owner.")
    st.stop()

# App Configuration
st.set_page_config(
    page_title="DDR Report Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    h1 {
        color: #1E3A8A;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize output and input directories using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------
# STATE MANAGEMENT
# -----------------
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'data1' not in st.session_state:
    st.session_state.data1 = None
if 'data2' not in st.session_state:
    st.session_state.data2 = None
if 'report_json' not in st.session_state:
    st.session_state.report_json = None
if 'html_content' not in st.session_state:
    st.session_state.html_content = None

# -----------------
# SIDEBAR SETTINGS
# -----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2889/2889312.png", width=80)
    st.header("⚙️ Agentic-RAG Settings")
    st.write("Configure connection & LLM parameters.")
    
    with st.expander("🔑 Access Control", expanded=True):
        st.success("API Key securely loaded from Streamlit Secrets.")
        
    with st.expander("🧠 Model Params", expanded=False):
        model_name = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192"])
        temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    st.markdown("---")
    st.markdown("### System Architecture")
    st.caption("- **Parser**: PyMuPDF OCR")
    st.caption(f"- **Model**: {model_name}")
    st.caption("- **Output Formatter**: Jinja2 + HTML5")

    if st.button("🔄 Reset App State"):
        st.session_state.processing_complete = False
        st.session_state.data1 = None
        st.session_state.data2 = None
        st.session_state.report_json = None
        st.session_state.html_content = None
        st.rerun()

# -----------------
# MAIN UI
# -----------------
st.title("📄 Detail Diagnostic Report (DDR) Engine")
st.markdown("Upload your raw *Site Inspection* and *Thermal Imaging* PDFs. Our AI agent will extract the text, analyze the data layout, and synthesize a single client-ready diagnostic report.")

# Set up tabs for a cleaner workflow
tab_upload, tab_preview, tab_final = st.tabs(["📁 1. Upload & Process", "🔍 2. Raw Data Extracted", "📜 3. Final Report"])

with tab_upload:
    st.header("Document Ingestion")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📌 Site Inspection Report")
        inspection_file = st.file_uploader("Upload Inspection PDF", type=["pdf"], key="insp")
    
    with col2:
        st.info("🌡️ Thermal Imaging Report")
        thermal_file = st.file_uploader("Upload Thermal PDF", type=["pdf"], key="therm")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("⚡ Synthesize & Generate DDR", use_container_width=True):
        if not inspection_file or not thermal_file:
            st.warning("⚠️ Please upload BOTH the Inspection Report and the Thermal Report PDFs to proceed.")
        else:
            # Step 1: Save the uploaded files to disk
            inspection_path = os.path.join(INPUT_DIR, "inspection_streamlit.pdf")
            thermal_path = os.path.join(INPUT_DIR, "thermal_streamlit.pdf")
            
            with open(inspection_path, "wb") as f:
                f.write(inspection_file.read())
            with open(thermal_path, "wb") as f:
                f.write(thermal_file.read())
    
            with st.status("Initializing AI Pipeline...", expanded=True) as status:
                try:
                    # Parse
                    st.write("🔍 Parsing Inspection PDF text & images...")
                    parser1 = PDFParser(inspection_path)
                    data1 = parser1.parse()
                    
                    st.write("🔍 Parsing Thermal PDF text & images...")
                    parser2 = PDFParser(thermal_path)
                    data2 = parser2.parse()
                    
                    st.session_state.data1 = data1
                    st.session_state.data2 = data2
                    
                    # Synthesize
                    st.write(f"🧠 Synthesizing data with Groq ({model_name})...")
                    agent = DDRAgent()
                    
                    agent.model = model_name
                    report_json = agent.run_agent(data1["text"], data2["text"], groq_api_key)
                    st.session_state.report_json = report_json
                    
                    # Generate HTML
                    st.write("🎨 Designing final interactive HTML Report...")
                    gen = ReportGenerator()
                    html_path = gen.generate_html(
                        report_json=report_json, 
                        inspection_images=data1["images"], 
                        thermal_images=data2["images"]
                    )
                    
                    if html_path:
                        # Read the final HTML
                        with open(html_path, "r", encoding="utf-8") as html_f:
                            st.session_state.html_content = html_f.read()
                            
                        st.session_state.processing_complete = True
                        status.update(label="Report Generated Successfully!", state="complete", expanded=False)
                        st.balloons()
                        st.toast("✅ Report Generation Complete!")
                        st.success("Analysis Complete! Head over to the 'Final Report' tab.")
                    else:
                        status.update(label="HTML Generation Failed", state="error", expanded=True)
                        st.error("There was an error generating the final HTML. Check system logs.")
                
                except Exception as e:
                    status.update(label="System Error Encountered", state="error", expanded=True)
                    st.error(f"An unexpected error occurred during pipeline execution:\n\n{repr(e)}")


with tab_preview:
    if st.session_state.processing_complete:
        st.header("Raw Extracted Intelligence")
        
        # Display high-level metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Inspection Chars", len(st.session_state.data1["text"]))
        col_m2.metric("Inspection Images", len(st.session_state.data1["images"]))
        col_m3.metric("Thermal Chars", len(st.session_state.data2["text"]))
        col_m4.metric("Thermal Images", len(st.session_state.data2["images"]))

        st.markdown("---")
        
        col_raw1, col_raw2 = st.columns(2)
        with col_raw1:
            with st.expander("📄 View Extracted Inspection Text"):
                st.text_area("Inspection Text", st.session_state.data1["text"], height=300, disabled=True)
        with col_raw2:
            with st.expander("🌡️ View Extracted Thermal Text"):
                st.text_area("Thermal Text", st.session_state.data2["text"], height=300, disabled=True)
                
        st.markdown("### 🤖 LLM Synthesized JSON Object")
        st.json(st.session_state.report_json)

    else:
        st.info("Upload documents and generate a report to see raw data extracted by the PyMuPDF parsers.")


with tab_final:
    if st.session_state.processing_complete:
        st.header("Client-Ready DDR Report")
        
        # Action buttons
        dl_col1, dl_col2 = st.columns([1, 4])
        with dl_col1:
            st.download_button(
                label="📥 Download HTML Report",
                data=st.session_state.html_content,
                file_name="DDR_Final_Report.html",
                mime="text/html",
                type="primary",
                use_container_width=True
            )
            
        st.markdown("---")
        # Ensure the preview occupies a good chunk of real estate
        components.html(st.session_state.html_content, height=1000, scrolling=True)
    else:
        st.info("Final HTML Report will appear here once generated.")
