import os
import argparse
import logging
import json
from dotenv import load_dotenv

from src.parser import PDFParser
from src.agent import DDRAgent
from src.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main entry point for the Detailed Diagnostic Report (DDR) generator.
    Parses input PDFs -> Uses LLM to combine data -> Generates HTML report.
    """
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="DDR Generator: Generate AI-powered building diagnostic reports.")
    parser.add_argument("--inspection", type=str, default="data/input/inspection.pdf",
                        help="Path to the Site Inspection Report PDF")
    parser.add_argument("--thermal", type=str, default="data/input/thermal.pdf",
                        help="Path to the Thermal Imaging Report PDF")
    
    args = parser.parse_args()
    
    logging.info("==================================================")
    logging.info(" Starting DDR Generator System Pipeline")
    logging.info("==================================================")
    
    inspection_pdf = args.inspection
    thermal_pdf = args.thermal
    
    # 1. Validation
    missing_files = []
    if not os.path.exists(inspection_pdf):
        missing_files.append(inspection_pdf)
    if not os.path.exists(thermal_pdf):
        missing_files.append(thermal_pdf)
        
    if missing_files:
        logging.error(f"Missing input files: {', '.join(missing_files)}")
        logging.error("Please provide valid paths using --inspection and --thermal arguments.")
        return

    # 2. Parse PDFs and extract text and images
    logging.info(f"[Step 1/3] Parsing PDFs: '{inspection_pdf}' and '{thermal_pdf}'...")
    inspection_parser = PDFParser(inspection_pdf)
    inspection_data = inspection_parser.parse()
    
    thermal_parser = PDFParser(thermal_pdf)
    thermal_data = thermal_parser.parse()

    # 3. Call Groq LLM to generate the report content
    logging.info("[Step 2/3] Analyzing data with Groq LLM...")
    api_key = os.getenv("GROQ_API_KEY")
    agent = DDRAgent()
    report_json = agent.run_agent(
        inspection_text=inspection_data["text"],
        thermal_text=thermal_data["text"],
        api_key=api_key
    )
    
    # Save intermediate JSON for debugging/audit
    os.makedirs("data/output", exist_ok=True)
    with open("data/output/intermediate_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=4)

    # 4. Generate HTML Report
    logging.info("[Step 3/3] Assembling HTML Report...")
    report_gen = ReportGenerator()
    html_path = report_gen.generate_html(
        report_json=report_json,
        inspection_images=inspection_data["images"],
        thermal_images=thermal_data["images"],
        output_filename="DDR_Report.html"
    )
    
    if html_path:
        logging.info("==================================================")
        logging.info(f" Pipeline Complete! Final report saved to: {html_path}")
        logging.info("==================================================")
    else:
        logging.error("Failed to generate HTML report.")

if __name__ == "__main__":
    main()
