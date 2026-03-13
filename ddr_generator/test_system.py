import os
import sys
import json
import importlib.util

def print_result(success, message):
    if success:
        print(f"[PASS] - {message}")
    else:
        print(f"[FAIL] - {message}")
    return success

def run_tests():
    print("Starting Diagnostic Tests...\n")
    passed = 0
    total = 9

    # 1. Check required modules
    modules = ['fitz', 'groq', 'jinja2', 'dotenv']
    modules_success = True
    missing_modules = []
    for mod in modules:
        if importlib.util.find_spec(mod) is None:
            modules_success = False
            missing_modules.append(mod)
    
    if modules_success:
        passed += print_result(True, "All required Python modules are installed (fitz, groq, jinja2, dotenv)")
    else:
        print_result(False, f"Missing Python modules: {', '.join(missing_modules)}")

    # 2. Check .env file and GROQ_API_KEY
    env_success = False
    env_msg = ""
    if os.path.exists(".env"):
        from dotenv import dotenv_values
        config = dotenv_values(".env")
        if "GROQ_API_KEY" in config and config["GROQ_API_KEY"] and config["GROQ_API_KEY"] != "your_api_key_here":
            env_success = True
            env_msg = ".env file exists and GROQ_API_KEY is set"
        else:
            env_msg = "GROQ_API_KEY is missing or empty in .env"
    else:
        env_msg = ".env file does not exist"
        
    passed += print_result(env_success, env_msg)

    # 3. Check sample_data/ folder and mock PDFs
    sample_success = False
    inspection_pdf = "sample_data/mock_inspection_report.pdf"
    thermal_pdf = "sample_data/mock_thermal_report.pdf"
    if os.path.exists("sample_data") and os.path.exists(inspection_pdf) and os.path.exists(thermal_pdf):
        sample_success = True
        sample_msg = "sample_data/ folder exists with both mock PDFs"
    else:
        sample_msg = "sample_data/ or mock PDFs are missing"
        
    passed += print_result(sample_success, sample_msg)

    # Need parser to check text and images
    parser_success = False
    text_extracted = False
    images_extracted = False
    
    if sample_success:
        try:
            from src.parser import PDFParser
            parser1 = PDFParser(inspection_pdf)
            res1 = parser1.parse()
            
            parser2 = PDFParser(thermal_pdf)
            res2 = parser2.parse()
            
            if len(res1["text"]) > 0 and len(res2["text"]) > 0:
                text_extracted = True
                
            # If the parser executed successfully and returned a list for images
            if isinstance(res1.get("images"), list) and isinstance(res2.get("images"), list):
                images_extracted = True
        except Exception as e:
             print_result(False, f"Parser exception: {e}")

    # 4. Check text extraction
    passed += print_result(text_extracted, "parser.py can successfully extract text from both mock PDFs")
    
    # 5. Check image extraction
    passed += print_result(images_extracted, "parser.py can successfully handle image extraction from both mock PDFs")

    # 6. Check Groq API Connection
    groq_success = False
    if env_success:
        try:
            from groq import Groq
            from dotenv import load_dotenv
            load_dotenv()
            client = Groq()
            # Simple test call
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": "Just reply with one word: working."}],
                model="llama-3.3-70b-versatile",
                max_tokens=10
            )
            if chat_completion.choices:
                groq_success = True
        except Exception as e:
             pass
             
    passed += print_result(groq_success, "Groq API connection works with a simple test call")

    # 7. Check agent returns valid JSON
    agent_success = False
    from src.agent import DDRAgent
    try:
        agent = DDRAgent()
        if text_extracted and groq_success:
            from dotenv import load_dotenv
            load_dotenv()
            json_res = agent.run_agent("mock inspection", "mock thermal", os.getenv("GROQ_API_KEY", ""))
            required_keys = ["property_summary", "area_observations", "root_cause", 
                           "severity_assessment", "recommended_actions", "additional_notes", "missing_information"]
            if all(key in json_res for key in required_keys):
                agent_success = True
            else:
                pass
    except Exception as e:
        pass
        
    passed += print_result(agent_success, "agent.py returns valid JSON with all 7 required DDR sections")

    # 8. Check report_generator
    report_success = False
    from src.report_generator import ReportGenerator
    output_html = "data/output/DDR_Report.html"
    try:
        gen = ReportGenerator()
        dummy_json = {
            "property_summary": "Test Summary",
            "area_observations": "Test Obs",
            "root_cause": "Test Cause",
            "severity_assessment": "High",
            "recommended_actions": "Fix it",
            "additional_notes": "None",
            "missing_information": "N/A"
        }
        res_path = gen.generate_html(dummy_json, [], [], "DDR_Report.html")
        if res_path and os.path.exists(res_path):
            report_success = True
    except Exception as e:
        pass
        
    passed += print_result(report_success, "report_generator.py creates DDR_Report.html in output folder")

    # 9. HTML Report has 7 sections
    html_success = False
    if report_success:
        try:
            with open(output_html, "r", encoding="utf-8") as f:
                content = f.read()
                sections = ["Property Issue Summary", "Area-wise Observations", 
                          "Probable Root Cause", "Severity Assessment", 
                          "Recommended Actions", "Additional Notes", 
                          "Missing or Unclear Information"]
                if all(sec in content for sec in sections):
                    html_success = True
        except Exception as e:
            pass
            
    passed += print_result(html_success, "Final HTML report contains all 7 sections")

    print(f"\nSummary:")
    if passed == total:
        print(f"{passed}/{total} tests passed - System is READY")
    else:
        print(f"{passed}/{total} tests passed - Fix errors first")

if __name__ == "__main__":
    run_tests()
