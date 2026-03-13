import os
import fitz  # PyMuPDF

def create_mock_inspection():
    doc = fitz.open()
    page = doc.new_page()
    
    text = """
    SITE INSPECTION REPORT
    Date: 2023-10-25
    Inspector: John Doe
    Property: 123 Main St, Springfield
    
    Observation 1: Roof
    - The shingle roof on the south-facing side shows signs of wear and minor curling.
    - Flashing around the chimney appears loose.
    - Gutters on the east side are clogged with debris, causing some water to spill over.
    
    Observation 2: Basement
    - A musty smell was noted upon entering the basement.
    - There are visible water stains on the north foundation wall, extending about 2 feet up from the floor.
    - Sump pump pit has some standing water, but the pump appears operational.
    
    Observation 3: Electrical
    - Main electrical panel in the garage is outdated (estimated 1980s).
    - Several circuits are double-tapped.
    - Exposed wiring noted near the water heater.
    """
    
    page.insert_text(fitz.Point(50, 50), text, fontsize=12)
    
    os.makedirs("sample_data", exist_ok=True)
    doc.save("sample_data/mock_inspection_report.pdf")
    doc.close()

def create_mock_thermal():
    doc = fitz.open()
    page = doc.new_page()
    
    text = """
    THERMAL IMAGING REPORT
    Date: 2023-10-25
    Technician: Jane Smith
    Property: 123 Main St, Springfield
    
    Area: Roof / Attic
    - Thermal signature indicates a significant cold spot near the chimney flashing area, 
      suggesting ongoing moisture intrusion or severe insulation absence.
    - Temperature delta: -5 degrees F compared to surrounding roof deck.
    
    Area: Basement
    - Infrared scan of the north foundation wall revels a widespread cool anomaly 
      consistent with active moisture behind the drywall/concrete. 
    - The moisture signature is more extensive than the visible stains suggest.
    
    Area: Electrical Panel
    - Thermal scan of the main breaker panel shows two breakers operating at elevated 
      temperatures (approx 140 degrees F), indicating potential overloading or loose connections.
    """
    page.insert_text(fitz.Point(50, 50), text, fontsize=12)
    
    os.makedirs("sample_data", exist_ok=True)
    doc.save("sample_data/mock_thermal_report.pdf")
    doc.close()

if __name__ == "__main__":
    print("Generating mock PDFs...")
    create_mock_inspection()
    create_mock_thermal()
    print("Successfully generated mock PDFs in sample_data/ directory.")
