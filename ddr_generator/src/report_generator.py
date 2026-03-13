import os
import logging
from jinja2 import Environment, FileSystemLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ReportGenerator:
    def __init__(self, template_dir: str = None, output_dir: str = None):
        """Initializes the Report Generator with template and output directories."""
        self.template_dir = template_dir or os.path.join(BASE_DIR, "templates")
        self.output_dir = output_dir or os.path.join(BASE_DIR, "data", "output")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup Jinja2 Environment
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        self.template_name = "report_template.html"

    def match_images_to_areas(self, area_text: str, images: list) -> list:
        """
        A heuristic to match images to specific area observations. 
        In a real scenario, this might use NLP or bounding boxes to map text to images accurately.
        For now, we just pass all images or attempt to filter based on keywords.
        """
        # For a basic implementation, we just return all images or a subset if we had metadata
        # Ideally, the LLM output 'area_observations' might be a dictionary or list 
        # where we could inject images. For this string-based template, we just pass images.
        return images

    def generate_html(self, report_json: dict, inspection_images: list, thermal_images: list, output_filename: str = "DDR_Report.html"):
        """
        Takes the JSON from the agent and extracted images, populates the template,
        and saves it to the output directory.
        """
        try:
            template = self.env.get_template(self.template_name)
            
            # Combine all images for simple displaying
            all_images = inspection_images + thermal_images
            
            # Note: For placing images under matching areas, the template and JSON need to 
            # support lists/dicts for 'area_observations'. For now, we will pass them 
            # as a separate variable to the template so it can render them.
            
            html_content = template.render(
                summary=report_json.get("property_summary", "N/A"),
                observations=report_json.get("area_observations", "N/A"),
                root_cause=report_json.get("root_cause", "N/A"),
                severity=report_json.get("severity_assessment", "N/A"),
                recommendations=report_json.get("recommended_actions", "N/A"),
                notes=report_json.get("additional_notes", "N/A"),
                missing_info=report_json.get("missing_information", "N/A"),
                images=all_images  # Added images to context
            )
            
            output_filepath = os.path.join(self.output_dir, output_filename)
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            logging.info(f"Successfully generated HTML report at: {output_filepath}")
            return output_filepath
            
        except Exception as e:
            logging.error(f"Error generating HTML report: {e}")
            return None
