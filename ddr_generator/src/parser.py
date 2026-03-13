import os
import fitz  # PyMuPDF
import logging
from typing import Dict, List, Any

# Configure logging for informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PDFParser:
    def __init__(self, file_path: str, output_image_dir: str = "data/output/images/"):
        """
        Initializes the PDF parser with the target file and the output directory for images.
        """
        self.file_path = file_path
        self.output_image_dir = output_image_dir
        
        # Ensure output directory exists before writing to it
        os.makedirs(self.output_image_dir, exist_ok=True)

    def parse(self) -> Dict[str, Any]:
        """
        Extracts all text and images from the provided PDF file.
        Returns a dictionary with 'text' and 'images' keys.
        """
        result = {
            "text": "",
            "images": []
        }
        
        if not os.path.exists(self.file_path):
            logging.error(f"File not found: {self.file_path}")
            return result

        try:
            doc = fitz.open(self.file_path)
            logging.info(f"Successfully opened PDF: {self.file_path}")
        except Exception as e:
            logging.error(f"Failed to open PDF or corrupted file '{self.file_path}': {e}")
            return result

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 1. Extract all text from the current page
                result["text"] += page.get_text() + "\n"
                
                # 2. Extract all images from the current page
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Find the image positions on the page (can appear multiple times playfully)
                        rects = page.get_image_rects(xref)
                        
                        for rect_index, rect in enumerate(rects):
                            position = {
                                "x0": round(rect.x0, 2), 
                                "y0": round(rect.y0, 2), 
                                "x1": round(rect.x1, 2), 
                                "y1": round(rect.y1, 2)
                            }
                            
                            # 3. Create a descriptive filename including page and iteration
                            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                            image_filename = f"{base_name}_page{page_num+1}_img{img_index+1}_pos{rect_index+1}.{image_ext}"
                            image_filepath = os.path.join(self.output_image_dir, image_filename)
                            
                            # Save the extracted image to the specified output folder
                            with open(image_filepath, "wb") as image_file:
                                image_file.write(image_bytes)
                                
                            # 4. Tag each image with its page number and position bounding box
                            image_data = {
                                "filepath": image_filepath,
                                "page": page_num + 1,
                                "position": position
                            }
                            result["images"].append(image_data)
                            
                    except Exception as img_e:
                        logging.warning(f"Failed to extract image xref {xref} on page {page_num+1}: {img_e}")
                        
        except Exception as e:
            logging.error(f"Error while parsing contents of PDF '{self.file_path}': {e}")
        finally:
            doc.close()
            logging.info(f"Finished parsing. Extracted {len(result['text'])} chars and {len(result['images'])} images.")
            
        return result

# Simple test block (optional)
if __name__ == "__main__":
    test_pdf = "data/input/sample.pdf"
    
    # Touch a sample PDF location to simulate testing
    if not os.path.exists(test_pdf):
        print("Note: Provide a valid PDF file at 'data/input/sample.pdf' to test the parser standalone.")
    else:
        parser = PDFParser(test_pdf)
        data = parser.parse()
        print(f"Extracted {len(data['text'])} characters of text and {len(data['images'])} images.")
