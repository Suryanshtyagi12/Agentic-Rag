import os
import json
import logging
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DDRAgent:
    def __init__(self):
        """Initializes the DDR Agent properties."""
        self.model = "llama-3.3-70b-versatile"
        
        self.system_prompt = (
            "You are an expert building inspector. Generate a structured DDR report "
            "in JSON format with exactly these keys: property_summary, area_observations, "
            "root_cause, severity_assessment, recommended_actions, additional_notes, "
            "missing_information. Never invent facts. Write missing info as Not Available. "
            "Flag conflicts clearly. Use simple client-friendly language."
        )

    def run_agent(self, inspection_text: str, thermal_text: str, api_key: str) -> dict:
        """
        Takes parsed text from both documents and the API key, merges them, removes duplicates,
        flags conflicts, and returns a structured JSON dictionary.
        """
        if not api_key:
            logging.error("API Key must be provided to run_agent.")
            return {"error": "Missing API Key"}
            
        client = Groq(api_key=api_key)
        user_prompt = (
            "Please analyze, merge, and structure the following two reports into the requested JSON format.\n"
            "Merge the data logically, remove obvious duplicates, and flag any conflicting information between "
            "the visual inspection and the thermal imaging.\n\n"
            f"--- START SITE INSPECTION REPORT ---\n{inspection_text}\n--- END SITE INSPECTION REPORT ---\n\n"
            f"--- START THERMAL IMAGING REPORT ---\n{thermal_text}\n--- END THERMAL IMAGING REPORT ---\n"
        )

        try:
            logging.info(f"Calling Groq API using model '{self.model}'...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            response_content = chat_completion.choices[0].message.content
            
            # The response should be a valid JSON string due to response_format
            report_data = json.loads(response_content)
            logging.info("Successfully received and parsed JSON from Groq API.")
            return report_data
            
        except Exception as e:
            logging.error(f"Error communicating with Groq API or parsing JSON: {e}")
            return {
                "property_summary": "Error generating report.",
                "area_observations": "Error",
                "root_cause": "Error",
                "severity_assessment": "Error",
                "recommended_actions": "Error",
                "additional_notes": "Error",
                "missing_information": "Error"
            }
