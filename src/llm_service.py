import os
import json
import logging
from typing import Dict, Type, TypeVar, Optional 
from pydantic import BaseModel
# Google Generative AI API
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Type variable for Pydantic models (needed for type hints in the original code)
T = TypeVar('T', bound=BaseModel)

class LLMService:
    """Centralized service for LLM interactions."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
    
    async def generate_content(self, 
                              prompt: str, 
                              model_name: str = "gemini-2.0-flash",
                              system_instruction: str = "",
                              response_model: Optional[Type[T]] = None, # Make it Optional explicitly
                              temperature: float = 0.7) -> Dict:
        """Generate content using the Gemini API with proper error handling."""
        response_text = "No response"
        try:
            response = self.client.models.generate_content(
                contents=prompt,
                model=model_name,
                
                config=types.GenerateContentConfig(
                    # system_instruction=system_instruction,
                    temperature=temperature,
                    response_mime_type='application/json',
                    response_schema=response_model,
                    max_output_tokens=120000
                )
            )
            
            response_text = response.text
            
            # Validate and parse JSON response using Pydantic model if provided
            if response_model:
                validated_response = response_model.model_validate_json(response_text)
                return validated_response.model_dump()
            
            # If no model provided, try to parse as JSON
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError from Gemini response: {e}, response text: {response_text}")
            return {"error": f"Could not decode JSON: {str(e)}", "raw_response": response_text}
        except Exception as e:
            logger.error(f"Error in LLM service: {e}")
            return {"error": str(e)}