# src/generation/llm_service.py
import os
import json
import logging
from typing import Dict, Type, TypeVar, Optional
from pydantic import BaseModel, ValidationError # Added ValidationError
# Google Generative AI API
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

class LLMService:
    """Centralized service for LLM interactions using Google GenAI."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash-latest"):
        """Initializes the LLM Service."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.critical("GOOGLE_API_KEY not found. LLMService cannot be initialized.")
            raise ValueError("API key not found. Please set GOOGLE_API_KEY environment variable or pass it during initialization.")

        # Configure the library
        self.client = genai.Client(api_key=self.api_key)
        self.default_model_name = os.getenv("MODEL_GEMINI_PRO")
        logger.info(f"LLMService initialized with default model: {self.default_model_name}")
        # Note: The client object is not explicitly needed for the simple generate_content method used here.
        # If using other genai features (like listing models, embeddings), you might instantiate client = genai.Client()

    async def generate_content(self,
                               prompt: str,
                               response_model: Optional[Type[T]] = None, # Expecting a Pydantic model type
                               model_name: Optional[str] = None,
                               temperature: float = 1,
                               system_instruction: Optional[str] = None) -> Optional[Dict]: # Return type is Dict or None on failure
        """
        Generate content using the Gemini API, with JSON mode and Pydantic validation if a model is provided.
        Uses asyncio.to_thread to run the synchronous SDK call in an async context.
        Returns the validated data as a dictionary or None if validation/generation fails.
        """
        """Generate content using the Gemini API with proper error handling."""
        response_text = "No response"
        try:
            response = self.client.models.generate_content(
                contents=prompt,
                model=self.default_model_name,
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

    def generate_content_text(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 1,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate plain text content using the Gemini API.
        This method returns a raw text response without any model validation or JSON parsing.
        """
        try:
            # Use the provided model name or the default model
            model_to_use = model_name or self.default_model_name

            # Call the Gemini API
            response = self.client.models.generate_content(
                contents=prompt,
                model=model_to_use,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type='text/plain',  # Request plain text response
                    max_output_tokens=4096
                )
            )

            # Return the plain text response
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error in LLMService.generate_content_text: {e}")
            return f"Error: {str(e)}"