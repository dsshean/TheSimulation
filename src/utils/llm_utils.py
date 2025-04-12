import logging
import json
from typing import Optional, Type, TypeVar, Dict, Any
from pydantic import BaseModel, ValidationError

# Import LLMService type hint if needed, or use 'Any'/'object' if circular imports are an issue
# from src.llm_service import LLMService # Assuming this doesn't create circular dependencies

logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

async def generate_and_validate_llm_response(
    llm_service: Any, # Use 'LLMService' if import is safe, otherwise 'Any'
    prompt: str,
    response_model: Optional[Type[T]] = None,
    system_instruction: str = "Generate JSON data matching the requirements.",
    operation_description: str = "LLM Generation",
    temperature: float = 0.7,
    model_name: str = "gemini-1.5-flash" # Or your preferred default
) -> Optional[Dict]:
    """
    Calls the LLM service, validates the response against a Pydantic model if provided,
    handles errors, and returns the validated dictionary or None on failure.

    Args:
        llm_service: The initialized LLMService instance.
        prompt: The prompt to send to the LLM.
        response_model: The Pydantic model to validate the response against (optional).
        system_instruction: System instruction for the LLM (optional).
        operation_description: A description of the operation for logging purposes.
        temperature: The generation temperature.
        model_name: The specific LLM model to use.

    Returns:
        A dictionary containing the validated data, or None if generation/validation fails.
    """
    if not llm_service:
        logger.error(f"LLM Service not available for '{operation_description}'.")
        return None

    try:
        logger.debug(f"Attempting LLM call for '{operation_description}'...")

        # Construct the arguments for generate_content
        # Note: llm_service.generate_content handles Pydantic validation internally if response_model is passed
        response_dict = await llm_service.generate_content(
            prompt=prompt,
            model_name=model_name,
            system_instruction=system_instruction,
            response_model=response_model, # Pass the model directly
            temperature=temperature
        )

        # Check the response from llm_service.generate_content
        if not response_dict:
            logger.error(f"LLM service returned None/empty for '{operation_description}'. Prompt: {prompt[:100]}...")
            return None

        if "error" in response_dict:
            error_msg = response_dict.get("error", "Unknown LLM Error")
            raw_response = response_dict.get("raw_response", "")
            logger.error(f"LLM service error during '{operation_description}': {error_msg}. Raw: '{raw_response[:200]}...'")
            return None # Error dictionary already indicates failure

        # If no error key, the llm_service successfully generated and potentially validated
        logger.debug(f"Successfully received response for '{operation_description}'.")
        return response_dict # Return the dictionary (validated if model was provided)

    # Catch potential exceptions during the await call itself or unexpected issues
    except Exception as e:
        logger.error(f"Unexpected exception during LLM call/processing for '{operation_description}': {e}", exc_info=True)
        return None
