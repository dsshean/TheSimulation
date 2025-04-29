# src/tools/world_engine_tools.py
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from google.adk.tools import ToolContext
from google.adk.tools.tool_context import ToolContext
from GoogleNews import GoogleNews
from rich.console import Console

from src.generation.llm_service import LLMService
from src.tools.python_weather.client import Client  # Import the Client class
from src.tools.python_weather.constants import IMPERIAL

# Load environment variables from .env file
TURN_VALIDATION_RESULTS_KEY = "turn_validation_results"
logger = logging.getLogger(__name__)
googlenews = GoogleNews()
console = Console()

############ Probably not needed most moved to world state tools ############

def process_movement(origin: str, destination: str, tool_context: ToolContext) -> dict:
    """Calculates travel time, signals world time and location updates via state_delta. Returns a result dictionary."""
    console.print(f"[dim green]--- Tool: World Engine processing move from [i]{origin}[/i] to [i]{destination}[/i] ---[/dim green]")
    # !! Placeholder Logic !!
    travel_time_minutes = 30
    current_time_str = tool_context.state.get("world_time", "Day 1, 09:00") # Read current time
    new_time_str = f"{current_time_str} (+{travel_time_minutes}m)" # Simple placeholder

    # --- Prepare state delta ---
    state_changes = {
        "world_time": new_time_str,
        "simulacra_location": destination # Assuming this key is correct for the agent using this tool
    }
    # Add last update info to the delta as well
    result_message = f"Travel from {origin} to {destination} took {travel_time_minutes} minutes. You arrived at {destination} at {new_time_str}."
    state_changes["last_world_engine_update"] = {
        "status": "success",
        "duration": travel_time_minutes,
        "new_location": destination,
        "new_time": new_time_str,
        "message": result_message
    }

    # --- Assign to state_delta ---
    try:
        tool_context.state_delta = state_changes
        logger.info(f"Signaled state_delta for movement: {list(state_changes.keys())}")
    except Exception as e:
        logger.exception(f"Error assigning state_delta in process_movement: {e}")
        # Return error if assignment fails
        return {
            "status": "error",
            "message": f"Internal error signaling state update: {e}"
        }

    console.print(f"[dim green]--- Tool: World Engine signaled state update: Time=[b]{new_time_str}[/b], Location=[b]{destination}[/b] ---[/dim green]")
    # Return the result information (not the state delta)
    return None

def advance_time(minutes: int, tool_context: ToolContext) -> str:
    """Advances the world clock by a specified number of minutes, signaling update via state_delta. Called ONLY by Narration."""
    console.print(f"[dim green]--- Tool: World Engine advancing time by {minutes} minutes ---[/dim green]")
    current_time_str = tool_context.state.get("world_time", "Day 1, 09:00") # Read current time
    # !! Placeholder Logic !!
    new_time_str = f"{current_time_str} (+{minutes}m)"
    result_msg = f"Time advanced by {minutes} minutes. Current time is {new_time_str}."

    # --- Prepare state delta ---
    state_changes = {
        "world_time": new_time_str,
        "last_world_engine_update": {"status": "success", "message": result_msg}
    }

    # --- Assign to state_delta ---
    try:
        tool_context.state_delta = state_changes
        logger.info(f"Signaled state_delta for time advance: {list(state_changes.keys())}")
    except Exception as e:
        logger.exception(f"Error assigning state_delta in advance_time: {e}")
        return f"Error signaling state update: {e}" # Return error message

    console.print(f"[dim green]--- Tool: World Engine signaled state update: Time=[b]{new_time_str}[/b] ---[/dim green]")
    return None # Return the status message

def save_validation_results(
    validation_results: Dict[str, Any],
    tool_context: ToolContext
) -> Dict[str, str]:
    """
    Saves the provided dictionary of validation results for the current turn
    into the session state under the 'turn_validation_results' key,
    signaling the update via state_delta.

    Args:
        validation_results: A dictionary where keys are simulacra IDs and
                            values are their validation result objects
                            (e.g., {'validation_status': 'approved', ...}).
        tool_context: The context object providing access to session state actions.

    Returns:
        A dictionary confirming the success or failure of the operation.
    """
    if not isinstance(validation_results, dict):
        logger.error(f"Tool 'save_validation_results' received non-dict input: {type(validation_results)}")
        return {"status": "error", "message": "Input must be a dictionary."}

    try:
        logger.info(f"Tool 'save_validation_results': Signaling update for keys: {list(validation_results.keys())}")

        delta_update = {
            TURN_VALIDATION_RESULTS_KEY: validation_results
        }

        # --- Assign to state_delta ---
        tool_context.state_delta = delta_update
        logger.info(f"Signaled state_delta for key '{TURN_VALIDATION_RESULTS_KEY}'.")

        console.print(f"[dim green]--- Tool: Signaled save validation results for {len(validation_results)} simulacra ---[/dim green]")
        return None
    except Exception as e:
        logger.exception(f"Tool 'save_validation_results': Error signaling state update: {e}")
        return None
    
def save_single_validation_result(
    simulacra_id: str,
    validation_result: Dict[str, Any],
    tool_context: ToolContext
) -> Dict[str, str]:
    """
    Saves the validation result for a SINGLE simulacrum into the session state
    by adding it to the dictionary under the 'turn_validation_results' key.
    Uses state_delta to signal the update to the main loop for merging.

    Args:
        simulacra_id: The ID of the simulacrum being validated.
        validation_result: The validation result dictionary for this simulacrum
                           (e.g., {'validation_status': 'approved', ...}).
        tool_context: The context object providing access to session state actions.

    Returns:
        A dictionary confirming the success or failure of the operation.
    """
    if not isinstance(simulacra_id, str) or not simulacra_id:
        logger.error("Tool 'save_single_validation_result' received invalid simulacra_id.")
        return {"status": "error", "message": "Invalid simulacra_id provided."}
    if not isinstance(validation_result, dict):
        logger.error(f"Tool 'save_single_validation_result' received non-dict validation_result for {simulacra_id}: {type(validation_result)}")
        return {"status": "error", "message": "validation_result must be a dictionary."}

    try:
        logger.info(f"Tool 'save_single_validation_result': Signaling update for {simulacra_id}")

        delta_update = {
            TURN_VALIDATION_RESULTS_KEY: {
                simulacra_id: validation_result
            }
        }
        tool_context.state_delta = delta_update
        # ---

        console.print(f"[dim green]--- Tool: Signaled validation result save for {simulacra_id} ---[/dim green]")
        return None
    except Exception as e:
        logger.exception(f"Tool 'save_single_validation_result': Error signaling state update for {simulacra_id}: {e}")
        return None