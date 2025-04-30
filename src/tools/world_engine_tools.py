# src/tools/world_engine_tools.py
import logging
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext # Keep ToolContext for type hint
# <<< Import the State class to check its type >>>
from google.adk.sessions.state import State

logger = logging.getLogger(__name__)

# --- State Keys (Define or import if needed, ensure consistency) ---
TURN_VALIDATION_RESULTS_KEY = "turn_validation_results"

# --- Tool Definitions ---

# <<< NO DECORATOR HERE >>>
def save_single_validation_result(
    tool_context: ToolContext, simulacra_id: str, validation_result: Dict[str, Any]
) -> None: # <<< Return type hint is None
    """
    Logs the intent to save a validation result.
    Does NOT modify state_delta directly anymore.
    The agent calling this tool is responsible for outputting the result
    to be captured by its output_key.
    """
    # Basic argument validation
    if not simulacra_id or not isinstance(validation_result, dict):
        logger.error(
            "save_single_validation_result: Invalid arguments received. "
            f"simulacra_id='{simulacra_id}', type(validation_result)={type(validation_result)}"
        )
        # Optionally raise an error or return specific error info if needed by agent
        return # Return None on error

    # Log the action, but don't modify state here
    validation_key_for_sim = f"{TURN_VALIDATION_RESULTS_KEY}.{simulacra_id}"
    logger.info(
        f"Tool 'save_single_validation_result': Called for "
        f"'{simulacra_id}'. Agent should output result for key '{validation_key_for_sim}'."
    )
    # Return None to indicate successful execution of the logging action
    return None

# <<< NO DECORATOR HERE >>>
def read_state_key(tool_context: ToolContext, key: str) -> Optional[Any]:
    """
    Reads the value of a specific key from the current session state.
    Returns the value found or None if the key doesn't exist or an error occurs.
    Handles basic dot notation for nested keys (e.g., 'current_world_state.world_time').
    Attempts direct access to _value if standard access fails.
    """
    # Check if tool_context has state before using it
    if not hasattr(tool_context, 'state'):
        logger.error("read_state_key: ToolContext is missing 'state'. Context injection failed.")
        return None # Cannot proceed without state

    logger.info(f"Tool 'read_state_key': Attempting to read key '{key}'")
    if not key or not isinstance(key, str):
        logger.error(f"read_state_key: Invalid key provided (type: {type(key)}).")
        return None

    try:
        current_value = tool_context.state # This should be the State object
        key_parts = key.split('.')
        logger.debug(f"Attempting standard access for key parts: {key_parts}")

        for i, part in enumerate(key_parts):
            if isinstance(current_value, dict):
                # If it's already a dict (from previous iteration), use standard get
                logger.debug(f"Accessing part '{part}' in dict...")
                current_value = current_value.get(part)
            elif isinstance(current_value, State):
                # If it's the State object (likely first iteration), try its get method
                logger.debug(f"Accessing part '{part}' using State.get()...")
                current_value = current_value.get(part) # Use the State object's get method

                # --- Fallback to _value if standard get returns None ---
                if current_value is None and hasattr(tool_context.state, '_value'):
                    logger.warning(f"State.get() returned None for '{part}'. Trying direct access to _value...")
                    # Only try _value for the *first* part of the key
                    if i == 0 and isinstance(tool_context.state._value, dict):
                         current_value = tool_context.state._value.get(part)
                         if current_value is not None:
                             logger.info(f"Successfully retrieved '{part}' directly from _value.")
                         else:
                             logger.warning(f"Key part '{part}' not found directly in _value either.")
                    else:
                         # Avoid using _value for nested parts if the first part wasn't the State object
                         logger.warning(f"Not attempting _value access for nested part '{part}'.")

            else:
                # Cannot traverse further
                logger.warning(
                    f"read_state_key: Cannot access part '{part}' in non-dict/non-State value "
                    f"during path '{key}'. Current value type: {type(current_value)}"
                )
                return None # Invalid path

            # Check if the part was found in this iteration
            if current_value is None:
                logger.warning(
                    f"read_state_key: Key part '{part}' not found in path '{key}' (iteration {i})."
                )
                return None # Key part not found

        # If loop completes, current_value holds the final value
        value_type = type(current_value).__name__
        logger.info(f"Tool 'read_state_key': Successfully found value for key '{key}'. Type: {value_type}")
        return current_value

    except Exception as e:
        logger.error(f"read_state_key: Unexpected error accessing key '{key}': {e}", exc_info=True)
        return None
