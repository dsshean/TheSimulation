# src/tools/npc_tools.py (Interaction Resolver Tools)

from google.adk.tools.tool_context import ToolContext
from rich.console import Console
import json
import logging
from typing import List, Dict, Any
from src.loop_utils import format_iso_timestamp

console = Console()
logger = logging.getLogger(__name__)

WORLD_STATE_KEY = "current_world_state"
ACTION_VALIDATION_KEY_FORMAT = "simulacra_{}_action_validation"
INTERACTION_RESULT_KEY_FORMAT = "simulacra_{}_interaction_result"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location" # Define this
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status" 

def get_validated_interactions(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Retrieves validated 'talk' and 'interact' actions from session state,
    along with relevant context (world state, character states).
    It looks for state keys like 'simulacra_X_action_validation'.
    """
    console.print("[dim blue]--- Tool (NPC): Getting Validated Interactions ---[/dim blue]")
    interactions_to_process = []
    world_state = tool_context.state.get(WORLD_STATE_KEY, {})
    active_ids = tool_context.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
    processed_keys = [] # Keep track of keys we read

    logger.info(f"Checking for validated actions for IDs: {active_ids}")

    for sim_id in active_ids:
        validation_key = ACTION_VALIDATION_KEY_FORMAT.format(sim_id)
        validation_data = tool_context.state.get(validation_key)

        if isinstance(validation_data, dict) and validation_data.get("validation_status") in ["approved", "modified"]:
            original_intent = validation_data.get("original_intent", {})
            action_type = original_intent.get("action_type")
            if action_type in ["talk", "interact"]:
                logger.info(f"Found validated '{action_type}' action for {sim_id} to process.")
                interactions_to_process.append(validation_data)
                processed_keys.append(validation_key)
            else:
                 if action_type: logger.info(f"Skipping validated physical action '{action_type}' for {sim_id} (handled by Executor).")
                 processed_keys.append(validation_key) # Mark as processed even if skipped
        elif validation_data:
             logger.warning(f"Action for {sim_id} at key '{validation_key}' was not approved/modified or is invalid: {validation_data}")
             processed_keys.append(validation_key) # Mark as processed to potentially clear later
    current_time_iso = world_state.get("world_time", "Unknown Time")
    context_for_llm = {
        "current_time": format_iso_timestamp(current_time_iso), # Use formatted time
        "npcs": world_state.get("npcs", {}), # Use 'npcs' key
        "simulacra_states": {}, # Populate below
        "location_details": world_state.get("location_details", {}) # Use 'location_details' key
    }

    for sim_id in active_ids:
         location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
         status_key = SIMULACRA_STATUS_KEY_FORMAT.format(sim_id)
         context_for_llm["simulacra_states"][sim_id] = {
             "location": tool_context.state.get(location_key, "Unknown Location"),
             "status": tool_context.state.get(status_key, {"condition": "Unknown", "mood": "Unknown"})
         }

    result = {
        "status": "success",
        "interactions": interactions_to_process,
        "context": context_for_llm
    }
    logger.debug(f"Returning {len(interactions_to_process)} interactions for processing with context: {json.dumps(context_for_llm, indent=2)}") # Log context formatted
    return result

def update_interaction_results(results: List[Dict[str, Any]], tool_context: ToolContext) -> Dict[str, Any]:
    """
    Updates the session state with the results of processed interactions
    (e.g., dialogue responses, object state changes) and clears the
    corresponding action validation keys.

    Args:
        results: A list of dictionaries, where each dict contains at least
                 'simulacra_id' and the 'result_payload' (the outcome determined
                 by the NpcAgent LLM, e.g., {"dialogue": "NPC says hi", "npc_state_change": {...}}).
        tool_context: The tool context.

    Returns:
        A status dictionary.
    """
    console.print(f"[dim blue]--- Tool (NPC): Updating State with {len(results)} Interaction Results ---[/dim blue]")
    state_changes = {}
    errors = []

    if not isinstance(results, list):
        logger.error(f"Invalid results format: {results}. Must be a list of dicts.")
        return {"status": "error", "message": "Invalid results format."}

    for result_item in results:
        if not isinstance(result_item, dict):
            logger.warning(f"Skipping invalid item in results list: {result_item}")
            continue

        sim_id = result_item.get("simulacra_id")
        payload = result_item.get("result_payload")

        if not sim_id:
            logger.warning(f"Missing 'simulacra_id' in result item: {result_item}")
            errors.append("Result item missing simulacra_id.")
            continue
        if payload is None: # Payload could be empty dict/list/None legitimately
             logger.warning(f"Missing 'result_payload' in result item for {sim_id}: {result_item}")
             errors.append(f"Result item missing payload for {sim_id}.")
             continue

        # Store the result payload under the specific key
        result_key = INTERACTION_RESULT_KEY_FORMAT.format(sim_id)
        state_changes[result_key] = payload
        logger.info(f"Prepared state update for '{result_key}'.")

        # Mark the corresponding validation key for clearing
        validation_key = ACTION_VALIDATION_KEY_FORMAT.format(sim_id)
        state_changes[validation_key] = None
        logger.info(f"Prepared clearing of validation key '{validation_key}'.")

        pass


    # Update the state directly via context
    try:
        tool_context.state.update(state_changes)
        logger.info(f"Applied interaction result state changes: {list(state_changes.keys())}")
        status = "success"
    except Exception as e:
        logger.exception(f"Error applying state changes for interaction results: {e}")
        status = "error"
        errors.append(f"Failed to apply state changes: {e}")


    final_result = {"status": status}
    if errors:
        final_result["errors"] = errors
    return final_result