# src/tools/npc_tools.py (Interaction Resolver Tools)

from google.adk.tools.tool_context import ToolContext
from rich.console import Console
import json
import logging
from typing import List, Dict, Any

console = Console()
logger = logging.getLogger(__name__)

# State key formats/names
WORLD_STATE_KEY = "current_world_state"
# Key format where WE writes validation status
ACTION_VALIDATION_KEY_FORMAT = "simulacra_{}_action_validation"
# Key format where this agent writes interaction results
INTERACTION_RESULT_KEY_FORMAT = "simulacra_{}_interaction_result"
# Assumes this key holds ['id1', 'id2'] - needed to find validation keys
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"

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
            action_type = validation_data.get("action_type")
            # Only process 'talk' or 'interact' types here
            if action_type in ["talk", "interact"]:
                logger.info(f"Found validated '{action_type}' action for {sim_id} to process.")
                interactions_to_process.append(validation_data) # Pass the whole validation dict
                processed_keys.append(validation_key)
            else:
                 # Log physical actions validated but not handled here
                 if action_type: logger.info(f"Skipping validated physical action '{action_type}' for {sim_id} (handled by Executor).")
                 processed_keys.append(validation_key) # Mark as processed even if skipped
        elif validation_data:
             # Action might be rejected or invalid format
             logger.warning(f"Action for {sim_id} at key '{validation_key}' was not approved/modified or is invalid: {validation_data}")
             processed_keys.append(validation_key) # Mark as processed to potentially clear later


    # Optionally clear the validation keys now or let the update tool do it
    # for key in processed_keys:
    #    if tool_context.actions: tool_context.actions.state_delta[key] = None

    # Extract relevant parts of world state for the LLM context
    context_for_llm = {
        "world_description": world_state.get("description", ""),
        "current_time": tool_context.state.get("world_time", "Unknown"),
        "npc_states": tool_context.state.get("npc_states", {}),
        "simulacra_states": {}, # Populate with status of relevant sims
        "location_data": world_state.get("location_data", {})
    }
    # Populate simulacra states for context
    for sim_id in active_ids:
         context_for_llm["simulacra_states"][sim_id] = {
             "location": tool_context.state.get(f"simulacra_{sim_id}_location"),
             "status": tool_context.state.get(f"simulacra_{sim_id}_status")
         }


    result = {
        "status": "success",
        "interactions": interactions_to_process,
        "context": context_for_llm
    }
    logger.debug(f"Returning {len(interactions_to_process)} interactions for processing.")
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

        # --- TODO: Apply state changes derived *from* the payload ---
        # Example: If payload contains {"npc_state_change": {"Merchant Bob": {"activity": "closing stall"}}}
        # You would need logic here to merge this into tool_context.state["npc_states"]
        # Example: If payload contains {"object_state_change": {"Computer": {"power": "off"}}}
        # You would need logic here to merge this into tool_context.state["object_states"]
        # This requires defining how the LLM should structure state change requests in result_payload.
        pass # Add state update logic based on payload structure here


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