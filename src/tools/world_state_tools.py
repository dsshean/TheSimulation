# src/tools/world_state_tools.py (Executor Tool)

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from google.adk.tools.tool_context import ToolContext
from rich.console import Console

# --- ADD/VERIFY Constants Definitions HERE ---
WORLD_STATE_KEY = "current_world_state" # Make sure this line exists and is spelled correctly
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent"
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
ACTION_VALIDATION_KEY_FORMAT = "simulacra_{}_validation_result" # Ensure this matches simulation_loop.py
# --- END Constants Definitions ---

console = Console()
logger = logging.getLogger(__name__)

DEFAULT_TIME_INCREMENT_SECONDS = 60 * 5 # 5 minutes

# --- Function: update_and_get_world_state ---
def update_and_get_world_state(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Updates the world state (time, potentially other dynamics) and returns the complete current state.
    """
    console.print("[dim blue]--- Tool (WorldState): Updating and getting full world state... ---[/dim blue]")
    state = tool_context.state

    # --- Time Advancement ---
    # Uses WORLD_STATE_KEY defined above
    world_state = state.get(WORLD_STATE_KEY, {})
    current_time_str = world_state.get("world_time", None)
    new_time_str = current_time_str # Default to original if parsing fails

    if current_time_str:
        try:
            current_time_dt = datetime.fromisoformat(current_time_str)
            time_increment = timedelta(seconds=DEFAULT_TIME_INCREMENT_SECONDS)
            new_time_dt = current_time_dt + time_increment
            new_time_str = new_time_dt.isoformat()
            world_state["world_time"] = new_time_str
            logger.info(f"Advanced world time from {current_time_str} to {new_time_str}")
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing/advancing time tick '{current_time_str}': {e}. Returning original.")
        except Exception as e:
             logger.exception(f"Unexpected error during time advancement for '{current_time_str}': {e}")
    else:
        logger.warning("World time not found in state. Cannot advance time.")

    # --- Other State Updates (Optional) ---
    # world_state["weather"] = "Slightly cloudy" # Example

    # Uses WORLD_STATE_KEY defined above
    state[WORLD_STATE_KEY] = world_state
    console.print("[dim blue]--- Tool (WorldState): Finished updating world state. ---[/dim blue]")
    return world_state
# --- End function ---


# --- OBSOLETE Function: execute_physical_actions_batch ---
# This function is no longer needed as Phase 4b is handled by world_execution_agent without tools.
# def execute_physical_actions_batch(
#     actions_batch: List[Dict[str, Any]],
#     tool_context: ToolContext
# ) -> Dict[str, Any]:
#     """
#     (OBSOLETE) Executes a batch of approved physical actions, updating the relevant state.
#     """
#     console.print("[dim yellow]--- Tool (WorldState - OBSOLETE): execute_physical_actions_batch called. ---[/dim yellow]")
#     state = tool_context.state
#     results = {"executed_actions": 0, "failed_actions": 0, "details": []}
#
#     if not isinstance(actions_batch, list):
#         logger.error(f"execute_physical_actions_batch received invalid input type: {type(actions_batch)}. Expected list.")
#         results["error"] = "Invalid input type for actions_batch."
#         return results
#
#     for action_detail in actions_batch:
#         sim_id = action_detail.get("sim_id")
#         action_type = action_detail.get("action_type")
#         details = action_detail.get("details", {}) # Original intent details
#
#         if not sim_id or not action_type:
#             logger.warning(f"Skipping invalid action in batch: {action_detail}")
#             results["failed_actions"] += 1
#             results["details"].append({"sim_id": sim_id, "status": "failed", "reason": "Missing sim_id or action_type"})
#             continue
#
#         try:
#             if action_type == "move":
#                 destination = details.get("destination")
#                 origin = details.get("origin", "Unknown") # Get origin from intent if available
#                 location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
#
#                 if destination:
#                     # Update the simulacra's location in the state
#                     state[location_key] = destination
#                     logger.info(f"Executed move for {sim_id}: {origin} -> {destination}")
#                     results["executed_actions"] += 1
#                     results["details"].append({"sim_id": sim_id, "action": "move", "status": "success", "destination": destination})
#                     # --- Clear intent after execution ---
#                     intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id)
#                     if intent_key in state: del state[intent_key]
#                     # --- End Clear ---
#                 else:
#                     logger.warning(f"Move action for {sim_id} missing destination in details: {details}")
#                     results["failed_actions"] += 1
#                     results["details"].append({"sim_id": sim_id, "action": "move", "status": "failed", "reason": "Missing destination"})
#
#             elif action_type == "interact":
#                 # --- Placeholder for interaction execution ---
#                 target = details.get("target") # e.g., object ID or NPC ID
#                 interaction_type = details.get("interaction_type") # e.g., 'pickup', 'use', 'open'
#                 logger.info(f"Executing interaction for {sim_id}: Type={interaction_type}, Target={target}")
#                 # Add logic here to modify world state based on interaction
#                 # e.g., change object state, update NPC relationship, add item to inventory
#                 # Example: if target in state.get(WORLD_STATE_KEY, {}).get("objects", {}):
#                 #             state[WORLD_STATE_KEY]["objects"][target]["state"] = "used"
#                 results["executed_actions"] += 1
#                 results["details"].append({"sim_id": sim_id, "action": "interact", "status": "success", "target": target})
#                 # --- Clear intent after execution ---
#                 intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id)
#                 if intent_key in state: del state[intent_key]
#                 # --- End Clear ---
#                 # --- End Placeholder ---
#
#             else:
#                 logger.warning(f"Unsupported physical action type '{action_type}' for {sim_id}")
#                 results["failed_actions"] += 1
#                 results["details"].append({"sim_id": sim_id, "action": action_type, "status": "failed", "reason": "Unsupported action type"})
#
#         except Exception as exec_e:
#             logger.exception(f"Error executing action for {sim_id}: {action_detail}")
#             results["failed_actions"] += 1
#             results["details"].append({"sim_id": sim_id, "action": action_type, "status": "failed", "reason": str(exec_e)})
#
#     console.print(f"[dim blue]--- Tool (WorldState): Finished executing batch. Success: {results['executed_actions']}, Failed: {results['failed_actions']} ---[/dim blue]")
#     return results
# --- End function ---