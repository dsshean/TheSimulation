from google.adk.tools.tool_context import ToolContext
from rich.console import Console
import json
import logging
from typing import List, Dict, Any

console = Console()
logger = logging.getLogger(__name__)

# State keys read by this tool (ensure these match simulation_loop.py)
WORLD_STATE_KEY = "current_world_state"
INTERACTION_RESULT_KEY_FORMAT = "simulacra_{}_interaction_result"
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_GOAL_KEY_FORMAT = "simulacra_{}_goal"
SIMULACRA_PERSONA_KEY_FORMAT = "simulacra_{}_persona"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent"
SIMULACRA_MONOLOGUE_KEY_FORMAT = "last_simulacra_{}_monologue"
ACTION_VALIDATION_KEY_FORMAT = "simulacra_{}_validation_result"
SIMULACRA_NARRATION_KEY_FORMAT = "simulacra_{}_last_narration"

def get_narration_context(
    tool_context: ToolContext,
    target_simulacra_id: str # <<< Expect the target ID
) -> Dict[str, Any]:
    """
    Gathers context specifically for the target_simulacra_id for narration.
    Includes world state, and the target simulacra's status, goal, persona,
    last monologue, intent, validation, and interaction results for the turn.
    Optionally clears the interaction result key for the target simulacra after reading.
    """
    console.print(f"[dim magenta]--- Tool (Narrator): Getting Context for '{target_simulacra_id}' ---[/dim magenta]")
    context = {"status": "error", "error_message": f"Context gathering failed for {target_simulacra_id}"} # Default error
    state_keys_to_clear = {}
    warnings = []

    if not target_simulacra_id:
         logger.error("get_narration_context called without target_simulacra_id!")
         return {"status": "error", "error_message": "Missing target_simulacra_id"}

    try:
        state = tool_context.state

        # 1. Get Final World State (Global Context)
        world_state = state.get(WORLD_STATE_KEY, {})
        if not isinstance(world_state, dict):
            logger.warning(f"'{WORLD_STATE_KEY}' invalid in state.")
            world_state = {"error": f"'{WORLD_STATE_KEY}' missing or invalid."}
        narration_world_context = {
            "world_time": world_state.get("world_time", "Unknown Time"),
            "location_details": world_state.get("location_details", {})
        }

        # 2. Fetch data specific to the target simulacra
        sim_location = state.get(SIMULACRA_LOCATION_KEY_FORMAT.format(target_simulacra_id), "Unknown Location")
        sim_goal = state.get(SIMULACRA_GOAL_KEY_FORMAT.format(target_simulacra_id), "Unknown Goal")
        sim_persona = state.get(SIMULACRA_PERSONA_KEY_FORMAT.format(target_simulacra_id), {})
        sim_status = state.get(SIMULACRA_STATUS_KEY_FORMAT.format(target_simulacra_id), {})
        sim_monologue = state.get(SIMULACRA_MONOLOGUE_KEY_FORMAT.format(target_simulacra_id), "[No monologue recorded]")
        sim_intent = state.get(SIMULACRA_INTENT_KEY_FORMAT.format(target_simulacra_id), None)
        sim_validation = state.get(ACTION_VALIDATION_KEY_FORMAT.format(target_simulacra_id), None)

        # 3. Get Interaction Result for the target Simulacra
        interaction_result_key = INTERACTION_RESULT_KEY_FORMAT.format(target_simulacra_id)
        sim_interaction = state.get(interaction_result_key, None)
        if sim_interaction is not None:
            # Decide if you want to clear this state key after narration
            # state_keys_to_clear[interaction_result_key] = None # Uncomment to clear
            logger.info(f"Interaction result retrieved for {target_simulacra_id} from '{interaction_result_key}'.")

        # 4. Construct the focused context dictionary
        context = {
            "status": "success", # Indicate success
            "target_simulacra_id": target_simulacra_id,
            "world_state": narration_world_context,
            "simulacra_context": {
                "location": sim_location,
                "goal": sim_goal,
                "persona": sim_persona,
                "status": sim_status,
                "last_monologue": sim_monologue,
                "intent_this_turn": sim_intent,
                "validation_result": sim_validation,
                "interaction_result": sim_interaction
            }
        }
        logger.info(f"Context gathered successfully for {target_simulacra_id}")

        # 5. Clear state keys if marked
        if state_keys_to_clear:
            if tool_context.actions:
                tool_context.actions.state_delta.update(state_keys_to_clear)
                logger.info(f"Signaled clearing of state keys: {list(state_keys_to_clear.keys())}")
            else:
                 logger.warning(f"ToolContext.actions is None, cannot signal state clearing for {target_simulacra_id}.")
                 warnings.append("System Warning: Could not signal clearing of state keys.")

        if warnings:
            context["warnings"] = warnings

    except Exception as e:
        logger.exception(f"Error gathering narration context for {target_simulacra_id}: {e}")
        context = {"status": "error", "error_message": f"Exception gathering context for {target_simulacra_id}: {e}"}

    logger.debug(f"Returning narration context for {target_simulacra_id}.")
    return context

def save_narration(
    narratives: Dict[str, str],
    tool_context: ToolContext
) -> Dict[str, str]:
    """
    Saves the provided dictionary of final turn narratives for each simulacra
    into the session state under keys formatted like 'simulacra_<sim_id>_last_narration'.

    Args:
        narratives: A dictionary where keys are simulacra IDs and values
                    are the final narrative strings for that simulacrum.
        tool_context: The context object providing access to session state.

    Returns:
        A dictionary confirming the success or failure of the operation.
    """
    if not isinstance(narratives, dict):
        logger.error(f"Tool 'save_narration' received non-dict input: {type(narratives)}")
        return {"status": "error", "message": "Input must be a dictionary."}

    state_changes = {}
    saved_count = 0
    try:
        for sim_id, narrative_text in narratives.items():
            if isinstance(narrative_text, str):
                narration_key = SIMULACRA_NARRATION_KEY_FORMAT.format(sim_id)
                state_changes[narration_key] = narrative_text
                saved_count += 1
            else:
                logger.warning(f"Tool 'save_narration': Skipping non-string narrative for {sim_id}")

        if state_changes:
            logger.info(f"Tool 'save_narration': Saving narratives for keys: {list(state_changes.keys())}")
            # Directly update the state via tool_context
            tool_context.state.update(state_changes)
            console.print(f"[dim green]--- Tool: Saved narratives for {saved_count} simulacra ---[/dim green]")
            return {"status": "success", "message": f"Saved narratives for {saved_count} simulacra."}
        else:
            logger.warning("Tool 'save_narration': No valid narratives provided to save.")
            return {"status": "warning", "message": "No valid narratives provided to save."}

    except Exception as e:
        logger.exception(f"Tool 'save_narration': Error saving state: {e}")
        return {"status": "error", "message": f"Failed to save narratives: {e}"}