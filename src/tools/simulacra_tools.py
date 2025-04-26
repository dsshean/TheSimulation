# src/tools/simulacra_tools.py (Correct ID reading in check_self_status)
import json
import logging
import os  # Add os import
from typing import Any, Dict

from google.adk.tools.tool_context import ToolContext
from rich.console import Console

from src.generation.llm_service import LLMService

console = Console()
        
logger = logging.getLogger(__name__) # Ensure logger is defined

# --- MODIFIED: State key formats expect base ID (e.g., 'eleanor_vance') ---
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
SIMULACRA_GOAL_KEY_FORMAT = "simulacra_{}_goal"
SIMULACRA_PERSONA_KEY_FORMAT = "simulacra_{}_persona"
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent" # Used by intent tools
SIMULACRA_MONOLOGUE_KEY_FORMAT = "last_simulacra_{}_monologue" # Keep 'last_' prefix
# --- END MODIFIED ---
WORLD_STATE_KEY = "current_world_state"

# --- WORKAROUND: Define temp file path ---
TEMP_LOCATION_FILE = "temp_sim_locations.json"
# ---

# --- Tools ---

def generate_internal_monologue(
    simulacra_id: str, # Will now be 'eleanor_vance', 'eleanor_vance_2', etc.
    tool_context: ToolContext,
    **kwargs
) -> str:
    """
    Generates a brief internal monologue based on the Simulacra's current context
    (fetched from state) AND background persona (fetched from state), using an LLMService.
    Saves the monologue to the specific 'last_simulacra_{id}_monologue' state key.
    Ignores any extra keyword arguments passed.
    """
    # --- ADDED: Log if unexpected args are received ---
    if kwargs:
        logger.warning(f"generate_internal_monologue for {simulacra_id} received unexpected arguments: {kwargs.keys()}. Ignoring them.")
    # --- END ADDED ---

    console.print(f"[dim blue]--- Tool ({simulacra_id}): Generating internal monologue ---[/dim blue]")
    monologue = "Error: Could not generate monologue." # Default error message
    # Use the updated format string directly with the received simulacra_id
    state_key = SIMULACRA_MONOLOGUE_KEY_FORMAT.format(simulacra_id) # -> 'last_simulacra_eleanor_vance_monologue'

    if not simulacra_id:
        logger.error("Simulacra ID not provided to generate_internal_monologue!")
        tool_context.state[state_key] = "Error: Missing ID for monologue generation."
        return "Error: Missing ID."

    try:
        # --- Use received simulacra_id directly for formatting keys ---
        goal_key = SIMULACRA_GOAL_KEY_FORMAT.format(simulacra_id)       # -> 'simulacra_eleanor_vance_goal'
        location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(simulacra_id) # -> 'simulacra_eleanor_vance_location'
        persona_key = SIMULACRA_PERSONA_KEY_FORMAT.format(simulacra_id)   # -> 'simulacra_eleanor_vance_persona'
        # ---

        current_goal = tool_context.state.get(goal_key, "Goal unknown")
        current_location = tool_context.state.get(location_key, "Location unknown")
        persona_data = tool_context.state.get(persona_key, {})
        world_state = tool_context.state.get(WORLD_STATE_KEY, {})
        current_time = world_state.get("world_time", "Time unknown")
        location_details = world_state.get("location_details", {})
        setting_description = location_details.get(current_location, "No description available for this location.")
        # --- End Fetch context ---

        logger.info(f"({simulacra_id}) Generating internal monologue for goal: {current_goal} at {current_location}")

        # --- MODIFIED: Format Persona Summary from simplified dict ---
        persona_summary = "No background available."
        if persona_data and isinstance(persona_data, dict):
            # Directly access fields from the simplified persona_data dict
            name = persona_data.get("Name", "Unknown")
            traits = persona_data.get("Personality_Traits", [])
            occupation = persona_data.get("Occupation", "Unknown")
            traits_str = ', '.join(traits) if isinstance(traits, list) else "Unknown traits"
            persona_summary = f"I am {name}, a {occupation}. My traits are: {traits_str}."
        elif persona_data:
             persona_summary = f"Background: {str(persona_data)}" # Fallback if not dict
        # --- END MODIFIED ---

        # --- LLM Call Prompt ---
        prompt = f"""
        Current Time: {current_time}
        Current Location: {current_location} ({setting_description})
        My Goal: {current_goal}
        My Background: {persona_summary}

        Based on the above, generate a brief, first-person internal monologue (1-2 sentences) reflecting my current thoughts or feelings. Focus on the immediate situation or goal.
        """
        llm_service = LLMService()
        # --- MODIFIED: Revert to original method name ---
        generated_text = llm_service.generate_content_text(prompt=prompt)
        # --- END MODIFIED ---
        # --- End LLM Call Logic ---

        # --- Process LLM Result (uses correctly formatted state_key) ---
        if generated_text and generated_text.strip():
            monologue = generated_text.strip()
            tool_context.state[state_key] = monologue
            console.print(f"[dim blue]--- Tool ({simulacra_id}): Monologue generated: '[italic]{monologue}[/italic]' and saved to '{state_key}' ---[/dim blue]")
        else:
            logger.warning(f"Monologue generation for {simulacra_id} returned empty result.")
            monologue = "[No specific thought generated]"
            tool_context.state[state_key] = monologue

    except Exception as e:
        logger.exception(f"Error during internal monologue generation for {simulacra_id}: {e}")
        console.print(f"[bold red]--- Tool Error (generate_internal_monologue for {simulacra_id}): {e} ---[/bold red]")
        # Save error state?
        tool_context.state[state_key] = f"Error generating monologue: {e}"
        return f"Error: {e}"

    return monologue

# --- Ensure other tools also use the received simulacra_id with the updated format strings ---
def check_self_status(simulacra_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Retrieves the current status, location, goal, and full persona for the specified Simulacra ID.
    Requires the simulacra_id to be provided as an argument.
    WORKAROUND: Reads location primarily from temp_sim_locations.json.
    """
    console.print(f"[dim blue]--- Tool ({simulacra_id}): Checking self status ---[/dim blue]")

    if not simulacra_id:
        logger.error("Simulacra ID not provided to check_self_status!")
        return {"error": "Missing required 'simulacra_id' argument."}

    status_key = SIMULACRA_STATUS_KEY_FORMAT.format(simulacra_id)
    location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(simulacra_id)
    goal_key = SIMULACRA_GOAL_KEY_FORMAT.format(simulacra_id)
    persona_key = SIMULACRA_PERSONA_KEY_FORMAT.format(simulacra_id)

    try:
        # Get other info from context as usual
        status_info = tool_context.state.get(status_key, {"error": f"Status not found for {simulacra_id}"})
        goal_info = tool_context.state.get(goal_key, f"Goal unknown for {simulacra_id}")
        persona_info = tool_context.state.get(persona_key, {"error": f"Persona details not found for {simulacra_id}"})

        # --- WORKAROUND: Determine location ---
        location_info = f"Location unknown for {simulacra_id}" # Default
        location_source = "Default"
        try:
            # Try reading from the temp file first
            if os.path.exists(TEMP_LOCATION_FILE):
                with open(TEMP_LOCATION_FILE, "r") as f:
                    temp_locations = json.load(f)
                    if simulacra_id in temp_locations:
                        location_info = temp_locations[simulacra_id]
                        location_source = "File"
                        logger.debug(f"WORKAROUND: Used location '{location_info}' from file for {simulacra_id}.")
                    else:
                        logger.warning(f"WORKAROUND: {simulacra_id} not found in {TEMP_LOCATION_FILE}.")
                        location_source = "File (Not Found)"
            else:
                 logger.warning(f"WORKAROUND: Temp location file {TEMP_LOCATION_FILE} not found.")
                 location_source = "File (Missing)"

            # Fallback to tool_context.state only if file read failed/didn't provide location
            if location_source not in ["File"]:
                context_location = tool_context.state.get(location_key)
                if context_location:
                    location_info = context_location
                    location_source = "ToolContext"
                    logger.debug(f"WORKAROUND: Fell back to location '{location_info}' from tool_context for {simulacra_id}.")
                else:
                    # Keep the default "Location unknown..."
                    location_source = "ToolContext (Missing)"
                    logger.debug(f"WORKAROUND: Location for {simulacra_id} also missing from tool_context.")

        except (FileNotFoundError, json.JSONDecodeError, Exception) as file_e:
            logger.error(f"WORKAROUND: Error reading temp location file {TEMP_LOCATION_FILE}: {file_e}")
            location_source = f"File Error ({type(file_e).__name__})"
            # Attempt fallback to context state on error
            context_location = tool_context.state.get(location_key)
            if context_location:
                 location_info = context_location
                 location_source += " -> Context Fallback"
                 logger.debug(f"WORKAROUND: Fell back to location '{location_info}' from tool_context after file error for {simulacra_id}.")
            else:
                 location_source += " -> Context Missing"
        # --- END WORKAROUND ---

        full_status = {
            "id": simulacra_id,
            "current_location": location_info, # Use the determined location
            "current_goal": goal_info,
            "status_summary": status_info,
            "full_persona": persona_info
        }

        logger.info(f"Status check for {simulacra_id}: Location='{location_info}' (Source: {location_source}), Goal='{goal_info}'")
        console.print(f"[dim blue]--- Tool ({simulacra_id}): Status retrieved (Location Source: {location_source}) ---[/dim blue]")
        return full_status

    except Exception as e:
        logger.exception(f"Error during check_self_status for {simulacra_id}: {e}")
        console.print(f"[bold red]--- Tool Error (check_self_status for {simulacra_id}): {e} ---[/bold red]")
        return {"error": f"Failed to check status: {e}"}

# --- ACTION TOOLS USE ID ARGUMENT ---
def attempt_move_to(
    simulacra_id: str, # Requires ID as argument
    destination: str,
    tool_context: ToolContext
) -> dict:
    """
    Writes move intent to state key 'simulacra_{simulacra_id}_intent'.
    Requires simulacra_id argument. Returns status.
    """
    # *** Use the provided simulacra_id argument ***
    if not simulacra_id:
        logger.error("Simulacra ID argument missing in attempt_move_to!")
        return {"status": "error", "message": "Simulacra ID argument missing."}

    # Get location using the provided ID
    location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(simulacra_id)
    intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(simulacra_id)
    current_location = tool_context.state.get(location_key, "Unknown Location")

    console.print(f"[dim blue]--- Tool ({simulacra_id}): Intends move to [i]{destination}[/i]. Writing to state key '{intent_key}' ---[/dim blue]")

    action_details = {
        "action_type": "move",
        "destination": destination,
        "origin": current_location,
        "simulacra_id": simulacra_id # Include the ID in the details
    }
    try:
        tool_context.state[intent_key] = action_details
        logger.info(f"Intent for {simulacra_id} written to state key '{intent_key}'.")
        return {"status": "intent_set", "key": intent_key, "intent": action_details}
    except Exception as e:
         logger.error(f"Failed to write intent to state for key '{intent_key}': {e}")
         return {"status": "error", "message": f"Failed to write intent state: {e}"}


async def attempt_talk_to(
    simulacra_id: str, # Requires ID as argument
    npc_name: str,
    message: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Writes talk intent to state key 'simulacra_{simulacra_id}_intent'.
    Requires simulacra_id argument. Returns status.
    """
    # *** Use the provided simulacra_id argument ***
    if not simulacra_id:
        logger.error("Simulacra ID argument missing in attempt_talk_to!")
        return {"status": "error", "message": "Simulacra ID argument missing."}

    # --- Use received simulacra_id directly ---
    intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(simulacra_id)
    current_location = tool_context.state.get(SIMULACRA_LOCATION_KEY_FORMAT.format(simulacra_id), "Unknown Location")

    console.print(f"[dim blue]--- Tool ({simulacra_id}): Intends talk to [i]{npc_name}[/i]. Writing to state key '{intent_key}'---[/dim blue]")
    console.print(f"[dim blue]    Message: '[italic]{message}[/italic]'[/dim blue]")

    action_details = {
        "action_type": "talk",
        "target_npc": npc_name,
        "message": message,
        "location": current_location,
        "simulacra_id": simulacra_id
    }
    try:
        tool_context.state[intent_key] = action_details
        logger.info(f"Intent for {simulacra_id} written to state key '{intent_key}'.")
        return {"status": "intent_set", "key": intent_key, "intent": action_details}
    except Exception as e:
         logger.error(f"Failed to write intent to state for key '{intent_key}': {e}")
         return {"status": "error", "message": f"Failed to write intent state: {e}"}


async def attempt_interact_with(
    simulacra_id: str, # Requires ID as argument
    object_name: str,
    interaction_type: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Writes interact intent to state key 'simulacra_{simulacra_id}_intent'.
    Requires simulacra_id argument. Returns status.
    """
    # *** Use the provided simulacra_id argument ***
    if not simulacra_id:
        logger.error("Simulacra ID argument missing in attempt_interact_with!")
        return {"status": "error", "message": "Simulacra ID argument missing."}

    # --- Use received simulacra_id directly ---
    intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(simulacra_id)
    current_location = tool_context.state.get(SIMULACRA_LOCATION_KEY_FORMAT.format(simulacra_id), "Unknown Location")

    console.print(f"[dim blue]--- Tool ({simulacra_id}): Intends to '{interaction_type}' [i]{object_name}[/i] at {current_location}. Writing to state key '{intent_key}'---[/dim blue]")

    action_details = {
        "action_type": "interact",
        "target_object": object_name,
        "interaction": interaction_type,
        "location": current_location,
        "simulacra_id": simulacra_id
    }
    try:
        tool_context.state[intent_key] = action_details
        logger.info(f"Intent for {simulacra_id} written to state key '{intent_key}'.")
        return {"status": "intent_set", "key": intent_key, "intent": action_details}
    except Exception as e:
         logger.error(f"Failed to write intent to state for key '{intent_key}': {e}")
         return {"status": "error", "message": f"Failed to write intent state: {e}"}

