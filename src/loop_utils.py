import glob
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple # Added Tuple

import pytz
from google.generativeai import types
from rich.console import Console
from timezonefinder import TimezoneFinder
# from pydantic import BaseModel, ValidationError # No longer needed here

# --- Constants (Moved/Duplicated from simulation_async for loading logic) ---
# Consider a central config module in the future
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project Root
STATE_DIR = os.path.join(BASE_DIR, "data", "states")
LIFE_SUMMARY_DIR = os.path.join(BASE_DIR, "data", "life_summaries")
WORLD_CONFIG_DIR = os.path.join(BASE_DIR, "data") # World configs are in the main data dir

# State Keys (Moved/Duplicated)
WORLD_STATE_KEY = "current_world_state"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
LOCATION_DETAILS_KEY = "location_details"
SIMULACRA_PROFILES_KEY = "simulacra_profiles"
CURRENT_LOCATION_KEY = "current_location"
HOME_LOCATION_KEY = "home_location"
WORLD_TEMPLATE_DETAILS_KEY = "world_template_details"
LOCATION_KEY = "location" # Used in world_config
DEFAULT_HOME_LOCATION_NAME = "At home"
DEFAULT_HOME_DESCRIPTION = "You are at home. It's a cozy place with familiar surroundings."

# Ensure directories exist (safe check)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)
os.makedirs(WORLD_CONFIG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
console = Console() # Add console for loading messages

def print_event_details(
    event: Any, # Use Any temporarily to bypass type errors
    phase_name: str,
    console: Console,
    logger: logging.Logger,
    max_content_length: int = 5000,
    max_response_length: int = 5000
):
    logger.debug(f"[{phase_name}] Received event of type: {type(event)}")

    agent_id = getattr(event, 'author', 'UnknownAuthor')
    is_final = getattr(event, 'is_final_response', lambda: False)() # Call if it's a method
    actions = getattr(event, 'actions', None)
    content = getattr(event, 'content', None)

    logger.debug(f"{phase_name} Event ({agent_id}): Final={is_final}, Actions={actions}, Content={str(content)[:max_content_length]}...")

    if content and hasattr(content, 'parts') and content.parts:
        # Assuming content.parts structure is somewhat stable (like google.generativeai.types.Content)
        part = content.parts[0]
        if hasattr(part, 'function_call') and part.function_call:
            tool_call = part.function_call
            # Ensure args is dictionary-like before converting
            args_dict = getattr(tool_call, 'args', {})
            try:
                args_display = dict(args_dict)
            except (TypeError, ValueError):
                args_display = str(args_dict) # Fallback to string representation
            console.print(f"[dim blue]  {phase_name} ({agent_id}) -> Tool Call: {getattr(tool_call, 'name', 'UnknownTool')} with args: {args_display}[/dim blue]")
        elif hasattr(part, 'function_response') and part.function_response:
            tool_response = part.function_response
            response_content = getattr(tool_response, 'response', {})
            try:
                response_str = json.dumps(dict(response_content))
            except (TypeError, ValueError):
                 response_str = str(response_content) # Fallback to string
            response_display = response_str[:max_response_length] + ('...' if len(response_str) > max_response_length else '')
            console.print(f"[dim green]  {phase_name} ({agent_id}) <- Tool Response: {getattr(tool_response, 'name', 'UnknownTool')} -> {response_display}[/dim green]")
        elif is_final and hasattr(part, 'text'):
            text_content = getattr(part, 'text', '')
            console.print(f"[dim cyan]  {phase_name} ({agent_id}) Final Output: {text_content if text_content else '[No text output]'}[/dim cyan]")
        elif hasattr(part, 'text'):
            text_content = getattr(part, 'text', '')
            console.print(f"[dim cyan]  {phase_name} ({agent_id}) Text Output: {text_content if text_content else '[No text output]'}[/dim cyan]")
    elif is_final:
         logger.debug(f"{phase_name} ({agent_id}) Final event with no standard parts. Event: {event}")

def parse_json_output_last(
    raw_text: Optional[str],
    phase_name: str,
    agent_name: str,
    console: Console,
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON from an agent's text output, robustly handling
    markdown code fences and preceding/trailing text.
    Prioritizes the LAST JSON block found.
    Returns the parsed dictionary or None on failure.
    """
    if not raw_text:
        console.print(f"[yellow]{phase_name}: {agent_name} returned empty final response.[/yellow]")
        return None

    text_to_parse = raw_text.strip()
    logger.debug(f"[{phase_name}/{agent_name}] Raw output: {repr(text_to_parse)}")

    json_string_to_parse = None

    # --- More Robust JSON Extraction - Prioritize LAST block ---
    # 1. Look for ```json ... ``` blocks
    json_matches = re.findall(r"```json\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
    if json_matches:
        # Take the last match found
        json_string_to_parse = json_matches[-1].strip()
        logger.debug(f"[{phase_name}/{agent_name}] Extracted LAST JSON block using ```json regex.")
    else:
        # 2. If no ```json block, look for ``` ... ``` blocks
        code_matches = re.findall(r"```\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
        if code_matches:
            # Take the last match found
            json_string_to_parse = code_matches[-1].strip()
            logger.debug(f"[{phase_name}/{agent_name}] Extracted LAST JSON block using ``` regex.")
        else:
            # 3. If no markdown block, find the LAST '{' and try parsing from there
            # This is less reliable but better than taking the first one
            last_brace = text_to_parse.rfind('{')
            if last_brace != -1:
                # Attempt to find the matching closing brace (simple heuristic)
                # Find the last '}' after the last '{'
                last_close_brace = text_to_parse.rfind('}', last_brace)
                if last_close_brace != -1:
                    potential_json = text_to_parse[last_brace : last_close_brace + 1]
                    json_string_to_parse = potential_json
                    logger.debug(f"[{phase_name}/{agent_name}] No markdown found, trying from last '{{' to last '}}'.")
                else:
                    # Fallback: try parsing from last '{' to the end
                    potential_json = text_to_parse[last_brace:]
                    json_string_to_parse = potential_json
                    logger.debug(f"[{phase_name}/{agent_name}] No markdown found, trying from last '{{' to end (fallback).")
            else:
                # 4. If no '{' found at all
                logger.error(f"{phase_name}: Could not find JSON object start '{{' in output from {agent_name}.")
                console.print(f"[bold red]{phase_name}: Error parsing {agent_name} JSON output (no '{{' found).[/bold red]")
                console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
                return None
    # --- End Robust Extraction ---

    if json_string_to_parse is None:
         # This case should be rare if step 3 or 4 found something, but added for safety
         logger.error(f"{phase_name}: Could not extract any potential JSON string from {agent_name}.")
         console.print(f"[bold red]{phase_name}: Error extracting potential JSON from {agent_name} output.[/bold red]")
         console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
         return None

    logger.debug(f"[{phase_name}/{agent_name}] Attempting to parse: {repr(json_string_to_parse)}")
    try:
        # Attempt to repair common issues like trailing commas (requires external library or more complex logic)
        # For now, just try standard parsing
        parsed_data = json.loads(json_string_to_parse)
        if not isinstance(parsed_data, dict):
            raise ValueError(f"Parsed JSON is not a dictionary (type: {type(parsed_data)}).")
        logger.info(f"[{phase_name}/{agent_name}] Successfully parsed JSON.")
        return parsed_data
    except (json.JSONDecodeError, ValueError) as json_error:
        logger.exception(f"{phase_name}: Failed to parse {agent_name} output as JSON: {json_error}")
        console.print(f"[bold red]{phase_name}: Error parsing {agent_name} JSON output.[/bold red]")
        console.print(f"[dim]Attempted to parse:[/dim]\n[yellow]{json_string_to_parse}[/yellow]")
        console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
        return None

def parse_json_output(
    raw_text: Optional[str],
    phase_name: str,
    agent_name: str,
    console: Console,
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON from an agent's text output, robustly handling
    markdown code fences and preceding/trailing text.
    Returns the parsed dictionary or None on failure.
    """
    if not raw_text:
        console.print(f"[yellow]{phase_name}: {agent_name} returned empty final response.[/yellow]")
        return None

    text_to_parse = raw_text.strip()
    logger.debug(f"[{phase_name}/{agent_name}] Raw output: {repr(text_to_parse)}")

    # --- More Robust JSON Extraction ---
    # 1. Look for ```json ... ``` block first
    match = re.search(r"```json\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
    if match:
        json_string_to_parse = match.group(1).strip()
        logger.debug(f"[{phase_name}/{agent_name}] Extracted JSON block using ```json regex.")
    else:
        # 2. If no ```json block, look for ``` ... ``` block
        match = re.search(r"```\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
        if match:
            json_string_to_parse = match.group(1).strip()
            logger.debug(f"[{phase_name}/{agent_name}] Extracted JSON block using ``` regex.")
        else:
            # 3. If no markdown block, find the first '{' and try parsing from there
            start_brace = text_to_parse.find('{')
            if start_brace != -1:
                # Attempt to find the matching closing brace (simple heuristic)
                # This might fail for complex nested structures if there's trailing text
                potential_json = text_to_parse[start_brace:]
                # Try to find a balanced closing brace - this is tricky without full parsing
                # A simpler approach: just try parsing the substring from the first brace
                json_string_to_parse = potential_json
                logger.debug(f"[{phase_name}/{agent_name}] No markdown found, trying from first '{{'.")
            else:
                # 4. If no '{' found at all
                logger.error(f"{phase_name}: Could not find JSON object start '{{' in output from {agent_name}.")
                console.print(f"[bold red]{phase_name}: Error parsing {agent_name} JSON output (no '{{' found).[/bold red]")
                console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
                return None
    # --- End Robust Extraction ---

    logger.debug(f"[{phase_name}/{agent_name}] Attempting to parse: {repr(json_string_to_parse)}")
    try:
        parsed_data = json.loads(json_string_to_parse)
        if not isinstance(parsed_data, dict):
            raise ValueError(f"Parsed JSON is not a dictionary (type: {type(parsed_data)}).")
        # console.print(f"[dim]  {phase_name} ({agent_name}) Final Output (Parsed JSON): {len(parsed_data)} results received.[/dim]") # Less verbose success
        logger.info(f"[{phase_name}/{agent_name}] Successfully parsed JSON.")
        return parsed_data
    except (json.JSONDecodeError, ValueError) as json_error:
        logger.exception(f"{phase_name}: Failed to parse {agent_name} output as JSON: {json_error}")
        console.print(f"[bold red]{phase_name}: Error parsing {agent_name} JSON output.[/bold red]")
        console.print(f"[dim]Attempted to parse:[/dim]\n[yellow]{json_string_to_parse}[/yellow]")
        console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
        return None

# --- Simulation State Loading & Initialization Functions ---

def create_blank_simulation_state(new_uuid: str) -> Dict[str, Any]:
    """Creates a dictionary representing a minimal blank simulation state."""
    logger.info(f"Generating blank state structure for new UUID: {new_uuid}")
    # Define a default starting location if none exists yet
    default_location_id = "default_start_location"
    default_location = {
        "name": "A Starting Point",
        "description": "An initial location within the simulation.",
        "objects_present": [],
        "connected_locations": []
    }
    return {
      "world_instance_uuid": new_uuid,
      "location_details": {default_location_id: default_location.copy()}, # Use copy
      "objects": {},
      "active_simulacra_ids": [],
      "world_time": 0.0,
      "narrative_log": [],
      "simulacra": {},
      "npcs": {},
      "current_world_state": {
        "location_details": {default_location_id: default_location.copy()} # Use copy
      },
      "world_template_details": {
        "description": "Initial Blank State",
        "rules": {}
      },
      "simulacra_profiles": {}
    }

def load_json_file(path: str, default: Optional[Any] = None) -> Optional[Any]:
    """Loads JSON from a file, returning default if file not found or invalid."""
    if not os.path.exists(path):
        logger.debug(f"File not found: {path}. Returning default.")
        return default
    try:
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {path}: {e}. Returning default.")
        return default
    except Exception as e:
        logger.error(f"Error loading file {path}: {e}. Returning default.")
        return default

def save_json_file(path: str, data: Any):
    """Saves data to a JSON file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved data to {path}")
    except Exception as e:
        logger.error(f"Error saving file {path}: {e}")
        raise

def find_latest_file(pattern: str) -> Optional[str]:
    """Finds the most recently modified file matching a glob pattern."""
    try:
        list_of_files = glob.glob(pattern)
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        logger.error(f"Error finding latest file for pattern '{pattern}': {e}", exc_info=True)
        return None

def find_latest_simulation_state_file(state_dir: str = STATE_DIR) -> Optional[str]:
    """Finds the most recently modified simulation state file."""
    try:
        os.makedirs(state_dir, exist_ok=True)
        state_file_pattern = os.path.join(state_dir, "simulation_state_*.json")
        list_of_files = glob.glob(state_file_pattern)
        if not list_of_files:
            logger.info(f"No existing simulation state files found in {state_dir}.")
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        logger.info(f"Found latest simulation state file: {latest_file}")
        return latest_file
    except Exception as e:
        logger.error(f"Error finding latest state file in {state_dir}: {e}")
        return None

def ensure_state_structure(state_dict: Dict[str, Any]) -> bool:
    """Checks and adds missing essential keys/structures to a state dictionary."""
    modified = False
    if not isinstance(state_dict, dict): return False

    # Ensure top-level keys exist
    required_top_level_keys = {
        "world_instance_uuid": str(uuid.uuid4()), # Default to new UUID if missing
        "location_details": {},
        "objects": {},
        "active_simulacra_ids": [],
        "world_time": 0.0,
        "narrative_log": [],
        "simulacra": {},
        "npcs": {},
        "current_world_state": {},
        "world_template_details": {"description": "Default", "rules": {}},
        "simulacra_profiles": {}
    }
    for key, default_value in required_top_level_keys.items():
        if key not in state_dict:
            state_dict[key] = default_value
            logger.warning(f"Added missing top-level key '{key}'.")
            modified = True
        # Ensure correct types for collections
        elif key in ["location_details", "objects", "simulacra", "npcs", "current_world_state", "world_template_details", "simulacra_profiles"] and not isinstance(state_dict[key], dict):
            state_dict[key] = default_value
            logger.warning(f"Corrected type for top-level key '{key}' to dict.")
            modified = True
        elif key in ["active_simulacra_ids", "narrative_log"] and not isinstance(state_dict[key], list):
            state_dict[key] = default_value
            logger.warning(f"Corrected type for top-level key '{key}' to list.")
            modified = True
        elif key == "world_time" and not isinstance(state_dict[key], (int, float)):
            state_dict[key] = default_value
            logger.warning(f"Corrected type for top-level key '{key}' to float.")
            modified = True

    # Ensure nested structure for current_world_state
    world_state_dict = state_dict.get(WORLD_STATE_KEY, {})
    if not isinstance(world_state_dict, dict):
        state_dict[WORLD_STATE_KEY] = {}
        world_state_dict = state_dict[WORLD_STATE_KEY]
        modified = True

    if LOCATION_DETAILS_KEY not in world_state_dict:
        world_state_dict[LOCATION_DETAILS_KEY] = {}
        logger.warning(f"Added missing '{LOCATION_DETAILS_KEY}' key to '{WORLD_STATE_KEY}'.")
        modified = True
    elif not isinstance(world_state_dict[LOCATION_DETAILS_KEY], dict):
         world_state_dict[LOCATION_DETAILS_KEY] = {}
         logger.warning(f"Corrected type for '{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}' to dict.")
         modified = True

    # Ensure nested structure for world_template_details
    template_details = state_dict.get(WORLD_TEMPLATE_DETAILS_KEY, {})
    if not isinstance(template_details, dict):
        state_dict[WORLD_TEMPLATE_DETAILS_KEY] = {"description": "Default", "rules": {}}
        template_details = state_dict[WORLD_TEMPLATE_DETAILS_KEY]
        modified = True
    if "description" not in template_details:
        template_details["description"] = "Default"
        logger.warning(f"Added missing 'description' key to '{WORLD_TEMPLATE_DETAILS_KEY}'.")
        modified = True
    if "rules" not in template_details:
        template_details["rules"] = {}
        logger.warning(f"Added missing 'rules' key to '{WORLD_TEMPLATE_DETAILS_KEY}'.")
        modified = True
    elif not isinstance(template_details["rules"], dict):
        template_details["rules"] = {}
        logger.warning(f"Corrected type for '{WORLD_TEMPLATE_DETAILS_KEY}.rules' to dict.")
        modified = True

    # --- ADDED: Ensure 'mood' key exists with a default ---
    if "mood" not in template_details:
        default_mood = "neutral_descriptive" # Or choose your preferred default
        template_details["mood"] = default_mood
        logger.warning(f"Added missing 'mood' key to '{WORLD_TEMPLATE_DETAILS_KEY}' with default value '{default_mood}'.")
        modified = True
    
    # Ensure last_interjection_sim_time for each simulacra
    for sim_id, sim_data in state_dict.get("simulacra", {}).items():
        if isinstance(sim_data, dict) and "last_interjection_sim_time" not in sim_data:
            sim_data["last_interjection_sim_time"] = 0.0
            modified = True
        if isinstance(sim_data, dict) and "next_simple_timer_interjection_sim_time" not in sim_data:
            sim_data["next_simple_timer_interjection_sim_time"] = 0.0 # Fire early on first load
            modified = True
    # --- END ADDED ---

    return modified

def load_or_initialize_simulation(instance_uuid_arg: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads or initializes simulation state based on world config and life summaries.

    1. Determines the target UUID (from arg, latest world config, or new).
    2. Finds and loads the corresponding world_config JSON.
    3. Finds corresponding life_summary JSON files.
    4. Checks for an existing simulation_state JSON in STATE_DIR.
    5. If state exists, loads it.
    6. If state doesn't exist, creates a new one populated from world_config and life_summaries.
    7. Ensures the final state structure is valid.
    8. Saves the state if it was newly created or modified during structure check.

    Args:
        instance_uuid_arg: The specific UUID provided via command line, or None.

    Returns:
        A tuple containing:
        - The loaded or initialized state dictionary (or None on critical failure).
        - The path to the state file used or created (or None on critical failure).
    """
    logger.info("Starting simulation state loading/initialization.")
    world_config_path: Optional[str] = None
    target_uuid: Optional[str] = instance_uuid_arg
    world_config: Optional[Dict[str, Any]] = None
    state_file_path: Optional[str] = None
    loaded_state_data: Optional[Dict[str, Any]] = None
    created_new_state = False
    state_modified = False

    # --- 1. Determine Target UUID ---
    if not target_uuid: # If no specific UUID was provided via argument
        logger.info("No instance UUID specified, searching for the latest world config...")
        world_config_pattern = os.path.join(WORLD_CONFIG_DIR, "world_config_*.json")
        latest_world_config_path = find_latest_file(world_config_pattern)
        if latest_world_config_path:
            try:
                filename = os.path.basename(latest_world_config_path)
                # Assumes format world_config_UUID.json
                # We will load the file first to get the definitive UUID from its content
                world_config_path = latest_world_config_path
                logger.info(f"Found latest world config file: {world_config_path}. Will load to determine UUID.")
                console.print(f"Found latest world config file: [cyan]{filename}[/cyan]")
            except IndexError: # Should not happen with find_latest_file, but safety check
                logger.error(f"Could not process latest world config filename: {latest_world_config_path}")
                console.print(f"[bold red]Error:[/bold red] Could not process filename {latest_world_config_path}.")
                return None, None
        else:
            logger.warning("No world config files found. Will create a new simulation instance.")
            target_uuid = str(uuid.uuid4())
            console.print(f"[yellow]Warning:[/yellow] No world config found. Creating new simulation with UUID: {target_uuid}")
            # No world_config_path, will use defaults later
    else:
        # UUID was provided, construct the expected world config path
        world_config_path = os.path.join(WORLD_CONFIG_DIR, f"world_config_{target_uuid}.json")
        if not os.path.exists(world_config_path):
            logger.error(f"Specified world config file not found: {world_config_path}")
            console.print(f"[bold red]Error:[/bold red] World config for UUID {target_uuid} not found at {world_config_path}.")
            return None, None

    # --- 2. Load World Config (if path exists) ---
    if world_config_path and os.path.exists(world_config_path):
        world_config = load_json_file(world_config_path)
        if world_config is None:
            logger.error(f"Failed to load or parse world config file: {world_config_path}")
            console.print(f"[bold red]Error:[/bold red] Failed to load world config file {world_config_path}.")
            return None, None

        # Verify UUID match if arg was provided
        config_uuid = world_config.get("world_instance_uuid")
        if not config_uuid:
             logger.error(f"World config file {world_config_path} is missing 'world_instance_uuid'.")
             console.print(f"[bold red]Error:[/bold red] World config file missing 'world_instance_uuid'.")
             # If we found the latest file but it has no UUID, we can't proceed reliably.
             return None, None
        if instance_uuid_arg and config_uuid != instance_uuid_arg:
            logger.error(f"UUID mismatch: Arg specified {instance_uuid_arg}, but file {world_config_path} contains {config_uuid}")
            console.print(f"[bold red]Error:[/bold red] UUID mismatch in world config file.")
            return None, None
        target_uuid = config_uuid # Use UUID from file as definitive source
        logger.info(f"Successfully loaded world config for UUID: {target_uuid}")
        if not instance_uuid_arg: # If we loaded the latest, print the UUID now
            console.print(f"Using world config for UUID: [cyan]{target_uuid}[/cyan]")

    elif not world_config_path and target_uuid: # Case where no config was found and new UUID generated
         logger.info(f"Proceeding with new simulation UUID {target_uuid} without a world config file.")
         world_config = {} # Use empty dict as default
    # else: world_config_path existed but file didn't - error handled above

    # --- 3. Find Life Summaries ---
    life_summary_pattern = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_*_{target_uuid}.json")
    life_summary_files = glob.glob(life_summary_pattern)
    logger.info(f"Found {len(life_summary_files)} life summary file(s) for world UUID {target_uuid}.")
    if not life_summary_files:
        logger.warning(f"No life summary files found for world {target_uuid}. Simulation may start without agents if creating new state.")

    life_summaries = []
    for ls_file in life_summary_files:
        summary_data = load_json_file(ls_file)
        if summary_data:
            life_summaries.append(summary_data)
        else:
            logger.error(f"Failed to load life summary file: {ls_file}")
            console.print(f"[yellow]Warning:[/yellow] Could not load life summary {os.path.basename(ls_file)}. Skipping.")

    # --- 4. Check for Existing Simulation State ---
    state_file_path = os.path.join(STATE_DIR, f"simulation_state_{target_uuid}.json")
    if os.path.exists(state_file_path):
        # --- 5. Load Existing State ---
        logger.info(f"Found existing simulation state file: {state_file_path}. Loading...")
        loaded_state_data = load_json_file(state_file_path)
        if loaded_state_data is None:
            logger.error(f"Error loading existing simulation state file {state_file_path}. Will attempt to initialize anew.", exc_info=True)
            console.print(f"[bold red]Error:[/bold red] Failed to load existing state file {state_file_path}. Attempting to create a new one.")
            # Fall through to create new state
        else:
            # Verify UUID match within the loaded state
            state_uuid = loaded_state_data.get("world_instance_uuid")
            if not state_uuid:
                 logger.critical(f"State file {state_file_path} is missing 'world_instance_uuid'. Cannot proceed.")
                 console.print(f"[bold red]Error:[/bold red] State file is missing the 'world_instance_uuid' key.")
                 return None, None
            if state_uuid != target_uuid:
                 logger.critical(f"UUID mismatch! Expected '{target_uuid}', but state file contains '{state_uuid}'.")
                 console.print(f"[bold red]Error:[/bold red] UUID mismatch between state file content ('{state_uuid}') and expected UUID ('{target_uuid}').")
                 return None, None

            logger.info(f"Successfully loaded simulation state for UUID: {target_uuid}")
            console.print(f"Loaded existing simulation state from: [green]{state_file_path}[/green]")
            # State is loaded, proceed to structure check
    else:
        # --- 6. Create New State ---
        logger.info(f"No existing simulation state found for UUID {target_uuid}. Creating new state...")
        console.print(f"No simulation state file found for UUID {target_uuid}. Initializing new state...")
        loaded_state_data = create_blank_simulation_state(target_uuid)
        created_new_state = True # Mark that we created it

        # Populate from world_config (use defaults if world_config is empty)
        loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY]["description"] = world_config.get("description", "Default World")
        loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY]["rules"] = world_config.get("rules", {})
        
        # Explicitly copy world_type, sub_genre, and location details from the top level of world_config
        loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY]["world_type"] = world_config.get("world_type", "real")
        loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY]["sub_genre"] = world_config.get("sub_genre", "realtime")
        
        # Ensure LOCATION_KEY exists before populating
        loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY].setdefault(LOCATION_KEY, {})
        loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]["city"] = world_config.get(LOCATION_KEY, {}).get("city", "Unknown City")
        loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]["state"] = world_config.get(LOCATION_KEY, {}).get("state", "Unknown State")
        loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]["country"] = world_config.get(LOCATION_KEY, {}).get("country", "Unknown Country")
        loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]["coordinates"] = world_config.get(LOCATION_KEY, {}).get("coordinates", {})
        
        logger.info(f"Populated world_type '{loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY]['world_type']}', sub_genre '{loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY]['sub_genre']}', and location details from world_config into {WORLD_TEMPLATE_DETAILS_KEY}.")

        # --- ADDED: Explicitly copy mood from world_config if it exists ---
        # The 'mood' might not be at the top level of world_config, but rather within a 'world_template_details' if it was saved from an old state.
        # For new configs, we might want to define a top-level mood or a default.
        # For now, let's assume mood is handled by ensure_state_structure or is not a top-level config item.
        # If 'mood' is intended to be a top-level item in world_config.json, add:
        # loaded_state_data[WORLD_TEMPLATE_DETAILS_KEY]["mood"] = world_config.get("mood", "neutral")

        # Initialize other necessary top-level state structures
        loaded_state_data.setdefault("simulacra", {})
        loaded_state_data.setdefault("npcs", {})
        loaded_state_data.setdefault("objects", {})
        loaded_state_data.setdefault("narrative_log", [])
        loaded_state_data.setdefault("world_time", 0.0)
        loaded_state_data.setdefault(ACTIVE_SIMULACRA_IDS_KEY, [])
        loaded_state_data.setdefault(SIMULACRA_PROFILES_KEY, {})
        loaded_state_data.setdefault(WORLD_STATE_KEY, {}).setdefault(LOCATION_DETAILS_KEY, {})

        # --- Populate location_details from world_config if available ---
        if world_config.get(LOCATION_KEY): # Check if 'location' key exists in world_config
             primary_loc_city = world_config[LOCATION_KEY].get("city")
             if primary_loc_city: # Use city name as ID if available
                 primary_loc_id = primary_loc_city 
             else: # Fallback to a generic ID if city is not specified
                 primary_loc_id = "default_start_location"
                 
             primary_loc_name = world_config[LOCATION_KEY].get("city", "A Starting Point")
             primary_loc_desc = world_config.get("description", "An initial location.") 

             loaded_state_data["location_details"] = {
                 primary_loc_id: {
                     "name": primary_loc_name,
                     "description": primary_loc_desc,
                     "objects_present": [],
                     "connected_locations": []
                 }
             }
             loaded_state_data[WORLD_STATE_KEY][LOCATION_DETAILS_KEY] = loaded_state_data["location_details"].copy()
             logger.info(f"Populated initial location '{primary_loc_id}' from world_config.")
        # ---

        # Populate from life_summaries
        default_start_loc_id = list(loaded_state_data["location_details"].keys())[0] if loaded_state_data["location_details"] else "default_start_location"
        default_start_loc_name = loaded_state_data["location_details"].get(default_start_loc_id, {}).get("name", "A Starting Point")


        for summary in life_summaries:
            sim_id = summary.get("simulacra_id")
            persona = summary.get("persona_details")
            if not sim_id or not persona:
                logger.warning(f"Skipping life summary due to missing 'simulacra_id' or 'persona_details': {summary.get('file_origin', 'Unknown file')}")
                continue

            loaded_state_data[ACTIVE_SIMULACRA_IDS_KEY].append(sim_id)

            # Populate simulacra dictionary (runtime state)
            loaded_state_data["simulacra"][sim_id] = {
                "id": sim_id,
                "name": persona.get("Name", "Unnamed Simulacra"),
                "persona": persona,
                "location": default_start_loc_id, # Start at the default location ID
                "home_location": default_start_loc_id, # Assume home is the start for now
                "status": "idle", # Initial status
                "current_action_end_time": 0.0, # Initialize action time
                "goal": persona.get("Initial_Goal", "Determine long term goals."), # Use Initial_Goal from persona if available
                "last_observation": f"You find yourself in {default_start_loc_name}.", # Basic observation using location name
                "memory_log": [], # Start with empty memory log
                "pending_results": {}
            }

            # Populate simulacra_profiles dictionary (persistent profile info)
            loaded_state_data[SIMULACRA_PROFILES_KEY][sim_id] = {
                "persona_details": persona,
                "current_location": default_start_loc_id, # Use location ID
                "home_location": default_start_loc_id, # Use location ID
                "last_observation": loaded_state_data["simulacra"][sim_id]["last_observation"], # Use the same initial observation
                "goal": loaded_state_data["simulacra"][sim_id]["goal"], # Store initial goal here too
                "memory_log": [] # Store memory log here too? Decide on single source of truth.
            }

        logger.info(f"Initialized new simulation state for UUID {target_uuid} with {len(loaded_state_data[ACTIVE_SIMULACRA_IDS_KEY])} simulacra.")
        console.print(f"Initialized new simulation state with [bold]{len(loaded_state_data[ACTIVE_SIMULACRA_IDS_KEY])}[/bold] simulacra.")

    # --- 7. Ensure State Structure ---
    if loaded_state_data:
        state_modified = ensure_state_structure(loaded_state_data)
    else:
        logger.critical("State data is None after loading/initialization attempt. Cannot proceed.")
        console.print("[bold red]Critical Error:[/bold red] Failed to obtain state data.")
        return None, None

    # --- 8. Save if New or Modified ---
    if created_new_state or state_modified:
        logger.info(f"Saving {'new' if created_new_state else 'modified'} state file: {state_file_path}")
        try:
            save_json_file(state_file_path, loaded_state_data)
        except Exception as e:
            logger.error(f"Failed to save state file {state_file_path}: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/bold red] Failed to save state file {state_file_path}.")
            # Decide if this is critical - maybe return None if save fails?
            # For now, we'll return the in-memory state but log the error.
            # return None, None # Uncomment this to make saving mandatory

    return loaded_state_data, state_file_path

# --- ADDED: get_nested function ---
def get_nested(data: Dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve nested dictionary values."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and isinstance(key, int):
            try:
                current = current[key]
            except IndexError: return default
        else: return default
        if current is None: # Stop early if a key is missing
            return default
    return current if current is not None else default
# ---

# --- Timestamp and Timezone Functions ---

def format_iso_timestamp(iso_str: Optional[str]) -> str:
    """Formats an ISO timestamp string into 'YYYY-MM-DD h:MM AM/PM'."""
    if not iso_str:
        return "Unknown Time"
    try:
        # Handle 'Z' explicitly for UTC
        if iso_str.endswith('Z'):
            dt_obj = datetime.fromisoformat(iso_str[:-1] + '+00:00')
        else:
            dt_obj = datetime.fromisoformat(iso_str)

        # If timezone is naive, assume UTC (though ISO format should ideally have offset)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)

        # Format without timezone information for simplicity here
        return dt_obj.strftime("%Y-%m-%d %I:%M %p")
    except (ValueError, TypeError):
        logging.warning(f"Could not parse timestamp for formatting: {iso_str}")
        return iso_str

def format_localized_timestamp(
    iso_str: Optional[str],
    timezone_str: str = 'UTC', # Default to UTC if no specific timezone provided
    display_format: str = "%Y-%m-%d %I:%M:%S %p %Z" # Example: 2023-10-27 03:45:10 PM EDT
) -> str:
    """
    Formats an ISO timestamp string into a human-readable, localized time.

    Args:
        iso_str: The ISO timestamp string (assumed to be UTC or timezone-aware).
        timezone_str: The target IANA timezone name (e.g., 'America/New_York').
        display_format: The strftime format string for the output.

    Returns:
        The formatted, localized time string or an error message.
    """
    if not iso_str:
        return "Unknown Time"

    try:
        # 1. Parse the ISO string into a timezone-aware datetime object
        # Handle 'Z' for UTC explicitly if present
        if iso_str.endswith('Z'):
            utc_dt = datetime.fromisoformat(iso_str[:-1] + '+00:00')
        else:
            # Assume it's already timezone-aware or implicitly UTC if offset is missing
            # (datetime.fromisoformat handles offsets like +00:00)
            utc_dt = datetime.fromisoformat(iso_str)
            # If the parsed object is naive, assume it's UTC
            if utc_dt.tzinfo is None:
                utc_dt = utc_dt.replace(tzinfo=timezone.utc)


        # 2. Get the target timezone object
        try:
            target_tz = pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone '{timezone_str}'. Falling back to UTC.")
            target_tz = pytz.utc
            timezone_str = 'UTC' # Update for display

        # 3. Convert the datetime object to the target timezone
        local_dt = utc_dt.astimezone(target_tz)

        # 4. Format the localized datetime object
        return local_dt.strftime(display_format)

    except ValueError:
        logger.warning(f"Could not parse ISO timestamp: {iso_str}")
        return f"Invalid Time ({iso_str})"
    except Exception as e:
        logger.error(f"Error formatting localized time for '{iso_str}' with tz '{timezone_str}': {e}")
        return f"Time Error ({iso_str})"

def get_timezone_from_location(
    location_dict: Optional[Dict[str, Any]],
    tf_instance: TimezoneFinder # Pass the instance
) -> Optional[str]:
    """
    Attempts to find the IANA timezone from location details using coordinates.

    Args:
        location_dict: Dictionary containing location details (must have 'coordinates').
        tf_instance: An initialized TimezoneFinder instance.

    Returns:
        The IANA timezone string or None if not found/error.
    """
    if not location_dict:
        logger.debug("get_timezone_from_location: location_dict is None.")
        return None

    coords = location_dict.get("coordinates")
    if coords and coords.get("latitude") is not None and coords.get("longitude") is not None:
        lat = coords["latitude"]
        lon = coords["longitude"]
        try:
            # Use the passed instance
            found_tz = tf_instance.timezone_at(lng=lon, lat=lat)
            if found_tz:
                logger.debug(f"Found timezone '{found_tz}' from coordinates ({lat}, {lon})")
                return found_tz
            else:
                logger.warning(f"timezonefinder returned None for coordinates ({lat}, {lon}).")
                return None
        except Exception as tz_err:
            logger.error(f"Error using timezonefinder for coords ({lat}, {lon}): {tz_err}")
            return None
    else:
        city = location_dict.get("city", "N/A") # Use city for logging if available
        logger.debug(f"Coordinates missing or incomplete for location '{city}'. Cannot determine timezone.")
        return None
