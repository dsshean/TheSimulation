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

from .config import (
    STATE_DIR,
    LIFE_SUMMARY_DIR,
    WORLD_CONFIG_DIR,
    WORLD_STATE_KEY,
    ACTIVE_SIMULACRA_IDS_KEY,
    LOCATION_DETAILS_KEY,
    SIMULACRA_PROFILES_KEY,
    WORLD_TEMPLATE_DETAILS_KEY,
    LOCATION_KEY
)

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
    raw_text: Optional[str]
) -> Optional[str]:
    """
    Attempts to extract a potential JSON string from an agent's text output,
    robustly handling markdown code fences and preceding/trailing text.
    Prioritizes the LAST JSON block found.
    Returns the potential JSON string or None if no suitable candidate is found.
    """
    if not raw_text:
        return None

    text_to_parse = raw_text.strip()
    json_string_to_parse = None

    # 1. Look for ```json ... ``` blocks
    json_matches = re.findall(r"```json\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
    if json_matches:
        # Take the last match found
        json_string_to_parse = json_matches[-1].strip()
    else:
        # 2. If no ```json block, look for ``` ... ``` blocks
        code_matches = re.findall(r"```\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
        if code_matches:
            # Take the last match found
            json_string_to_parse = code_matches[-1].strip()
        else:
            # 3. If no markdown block, find the LAST '{' and try from there
            last_brace = text_to_parse.rfind('{')
            if last_brace != -1:
                # Attempt to find the matching closing brace (simple heuristic)
                # Find the last '}' after the last '{'
                last_close_brace = text_to_parse.rfind('}', last_brace)
                if last_close_brace != -1:
                    potential_json = text_to_parse[last_brace : last_close_brace + 1]
                    json_string_to_parse = potential_json
                else:
                    # Fallback: try from last '{' to the end
                    potential_json = text_to_parse[last_brace:]
                    json_string_to_parse = potential_json
            else:
                # 4. If no '{' found at all
                return None
    return json_string_to_parse

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
      # "npcs": {}, # NPCs are now ephemeral, removed from persistent state
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
        # "npcs": {}, # NPCs are now ephemeral
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
        elif key in ["location_details", "objects", "simulacra", "current_world_state", "world_template_details", "simulacra_profiles"] and not isinstance(state_dict[key], dict):
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
        default_mood = "Real world" # Or choose your preferred default
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
    logger.info("Starting simulation state loading/initialization.")
    world_config_path: Optional[str] = None
    target_uuid: Optional[str] = instance_uuid_arg
    world_config: Optional[Dict[str, Any]] = None
    state_file_path: Optional[str] = None
    loaded_state_data: Optional[Dict[str, Any]] = None
    created_new_state = False
    state_modified_during_load = False

    if not target_uuid:
        logger.info("No instance UUID specified, searching for the latest world config...")
        world_config_pattern = os.path.join(WORLD_CONFIG_DIR, "world_config_*.json")
        latest_world_config_path = find_latest_file(world_config_pattern)
        if latest_world_config_path:
            world_config_path = latest_world_config_path
            logger.info(f"Found latest world config file: {world_config_path}.")
            console.print(f"Found latest world config file: [cyan]{os.path.basename(world_config_path)}[/cyan]")
        else:
            logger.warning("No world config files found. Creating new simulation instance.")
            target_uuid = str(uuid.uuid4())
            console.print(f"[yellow]Warning:[/yellow] No world config found. Creating new simulation with UUID: {target_uuid}")
    else:
        world_config_path = os.path.join(WORLD_CONFIG_DIR, f"world_config_{target_uuid}.json")
        if not os.path.exists(world_config_path):
            logger.error(f"Specified world config file not found: {world_config_path}")
            console.print(f"[bold red]Error:[/bold red] World config for UUID {target_uuid} not found.")
            return None, None

    if world_config_path and os.path.exists(world_config_path):
        world_config = load_json_file(world_config_path)
        if world_config is None:
            logger.error(f"Failed to load world config: {world_config_path}")
            return None, None
        config_uuid = world_config.get("world_instance_uuid")
        if not config_uuid:
             logger.error(f"World config {world_config_path} missing 'world_instance_uuid'.")
             return None, None
        if instance_uuid_arg and config_uuid != instance_uuid_arg:
            logger.error(f"UUID mismatch: Arg={instance_uuid_arg}, File={config_uuid}")
            return None, None
        target_uuid = config_uuid
        logger.info(f"Successfully loaded world config for UUID: {target_uuid}")
        if not instance_uuid_arg: console.print(f"Using world config for UUID: [cyan]{target_uuid}[/cyan]")
    elif not world_config_path and target_uuid:
         logger.info(f"Proceeding with new UUID {target_uuid} without a world config file (using defaults).")
         world_config = {"world_instance_uuid": target_uuid} # Minimal default
    else:
        logger.error("Critical error determining world_config.")
        return None, None

    life_summary_pattern = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_*_{target_uuid}.json")
    life_summary_files = glob.glob(life_summary_pattern)
    logger.info(f"Found {len(life_summary_files)} life summary file(s) for UUID {target_uuid}.")
    life_summaries = [ls_data for ls_file in life_summary_files if (ls_data := load_json_file(ls_file))]

    state_file_path = os.path.join(STATE_DIR, f"simulation_state_{target_uuid}.json")
    if os.path.exists(state_file_path):
        logger.info(f"Found existing state file: {state_file_path}. Loading...")
        loaded_state_data = load_json_file(state_file_path)
        if loaded_state_data is None:
            logger.error(f"Error loading existing state {state_file_path}. Initializing anew.")
        else:
            state_uuid = loaded_state_data.get("world_instance_uuid")
            if not state_uuid or state_uuid != target_uuid:
                 logger.critical(f"UUID mismatch in state file! Expected '{target_uuid}', found '{state_uuid}'. Forcing new state.")
                 loaded_state_data = None # Force new state
            else:
                logger.info(f"Successfully loaded state for UUID: {target_uuid}")
                console.print(f"Loaded existing state from: [green]{state_file_path}[/green]")
    
    if loaded_state_data is None:
        logger.info(f"No usable existing state for UUID {target_uuid}. Creating new state...")
        console.print(f"No usable state file for UUID {target_uuid}. Initializing new state...")
        loaded_state_data = create_blank_simulation_state(target_uuid)
        created_new_state = True
        state_modified_during_load = True 

        # Populate simulacra for a NEW state
        # setup_simulation.py should create the initial location_details and objects in the state file.
        # This part only adds simulacra based on life summaries.
        default_start_loc_id = "default_start_location" # Fallback
        # Check if location_details exists and has keys, from create_blank_simulation_state or setup_simulation
        loc_details_in_new_state = get_nested(loaded_state_data, WORLD_STATE_KEY, LOCATION_DETAILS_KEY)
        if loc_details_in_new_state and isinstance(loc_details_in_new_state, dict) and loc_details_in_new_state.keys():
            default_start_loc_id = list(loc_details_in_new_state.keys())[0]
        
        for summary in life_summaries:
            sim_id = summary.get("simulacra_id")
            persona = summary.get("persona_details")
            if not sim_id or not persona: continue
            loaded_state_data.setdefault(ACTIVE_SIMULACRA_IDS_KEY, []).append(sim_id)
            loaded_state_data.setdefault("simulacra", {})[sim_id] = {
                "id": sim_id, "name": persona.get("Name", sim_id), "persona": persona,
                "location": default_start_loc_id, "home_location": default_start_loc_id,
                "status": "idle", "current_action_end_time": 0.0,
                "goal": persona.get("Initial_Goal", "Determine goals."),
                "last_observation": f"Waking up in {get_nested(loc_details_in_new_state, default_start_loc_id, 'name', default=default_start_loc_id)}.",
                "memory_log": [], "pending_results": {}
            }
            loaded_state_data.setdefault(SIMULACRA_PROFILES_KEY, {})[sim_id] = {
                "persona_details": persona, "current_location": default_start_loc_id,
                "home_location": default_start_loc_id,
                "last_observation": loaded_state_data["simulacra"][sim_id]["last_observation"],
                "goal": loaded_state_data["simulacra"][sim_id]["goal"], "memory_log": []
            }
        logger.info(f"Initialized new state for UUID {target_uuid} with {len(get_nested(loaded_state_data, ACTIVE_SIMULACRA_IDS_KEY, default=[]))} simulacra.")

    # --- Sync authoritative fields from world_config and reset objects ---
    if loaded_state_data and world_config:
        logger.info(f"Syncing from world_config and resetting objects for UUID: {target_uuid}")
        wtd = loaded_state_data.setdefault(WORLD_TEMPLATE_DETAILS_KEY, {})

        # These fields are ALWAYS taken from world_config
        for key, default_wc_val in [("world_type", "fictional"), ("sub_genre", "turn_based"), 
                                    ("description", "A simulated world."), ("rules", {})]:
            wc_val = world_config.get(key, default_wc_val)
            if wtd.get(key) != wc_val:
                wtd[key] = wc_val
                state_modified_during_load = True
        
        loc_key_data = wtd.setdefault(LOCATION_KEY, {})
        wc_loc = world_config.get(LOCATION_KEY, {})
        for key, default_wc_val in [("city", "Unknown City"), ("state", "Unknown State"), 
                                    ("country", "Unknown Country"), ("coordinates", {})]:
            wc_val = wc_loc.get(key, default_wc_val)
            if loc_key_data.get(key) != wc_val:
                loc_key_data[key] = wc_val
                state_modified_during_load = True
        logger.info(f"Synced WORLD_TEMPLATE_DETAILS_KEY from world_config: type='{wtd.get('world_type')}', city='{loc_key_data.get('city')}'")

        # ALWAYS reset objects from world_config.json
        # This assumes world_config.json is the source of truth for initial object states.
        # If world_config.json does NOT contain objects, this will result in an empty object dict.
        loaded_state_data["objects"] = {} # Blank out existing objects from loaded state
        config_objects = world_config.get("objects", {})
        if config_objects: # Only populate if world_config has objects
            for obj_id, obj_data_template in config_objects.items():
                loaded_state_data["objects"][obj_id] = json.loads(json.dumps(obj_data_template)) # Deep copy
                if "id" not in loaded_state_data["objects"][obj_id]:
                     loaded_state_data["objects"][obj_id]["id"] = obj_id
            logger.info(f"Reset and populated 'objects' from world_config.json ({len(loaded_state_data['objects'])} objects).")
            state_modified_during_load = True
        else:
            logger.info("'objects' key not found or empty in world_config.json. State 'objects' will be empty.")
            # If objects were blanked from a loaded state, and world_config has no objects,
            # this is a modification.
            if not created_new_state: # Only mark as modified if we blanked a previously non-empty dict
                 state_modified_during_load = True

        # location_details (in-game map) are NOT reset from world_config here by default,
        # as they are managed in simulation_state.json.
        # If world_config *were* to define the master map, you would add:
        # if "location_details" in world_config:
        #    loaded_state_data.setdefault(WORLD_STATE_KEY, {})
        #    loaded_state_data[WORLD_STATE_KEY][LOCATION_DETAILS_KEY] = json.loads(json.dumps(world_config["location_details"]))
        #    state_modified_during_load = True
        #    logger.info(f"Reset state's '{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}' from world_config.")

    if loaded_state_data:
        structure_modified = ensure_state_structure(loaded_state_data)
        state_modified_during_load = state_modified_during_load or structure_modified
    else:
        logger.critical("State data is None. Cannot proceed.")
        return None, None

    if created_new_state or state_modified_during_load:
        logger.info(f"Saving {'new' if created_new_state else 'modified'} state file: {state_file_path}")
        try:
            save_json_file(state_file_path, loaded_state_data)
        except Exception as e:
            logger.error(f"Failed to save state file {state_file_path}: {e}", exc_info=True)
    
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
