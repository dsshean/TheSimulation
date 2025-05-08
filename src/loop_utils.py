import os
import json
import time
import logging
import uuid
import glob
import copy
from typing import Any, Dict, Optional, List, Tuple # Added Tuple
from .file_utils import ensure_dir_exists, get_data_dir, get_states_dir, get_world_config_dir, get_life_summary_dir
from .state_loader import parse_location_string # Assuming this is correctly placed or imported
import re
# Configure logging
logger = logging.getLogger(__name__)

# --- Directory Setups ---
DATA_DIR = get_data_dir()
STATE_DIR = get_states_dir()
WORLD_CONFIG_DIR = get_world_config_dir()
LIFE_SUMMARY_DIR = get_life_summary_dir()

# --- Constants for New Simulation State ---
DEFAULT_SIMULACRUM_CURRENT_LOCATION = "Home_01"
DEFAULT_SIMULACRUM_LOCATION_DETAILS = "Your home and bedroom."
DEFAULT_SIMULACRUM_LAST_OBSERVATION = "Waking up in Home_01."
DEFAULT_SIMULACRUM_GOAL = "Determine goals."
DEFAULT_SIMULACRUM_HOME_LOCATION = "Home_01"
DEFAULT_SIMULACRUM_STATUS = "idle"

NEW_SIMULATION_STATE_TEMPLATE = {
  "world_instance_uuid": "",
  "active_simulacra_ids": [],
  "world_time": 0.0,
  "narrative_log": [],
  "world_template_details": {
    "description": "A new world is beginning.",
    "rules": {
      "allow_teleportation": False,
      "time_progression_rate": 1.0,
      "weather_effects_travel": True,
      "historical_date": None
    },
    "world_type": "unknown",
    "sub_genre": "unknown",
    "location": {
      "city": "Unknown",
      "state": "Unknown",
      "country": "Unknown"
    },
    "mood": "Anticipatory"
  },
  "simulacra_profiles": {},
  "objects": [] # Will be populated from world_config's initial_objects
}

DEFAULT_SIMULACRUM_RUNTIME_STATE = {
    "current_location": DEFAULT_SIMULACRUM_CURRENT_LOCATION,
    "location_details": DEFAULT_SIMULACRUM_LOCATION_DETAILS,
    "last_observation": DEFAULT_SIMULACRUM_LAST_OBSERVATION,
    "goal": DEFAULT_SIMULACRUM_GOAL,
    "memory_log": [],
    "home_location": DEFAULT_SIMULACRUM_HOME_LOCATION, # "location" key will be same as current_location initially
    "status": DEFAULT_SIMULACRUM_STATUS,
    "current_action_end_time": 0.0,
    "pending_results": {},
    "last_interjection_sim_time": 0.0,
    "next_simple_timer_interjection_sim_time": 0.0
}


def load_json_file(path: str, default=None):
    """Loads a JSON file from the given path."""
    if not os.path.exists(path):
        logger.debug(f"File not found: {path}. Returning default.")
        return default
    try:
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {path}. File might be corrupted or not valid JSON.")
        return default
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {path}: {e}")
        return default

def save_json_file(path: str, data: dict) -> bool:
    """Saves a dictionary to a JSON file."""
    try:
        ensure_dir_exists(os.path.dirname(path))
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Successfully saved JSON to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {path}: {e}")
        return False

def create_blank_simulation_state(
    target_uuid: str,
    world_config_data: dict,
    all_life_summaries_data: list[dict]
) -> dict:
    """
    Creates a new simulation state dictionary based on the template,
    populated with data from world_config and life_summaries.
    """
    logger.info(f"Creating a new simulation state from template for UUID: {target_uuid}")
    state = copy.deepcopy(NEW_SIMULATION_STATE_TEMPLATE)

    state["world_instance_uuid"] = target_uuid

    # Populate world_template_details from world_config_data
    state["world_template_details"]["description"] = world_config_data.get("world_description", state["world_template_details"]["description"])
    state["world_template_details"]["rules"] = world_config_data.get("rules", state["world_template_details"]["rules"])
    state["world_template_details"]["world_type"] = world_config_data.get("world_type", state["world_template_details"]["world_type"])
    state["world_template_details"]["sub_genre"] = world_config_data.get("sub_genre", state["world_template_details"]["sub_genre"])
    
    world_config_location_value = world_config_data.get("location")
    if isinstance(world_config_location_value, str):
        parsed_location = parse_location_string(world_config_location_value)
        if parsed_location: # parse_location_string returns a dict, potentially with None values
            state["world_template_details"]["location"] = parsed_location
        else: # Keep template default if parsing fails
            logger.warning(f"Could not parse location string '{world_config_location_value}' from world_config. Using default location for new state.")
    elif isinstance(world_config_location_value, dict):
        state["world_template_details"]["location"] = world_config_location_value # Assume it's already the correct dict structure
    # If world_config_location_value is None or not str/dict, the template default (from NEW_SIMULATION_STATE_TEMPLATE) will remain.
    # No explicit else needed here if the template already has a satisfactory default.

    state["world_template_details"]["mood"] = world_config_data.get("mood", state["world_template_details"]["mood"])

    # Populate objects from world_config_data
    state["objects"] = world_config_data.get("initial_objects", [])

    # Populate simulacra_profiles from life_summaries_data
    for summary_data in all_life_summaries_data:
        sim_id = summary_data.get("sim_id")
        if not sim_id:
            logger.warning(f"Life summary missing sim_id: {summary_data.get('persona_details', {}).get('Name', 'Unknown Name')}. Skipping.")
            continue
        
        state["active_simulacra_ids"].append(sim_id)
        sim_profile = copy.deepcopy(DEFAULT_SIMULACRUM_RUNTIME_STATE)
        sim_profile["persona_details"] = summary_data.get("persona_details", {})
        sim_profile["location"] = sim_profile["current_location"] # Ensure 'location' matches 'current_location' initially
        state["simulacra_profiles"][sim_id] = sim_profile
    return state


def ensure_state_structure(state: dict, instance_uuid: str, active_sim_ids_from_summaries: list[str]) -> bool:
    """
    Ensures the simulation state dictionary has all necessary top-level keys and basic structures.
    Adds missing keys with default values.
    Returns True if the state was modified, False otherwise.
    """
    modified = False
    required_top_level_keys = {
        "world_instance_uuid": instance_uuid,
        "active_simulacra_ids": [],
        "world_time": 0.0,
        "narrative_log": [],
        "world_template_details": {}, # Will be further checked
        "simulacra_profiles": {},   # Will be further checked
        "objects": []
    }

    for key, default_value in required_top_level_keys.items():
        if key not in state:
            state[key] = default_value
            logger.info(f"ensure_state_structure: Added missing top-level key '{key}' with default.")
            modified = True

    # Ensure world_template_details and its sub-keys
    if "world_template_details" not in state or not isinstance(state["world_template_details"], dict):
        state["world_template_details"] = copy.deepcopy(NEW_SIMULATION_STATE_TEMPLATE["world_template_details"])
        modified = True
    details_template = NEW_SIMULATION_STATE_TEMPLATE["world_template_details"]
    for key, default_value in details_template.items():
        if key not in state["world_template_details"]:
            state["world_template_details"][key] = copy.deepcopy(default_value)
            modified = True

    # Ensure simulacra_profiles structure
    if "simulacra_profiles" not in state or not isinstance(state["simulacra_profiles"], dict):
        state["simulacra_profiles"] = {}
        modified = True
    
    # Ensure all active simulacra (from summaries) have a profile entry
    # This is more about ensuring entries exist if somehow missed during initial load/creation
    # The create_blank_simulation_state should handle the initial population.
    for sim_id in active_sim_ids_from_summaries:
        if sim_id not in state["simulacra_profiles"]:
            state["simulacra_profiles"][sim_id] = copy.deepcopy(DEFAULT_SIMULACRUM_RUNTIME_STATE)
            # Attempt to find matching persona_details (though ideally this is set during creation)
            # This is a fallback.
            logger.warning(f"ensure_state_structure: Added missing profile for sim_id '{sim_id}'. Persona details might be missing if not set earlier.")
            modified = True
        
        profile = state["simulacra_profiles"][sim_id]
        if "persona_details" not in profile:
            profile["persona_details"] = {} # Basic default
            modified = True

        # Ensure basic runtime keys for each simulacrum profile (more can be added as needed)
        template_sim_profile = DEFAULT_SIMULACRUM_RUNTIME_STATE # Using this as a reference for keys
        for key, default_value in template_sim_profile.items():
            if key not in profile: # Add missing runtime keys with their defaults
                profile[key] = copy.deepcopy(default_value)
                modified = True
    return modified


def sync_world_config_to_state(state: dict, world_config: dict) -> bool:
    """
    Syncs relevant details from the world_config into the simulation state.
    This is especially important for initial_objects and world_template_details.
    Returns True if the state was modified, False otherwise.
    """
    modified = False
    logger.debug(f"Syncing world_config (UUID: {world_config.get('world_instance_uuid')}) to state (UUID: {state.get('world_instance_uuid')}).")
    # Handle location syncing carefully
    world_config_location = world_config.get("location")
    parsed_wc_location = None
    if isinstance(world_config_location, str):
        parsed_wc_location = parse_location_string(world_config_location)
    elif isinstance(world_config_location, dict): # It might already be a parsed dictionary
        parsed_wc_location = world_config_location
    # Sync world_template_details
    new_details = {
        "description": world_config.get("world_description", state.get("world_template_details", {}).get("description")),
        "rules": world_config.get("rules", state.get("world_template_details", {}).get("rules", {})),
        "world_type": world_config.get("world_type", state.get("world_template_details", {}).get("world_type")),
        "sub_genre": world_config.get("sub_genre", state.get("world_template_details", {}).get("sub_genre")),
        "location": parsed_wc_location or state.get("world_template_details", {}).get("location"),
        "mood": world_config.get("mood", state.get("world_template_details", {}).get("mood"))
    }
    if state.get("world_template_details") != new_details:
        state["world_template_details"] = new_details
        modified = True
        logger.info("Updated state.world_template_details from world_config.")

    # Sync initial_objects - This typically means resetting objects in state to match world_config
    # This is a destructive update for state["objects"] based on world_config.
    initial_objects = world_config.get("initial_objects", [])
    if state.get("objects") != initial_objects: # Simple comparison, might need deep compare for complex objects
        state["objects"] = copy.deepcopy(initial_objects) # Ensure a copy is stored
        modified = True
        logger.info("Updated state.objects from world_config.initial_objects.")

    return modified


def find_latest_file(pattern: str) -> str | None:
    """Finds the most recently modified file matching the pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def load_or_initialize_simulation(instance_uuid_arg: str | None) -> tuple[dict | None, str]:
    """
    Loads an existing simulation state or initializes a new one.
    - If instance_uuid_arg is provided, it tries to load that specific simulation.
    - If not, it tries to load the latest simulation.
    - If no simulation is found, it initializes a new one (if possible).

    Returns a tuple: (loaded_state_data, state_file_path)
    If loading/initialization fails critically, loaded_state_data might be None.
    """
    target_uuid = None
    world_config = None
    world_config_path = None
    state_modified_during_load = False # Tracks if any changes made that require saving
    created_new_state = False

    # 1. Determine target_uuid and load world_config
    if instance_uuid_arg:
        logger.info(f"Attempting to load simulation for specified UUID: {instance_uuid_arg}")
        world_config_path = os.path.join(WORLD_CONFIG_DIR, f"world_config_{instance_uuid_arg}.json")
        world_config = load_json_file(world_config_path)
        if world_config and world_config.get("world_instance_uuid") == instance_uuid_arg:
            target_uuid = instance_uuid_arg
        else:
            logger.error(f"World config for UUID {instance_uuid_arg} not found or UUID mismatch. Cannot proceed with this instance.")
            return None, "" # Critical failure
    else:
        logger.info("No instance UUID provided. Attempting to load the latest simulation.")
        world_config_pattern = os.path.join(WORLD_CONFIG_DIR, "world_config_*.json")
        latest_world_config_path = find_latest_file(world_config_pattern)
        if latest_world_config_path:
            world_config = load_json_file(latest_world_config_path)
            if world_config and world_config.get("world_instance_uuid"):
                target_uuid = world_config["world_instance_uuid"]
                world_config_path = latest_world_config_path
                logger.info(f"Found latest world config: {os.path.basename(latest_world_config_path)} with UUID: {target_uuid}")
            else:
                logger.warning(f"Latest world config file {latest_world_config_path} is invalid or missing UUID. Cannot load.")
        else:
            logger.info("No existing world_config files found. A new simulation will need to be set up first (e.g., via setup_simulation.py).")
            # This path implies no simulation can be started unless setup_simulation.py creates one.
            # For now, we won't auto-create a world_config here.
            return None, "" # Critical: No world to load or base a new state on.

    if not target_uuid or not world_config:
        logger.error("Failed to determine a target UUID or load its world configuration. Cannot initialize simulation state.")
        return None, ""

    # 2. Load Life Summaries for the target_uuid
    life_summary_pattern = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_*_{target_uuid}.json")
    life_summary_files = glob.glob(life_summary_pattern)
    life_summaries = [] # Initialize an empty list
    if life_summary_files:
        for ls_file in life_summary_files:
            ls_data = load_json_file(ls_file)
            if ls_data is None:
                logger.warning(f"load_json_file returned None for: {os.path.basename(ls_file)}")
                continue # Skip to next file if data couldn't be loaded
            
            # Strictly look for "sim_id"
            sim_id_from_file = ls_data.get("sim_id")
            logger.debug(f"Processing life summary file: {os.path.basename(ls_file)}. Found sim_id: '{sim_id_from_file}'")

            if sim_id_from_file: 
                life_summaries.append(ls_data) # CORRECTED: Append valid data
                logger.info(f"Loaded life summary: {os.path.basename(ls_file)} for sim_id: {sim_id_from_file}")
            else:
                logger.warning(f"Could not load or validate life summary (missing 'sim_id'): {os.path.basename(ls_file)}")
    else:
        logger.warning(f"No life summary files found for UUID {target_uuid} matching pattern {life_summary_pattern}.")
        # This might be an issue if we need to create a new state, as there would be no simulacra.

    # 3. Attempt to load existing simulation_state.json
    state_file_path = os.path.join(STATE_DIR, f"simulation_state_{target_uuid}.json")
    loaded_state_data = load_json_file(state_file_path)

    if loaded_state_data is None and os.path.exists(state_file_path): # File exists but failed to load (e.g. corrupt)
        logger.warning(f"Existing state file at {state_file_path} could not be loaded/parsed. Will create a new one.")
        created_new_state = True # Treat as new state creation
    elif not os.path.exists(state_file_path):
        logger.info(f"No existing state file found at {state_file_path}. Will create a new one.")
        loaded_state_data = None
        created_new_state = True
    elif loaded_state_data.get("world_instance_uuid") != target_uuid:
        logger.warning(f"State file UUID ({loaded_state_data.get('world_instance_uuid')}) "
                       f"does not match target UUID ({target_uuid}). Creating new state.")
        loaded_state_data = None # Discard mismatched state
        created_new_state = True

    if loaded_state_data is None:
        # This is where a new state is created
        if not world_config:
            logger.error(f"Cannot create new simulation state: World config for UUID {target_uuid} not loaded/found.")
            return None, state_file_path
        if not life_summaries:
            logger.error(f"Cannot create new simulation state for UUID {target_uuid}: No life summaries found. At least one simulacrum is required.")
            return None, state_file_path

        logger.info(f"Initializing new simulation state for UUID: {target_uuid}")
        loaded_state_data = create_blank_simulation_state(target_uuid, world_config, life_summaries)
        state_modified_during_load = True # New state is inherently a modification
        created_new_state = True # Explicitly set

        # active_sim_ids are already populated by create_blank_simulation_state
        # persona_details are also populated by create_blank_simulation_state

    # 4. Ensure state structure and sync with world_config
    # Get active sim_ids from summaries for ensure_state_structure, as it might be needed if state was loaded but incomplete
    active_sim_ids_from_summaries = [ls.get("sim_id") for ls in life_summaries if ls.get("sim_id")]
    
    if loaded_state_data: # If state was loaded or newly created successfully
        structure_modified = ensure_state_structure(loaded_state_data, target_uuid, active_sim_ids_from_summaries)
        state_modified_during_load = state_modified_during_load or structure_modified

        if world_config:
            state_modified_by_sync = sync_world_config_to_state(loaded_state_data, world_config)
            state_modified_during_load = state_modified_during_load or state_modified_by_sync
        else:
            # This case should ideally not be hit if target_uuid was derived from a loaded world_config
            logger.warning(f"World config for UUID {target_uuid} not available for final state sync. State may be incomplete.")
    else: # Failed to load or create state
        logger.error(f"Failed to load or create a simulation state for UUID {target_uuid}.")
        return None, state_file_path


    # 5. Save if modified during load (e.g., new state, structure ensured, or synced)
    if state_modified_during_load and loaded_state_data:
        logger.info(f"Simulation state for {target_uuid} was modified during load/initialization. Saving changes to {state_file_path}.")
        save_json_file(state_file_path, loaded_state_data)
    elif created_new_state and loaded_state_data: # Also save if it's a new state, even if no "modifications" per se
        logger.info(f"Newly created simulation state for {target_uuid}. Saving to {state_file_path}.")
        save_json_file(state_file_path, loaded_state_data)


    return loaded_state_data, state_file_path

def parse_json_output_last(text_output: str) -> Optional[Dict[Any, Any]]:
    """
    Attempts to parse a JSON object from a string.
    Prioritizes JSON within markdown code fences (```json ... ``` or ``` ... ```).
    If not found, attempts to find the last complete JSON object in the string.
    """
    if not text_output:
        logger.debug("parse_json_output_last received empty text_output.")
        return None

    # 1. Attempt to extract from markdown code fences
    # Regex to find ```json ... ``` or ``` ... ``` and capture the content within
    # re.DOTALL allows '.' to match newlines
    # The main capturing group (group 1) will be the content inside the code fence
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text_output, re.DOTALL | re.IGNORECASE)
    
    json_to_parse = None

    if fence_match:
        json_to_parse = fence_match.group(1).strip()
        logger.debug(f"Extracted from markdown fence: '{json_to_parse[:200]}...'")
    else:
        # 2. If no markdown fence, try to find the last JSON object (heuristic)
        logger.debug(f"No markdown fence found in: '{text_output[:200]}...'. Trying heuristic.")
        last_brace = text_output.rfind('{')
        if last_brace != -1:
            open_braces = 0
            candidate = ""
            for char in text_output[last_brace:]:
                candidate += char
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1
                    if open_braces == 0:
                        json_to_parse = candidate # Found a balanced pair
                        logger.debug(f"Heuristic found potential JSON object: '{json_to_parse[:200]}...'")
                        break 
            if open_braces != 0: # Didn't find a balanced pair
                json_to_parse = None 
                logger.debug(f"No balanced JSON object found after last '{{'. Candidate was: {candidate[:200]}")
        
        if not json_to_parse:
            # Fallback: if no object found, try to find the last array '[' ... ']'
            last_bracket = text_output.rfind('[')
            if last_bracket != -1:
                open_brackets = 0
                candidate_arr = ""
                for char_arr in text_output[last_bracket:]:
                    candidate_arr += char_arr
                    if char_arr == '[':
                        open_brackets += 1
                    elif char_arr == ']':
                        open_brackets -= 1
                        if open_brackets == 0:
                            json_to_parse = candidate_arr
                            logger.debug(f"Heuristic found potential JSON array: '{json_to_parse[:200]}...'")
                            break
                if open_brackets != 0:
                    json_to_parse = None
                    logger.debug(f"No balanced JSON array found after last '['. Candidate was: {candidate_arr[:200]}")

    if json_to_parse:
        try:
            logger.debug(f"Attempting to parse: '{json_to_parse[:200]}...'")
            parsed_dict = json.loads(json_to_parse)
            if isinstance(parsed_dict, dict):
                logger.debug(f"Successfully parsed into dict: {str(parsed_dict)[:200]}")
                return parsed_dict
            else:
                # This case is if json.loads returns a list, int, string, etc., but we expect a dict.
                logger.warning(f"Parsed JSON is not a dictionary: Type {type(parsed_dict)}, Value: {str(parsed_dict)[:200]}")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"Final JSON parsing attempt failed for: '{json_to_parse[:200]}...'. Error: {e}")
            return None
    
    logger.debug(f"Could not find or parse a valid JSON object from text: {text_output[:200]}...")
    return None

def get_nested(data: Dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve nested dictionary values."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list): # Added check for list and integer key
            if isinstance(key, int):
                try:
                    current = current[key]
                except IndexError:
                    return default
            else: # Key is not an int, cannot index list
                return default
        else: # Current is not a dict or list, cannot go deeper
            return default
        
        if current is None: # Stop early if a key is missing or .get() returned None
            return default
            
    return current # Return current value, or default if any key was missing