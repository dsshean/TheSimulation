import os
import json
import time
import logging
import uuid
import glob
import copy
from typing import Any, Dict, Optional, List, Tuple # Added Tuple
from datetime import datetime, timezone # Added for real_world_start_utc
from .file_utils import ensure_dir_exists, get_data_dir, get_states_dir, get_world_config_dir, get_life_summary_dir
from .config import ( # Added imports from config
    WORLD_STATE_KEY, LOCATION_DETAILS_KEY, DEFAULT_HOME_LOCATION_NAME, DEFAULT_HOME_DESCRIPTION,
    WORLD_FEEDS_KEY # Added WORLD_FEEDS_KEY
)
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
DEFAULT_SIMULACRUM_LAST_OBSERVATION = "You are at Home_01."
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
    "mood": "The real world and general slice of life."
  },
  "simulacra_profiles": {},
  "objects": [], # Static objects populated from world_config's initial_objects
  WORLD_STATE_KEY: { # Using constant for current_world_state
      LOCATION_DETAILS_KEY: {}, # e.g., {"Home_01": {"id": "Home_01", "name": "Home", "description": "...", ...}}
      WORLD_FEEDS_KEY: { # Using constant for world_feeds
          "weather": {"condition": "Weather is calm.", "temperature_celsius": 20, "forecast_short": "Stable conditions."},
          "news_updates": [],
          "pop_culture_updates": [],
          "last_update_sim_time": 0.0
      }
      # Add other world-specific state here if needed, e.g., world_context
  },
  "pending_simulation_events": [] # Added for scheduled events
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

    # Populate initial location definitions into the state
    initial_locs = world_config_data.get("initial_location_definitions", {})
    logger.debug(f"[CreateBlankState] Raw initial_locs from world_config_data: {json.dumps(initial_locs)}")
    # Ensure the nested structure for location_details exists
    state.setdefault(WORLD_STATE_KEY, {}).setdefault(LOCATION_DETAILS_KEY, {})

    if initial_locs:
        for loc_id, loc_data in initial_locs.items():
            state[WORLD_STATE_KEY][LOCATION_DETAILS_KEY][loc_id] = copy.deepcopy(loc_data)
            logger.info(f"Initialized location '{loc_id}' from world_config into state.")
            logger.debug(f"[CreateBlankState] Copied loc_data for {loc_id}: {json.dumps(loc_data)}")
    else:
        logger.warning(f"[CreateBlankState] No 'initial_location_definitions' found or empty in world_config_data.")

    # Ensure Home_01 has a basic entry if not in initial_location_definitions
    if DEFAULT_HOME_LOCATION_NAME not in state[WORLD_STATE_KEY][LOCATION_DETAILS_KEY]:
        state[WORLD_STATE_KEY][LOCATION_DETAILS_KEY][DEFAULT_HOME_LOCATION_NAME] = {
            "id": DEFAULT_HOME_LOCATION_NAME, "name": "Home", 
            "description": DEFAULT_HOME_DESCRIPTION, "ambient_sound_description": "The quiet hum of household appliances.",
            "ephemeral_objects": [], "ephemeral_npcs": [], "connected_locations": []
        }
        logger.info(f"Initialized default '{DEFAULT_HOME_LOCATION_NAME}' in state as it was missing from world_config.")

    # For "real" and "realtime" simulations, anchor the narrative start time to the real world
    # using the timestamp from when the world_config was created.
    if state["world_template_details"].get("world_type") == "real" and \
       state["world_template_details"].get("sub_genre") == "realtime":
        # world_config_data is the raw dict from the world_config_*.json file
        setup_time = world_config_data.get("setup_timestamp_utc")
        if setup_time:
            state["world_template_details"]["real_world_start_utc"] = setup_time
            logger.info(f"Set real_world_start_utc from world_config setup_timestamp_utc: {setup_time}")
        else:
            # Fallback if setup_timestamp_utc is somehow missing from world_config_data
            current_time_iso = datetime.now(timezone.utc).isoformat()
            state["world_template_details"]["real_world_start_utc"] = current_time_iso
            logger.warning(f"world_config_data missing 'setup_timestamp_utc'. Using current time as real_world_start_utc: {current_time_iso}")

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
        "objects": [],
        WORLD_STATE_KEY: {} # Add basic check for current_world_state
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

    # Ensure current_world_state and its sub-keys (like location_details, world_feeds)
    if WORLD_STATE_KEY not in state or not isinstance(state[WORLD_STATE_KEY], dict):
        state[WORLD_STATE_KEY] = copy.deepcopy(NEW_SIMULATION_STATE_TEMPLATE[WORLD_STATE_KEY])
        modified = True
    current_world_state_template = NEW_SIMULATION_STATE_TEMPLATE[WORLD_STATE_KEY]
    for key, default_value in current_world_state_template.items():
        if key not in state[WORLD_STATE_KEY]:
            state[WORLD_STATE_KEY][key] = copy.deepcopy(default_value)
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

    # Sync initial_location_definitions
    config_initial_locs = world_config.get("initial_location_definitions", {})
    state_loc_details_path_tuple = (WORLD_STATE_KEY, LOCATION_DETAILS_KEY)

    # Ensure the path exists in state's current_world_state
    state_location_details = get_nested(state, *state_loc_details_path_tuple)
    if state_location_details is None:
        # If the entire location_details dictionary is missing, initialize it.
        # This is a bit defensive; ensure_state_structure should ideally handle this.
        state.setdefault(WORLD_STATE_KEY, {})[LOCATION_DETAILS_KEY] = {}
        state_location_details = state[WORLD_STATE_KEY][LOCATION_DETAILS_KEY]
        modified = True
        logger.info(f"Initialized '{'.'.join(state_loc_details_path_tuple)}' in state during sync.")
    else:
        logger.debug(f"[SyncWorldConfig] State's current location details before sync: {json.dumps(state_location_details)}")

    locations_updated_or_added = False
    for loc_id, loc_data_from_config in config_initial_locs.items():
        if loc_id not in state_location_details:
            # Location is new in config, add it fully to state
            state_location_details[loc_id] = copy.deepcopy(loc_data_from_config)
            # Ensure essential dynamic keys exist if not in config template
            for key_to_ensure, default_val in [
                ("ephemeral_objects", []), ("ephemeral_npcs", []),
                ("connected_locations", []), ("objects_present", [])]:
                if key_to_ensure not in state_location_details[loc_id]:
                    state_location_details[loc_id][key_to_ensure] = default_val
            locations_updated_or_added = True
            logger.info(f"[SyncWorldConfig] Added new location '{loc_id}' from config to state.")
        else:
            # Location exists in state, update only specific static fields if they differ
            # Preserve dynamic fields like ephemeral_objects, connected_locations, objects_present from state
            state_loc_entry = state_location_details[loc_id]
            for static_field in ["name", "description", "ambient_sound_description"]: # Add other static fields if any
                if static_field in loc_data_from_config and \
                   state_loc_entry.get(static_field) != loc_data_from_config[static_field]:
                    state_loc_entry[static_field] = loc_data_from_config[static_field]
                    locations_updated_or_added = True
                    logger.info(f"[SyncWorldConfig] Updated static field '{static_field}' for location '{loc_id}' from config.")
            # Ensure essential dynamic keys exist in the state entry if they were somehow missing
            for key_to_ensure, default_val in [("ephemeral_objects", []), ("ephemeral_npcs", []), ("connected_locations", []), ("objects_present", [])]:
                if key_to_ensure not in state_loc_entry:
                    state_loc_entry[key_to_ensure] = default_val
                    locations_updated_or_added = True
                    logger.info(f"[SyncWorldConfig] Ensured dynamic key '{key_to_ensure}' for existing location '{loc_id}'.")

    if locations_updated_or_added:
        # No need to call _update_state_value here as we are modifying state_location_details directly,
        # which is a reference to the nested dict in 'state'.
        modified = True
        logger.info("State's 'current_world_state.location_details' were updated/augmented from world_config.initial_location_definitions.")

    logger.debug(f"[SyncWorldConfig] State's location details AFTER sync attempt: {json.dumps(get_nested(state, *state_loc_details_path_tuple))}")

    # Fallback: Ensure Home_01 exists if it's somehow still missing after sync
    if DEFAULT_HOME_LOCATION_NAME not in get_nested(state, *state_loc_details_path_tuple, default={}):
        home_loc_data = { "id": DEFAULT_HOME_LOCATION_NAME, "name": "Home", "description": DEFAULT_HOME_DESCRIPTION, "ephemeral_objects": [], "ephemeral_npcs": [], "connected_locations": [] }
        # Add default ambient sound for Home_01 if it's being created here
        if "ambient_sound_description" not in home_loc_data: # Check if it wasn't already set by a more specific config
            home_loc_data["ambient_sound_description"] = "The gentle hum of domestic life."
        state[WORLD_STATE_KEY][LOCATION_DETAILS_KEY][DEFAULT_HOME_LOCATION_NAME] = home_loc_data
        modified = True
        logger.info(f"Ensured default '{DEFAULT_HOME_LOCATION_NAME}' exists in state's location_details during sync.")
    return modified


def find_latest_file(pattern: str) -> str | None:
    """Finds the most recently modified file matching the pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

# --- Refactored Load/Initialize Logic ---

def _determine_target_uuid_and_load_world_config(
    instance_uuid_arg: Optional[str]
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Determines the target UUID and loads the corresponding world_config.json.
    Returns (target_uuid, world_config_data, world_config_path).
    """
    logger.debug(f"Determining target UUID. Argument: {instance_uuid_arg}")
    target_uuid = None
    world_config = None
    world_config_path = None

    if instance_uuid_arg:
        logger.info(f"Attempting to load simulation for specified UUID: {instance_uuid_arg}")
        world_config_path = os.path.join(WORLD_CONFIG_DIR, f"world_config_{instance_uuid_arg}.json")
        world_config = load_json_file(world_config_path)
        if world_config and world_config.get("world_instance_uuid") == instance_uuid_arg:
            target_uuid = instance_uuid_arg
        else:
            logger.error(f"World config for UUID {instance_uuid_arg} not found or UUID mismatch. Cannot proceed with this instance.")
            return None, None, None
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
                return None, None, None # Ensure we don't proceed with invalid latest config
        else:
            logger.info("No existing world_config files found. A new simulation will need to be set up first (e.g., via setup_simulation.py).")
            return None, None, None

    if not target_uuid or not world_config: # Double check after all paths
        logger.error("Failed to determine a target UUID or load its world configuration.")
        return None, None, None
    
    return target_uuid, world_config, world_config_path


def _load_life_summaries(target_uuid: str) -> List[Dict[str, Any]]:
    """Loads all life summary files for a given target_uuid."""
    if not target_uuid:
        logger.error("Cannot load life summaries: target_uuid is None.")
        return []
        
    life_summary_pattern = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_*_{target_uuid}.json")
    logger.debug(f"Searching for life summaries with pattern: {life_summary_pattern}")

    life_summary_files = glob.glob(life_summary_pattern)
    life_summaries = []
    if life_summary_files:
        for ls_file in life_summary_files:
            ls_data = load_json_file(ls_file)
            if ls_data is None:
                logger.warning(f"load_json_file returned None for: {os.path.basename(ls_file)}")
                continue
            
            sim_id_from_file = ls_data.get("sim_id")
            logger.debug(f"Processing life summary file: {os.path.basename(ls_file)}. Found sim_id: '{sim_id_from_file}'")

            if sim_id_from_file: 
                life_summaries.append(ls_data)
                logger.info(f"Loaded life summary: {os.path.basename(ls_file)} for sim_id: {sim_id_from_file}")
            else:
                logger.warning(f"Could not load or validate life summary (missing 'sim_id'): {os.path.basename(ls_file)}")
    else:
        logger.warning(f"No life summary files found for UUID {target_uuid} matching pattern {life_summary_pattern}.")
    return life_summaries


def _load_or_create_simulation_state_file(
    state_file_path_to_check: str,
    target_uuid: str,
    world_config_data: Optional[Dict[str, Any]],
    life_summaries_data: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Attempts to load an existing state file or creates a new one if necessary.
    Returns (state_data, created_new_state_flag).
    """
    logger.debug(f"Attempting to load or create state file: {state_file_path_to_check} for UUID: {target_uuid}")
    # state_file_path is already constructed and passed as state_file_path_to_check
    loaded_state_data = load_json_file(state_file_path_to_check)
    created_new_state = False

    if loaded_state_data is None and os.path.exists(state_file_path_to_check):
        logger.warning(f"Existing state file at {state_file_path_to_check} could not be loaded/parsed. Will create a new one.")
        created_new_state = True
    elif not os.path.exists(state_file_path_to_check):
        logger.info(f"No existing state file found at {state_file_path_to_check}. Will create a new one.")
        loaded_state_data = None # Ensure it's None to trigger creation
        created_new_state = True
    elif loaded_state_data.get("world_instance_uuid") != target_uuid:
        logger.warning(f"State file UUID ({loaded_state_data.get('world_instance_uuid')}) "
                       f"does not match target UUID ({target_uuid}). Creating new state.")
        loaded_state_data = None
        created_new_state = True

    if loaded_state_data is None:
        if not world_config_data:
            logger.error(f"Cannot create new simulation state: World config for UUID {target_uuid} not loaded/found.")
            return None, True
        if not life_summaries_data:
            logger.error(f"Cannot create new simulation state for UUID {target_uuid}: No life summaries found. At least one simulacrum is required.")
            return None, True

        logger.info(f"Initializing new simulation state for UUID: {target_uuid}")
        loaded_state_data = create_blank_simulation_state(target_uuid, world_config_data, life_summaries_data)
        # created_new_state is already True if we reach here
    return loaded_state_data, created_new_state


def load_or_initialize_simulation(instance_uuid_arg: str | None) -> tuple[dict | None, str]:
    """
    Loads an existing simulation state or initializes a new one.
    Returns a tuple: (loaded_state_data, state_file_path)
    """
    state_modified_during_load = False
    state_file_path = "" # Initialize

    # 1. Determine target_uuid and load world_config
    target_uuid, world_config, _ = _determine_target_uuid_and_load_world_config(instance_uuid_arg)

    if not target_uuid or not world_config:
        logger.error("Failed to determine target UUID or load world config. Aborting simulation load.")
        return None, ""
    
    state_file_path = os.path.join(STATE_DIR, f"simulation_state_{target_uuid}.json")

    # 2. Load Life Summaries for the target_uuid
    life_summaries = _load_life_summaries(target_uuid)

    # 3. Attempt to load existing simulation_state.json or create a new one
    loaded_state_data, created_new_state = _load_or_create_simulation_state_file(
        state_file_path, target_uuid, world_config, life_summaries
    )
    state_modified_during_load = state_modified_during_load or created_new_state

    # 4. Ensure state structure and sync with world_config if state was loaded/created
    active_sim_ids_from_summaries = [ls.get("sim_id") for ls in life_summaries if ls.get("sim_id")]
    
    if loaded_state_data:
        structure_modified = ensure_state_structure(loaded_state_data, target_uuid, active_sim_ids_from_summaries)
        state_modified_during_load = state_modified_during_load or structure_modified

        # world_config is guaranteed to be non-None if target_uuid is set
        state_modified_by_sync = sync_world_config_to_state(loaded_state_data, world_config)
        state_modified_during_load = state_modified_during_load or state_modified_by_sync
    else:
        logger.error(f"Failed to load or create a simulation state for UUID {target_uuid}.")
        return None, state_file_path

    # 5. Save if modified during load (e.g., new state, structure ensured, or synced)
    if state_modified_during_load and loaded_state_data:
        logger.info(f"Simulation state for {target_uuid} was modified during load/initialization. Saving changes to {state_file_path}.")
        save_json_file(state_file_path, loaded_state_data)
    # No need for 'elif created_new_state' because state_modified_during_load will be true if created_new_state is true.

    return loaded_state_data, state_file_path

def parse_json_output_last(text: str) -> Optional[Dict[str, Any]]:
    """
    Try direct JSON parse first, then fallback to robust extraction and cleanup.
    """
    if not text or not text.strip():
        return None

    # Try direct parse (strip markdown fence if present)
    # Improved regex pattern to better handle code blocks with or without language specifier
    fence_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    fence_match = re.search(fence_pattern, text, re.DOTALL)
    
    # More robust handling of the match
    if fence_match:
        try:
            json_text = fence_match.group(1).strip()
            logger.debug(f"Found JSON in code block: {json_text[:50]}...")
        except (IndexError, AttributeError):
            logger.debug("Code block regex matched but couldn't extract content")
            json_text = text.strip()
    else:
        json_text = text.strip()
        logger.debug("No code block found, using entire text")

    try:
        # Try direct parse
        return json.loads(json_text)
    except Exception as e:
        logger.debug(f"Direct json.loads failed: {e}. Falling back to robust cleanup.")

    def clean_json_text(json_str: str) -> str:
        """Clean up common JSON formatting issues"""
        json_str = json_str.strip()
        # Fix the specific pattern: ] \n , \n "key" -> ], \n "key"
        json_str = re.sub(r']\s*\n\s*,\s*\n\s*"', '],\n    "', json_str)
        # Fix standalone commas on their own lines
        json_str = re.sub(r']\s*\n\s*,\s*\n', '],\n', json_str)
        # Fix missing commas between array elements and object properties
        json_str = re.sub(r'}\s*\n\s*,\s*\n\s*"', '},\n    "', json_str)

        # --- NEW: Fix for missing comma between a closing brace } and a new key " ---
        # e.g. "results": { ... } "discovered_connections": [ ... ]
        # becomes "results": { ... }, "discovered_connections": [ ... ]
        json_str = re.sub(r'(})\s*(")', r'\1,\2', json_str)
        # --- END NEW ---

        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        # Fix common LLM mistake: array closed with ] instead of }]
        json_str = re.sub(r'(\{\s*"[^"]+":\s*".*?"\s*\})\s*\]', r'\1}]', json_str)
        # Remove trailing commas at the end of objects/arrays
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        # --- NEW: Fix for arrays of objects not closed with '}]' ---
        # This targets the exact pattern in your log: ...}, ...}, ...] (should be ...}, ...}, ...}]
        json_str = re.sub(
            r'(\{\s*"[^"]+":\s*".*?"(?:,.*?)*\})\s*\]',
            lambda m: m.group(1) + '}]' if not m.group(1).endswith('}]') else m.group(0),
            json_str,
            flags=re.DOTALL
        )
        return json_str

    cleaned_json = clean_json_text(json_text)

    # Try parsing with cleanup
    try:
        logger.debug("Initial JSON parsing attempt...")
        result = json.loads(cleaned_json)
        logger.debug("JSON parsing successful")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"Initial JSON parsing attempt failed: {e}. Trying with fixups...")

    # Try more aggressive fixes for quote issues
    try:
        fixed_json = cleaned_json.replace('\\"', '"').replace('"', '\\"')
        fixed_json = re.sub(r'\\"([^"]*)\\":', r'"\1":', fixed_json)
        fixed_json = re.sub(r':\s*\\"([^"]*)\\"', r': "\1"', fixed_json)
        result = json.loads(fixed_json)
        logger.debug("JSON parsing successful after quote fixes")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"Second JSON parsing attempt (after quote fixes) failed: {e}")

    # Try extracting just the JSON object bounds
    try:
        start_idx = cleaned_json.find('{')
        if start_idx == -1:
            raise ValueError("No opening brace found")
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(cleaned_json)):
            if cleaned_json[i] == '{':
                brace_count += 1
            elif cleaned_json[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        if end_idx == -1:
            raise ValueError("No matching closing brace found")
        bounded_json = cleaned_json[start_idx:end_idx + 1]
        bounded_json = clean_json_text(bounded_json)
        result = json.loads(bounded_json)
        logger.debug("JSON parsing successful after brace bounding")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"Third JSON parsing attempt (brace bounding) failed: {e}")

    # Final brute-force fix: try to close any open arrays/objects at the end
    try:
        # If the last bracket is a ']', but the next char should be '}', add it
        if cleaned_json.count('{') > cleaned_json.count('}'):
            cleaned_json += '}'
        if cleaned_json.count('[') > cleaned_json.count(']'):
            cleaned_json += ']'
        # NEW: Try to balance braces/brackets more aggressively
        while cleaned_json.count('{') > cleaned_json.count('}'):
            cleaned_json += '}'
        while cleaned_json.count('[') > cleaned_json.count(']'):
            cleaned_json += ']'
        result = json.loads(cleaned_json)
        logger.debug("JSON parsing successful after brute-force closing")
        return result
    except Exception as e:
        logger.debug(f"Brute-force closing failed: {e}")

    # Final fallback: Try using demjson3 if available (much more lenient parser)
    try:
        import demjson3
        result = demjson3.decode(json_text)
        logger.debug("JSON parsing successful with demjson3")
        return result
    except ImportError:
        logger.debug("demjson3 not installed, skipping this fallback option")
    except Exception as e:
        logger.debug(f"demjson3 parsing failed: {e}")

    logger.warning(f"All JSON parsing attempts failed. Original: {json_text[:100]}...")
    return None

def get_nested(data: Dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve nested dictionary values."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list): 
            if isinstance(key, int):
                try:
                    current = current[key]
                except IndexError:
                    return default
            else: 
                return default
        else: 
            return default
        
        if current is None: 
            return default
            
    return current
