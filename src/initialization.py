# c:\Users\dshea\Desktop\TheSimulation\src\initialization.py
# src/initialization.py

import os
import json
import uuid
import logging
import glob
import re # <<< Added re for UUID extraction fallback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

# --- State Key Imports ---
# Use a central location or define directly if simple
# Example: from .state_keys import WORLD_STATE_KEY, ACTIVE_SIMULACRA_IDS_KEY, LOCATION_DETAILS_KEY
WORLD_STATE_KEY = "current_world_state"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
LOCATION_DETAILS_KEY = "location_details" # <<< New Key Definition
SIMULACRA_PROFILES_KEY = "simulacra_profiles" # <<< Added for consistency
CURRENT_LOCATION_KEY = "current_location" # <<< Added for consistency
HOME_LOCATION_KEY = "home_location" # <<< Added for consistency
WORLD_TEMPLATE_DETAILS_KEY = "world_template_details" # <<< Added for consistency
LOCATION_KEY = "location" # <<< Added for consistency (from world template)

logger = logging.getLogger(__name__)

# Define default directories relative to the project root or a known base path
# Assuming this file is in src/, data/ is one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Gets the directory containing src/
STATE_DIR = os.path.join(BASE_DIR, "data", "states")
LIFE_SUMMARY_DIR = os.path.join(BASE_DIR, "data", "life_summaries")
# STATE_FILE_PATTERN is now constructed within find_latest_simulation_state_file

def load_json_file(path: str, default: Optional[Any] = None) -> Optional[Any]:
    """Loads JSON from a file, returning default if file not found or invalid."""
    if not os.path.exists(path):
        logger.debug(f"File not found: {path}. Returning default.")
        return default
    try:
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {path}. Returning default.")
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

def load_world_template(template_path: str) -> Optional[Dict[str, Any]]:
    """Loads the world template configuration file."""
    logger.info(f"Loading world template from: {template_path}")
    template_data = load_json_file(template_path)
    if template_data is None:
        logger.critical(f"World template file not found or invalid at {template_path}. Cannot proceed.")
        # Return None instead of raising FileNotFoundError immediately
        # Let the caller handle the missing template if necessary
        return None
    if template_data.get("world_instance_uuid"): # Check if key exists and has a value
        logger.warning(f"Template file {template_path} contains a world_instance_uuid. This should ideally be null or absent in the template.")
        # Consider removing it: del template_data["world_instance_uuid"]
    logger.info("World template loaded successfully.")
    return template_data

# --- <<< MODIFIED FUNCTION SIGNATURE >>> ---
def find_latest_simulation_state_file(state_dir: str = STATE_DIR) -> Optional[str]:
    """
    Finds the most recently modified simulation state file in the specified directory.

    Args:
        state_dir: The directory to search for state files. Defaults to STATE_DIR.

    Returns:
        The path to the latest state file, or None if none are found or an error occurs.
    """
    try:
        os.makedirs(state_dir, exist_ok=True)
        # Construct pattern using the provided state_dir
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
# --- <<< END MODIFICATION >>> ---


def load_or_create_simulation_instance(
    world_template: Dict[str, Any],
    session_id: str, # Keep session_id for logging/state update
    instance_uuid_to_load: Optional[str] = None,
    state_dir: str = STATE_DIR # Allow overriding state directory
) -> Tuple[Dict[str, Any], str, str]:
    """
    Loads a specific simulation instance state or the latest one, or creates a new one.
    Ensures the state contains essential keys like location_details.

    Args:
        world_template: The loaded world template data.
        session_id: The current session ID for logging/state update.
        instance_uuid_to_load: If provided, attempts to load this specific instance.
        state_dir: The directory where state files are stored.

    Returns:
        A tuple containing: (simulation_state_dict, world_instance_uuid, state_file_path)

    Raises:
        FileNotFoundError: If a specified instance_uuid_to_load doesn't exist.
        ValueError: If state file content is invalid or has UUID mismatch.
        RuntimeError: If loading/creation fails unexpectedly.
        IOError/Exception: If saving the new/updated state fails.
    """
    state_file_path = None
    world_instance_uuid = None
    state = None
    state_file_pattern = os.path.join(state_dir, "simulation_state_*.json") # Use passed state_dir

    if instance_uuid_to_load:
        potential_path = os.path.join(state_dir, f"simulation_state_{instance_uuid_to_load}.json")
        if os.path.exists(potential_path):
            logger.info(f"Attempting to load specified instance: {instance_uuid_to_load}")
            state = load_json_file(potential_path)
            if state and state.get("world_instance_uuid") == instance_uuid_to_load:
                state_file_path = potential_path
                world_instance_uuid = instance_uuid_to_load
                logger.info(f"Successfully loaded specified instance state from {state_file_path}")
            else:
                err_msg = f"Specified state file {potential_path} is invalid or has UUID mismatch (Expected: {instance_uuid_to_load}, Found: {state.get('world_instance_uuid') if state else 'None'})."
                logger.error(err_msg)
                raise ValueError(f"Invalid state file found for specified UUID: {instance_uuid_to_load}")
        else:
            logger.error(f"Specified state file {potential_path} not found for UUID: {instance_uuid_to_load}")
            raise FileNotFoundError(f"Simulation state for specified UUID not found: {instance_uuid_to_load}")
    else:
        # Find latest file using the state_dir
        try:
            # --- <<< CALL MODIFIED FUNCTION >>> ---
            latest_state_file = find_latest_simulation_state_file(state_dir)
            # --- <<< END CALL MODIFICATION >>> ---
            if latest_state_file:
                logger.info(f"Attempting to load latest instance state: {latest_state_file}")
                state = load_json_file(latest_state_file)
                # Extract UUID from filename as fallback if missing in content
                uuid_from_filename = None
                match = re.search(r"simulation_state_([a-f0-9\-]+)\.json", os.path.basename(latest_state_file))
                if match: uuid_from_filename = match.group(1)

                if state and state.get("world_instance_uuid"):
                    world_instance_uuid = state.get("world_instance_uuid")
                    if uuid_from_filename and world_instance_uuid != uuid_from_filename:
                         logger.warning(f"UUID mismatch between state content ({world_instance_uuid}) and filename ({uuid_from_filename}). Using content UUID.")
                    state_file_path = latest_state_file
                    logger.info(f"Successfully loaded latest instance state (UUID: {world_instance_uuid})")
                elif uuid_from_filename:
                     logger.warning(f"Latest state file {latest_state_file} missing UUID in content. Using UUID from filename: {uuid_from_filename}.")
                     world_instance_uuid = uuid_from_filename
                     state_file_path = latest_state_file
                     if state is None: state = {} # Ensure state is a dict if file was empty/invalid
                     state["world_instance_uuid"] = world_instance_uuid # Add missing UUID
                else:
                    logger.warning(f"Latest state file {latest_state_file} invalid or missing UUID in content and filename. Will create new.")
                    state = None # Reset state to trigger creation
            else:
                 logger.info(f"No existing simulation state files found in {state_dir}. Will create new.")
                 state = None # Trigger creation
        except Exception as e:
            logger.error(f"Error finding/loading latest state file in {state_dir}: {e}")
            state = None # Trigger creation on error

    # --- Create New State if Needed ---
    if state is None:
        logger.info("Creating new simulation instance state.")
        world_instance_uuid = str(uuid.uuid4())
        state_file_path = os.path.join(state_dir, f"simulation_state_{world_instance_uuid}.json")
        logger.info(f"New instance UUID: {world_instance_uuid}")
        logger.info(f"New state file path: {state_file_path}")

        # --- Initialize location_details ---
        initial_location_details = {}
        wc_location = world_template.get(LOCATION_KEY)
        if isinstance(wc_location, dict):
            city = wc_location.get("city", "UnknownCity")
            state_code = wc_location.get("state")
            country = wc_location.get("country", "UnknownCountry")

            location_key_parts = [part for part in [city, state_code, country] if part]
            location_key = ", ".join(location_key_parts) if location_key_parts else "default_location"

            initial_location_details[location_key] = {
                "name": location_key,
                "description": f"The primary starting location: {location_key}. Further details pending.",
                "objects_present": [],
                "connected_locations": [],
                "coordinates": wc_location.get("coordinates", {"latitude": None, "longitude": None})
            }
            logger.info(f"Initialized '{LOCATION_DETAILS_KEY}' with primary location: '{location_key}'")
        else:
            logger.warning(f"Template '{LOCATION_KEY}' key missing/invalid. Initial '{LOCATION_DETAILS_KEY}' empty.")

        wc_rules = world_template.get("rules", {})
        if not isinstance(wc_rules, dict):
            logger.warning("Template 'rules' key missing/invalid. Using empty rules.")
            wc_rules = {}

        # Create the initial state structure
        state = {
            "world_instance_uuid": world_instance_uuid,
            "session_id": session_id,
            WORLD_TEMPLATE_DETAILS_KEY: {
                "world_type": world_template.get("world_type"),
                "sub_genre": world_template.get("sub_genre"),
                "description": world_template.get("description", "Default description."),
                "rules": wc_rules,
                LOCATION_KEY: wc_location
            },
            WORLD_STATE_KEY: {
                "world_time": datetime.now(timezone.utc).isoformat(),
                LOCATION_DETAILS_KEY: initial_location_details,
                "global_events": [],
                "setting_description": world_template.get("description", "Default description."),
                "world_rules": wc_rules
            },
            ACTIVE_SIMULACRA_IDS_KEY: [],
            SIMULACRA_PROFILES_KEY: {}
        }
        logger.info("Initialized new simulation state structure.")
        save_json_file(state_file_path, state) # Save immediately

    # --- State Loaded Successfully ---
    else:
        state_changed = False
        # Ensure essential keys exist using the helper function
        state_changed = ensure_state_structure(state)

        # Always update the session ID
        if state.get("session_id") != session_id:
            state["session_id"] = session_id
            logger.info(f"Updated session ID to {session_id} in loaded state.")
            state_changed = True

        if state_changed:
            logger.info("Saving loaded state file due to updates (missing keys/session ID).")
            save_json_file(state_file_path, state)

    # Final check
    if not state or not world_instance_uuid or not state_file_path:
         logger.critical("Failed to load or create a valid simulation instance state.")
         raise RuntimeError("Could not obtain a valid simulation state instance.")

    return state, world_instance_uuid, state_file_path


def create_life_summary_data(sim_id: str, world_uuid: str, persona_data: Dict[str, Any]) -> Dict[str, Any]:
    """Structures the data for a life summary file."""
    if not world_uuid:
        logger.error(f"Cannot create life summary for {sim_id}: world_instance_uuid is missing.")
        raise ValueError("Valid world_instance_uuid is required to create life summary.")
    if not persona_data:
         logger.warning(f"Creating life summary for {sim_id} with empty persona data.")
         persona_data = {}

    logger.info(f"Structuring life summary data for {sim_id} in world {world_uuid}")
    persona_details_key = "persona_details"
    if persona_details_key not in persona_data:
        logger.warning(f"Persona data for {sim_id} missing '{persona_details_key}'. Wrapping existing data.")
        persona_data = {persona_details_key: persona_data}

    return {
      "simulacra_id": sim_id,
      "world_instance_uuid": world_uuid,
      **persona_data,
      "generation_timestamp": datetime.now(timezone.utc).isoformat()
    }

def generate_unique_id(prefix: str = "id") -> str:
    """Generates a unique ID string with a prefix and a short UUID."""
    unique_part = str(uuid.uuid4()).split('-')[0]
    return f"{prefix}_{unique_part}"

# --- Function to ensure state structure (call from main3.py or simulation_async.py) ---
def ensure_state_structure(state: Dict[str, Any]) -> bool:
    """
    Checks and adds missing essential keys/structures to a loaded state dictionary.

    Args:
        state: The simulation state dictionary.

    Returns:
        True if the state was modified, False otherwise.
    """
    modified = False
    if not isinstance(state, dict):
        logger.error("ensure_state_structure called with non-dict state.")
        return False # Cannot modify

    if ACTIVE_SIMULACRA_IDS_KEY not in state:
        state[ACTIVE_SIMULACRA_IDS_KEY] = []
        logger.warning(f"Added missing '{ACTIVE_SIMULACRA_IDS_KEY}' key.")
        modified = True

    if WORLD_STATE_KEY not in state:
        state[WORLD_STATE_KEY] = {}
        logger.warning(f"Added missing '{WORLD_STATE_KEY}' key structure.")
        modified = True

    world_state_dict = state.get(WORLD_STATE_KEY, {})
    if not isinstance(world_state_dict, dict): # Ensure it's a dict
        logger.warning(f"'{WORLD_STATE_KEY}' is not a dict. Resetting.")
        state[WORLD_STATE_KEY] = {}
        world_state_dict = state[WORLD_STATE_KEY]
        modified = True

    if LOCATION_DETAILS_KEY not in world_state_dict:
        world_state_dict[LOCATION_DETAILS_KEY] = {}
        logger.warning(f"Added missing '{LOCATION_DETAILS_KEY}' key to '{WORLD_STATE_KEY}'.")
        modified = True

    if SIMULACRA_PROFILES_KEY not in state:
        state[SIMULACRA_PROFILES_KEY] = {}
        logger.warning(f"Added missing '{SIMULACRA_PROFILES_KEY}' key.")
        modified = True

    # Ensure narrative log exists
    if "narrative_log" not in state:
        state["narrative_log"] = []
        logger.warning("Added missing 'narrative_log' key.")
        modified = True
    elif not isinstance(state["narrative_log"], list):
        state["narrative_log"] = [] # Ensure it's a list
        logger.warning("Corrected 'narrative_log' key to be a list.")
        modified = True

    # Ensure objects exists
    if "objects" not in state:
        state["objects"] = {}
        logger.warning("Added missing 'objects' key.")
        modified = True
    elif not isinstance(state["objects"], dict):
        state["objects"] = {} # Ensure it's a dict
        logger.warning("Corrected 'objects' key to be a dict.")
        modified = True

    # Ensure world_details exists
    if WORLD_TEMPLATE_DETAILS_KEY not in state:
        state[WORLD_TEMPLATE_DETAILS_KEY] = {"description": "Default", "rules": {}}
        logger.warning(f"Added missing '{WORLD_TEMPLATE_DETAILS_KEY}' key.")
        modified = True
    elif not isinstance(state[WORLD_TEMPLATE_DETAILS_KEY], dict):
        state[WORLD_TEMPLATE_DETAILS_KEY] = {"description": "Default", "rules": {}}
        logger.warning(f"Corrected '{WORLD_TEMPLATE_DETAILS_KEY}' key to be a dict.")
        modified = True

    return modified
