import os
import json
import uuid
import logging
import glob
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

# Assuming world_state_tools defines these keys
try:
    from .tools.world_state_tools import (
        WORLD_STATE_KEY,
        ACTIVE_SIMULACRA_IDS_KEY
    )
except ImportError:
    from tools.world_state_tools import (
         WORLD_STATE_KEY,
         ACTIVE_SIMULACRA_IDS_KEY
     )

logger = logging.getLogger(__name__)

# --- Constants for Subdirectories ---
STATE_DIR = os.path.join("data", "states")
STATE_FILE_PATTERN = os.path.join(STATE_DIR, "simulation_state_*.json")
LIFE_SUMMARY_DIR = os.path.join("data", "life_summaries") # Keep for reference if needed here

# --- File I/O Helpers ---

def load_json_file(path: str, default: Optional[Any] = None) -> Optional[Any]:
    """Loads JSON from a file, returning default if file not found or invalid."""
    if not os.path.exists(path):
        logger.debug(f"File not found: {path}. Returning default.")
        return default
    try:
        # Specify encoding for broader compatibility
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
        # Specify encoding and handle non-serializable types like datetime
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved data to {path}")
    except Exception as e:
        logger.error(f"Error saving file {path}: {e}")
        # Re-raise after logging so the caller knows saving failed
        raise

# --- Template and Instance Management ---

def load_world_template(template_path: str) -> Dict[str, Any]:
    """Loads the world template configuration file."""
    logger.info(f"Loading world template from: {template_path}")
    template_data = load_json_file(template_path)
    if template_data is None:
        logger.critical(f"World template file not found or invalid at {template_path}. Cannot proceed.")
        raise FileNotFoundError(f"World template not found or invalid: {template_path}")
    # Ensure the template itself doesn't have an instance UUID
    if "world_instance_uuid" in template_data and template_data["world_instance_uuid"]:
        logger.warning(f"Template file {template_path} contains a world_instance_uuid. This should ideally be null or absent in the template.")
        # Optionally remove it: del template_data["world_instance_uuid"]
    logger.info("World template loaded successfully.")
    return template_data

def find_latest_simulation_state_file() -> Optional[str]:
    """Finds the most recently modified simulation state file in STATE_DIR."""
    try:
        # Ensure the state directory exists before globbing
        os.makedirs(STATE_DIR, exist_ok=True)
        list_of_files = glob.glob(STATE_FILE_PATTERN)
        if not list_of_files:
            logger.info(f"No existing simulation state files found in {STATE_DIR}.")
            return None
        # Find the file with the latest modification time
        latest_file = max(list_of_files, key=os.path.getmtime)
        logger.info(f"Found latest simulation state file: {latest_file}")
        return latest_file
    except Exception as e:
        logger.error(f"Error finding latest state file in {STATE_DIR}: {e}")
        return None

def load_or_create_simulation_instance(
    world_template: Dict[str, Any],
    session_id: str,
    instance_uuid_to_load: Optional[str] = None
) -> Tuple[Dict[str, Any], str, str]:
    """
    Loads a specific simulation instance state or the latest one, or creates a new one.

    Args:
        world_template: The loaded world template data.
        session_id: The current session ID for logging/state update.
        instance_uuid_to_load: If provided, attempts to load this specific instance.

    Returns:
        A tuple containing: (simulation_state_dict, world_instance_uuid, state_file_path)

    Raises:
        FileNotFoundError: If a specified instance_uuid_to_load doesn't exist.
        RuntimeError: If loading/creation fails unexpectedly.
        IOError/Exception: If saving the new/updated state fails.
    """
    state_file_path = None
    world_instance_uuid = None
    state = None

    # --- Attempt to Load ---
    if instance_uuid_to_load:
        potential_path = os.path.join(STATE_DIR, f"simulation_state_{instance_uuid_to_load}.json")
        if os.path.exists(potential_path):
            logger.info(f"Attempting to load specified instance: {instance_uuid_to_load}")
            state = load_json_file(potential_path)
            # Validate the loaded state against the requested UUID
            if state and state.get("world_instance_uuid") == instance_uuid_to_load:
                state_file_path = potential_path
                world_instance_uuid = instance_uuid_to_load
                logger.info(f"Successfully loaded specified instance state from {state_file_path}")
            else:
                # File exists but is invalid or UUID mismatch - treat as error for specific load
                logger.error(f"Specified state file {potential_path} is invalid or has UUID mismatch (Expected: {instance_uuid_to_load}, Found: {state.get('world_instance_uuid')}).")
                raise ValueError(f"Invalid state file found for specified UUID: {instance_uuid_to_load}")
        else:
            # Specific UUID requested but file not found - this is an error
            logger.error(f"Specified state file {potential_path} not found for UUID: {instance_uuid_to_load}")
            raise FileNotFoundError(f"Simulation state for specified UUID not found: {instance_uuid_to_load}")
    else:
        # No specific UUID requested, find and attempt to load the latest existing state file
        latest_state_file = find_latest_simulation_state_file()
        if latest_state_file:
            logger.info(f"Attempting to load latest instance state: {latest_state_file}")
            state = load_json_file(latest_state_file)
            # Validate loaded state
            if state and state.get("world_instance_uuid"):
                world_instance_uuid = state.get("world_instance_uuid")
                state_file_path = latest_state_file
                logger.info(f"Successfully loaded latest instance state (UUID: {world_instance_uuid})")
            else:
                logger.warning(f"Latest state file {latest_state_file} invalid or missing UUID. Will create new.")
                state = None # Reset state to trigger creation

    # --- Create New If Loading Failed or Not Attempted ---
    if state is None:
        logger.info("Creating new simulation instance state.")
        world_instance_uuid = str(uuid.uuid4())
        state_file_path = os.path.join(STATE_DIR, f"simulation_state_{world_instance_uuid}.json")
        logger.info(f"New instance UUID: {world_instance_uuid}")
        logger.info(f"New state file path: {state_file_path}")

        # Copy relevant data from the template
        initial_location_details = {}
        wc_location = world_template.get("location")
        if isinstance(wc_location, dict):
            city = wc_location.get("city", "UnknownCity")
            state_code = wc_location.get("state", "UnknownState")
            location_key = f"{city}, {state_code}" # Key for easy lookup
            initial_location_details[location_key] = wc_location # Store full details
        else:
            logger.warning("Template 'location' key missing/invalid. Initial location_details empty.")

        wc_rules = world_template.get("rules", {})
        if not isinstance(wc_rules, dict):
            logger.warning("Template 'rules' key missing/invalid. Using empty rules.")
            wc_rules = {}

        # Create the initial state structure, embedding template info
        state = {
            "world_instance_uuid": world_instance_uuid,
            "session_id": session_id,
            # Embed template details at creation time for reference
            "world_template_details": {
                "world_type": world_template.get("world_type"),
                "sub_genre": world_template.get("sub_genre"),
                "description": world_template.get("description", "Default description."),
                "rules": wc_rules,
                "location": wc_location # Store the original location structure too
            },
            # Dynamic world state part
            WORLD_STATE_KEY: {
                "world_time": datetime.now(timezone.utc).isoformat(),
                "location_details": initial_location_details, # Processed location for easy access
                "global_events": [],
                # Keep these for compatibility/direct access if needed by agents
                "setting_description": world_template.get("description", "Default description."),
                "world_rules": wc_rules
            },
            ACTIVE_SIMULACRA_IDS_KEY: []
            # Add other top-level keys needed at initialization if any
        }
        logger.info("Initialized new simulation state structure.")
        # Save the newly created state immediately (save_json_file raises on error)
        save_json_file(state_file_path, state)

    else:
        # --- State Loaded Successfully ---
        # Ensure essential keys exist (optional, but good for robustness)
        state_changed = False
        if ACTIVE_SIMULACRA_IDS_KEY not in state:
            state[ACTIVE_SIMULACRA_IDS_KEY] = []
            logger.warning(f"Added missing '{ACTIVE_SIMULACRA_IDS_KEY}' key to loaded state.")
            state_changed = True
        if WORLD_STATE_KEY not in state:
            state[WORLD_STATE_KEY] = {} # Add basic structure if missing
            logger.warning(f"Added missing '{WORLD_STATE_KEY}' key structure to loaded state.")
            state_changed = True
        # Add more checks for sub-keys within WORLD_STATE_KEY if needed...

        # Always update the session ID for the current run
        if state.get("session_id") != session_id:
            state["session_id"] = session_id
            logger.info(f"Updated session ID to {session_id} in loaded state.")
            state_changed = True

        if state_changed:
            logger.info("Saving loaded state file due to updates (missing keys/session ID).")
            # Save the updated state (save_json_file raises on error)
            save_json_file(state_file_path, state)

    # Final check after load/create attempt
    if not state or not world_instance_uuid or not state_file_path:
         # This should not happen if logic is correct, but as a safeguard
         logger.critical("Failed to load or create a valid simulation instance state.")
         raise RuntimeError("Could not obtain a valid simulation state instance.")

    return state, world_instance_uuid, state_file_path


def create_life_summary_data(sim_id: str, world_uuid: str, persona_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structures the data for a life summary file.
    Ensures it links to the specific world instance UUID.
    """
    if not world_uuid:
        logger.error(f"Cannot create life summary for {sim_id}: world_instance_uuid is missing.")
        raise ValueError("Valid world_instance_uuid is required to create life summary.")
    if not persona_data:
         logger.warning(f"Creating life summary for {sim_id} with empty persona data.")
         persona_data = {} # Ensure it's a dict

    logger.info(f"Structuring life summary data for {sim_id} in world {world_uuid}")
    # Note: The actual saving of this data is handled elsewhere (e.g., main.py or life_generator.py)
    return {
      "simulacra_id": sim_id,
      "world_instance_uuid": world_uuid, # Link to the specific world instance
      "persona": persona_data, # Assuming persona_data is the dict generated by life_generator
      "generation_timestamp": datetime.now(timezone.utc).isoformat()
    }

def generate_unique_id(prefix: str = "id") -> str:
    """Generates a unique ID string with a prefix and a short UUID."""
    # Generate a UUID and take the first 8 characters for brevity
    unique_part = str(uuid.uuid4()).split('-')[0]
    return f"{prefix}_{unique_part}"