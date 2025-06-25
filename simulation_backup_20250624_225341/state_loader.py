import json
import os
import glob
from uuid import UUID
from typing import Optional, List
import logging
from pathlib import Path

from pydantic import ValidationError

# Use absolute import based on project structure
from src.models import WorldConfig, SimulacraState, SimulationState, Location, Coordinates # Remove parse_location_string from here

logger = logging.getLogger(__name__)
DATA_DIR = Path("data") # Use pathlib for better path handling

def find_latest_world_config() -> Optional[Path]:
    """Finds the most recently created world config file."""
    try:
        list_of_files = list(DATA_DIR.glob('world_config_*.json'))
        if not list_of_files:
            return None
        # Sort by creation time (or modification time as fallback)
        latest_file = max(list_of_files, key=lambda p: p.stat().st_ctime)
        return latest_file
    except Exception as e:
        logger.error(f"Error finding latest world config in {DATA_DIR}: {e}", exc_info=True)
        return None

def load_simulation_state(instance_uuid: Optional[UUID] = None) -> Optional[SimulationState]:
    """
    Loads the world configuration and associated simulacra data for a given instance UUID.
    If UUID is None, attempts to load the latest instance.
    """
    world_config_path = None
    if instance_uuid:
        world_config_path = DATA_DIR / f"world_config_{instance_uuid}.json"
        if not world_config_path.is_file():
             logger.error(f"World config file not found for UUID {instance_uuid} at {world_config_path}")
             return None
    else:
        logger.info("No instance UUID provided, attempting to load the latest.")
        world_config_path = find_latest_world_config()
        if not world_config_path:
            logger.error(f"Could not find any world config files to load in {DATA_DIR}.")
            return None
        logger.info(f"Loading latest world config: {world_config_path}")

    try:
        # --- Load World Config ---
        logger.debug(f"Reading world config from: {world_config_path}")
        with open(world_config_path, 'r', encoding='utf-8') as f:
            world_data = json.load(f)

        # Use Pydantic for parsing/validation
        world_config = WorldConfig.parse_obj(world_data)
        loaded_instance_uuid = world_config.world_instance_uuid # Get UUID from loaded file
        logger.info(f"Successfully parsed world config for UUID: {loaded_instance_uuid}")

        # --- Load Associated Simulacra ---
        simulacra_list: List[SimulacraState] = []
        # Assuming simulacra files are named like 'life_summary_sim_xxxxxx.json'
        # and contain 'world_instance_uuid' matching the loaded world.
        sim_files_pattern = DATA_DIR / 'life_summary_*.json'
        logger.debug(f"Searching for simulacra files matching: {sim_files_pattern}")

        for sim_file_path in DATA_DIR.glob('life_summary_*.json'):
            logger.debug(f"Checking simulacra file: {sim_file_path}")
            try:
                with open(sim_file_path, 'r', encoding='utf-8') as f:
                    sim_data = json.load(f)

                # Check if this simulacra belongs to the loaded world *before* parsing fully
                if sim_data.get("world_instance_uuid") == str(loaded_instance_uuid):
                     logger.debug(f"Found matching simulacra: {sim_file_path}")
                     # Parse with Pydantic
                     sim_state = SimulacraState.parse_obj(sim_data)
                     simulacra_list.append(sim_state)
                # else: logger.debug(f"Skipping {sim_file_path}, world UUID mismatch.")

            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON file: {sim_file_path}")
            except ValidationError as ve:
                 logger.warning(f"Skipping simulacra file due to validation error: {sim_file_path} - {ve}")
            except Exception as e:
                 logger.warning(f"Skipping simulacra file due to unexpected error ({type(e).__name__}): {sim_file_path} - {e}")

        logger.info(f"Loaded {len(simulacra_list)} associated simulacra for world {loaded_instance_uuid}.")

        # Combine into the main SimulationState object
        simulation_state = SimulationState(
            world_config=world_config,
            simulacra=simulacra_list
        )
        return simulation_state

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from world config {world_config_path}: {e}")
        return None
    except ValidationError as ve:
        logger.error(f"Failed to validate world config {world_config_path}: {ve}")
        return None
    except Exception as e:
        logger.error(f"Failed to load simulation state from {world_config_path}: {e}", exc_info=True)
        return None

# --- Helper Function (moved from setup_simulation) ---
# Keep it close to the Location model it populates

def parse_location_string(location_str: str) -> dict:
    """Attempts to parse 'City, State' or 'City, Country' into a dict for the Location model."""
    parts = [p.strip() for p in location_str.split(',')]
    location_data = {"city": None, "state": None, "country": None} # Don't include coordinates here

    if not location_str:
        return location_data # Return empty if input is empty

    if len(parts) == 1:
        # Assume it's a city or a region name if only one part
        location_data["city"] = parts[0]
        # Simple guesses for country based on city name
        if parts[0].lower() in ["london", "paris", "tokyo", "berlin"]:
             location_data["country"] = {"london": "United Kingdom", "paris": "France", "tokyo": "Japan", "berlin": "Germany"}[parts[0].lower()]
    elif len(parts) == 2:
        location_data["city"] = parts[0]
        # Basic check: if 2 letters, assume state (US/Canada), otherwise country
        if len(parts[1]) == 2 and parts[1].isalpha():
             location_data["state"] = parts[1].upper()
             location_data["country"] = "United States" # Default assumption
        else:
             location_data["country"] = parts[1]
    elif len(parts) >= 3: # Assume City, State, Country
        location_data["city"] = parts[0]
        location_data["state"] = parts[1].upper() # Assume state is second part
        location_data["country"] = ", ".join(parts[2:]) # Join remaining parts as country

    # Ensure city is set if only country was somehow derived (less likely now)
    if not location_data["city"] and location_str:
        location_data["city"] = location_str # Fallback

    logger.debug(f"Parsed location string '{location_str}' into: {location_data}")
    return location_data