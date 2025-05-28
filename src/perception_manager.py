# src/perception_manager.py
import logging
from typing import Any, Dict, List, Optional

from .config import (CURRENT_LOCATION_KEY, LOCATION_DETAILS_KEY,
                     SIMULACRA_KEY, WORLD_STATE_KEY)
from .loop_utils import get_nested

logger = logging.getLogger(__name__)

class PerceptionManager:
    """
    Manages what each simulacrum perceives in its environment based on the global state.
    This is primarily logic-based.
    """
    def __init__(self, global_state_ref: Dict[str, Any]):
        self.state = global_state_ref  # Reference to the main state dictionary

    def get_percepts_for_simulacrum(self, perceiving_sim_id: str) -> Dict[str, Any]:
        """
        Generates a structured perceptual package for a given simulacrum.
        """
        perceiving_sim_data = get_nested(self.state, SIMULACRA_KEY, perceiving_sim_id)
        if not perceiving_sim_data:
            logger.warning(f"[PerceptionManager] Perceiving simulacrum '{perceiving_sim_id}' not found in state.")
            return {"error": f"Perceiving simulacrum '{perceiving_sim_id}' not found."}

        current_location_id = perceiving_sim_data.get(CURRENT_LOCATION_KEY)
        if not current_location_id:
            logger.warning(f"[PerceptionManager] Perceiving simulacrum '{perceiving_sim_id}' has no current location.")
            return {
                "current_location_id": None,
                "location_description": "You are in an undefined space.",
                "visible_simulacra": [],
                "visible_static_objects": [],
                "visible_ephemeral_objects": [],
                "visible_npcs": [],
                "recently_departed_simulacra": [], # New field
                "audible_events": [],
                "error": "Perceiving simulacrum has no current location."
            }

        percepts: Dict[str, Any] = {
            "current_location_id": current_location_id,
            "location_description": get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "description", default="An undescribed location."),
            "visible_simulacra": [],
            "visible_static_objects": [],
            "visible_ephemeral_objects": get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "ephemeral_objects", default=[]),
            "visible_npcs": get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "ephemeral_npcs", default=[]),
            "recently_departed_simulacra": [], # New field
            "audible_events": [],
        }

        all_simulacra = get_nested(self.state, SIMULACRA_KEY, default={})
        for sim_id, sim_data in all_simulacra.items():
            if sim_id == perceiving_sim_id:
                continue # Skip self

            other_sim_current_loc = sim_data.get(CURRENT_LOCATION_KEY)
            other_sim_previous_loc = sim_data.get("previous_location_id")

            if other_sim_current_loc == current_location_id:
                # This simulacrum is in the same location
                percepts["visible_simulacra"].append({
                    "id": sim_id,
                    "name": get_nested(sim_data, "persona_details", "Name", default=sim_id),
                    "status": sim_data.get("status", "unknown")
                })
            elif other_sim_previous_loc == current_location_id and other_sim_current_loc != current_location_id:
                # This simulacrum was here but has moved to a different location
                percepts["recently_departed_simulacra"].append({
                    "id": sim_id,
                    "name": get_nested(sim_data, "persona_details", "Name", default=sim_id),
                    "departed_to_location_id": other_sim_current_loc # Their new current location
                })

        for obj_data in self.state.get("objects", []):
            if isinstance(obj_data, dict) and obj_data.get("location") == current_location_id:
                percepts["visible_static_objects"].append(obj_data) # Append the whole object data

        ambient_sound_desc = get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "ambient_sound_description")
        if ambient_sound_desc:
            percepts["audible_events"].append({"source_id": "environment", "description": ambient_sound_desc, "type": "ambient"})

        return percepts