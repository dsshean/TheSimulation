# src/perception_manager.py
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .config import (CURRENT_LOCATION_KEY, LOCATION_DETAILS_KEY,
                     SIMULACRA_KEY, WORLD_STATE_KEY)
from .loop_utils import get_nested

logger = logging.getLogger(__name__)

class SynchronizedPerceptionManager:
    """
    Thread-safe perception manager that provides consistent views of the world state.
    Prevents race conditions between state updates and perception gathering.
    """
    def __init__(self, global_state_ref: Dict[str, Any], state_manager=None):
        self.state = global_state_ref  # Reference to the main state dictionary
        self.state_manager = state_manager  # Optional StateManager for locking
        self._perception_cache = {}
        self._cache_timeout = 0.1  # Cache valid for 100ms
        
    async def get_fresh_percepts(self, perceiving_sim_id: str) -> Dict[str, Any]:
        """
        Get fresh perception data with proper synchronization.
        """
        if self.state_manager:
            # Use state manager's lock for consistency
            async with self.state_manager._lock:
                return self._build_percepts(perceiving_sim_id)
        else:
            # Fallback to direct perception building
            return self._build_percepts(perceiving_sim_id)
    
    async def get_cached_percepts(self, perceiving_sim_id: str) -> Dict[str, Any]:
        """
        Get perception data with short-term caching to reduce state access.
        """
        current_time = time.time()
        cache_key = perceiving_sim_id
        
        # Check if cached data is still valid
        if cache_key in self._perception_cache:
            cached_data, cache_time = self._perception_cache[cache_key]
            if current_time - cache_time < self._cache_timeout:
                logger.debug(f"[PerceptionManager] Using cached percepts for {perceiving_sim_id}")
                return cached_data
        
        # Get fresh data and cache it
        fresh_percepts = await self.get_fresh_percepts(perceiving_sim_id)
        self._perception_cache[cache_key] = (fresh_percepts, current_time)
        
        return fresh_percepts
    
    def _build_percepts(self, perceiving_sim_id: str) -> Dict[str, Any]:
        """
        Build perception data from current state (internal method).
        """
        # Get state version for consistency checking
        state_version = self.state.get("state_version", 0)
        
        perceiving_sim_data = get_nested(self.state, SIMULACRA_KEY, perceiving_sim_id)
        if not perceiving_sim_data:
            logger.warning(f"[PerceptionManager] Perceiving simulacrum '{perceiving_sim_id}' not found in state.")
            return {
                "error": f"Perceiving simulacrum '{perceiving_sim_id}' not found.",
                "state_version": state_version
            }

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
                "recently_departed_simulacra": [],
                "audible_events": [],
                "error": "Perceiving simulacrum has no current location.",
                "state_version": state_version
            }

        percepts: Dict[str, Any] = {
            "current_location_id": current_location_id,
            "location_description": get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "description", default="An undescribed location."),
            "visible_simulacra": [],
            "visible_static_objects": [],
            "visible_ephemeral_objects": get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "ephemeral_objects", default=[]),
            "visible_npcs": get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "ephemeral_npcs", default=[]),
            "recently_departed_simulacra": [],
            "audible_events": [],
            "state_version": state_version,
            "perception_timestamp": time.time()
        }

        # Build list of visible simulacra with enhanced status information
        all_simulacra = get_nested(self.state, SIMULACRA_KEY, default={})
        for sim_id, sim_data in all_simulacra.items():
            if sim_id == perceiving_sim_id:
                continue # Skip self

            other_sim_current_loc = sim_data.get(CURRENT_LOCATION_KEY)
            other_sim_previous_loc = sim_data.get("previous_location_id")
            other_sim_status = sim_data.get("status", "unknown")
            other_sim_name = get_nested(sim_data, "persona_details", "Name", default=sim_id)

            if other_sim_current_loc == current_location_id:
                # This simulacrum is in the same location
                visible_sim = {
                    "id": sim_id,
                    "name": other_sim_name,
                    "status": other_sim_status
                }
                
                # Add current action information if available
                if other_sim_status == "busy":
                    current_action = sim_data.get("current_action_description", "doing something")
                    visible_sim["current_action"] = current_action
                    
                    # Add estimated completion time
                    action_end_time = sim_data.get("current_action_end_time")
                    current_sim_time = self.state.get("world_time", 0.0)
                    if action_end_time and action_end_time > current_sim_time:
                        visible_sim["action_time_remaining"] = action_end_time - current_sim_time
                
                percepts["visible_simulacra"].append(visible_sim)
                
            elif other_sim_previous_loc == current_location_id and other_sim_current_loc != current_location_id:
                # This simulacrum was here but has moved to a different location
                percepts["recently_departed_simulacra"].append({
                    "id": sim_id,
                    "name": other_sim_name,
                    "departed_to_location_id": other_sim_current_loc
                })

        # Add static objects
        for obj_data in self.state.get("objects", []):
            if isinstance(obj_data, dict) and obj_data.get("location") == current_location_id:
                percepts["visible_static_objects"].append(obj_data)

        # Add ambient sounds
        ambient_sound_desc = get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "ambient_sound_description")
        if ambient_sound_desc:
            percepts["audible_events"].append({
                "source_id": "environment", 
                "description": ambient_sound_desc, 
                "type": "ambient"
            })

        logger.debug(f"[PerceptionManager] Built percepts for {perceiving_sim_id} at {current_location_id}: "
                    f"{len(percepts['visible_simulacra'])} visible agents, "
                    f"{len(percepts['recently_departed_simulacra'])} recently departed")

        return percepts
    
    def get_percepts_for_simulacrum(self, perceiving_sim_id: str) -> Dict[str, Any]:
        """
        Synchronous version for backward compatibility.
        Generates a structured perceptual package for a given simulacrum.
        """
        return self._build_percepts(perceiving_sim_id)
    
    def invalidate_cache(self, agent_id: Optional[str] = None):
        """
        Invalidate perception cache for specific agent or all agents.
        """
        if agent_id:
            self._perception_cache.pop(agent_id, None)
        else:
            self._perception_cache.clear()


# Backward compatibility - keep original class as alias
class PerceptionManager(SynchronizedPerceptionManager):
    """
    Legacy PerceptionManager class for backward compatibility.
    """
    def __init__(self, global_state_ref: Dict[str, Any]):
        super().__init__(global_state_ref)
        
    def get_percepts_for_simulacrum(self, perceiving_sim_id: str) -> Dict[str, Any]:
        """
        Legacy method - synchronous perception gathering.
        Generates a structured perceptual package for a given simulacrum.
        """
        return self._build_percepts(perceiving_sim_id)