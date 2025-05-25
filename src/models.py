from pydantic import BaseModel, Field, field_validator, ValidationInfo, ConfigDict
from typing import List, Optional, Dict, Any, Union # ConfigDict was from previous attempt, not strictly needed for this one
from uuid import UUID
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

# --- Location and World Structures ---
# (Keeping these unchanged as they're well-structured)

class Coordinates(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class Location(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    coordinates: Coordinates = Field(default_factory=Coordinates)

class WorldRules(BaseModel):
    allow_teleportation: bool = False
    time_progression_rate: float = 1.0
    weather_effects_travel: bool = True
    historical_date: Optional[str] = None

class WorldConfig(BaseModel):
    world_instance_uuid: UUID
    world_type: str
    sub_genre: Optional[str] = None
    description: Optional[str] = None
    rules: WorldRules = Field(default_factory=WorldRules)
    location: Location = Field(default_factory=Location)
    setup_timestamp_utc: str

# --- Simulacra Structures ---

class PersonaDetails(BaseModel):
    Name: str = "Unknown"
    Age: Optional[int] = None
    Gender: Optional[str] = None
    Occupation: Optional[str] = None

    model_config = ConfigDict(extra='allow')

class SimulacraState(BaseModel):
    simulacra_id: str
    world_instance_uuid: UUID
    persona_details: PersonaDetails = Field(default_factory=PersonaDetails)
    current_mood: Optional[str] = "neutral"
    memories: List[Any] = Field(default_factory=list)
    goals: List[Any] = Field(default_factory=list)
    current_status: Optional[str] = None
    last_observation: Optional[str] = None
    current_location_id: str = "unknown_location"

    model_config = ConfigDict(extra='allow')

# --- Overall Simulation State ---

class SimulationState(BaseModel):
    world_config: WorldConfig
    simulacra: Dict[str, SimulacraState] = Field(default_factory=dict)  # Changed to dict for easier lookups
    current_simulation_time: float = 0.0  # Changed to float for easier time calculations
    last_real_update_time: Optional[float] = None  # Added for time tracking
    pending_actions: List[Dict[str, Any]] = Field(default_factory=list)  # For scheduled actions

# --- ADK Agent Output Models ---

class SimulacraIntentResponse(BaseModel):
    """Model for character agent output schema"""
    internal_monologue: str = Field(description="Character's thoughts, reasoning, and emotional responses")
    action_type: str = Field(description="Type of action being performed (move, use, look_around, etc.)")
    target_id: Optional[str] = Field(default=None, description="ID of the target object, NPC, or location")
    details: str = Field(default="", description="Additional details about the action")

class WorldEngineResponse(BaseModel):
    """Model for world engine agent output schema"""
    valid_action: bool = Field(description="Whether the action is valid and can be executed")
    duration: float = Field(default=0.0, ge=0.0, description="Duration of the action in simulation seconds.")
    results_str: str = Field(default="{}", description="A JSON string representing a dictionary of state changes (dot notation keys). Example: \"{}\" or \"{\\\"simulacra_profiles.sim_id.status\\\": \\\"idle\\\"}\".")
    outcome_description: str = Field(description="Objective description of what happened")
    scheduled_future_event_str: Optional[str] = Field(default=None, description="A JSON string representing a dictionary for a future event, or null if no event. Example: \"{\\\"event_type\\\": \\\"delivery\\\", ...}\" or null.")

    @property
    def results(self) -> Dict[str, Any]:
        try:
            return json.loads(self.results_str)
        except json.JSONDecodeError:
            logger.error(f"Error decoding results_str: {self.results_str}", exc_info=True)
            return {}

    @property
    def scheduled_future_event(self) -> Optional[Dict[str, Any]]:
        if self.scheduled_future_event_str is None:
            return None
        try:
            return json.loads(self.scheduled_future_event_str)
        except json.JSONDecodeError:
            logger.error(f"Error decoding scheduled_future_event_str: {self.scheduled_future_event_str}", exc_info=True)
            return None

    @field_validator('duration')
    @classmethod
    def duration_must_be_zero_if_invalid(cls, v: float, info: ValidationInfo):
        if 'valid_action' in info.data and not info.data['valid_action'] and v != 0.0:
            logger.debug(f"Action invalid, setting duration from {v} to 0.0")
            return 0.0
        return v

    @field_validator('results_str')
    @classmethod
    def results_str_must_be_empty_json_if_invalid(cls, v: str, info: ValidationInfo):
        if 'valid_action' in info.data and not info.data['valid_action'] and v != "{}":
            logger.debug(f"Action invalid, setting results_str from '{v}' to '{{}}'")
            return "{}"
        return v

# --- Narrator Output Models ---

class DiscoveredObject(BaseModel):
    id: str
    name: str
    description: str
    is_interactive: bool = Field(default=True)
    properties_str: Optional[str] = Field(default=None, description="A JSON string for additional properties (e.g., '{\"is_container\": true, \"is_openable\": false}'), or null if no properties.")

    @property
    def properties(self) -> Optional[Dict[str, Any]]:
        if self.properties_str is None:
            return None
        try:
            return json.loads(self.properties_str)
        except json.JSONDecodeError:
            logger.error(f"Error decoding DiscoveredObject properties_str: '{self.properties_str}'", exc_info=True)
            return None # Or return {} if an empty dict is preferred on error

class DiscoveredConnection(BaseModel):
    to_location_id_hint: str
    description: str
    travel_time_estimate_seconds: Optional[int] = None

class DiscoveredNPC(BaseModel):
    id: str
    name: str
    description: str
    is_interactive: bool = True

class NarrationResponse(BaseModel):
    """Model for narrator agent output schema"""
    narrative: str = Field(description="Stylized narrative description of the action outcome")
    discovered_objects: List[DiscoveredObject] = Field(default_factory=list, 
                                                  description="New objects discovered during look_around actions")
    discovered_connections: List[DiscoveredConnection] = Field(default_factory=list,
                                                         description="New connections discovered during look_around actions")
    discovered_npcs: List[DiscoveredNPC] = Field(default_factory=list,
                                            description="New NPCs discovered during look_around actions")

# --- Event and Message Models ---

class SimulationEvent(BaseModel):
    """Model for passing events between agents in the ADK system"""
    event_type: str
    source_agent: str
    target_agent: Optional[str] = None
    timestamp: float
    data: Dict[str, Any] = Field(default_factory=dict)

class SimulacraMessage(BaseModel):
    """Model for messages between simulacra"""
    from_simulacra_id: str
    to_simulacra_id: str
    content: str
    timestamp: float
    read: bool = False

# --- Session State Keys ---
# These constants define standardized keys for ADK session state

class StateKeys:
    """Constants for session state keys to maintain consistency"""
    WORLD_CONFIG = "world_config"
    CURRENT_SIM_TIME = "current_sim_time"
    LAST_REAL_UPDATE_TIME = "last_real_update_time"
    PENDING_ACTIONS = "pending_actions"
    SIMULATION_EVENTS = "simulation_events"
    # Keys for agent outputs or specific data points
    WORLD_ENGINE_RESULT = "world_engine_result"
    NARRATION_RESULT = "narration_result"
    NARRATION_INPUT_DATA = "narration_input_data" # For data passed to NarrationLLM
    CURRENT_ACTOR_ID = "current_actor_id" # For WorldResolutionPhase and NarrationPhase
    CURRENT_ACTOR_INTENT = "current_actor_intent" # For WorldResolutionPhase
    SIMULACRA_STATES = "simulacra_states"
    
    @staticmethod
    def simulacra_intent(simulacra_id: str) -> str:
        """Generate consistent state key for a simulacra's intent"""
        return f"simulacra_intent_{simulacra_id}"
    
    @staticmethod
    def simulacra_state(simulacra_id: str) -> str:
        """Generate consistent state key for a simulacra's state"""
        return f"simulacra_state_{simulacra_id}"
    
    @staticmethod
    def simulacra_observation(simulacra_id: str) -> str:
        """Generate consistent state key for a simulacra's last observation"""
        return f"simulacra_observation_{simulacra_id}"
    
class SearchResponseSchema(BaseModel):
    """Schema for search agent responses"""
    search_query: str = Field(description="The search query that was executed")
    search_results: List[str] = Field(description="List of search result summaries")
    answer: str = Field(description="Synthesized answer based on search results")
    sources: Optional[List[str]] = Field(default=None, description="Source URLs if available")