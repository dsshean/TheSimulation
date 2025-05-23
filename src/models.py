from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import List, Optional, Dict, Any, ClassVar, Union, Literal
from uuid import UUID
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# --- Location and World Structures ---

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
    historical_date: Optional[str] = None # Keep as string for now, consider datetime parsing later

class WorldConfig(BaseModel):
    world_instance_uuid: UUID
    world_type: str
    sub_genre: Optional[str] = None
    description: Optional[str] = None
    rules: WorldRules = Field(default_factory=WorldRules)
    location: Location = Field(default_factory=Location)
    setup_timestamp_utc: str # Keep as string for now

# --- Simulacra Structures ---
# Based on expected output from life_generator and potential state

class PersonaDetails(BaseModel):
    # Define fields based on what life_generator actually produces
    # Example fields:
    Name: str = "Unknown"
    Age: Optional[int] = None
    Gender: Optional[str] = None
    Occupation: Optional[str] = None
    
    model_config = ConfigDict(extra="allow") # Changed from "allow"

class SimulacraState(BaseModel):
    simulacra_id: str
    world_instance_uuid: UUID # Crucial for linking
    persona_details: PersonaDetails = Field(default_factory=PersonaDetails)
    current_mood: Optional[str] = "neutral" # Add mood field, default to neutral
    # Add other top-level fields from life_summary.json (goals, memories, status, etc.)
    # Use 'Any' or specific types/models if known
    memories: List[Any] = Field(default_factory=list)
    goals: List[Any] = Field(default_factory=list)
    current_status: Optional[str] = None
    
    model_config = ConfigDict(extra="allow") # Changed from "allow"

# --- Overall Simulation State ---

class SimulationState(BaseModel):
    world_config: WorldConfig
    simulacra: List[SimulacraState] = Field(default_factory=list)
    # Add other dynamic state elements if needed (e.g., current_time)
    current_simulation_time: Optional[datetime] = None

# --- Added from simulation_async.py ---
class ScheduledFutureEvent(BaseModel):
    event_type: str
    target_agent_id: Optional[str] = None
    location_id: str
    details: Dict[str, Any] = Field(default_factory=dict)
    estimated_delay_seconds: float

    model_config = ConfigDict(extra="allow")

class WorldEngineResponse(BaseModel):
    valid_action: bool
    duration: float = Field(ge=0.0)
    results: Dict[str, Any] = Field(default_factory=dict)  # Allow any values
    outcome_description: str
    scheduled_future_event: Optional[ScheduledFutureEvent] = None

    model_config = ConfigDict(extra="allow")

    # @model_validator(mode='after')
    # def check_duration_and_results_if_invalid(self) -> 'WorldEngineResponse':
    #     if not self.valid_action:
    #         if self.duration != 0.0:
    #             logger.debug(f"WorldEngineResponse: Invalid action returned non-zero duration ({self.duration}). Forcing to 0.0.")
    #             self.duration = 0.0
    #         if self.results:
    #             logger.debug(f"WorldEngineResponse: Invalid action returned non-empty results ({self.results}). Forcing to empty dict.")
    #             self.results = {}
    #     return self

class SimulacraIntentResponse(BaseModel):
    internal_monologue: str
    action_type: str
    target_id: Optional[str] = None
    details: str = ""

# --- Narrator Output Models (Added for Pydantic Validation) ---
class DiscoveredObject(BaseModel):
    id: str
    name: str
    description: str
    is_interactive: bool = True
    properties: Dict[str, Any] = Field(default_factory=dict)

class DiscoveredConnection(BaseModel):
    to_location_id_hint: str
    description: str
    travel_time_estimate_seconds: Optional[int] = None

class DiscoveredNPC(BaseModel):
    id: str
    name: str
    description: str
    is_interactive: bool = True # Based on prompt, though example output omits it. Making it default True.

class NarratorOutput(BaseModel):
    narrative: str
    discovered_objects: List[DiscoveredObject] = Field(default_factory=list)
    discovered_connections: List[DiscoveredConnection] = Field(default_factory=list)
    discovered_npcs: List[DiscoveredNPC] = Field(default_factory=list)