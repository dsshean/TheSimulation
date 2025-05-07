from pydantic import BaseModel, Field, FilePath, validator, ValidationInfo
from typing import List, Optional, Dict, Any # Removed FilePath as it's not used
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
    # Add other fields from the 'persona_details' section of life_summary.json
    class Config:
        extra = 'allow' # Allow extra fields not explicitly defined

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
    # Allow other fields that might exist in the JSON
    class Config:
        extra = 'allow'

# --- Overall Simulation State ---

class SimulationState(BaseModel):
    world_config: WorldConfig
    simulacra: List[SimulacraState] = Field(default_factory=list)
    # Add other dynamic state elements if needed (e.g., current_time)
    current_simulation_time: Optional[datetime] = None

# --- Added from simulation_async.py ---
class WorldEngineResponse(BaseModel):
    valid_action: bool
    duration: float = Field(ge=0.0)
    results: Dict[str, Any] = Field(default_factory=dict)
    outcome_description: str

    @validator('duration') # Pydantic v1 style validator
    @classmethod
    def duration_must_be_zero_if_invalid(cls, v: float, values: Dict[str, Any]): # Changed info to values for Pydantic v1
        if 'valid_action' in values and not values['valid_action'] and v != 0.0:
            # logger.warning(f"Invalid action returned non-zero duration ({v}). Forcing to 0.0.") # Logger not available here easily
            return 0.0
        return v

    @validator('results') # Pydantic v1 style validator
    @classmethod
    def results_must_be_empty_if_invalid(cls, v: Dict, values: Dict[str, Any]): # Changed info to values for Pydantic v1
        if 'valid_action' in values and not values['valid_action'] and v:
            # logger.warning(f"Invalid action returned non-empty results ({v}). Forcing to empty dict.") # Logger not available here easily
            return {}
        return v

class SimulacraIntentResponse(BaseModel):
    internal_monologue: str
    action_type: str
    target_id: Optional[str] = None
    details: str = ""