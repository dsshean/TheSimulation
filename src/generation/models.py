# src/generation/models.py
import calendar
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union
from datetime import date, datetime

# Use Pydantic v2 features
from pydantic import BaseModel, Field, conint, constr, model_validator, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Allowed verbs for Simulacra actions (can be expanded)
AllowedActionVerbs = Literal[
    "talk",         # Communication with another agent
    "move",         # Changing location (within env or to a new env)
    "wait",         # Explicitly doing nothing or pausing
    "observe",      # Focused perception of the environment or specific details
    "use",          # Interacting with an object (manipulating, consuming, operating)
    "think",        # Internal mental process (planning, analyzing, recalling)
    "search",       # Actively looking for an object, information, or person
    "read",         # Reading text from an object
    "write",        # Writing text onto an object or paper
    "rest",         # Physical resting or sleeping
    "get",          # Picking up or acquiring an item
    "drop",         # Letting go of or placing down an item
    "other"         # Any action not fitting the above categories
]

# --- Models for Life Generation ---

# Define constraint for summary length
ShortSummary = constr(min_length=5, max_length=5000) # Allow longer summaries

class Person(BaseModel):
    """Represents a person in the character's life."""
    name: str = Field(..., description="Full name of the person.")
    relationship: str = Field(..., description="Relationship to the main character (e.g., 'Mother', 'Father', 'Younger Brother', 'Best Friend', 'Spouse').")
    details: Optional[str] = Field(None, description="Brief relevant details (e.g., occupation, age relative to character, key personality trait).")

class InitialRelationshipsResponse(BaseModel):
    """Expected JSON response containing the initial family structure."""
    parents: List[Person] = Field(..., description="List of parents (usually 1 or 2).")
    siblings: List[Person] = Field(default_factory=list, description="List of siblings, if any.")

class PersonaDetailsResponse(BaseModel):
    """Defines the structure for a generated persona's basic details."""
    Name: str = Field(..., description="A plausible full name.")
    Age: conint(ge=1, le=120) = Field(..., description="A plausible current age between 1 and 120.")  # Min age 1
    Gender: Optional[str] = Field(None, description="The persona's gender (e.g., Male, Female, Non-binary, Other).")
    Occupation: str = Field(..., description="A plausible occupation (consistent with age).")
    Current_location: str = Field(..., description="A plausible current location (City, State/Country).")
    Personality_Traits: List[str] = Field(..., min_length=3, max_length=6, description="A list of 3-6 descriptive personality traits.")
    Birthplace: str = Field(..., description="A plausible birthplace (City, State/Country).")
    Birthdate: str = Field(..., description="The persona's full birthdate in ISO 8601 format (YYYY-MM-DD).")
    Birthtime: Optional[str] = Field(None, description="The persona's birth time (HH:MM, 24-hour format).")
    Education: Optional[str] = Field(None, description="Highest level of education achieved or currently pursuing (consistent with age).")
    Physical_Appearance: str = Field(..., description="Details about the persona's physical appearance.")
    Hobbies: str = Field(..., description="A description of hobbies or interests.")
    Skills: str = Field(..., description="A description of skills or proficiencies.")
    Languages: str = Field(..., description="Languages spoken by the persona.")
    Health_Status: str = Field(..., description="A brief description of the persona's health condition.")
    Family_Background: str = Field(..., description="Details about the persona's family background.")
    Life_Goals: str = Field(..., description="A description of the persona's life goals or aspirations.")
    Notable_Achievements: str = Field(..., description="A description of the persona's notable achievements.")
    Fears: str = Field(..., description="A description of the persona's fears or anxieties.")
    Strengths: str = Field(..., description="A description of the persona's strengths or positive attributes.")
    Weaknesses: str = Field(..., description="A description of the persona's weaknesses or challenges.")
    Ethnicity: str = Field(..., description="The persona's ethnicity or cultural background.")
    Religion: Optional[str] = Field(None, description="The persona's religious beliefs or affiliations.")
    Political_Views: Optional[str] = Field(None, description="The persona's political views or ideology.")
    Favorite_Foods: str = Field(..., description="A description of the persona's favorite foods or cuisines.")
    Favorite_Music: str = Field(..., description="A description of the persona's favorite music genres or artists.")
    Favorite_Books: str = Field(..., description="A description of the persona's favorite books or authors.")
    Favorite_Movies: str = Field(..., description="A description of the persona's favorite movies or genres.")
    Pet_Peeves: str = Field(..., description="A description of things that annoy or irritate the persona.")
    Dreams: str = Field(..., description="A description of the persona's dreams or aspirations for the future.")
    Past_Traumas: Optional[str] = Field(None, description="A description of significant past traumas or challenges.")

    @model_validator(mode="after")
    def validate_birthdate(self) -> "PersonaDetailsResponse":
        """Validate that Birthdate is in ISO 8601 format (YYYY-MM-DD)."""
        try:
            datetime.strptime(self.Birthdate, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid Birthdate format: {self.Birthdate}. Expected format is YYYY-MM-DD.")
        return self

class YearSummary(BaseModel):
    year: conint(ge=1) = Field(..., description="The calendar year.")
    location: str = Field(..., description="Primary city/region the persona lived in during this year. Be specific.") # Made mandatory
    summary: ShortSummary = Field(..., description="The summary for this year.")

class YearlySummariesResponse(BaseModel):
    birth_month: conint(ge=1, le=12) = Field(...)
    birth_day: conint(ge=1, le=31) = Field(...)
    summaries: List[YearSummary] = Field(...)

    @model_validator(mode='after')
    def check_day_valid_for_month(self) -> 'YearlySummariesResponse':
        """Validate birth day is valid for the birth month."""
        try:
            # Use a non-leap year for general validation, as birth year isn't directly available here for leap check
            ref_year = 2001 # Any non-leap year
            if hasattr(self, 'birth_month') and hasattr(self, 'birth_day'):
                # Check month validity first (already handled by conint)
                # Check day validity
                days_in_month = calendar.monthrange(ref_year, self.birth_month)[1]
                if not (1 <= self.birth_day <= days_in_month):
                    raise ValueError(f"Birth day {self.birth_day} is invalid for month {self.birth_month} (max {days_in_month}).")
        except Exception as e:
            logger.error(f"Date validation error in YearlySummariesResponse: {e}", exc_info=True)
            raise ValueError(f"Could not validate date components: {e}") from e
        return self

class MonthSummary(BaseModel):
    month: conint(ge=1, le=12) = Field(...)
    location: str = Field(..., description="Assumed location for this month (usually same as year).") # Made mandatory
    summary: ShortSummary = Field(...)

class MonthlySummariesResponse(BaseModel):
    summaries: List[MonthSummary] = Field(...)

class DaySummary(BaseModel):
    day: conint(ge=1, le=31) = Field(...)
    location: str = Field(..., description="Assumed location for this day (usually same as year/month).") # Made mandatory
    summary: ShortSummary = Field(...)

class DailySummariesResponse(BaseModel):
    summaries: List[DaySummary] = Field(...)

    # Optional: Add validator to check day based on month if month context were available here
    # Requires passing year/month or modifying structure. For now, relies on generator logic.

class HourActivity(BaseModel):
    hour: conint(ge=0, le=23) = Field(...)
    location: str = Field(..., description="Assumed location during this hour (usually same as day).") # Made mandatory
    activity: ShortSummary = Field(...)

class HourlyBreakdownResponse(BaseModel):
    activities: List[HourActivity] = Field(...)


# --- Models for ADK Simulation State/Interaction (Potentially simplified/combined) ---
# These might be used by the ADK agents directly or adapted.

class WorldReactionProfile(BaseModel):
    """Defines how the world reacts (Placeholder - Copied from provided script)."""
    consequence_severity: Literal["mild", "moderate", "severe"] = Field("moderate")
    social_responsiveness: Literal["hostile", "neutral", "friendly", "indifferent"] = Field("neutral")
    environmental_stability: Literal["stable", "dynamic", "chaotic"] = Field("dynamic")
    coincidence_frequency: Literal["rare", "occasional", "frequent"] = Field("occasional")
    challenge_level: Literal["easy", "normal", "difficult"] = Field("normal")
    narrative_tone: Literal["comedic", "dramatic", "mundane", "suspenseful"] = Field("mundane")
    opportunity_frequency: Literal["scarce", "normal", "abundant"] = Field("normal")
    serendipity: Literal["low", "medium", "high"] = Field("medium")
    world_awareness: Literal["invisible", "normal", "spotlight"] = Field("normal")
    karmic_response: Literal["strong", "moderate", "none"] = Field("moderate")
    npc_density_modifier: float = Field(1.0, ge=0.0)
    npc_proactivity_modifier: float = Field(0.5, ge=0.0, le=1.0)
    dialogue_quality_modifier: float = Field(1.0, ge=0.1, le=2.0)

    # Classmethod create_profile and get_description omitted for brevity, assume they exist as provided

class NPCEnvironmentStatus(BaseModel):
    """Represents the state of a specific NPC (Placeholder - Copied from provided script)."""
    name: str
    role: str
    status: str

class ImmediateEnvironment(BaseModel):
    """Represents immediate surroundings (Placeholder - Copied from provided script, fields omitted for brevity)."""
    current_location_name: str
    location_type: str
    # ... many other fields from provided script ...
    specific_npcs_present: List[NPCEnvironmentStatus] = Field(default_factory=list)
    # ... other fields ...

class WorldState(BaseModel):
    """Represents the broader world state (Placeholder - Copied from provided script, fields omitted for brevity)."""
    current_time: str
    current_date: str
    # ... many other fields ...


# --- Models for ADK Agent Responses (Potentially simplified/combined) ---

class ActionDetails(BaseModel):
    """Details about a Simulacra action."""
    target: Optional[str] = None
    utterance: Optional[str] = None
    target_location: Optional[str] = None
    item: Optional[str] = None
    manner: Optional[str] = None
    duration: Optional[str] = None

class ActionDecisionResponse(BaseModel):
    """Structured response for the Simulacra's action decision (Placeholder - Copied)."""
    thought_process: str
    action: AllowedActionVerbs
    action_details: Optional[ActionDetails] = None

# Note: The WorldStateResponse, ImmediateEnvironmentResponse, WorldStateChanges,
# WorldProcessUpdateResponse, EmotionAnalysisResponse, DayResponse models from the
# provided script are complex and might be specific to a different simulation engine.
# They are omitted here as the current ADK simulation uses direct state updates
# and simpler tool return values (dicts, strings). If needed, they could be
# integrated or adapted for use by the ADK agents/tools.