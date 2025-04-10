import calendar  # Import calendar if used in validator
import json
import logging
# Make sure List, Optional, Field, conint, constr are imported
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field, conint, constr, model_validator

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# --- ADDED Relationship Models ---
class Person(BaseModel):
    """Represents a person in the character's life."""
    name: str = Field(..., description="Full name of the person.")
    relationship: str = Field(..., description="Relationship to the main character (e.g., 'Mother', 'Father', 'Younger Brother', 'Best Friend', 'Spouse').")
    details: Optional[str] = Field(None, description="Brief relevant details (e.g., occupation, age relative to character, key personality trait).")

class InitialRelationshipsResponse(BaseModel):
    """Expected JSON response containing the initial family structure."""
    parents: List[Person] = Field(..., description="List of parents (usually 1 or 2).")
    siblings: List[Person] = Field(default_factory=list, description="List of siblings, if any.")
    # Optional: Add other key figures if desired later
    # other_key_figures: List[Person] = Field(default_factory=list)

class WorldReactionProfile(BaseModel): # ... (existing code) ...
    consequence_severity: Literal["mild", "moderate", "severe"] = Field("moderate", description="...")
    social_responsiveness: Literal["hostile", "neutral", "friendly", "indifferent"] = Field("neutral", description="...")
    environmental_stability: Literal["stable", "dynamic", "chaotic"] = Field("dynamic", description="...")
    coincidence_frequency: Literal["rare", "occasional", "frequent"] = Field("occasional", description="...")
    challenge_level: Literal["easy", "normal", "difficult"] = Field("normal", description="...")
    narrative_tone: Literal["comedic", "dramatic", "mundane", "suspenseful"] = Field("mundane", description="...")
    opportunity_frequency: Literal["scarce", "normal", "abundant"] = Field("normal", description="...")
    serendipity: Literal["low", "medium", "high"] = Field("medium", description="...")
    world_awareness: Literal["invisible", "normal", "spotlight"] = Field("normal", description="...")
    karmic_response: Literal["strong", "moderate", "none"] = Field("moderate", description="...")
    @classmethod
    def create_profile(cls, profile_name: str = "balanced") -> "WorldReactionProfile": # ... (existing code) ...
        profiles = {"balanced": {},"protagonist": {"social_responsiveness": "friendly","coincidence_frequency": "frequent","opportunity_frequency": "abundant","serendipity": "high","world_awareness": "spotlight","karmic_response": "strong"},"antagonist": {"social_responsiveness": "hostile","challenge_level": "difficult","narrative_tone": "dramatic","opportunity_frequency": "scarce","world_awareness": "spotlight"},"comedic": {"consequence_severity": "mild","coincidence_frequency": "frequent","narrative_tone": "comedic","serendipity": "high"},"dramatic": {"consequence_severity": "severe","environmental_stability": "chaotic","narrative_tone": "dramatic","challenge_level": "difficult"},"realistic": {"coincidence_frequency": "rare","karmic_response": "none","world_awareness": "invisible","serendipity": "low"},"serendipitous": {"coincidence_frequency": "frequent","opportunity_frequency": "abundant","serendipity": "high"},"challenging": {"consequence_severity": "severe","challenge_level": "difficult","opportunity_frequency": "scarce"}}
        if profile_name in profiles: return cls(**profiles[profile_name])
        else: logger.warning(f"Unknown profile '{profile_name}'. Using 'balanced' profile."); return cls()
    def get_description(self) -> str: # ... (existing code) ...
        descriptions = {"consequence_severity": { "mild": "Actions have minimal consequences", "moderate": "Actions have normal, expected consequences", "severe": "Actions have amplified consequences" },"social_responsiveness": { "hostile": "People tend to be unfriendly or antagonistic", "neutral": "People react normally based on circumstances", "friendly": "People tend to be helpful and accommodating", "indifferent": "People largely ignore the character" },"environmental_stability": { "stable": "Environment changes little, predictable", "dynamic": "Environment changes at a normal, realistic pace", "chaotic": "Environment frequently changes in unexpected ways" },"coincidence_frequency": { "rare": "Few coincidences, highly realistic cause-and-effect", "occasional": "Normal level of coincidences", "frequent": "Many coincidences (meeting just the right person, etc.)" },"challenge_level": { "easy": "Obstacles are simpler than expected", "normal": "Obstacles require appropriate effort", "difficult": "Obstacles require exceptional effort" },"narrative_tone": { "comedic": "Humorous situations tend to arise", "dramatic": "Emotionally significant events occur", "mundane": "Everyday, ordinary events predominate", "suspenseful": "Tense, uncertain situations develop" },"opportunity_frequency": { "scarce": "Few new opportunities present themselves", "normal": "Realistic number of opportunities", "abundant": "Many opportunities appear" },"serendipity": { "low": "Rarely stumble upon helpful things", "medium": "Occasionally find useful things by chance", "high": "Frequently make fortunate discoveries" },"world_awareness": { "invisible": "Character's actions go largely unnoticed", "normal": "Appropriate recognition of actions", "spotlight": "Character's actions receive unusual attention" },"karmic_response": { "strong": "Good/bad actions quickly lead to rewards/consequences", "moderate": "Some connection between moral choices and outcomes", "none": "No special connection between moral choices and outcomes" }}
        result = []
        for key, value in self.model_dump().items():
            if key in descriptions and value in descriptions[key]: result.append(f"{key.replace('_', ' ').title()}: {descriptions[key][value]}")
        return "\n".join(result)

class ImmediateEnvironment(BaseModel): # ... (existing code) ...
    current_location_name: str = Field(..., description="...")
    location_type: str = Field(..., description="...")
    indoor_outdoor: Literal["indoor", "outdoor"] = Field(..., description="...")
    noise_level: Literal["silent", "quiet", "moderate", "loud", "very loud"] = Field(..., description="...")
    lighting: str = Field(..., description="...")
    temperature_feeling: str = Field(..., description="...")
    air_quality: str = Field(..., description="...")
    present_people: List[str] = Field(default_factory=list, description="...")
    crowd_density: Literal["empty", "sparse", "moderate", "crowded", "packed"] = Field(..., description="...")
    social_atmosphere: str = Field(..., description="...")
    ongoing_activities: List[str] = Field(default_factory=list, description="...")
    nearby_objects: List[str] = Field(default_factory=list, description="...")
    available_services: List[str] = Field(default_factory=list, description="...")
    exit_options: List[str] = Field(default_factory=list, description="...")
    interaction_opportunities: List[str] = Field(default_factory=list, description="...")
    visible_features: List[str] = Field(default_factory=list, description="...")
    audible_sounds: List[str] = Field(default_factory=list, description="...")
    noticeable_smells: List[str] = Field(default_factory=list, description="...")
    seating_availability: str = Field(..., description="...")
    food_drink_options: List[str] = Field(default_factory=list, description="...")
    restroom_access: Literal["available", "unavailable", "unknown"] = Field(..., description="...")
    recent_changes: List[str] = Field(default_factory=list, description="...")
    ongoing_conversations: List[str] = Field(default_factory=list, description="...")
    attention_drawing_elements: List[str] = Field(default_factory=list, description="...")

class WorldState(BaseModel): # ... (existing code) ...
    current_time: str = Field(..., description="...")
    current_date: str = Field(..., description="...")
    city_name: str = Field(..., description="...")
    country_name: str = Field(..., description="...")
    region_name: Optional[str] = Field(None, description="...")
    weather_condition: str = Field(..., description="...")
    temperature: str = Field(..., description="...")
    forecast: Optional[str] = Field(None, description="...")
    social_climate: str = Field(..., description="...")
    economic_condition: str = Field(..., description="...")
    major_events: List[str] = Field(default_factory=list, description="...")
    local_news: List[str] = Field(default_factory=list, description="...")
    transportation_status: str = Field(..., description="...")
    utility_status: str = Field(..., description="...")
    public_announcements: List[str] = Field(default_factory=list, description="...")
    trending_topics: List[str] = Field(default_factory=list, description="...")
    current_cultural_events: List[str] = Field(default_factory=list, description="...")
    sports_events: List[str] = Field(default_factory=list, description="...")
    public_health_status: str = Field(..., description="...")
    public_safety_status: str = Field(..., description="...")

class WorldStateResponse(BaseModel): updated_world_state: WorldState

class ImmediateEnvironmentResponse(BaseModel): updated_environment: ImmediateEnvironment

class WorldUpdateResponse(BaseModel): time: str; location: str; weather: str; environment: str; events: str; social: str; economic: str; consequences: str; observations: str

class EmotionAnalysisResponse(BaseModel): primary_emotion: str; intensity: Literal['Low', 'Medium', 'High']; secondary_emotion: Optional[str] = None; emotional_update: str = Field(..., description="...")

class ActionDetails(BaseModel): manner: Optional[str] = Field(None, description="..."); duration: Optional[str] = Field(None, description="...")

class ActionDecisionResponse(BaseModel): thought_process: str = Field(..., description="..."); action: str = Field(..., description="..."); action_details: Optional[ActionDetails] = None

class DayResponse(BaseModel): reflect: str = Field(..., description="...")

class PersonaDetailsResponse(BaseModel):
    """Defines the structure for a generated persona."""
    Name: str = Field(..., description="A plausible full name.")
    Age: conint(ge=1, le=100) = Field(..., description="A plausible current age between 1 and 100.") # <<< ADDED AGE
    Occupation: str = Field(..., description="A plausible occupation (consistent with age).")
    Current_location: str = Field(..., description="A plausible current location (City, State/Country).")
    Personality_Traits: List[str] = Field(..., min_length=3, max_length=6, description="A list of 3-6 descriptive personality traits.")
    Birthplace: str = Field(..., description="A plausible birthplace (City, State/Country).")
    Education: Optional[str] = Field(None, description="Highest level of education achieved or currently pursuing (consistent with age).")

ShortSummary = constr(min_length=1, max_length=2500)

class YearSummary(BaseModel):
    year: conint(ge=1) = Field(..., description="The calendar year.")
    location: Optional[str] = Field(None, description="Primary city/region the persona lived in during this year.") # <<< ADDED
    summary: ShortSummary = Field(..., description="The summary for this year.")

class YearlySummariesResponse(BaseModel):
    birth_month: conint(ge=1, le=12) = Field(...)
    birth_day: conint(ge=1, le=31) = Field(...)
    summaries: List[YearSummary] = Field(...) # List of items that now include location

    @model_validator(mode='after')
    def check_day_valid_for_month(self) -> 'YearlySummariesResponse':
        # ... (validator remains the same) ...
        try:
            ref_year = 1999
            if hasattr(self, 'birth_month') and hasattr(self, 'birth_day'):
                 days_in_month = calendar.monthrange(ref_year, self.birth_month)[1]
                 if not (1 <= self.birth_day <= days_in_month): raise ValueError(f"Day {self.birth_day} invalid for month {self.birth_month}.")
        except Exception as e: raise ValueError(f"Could not validate day/month: {e}")
        return self

class MonthSummary(BaseModel):
    month: conint(ge=1, le=12) = Field(...)
    location: Optional[str] = Field(None, description="Assumed location for this month (usually same as year).") # <<< ADDED
    summary: ShortSummary = Field(...)

class MonthlySummariesResponse(BaseModel):
    summaries: List[MonthSummary] = Field(...) # List of items that now include location

class DaySummary(BaseModel):
    day: conint(ge=1, le=31) = Field(...)
    location: Optional[str] = Field(None, description="Assumed location for this day (usually same as year/month).") # <<< ADDED
    summary: ShortSummary = Field(...)

class DailySummariesResponse(BaseModel):
    summaries: List[DaySummary] = Field(...) # List of items that now include location

class HourActivity(BaseModel):
    hour: conint(ge=0, le=23) = Field(...)
    location: Optional[str] = Field(None, description="Assumed location during this hour (usually same as day).") # <<< ADDED
    activity: ShortSummary = Field(...)

class HourlyBreakdownResponse(BaseModel):
    activities: List[HourActivity] = Field(...) # List of items that now include location