# src/models.py
import calendar
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

# <<< Keep imports as is >>>
from pydantic import BaseModel, Field, conint, constr, model_validator, ValidationError # ConfigDict removed if not used elsewhere


logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

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
    Age: conint(ge=0, le=120) = Field(..., description="A plausible current age between 0 and 120.")
    Occupation: str = Field(..., description="A plausible occupation (consistent with age).")
    Current_location: str = Field(..., description="A plausible current location (City, State/Country).")
    Personality_Traits: List[str] = Field(..., min_length=3, max_length=6, description="A list of 3-6 descriptive personality traits.")
    Birthplace: str = Field(..., description="A plausible birthplace (City, State/Country).")
    Education: Optional[str] = Field(None, description="Highest level of education achieved or currently pursuing (consistent with age).")

ShortSummary = constr(min_length=1, max_length=3000) # Increased max length for summaries

class YearSummary(BaseModel):
    year: conint(ge=1) = Field(..., description="The calendar year.")
    location: Optional[str] = Field(None, description="Primary city/region the persona lived in during this year.")
    summary: ShortSummary = Field(..., description="The summary for this year.")

class YearlySummariesResponse(BaseModel):
    birth_month: conint(ge=1, le=12) = Field(...)
    birth_day: conint(ge=1, le=31) = Field(...)
    summaries: List[YearSummary] = Field(...)

    @model_validator(mode='after')
    def check_day_valid_for_month(self) -> 'YearlySummariesResponse':
        try:
            # Use a non-leap year for general validation, as birth year isn't directly available here
            ref_year = 2001
            if hasattr(self, 'birth_month') and hasattr(self, 'birth_day'):
                 # Check month validity first
                 if not (1 <= self.birth_month <= 12):
                      raise ValueError(f"Birth month {self.birth_month} is invalid.")
                 # Check day validity
                 days_in_month = calendar.monthrange(ref_year, self.birth_month)[1]
                 if not (1 <= self.birth_day <= days_in_month):
                      raise ValueError(f"Birth day {self.birth_day} is invalid for month {self.birth_month} (max {days_in_month}).")
        except Exception as e:
             # Log the error for debugging
             logger.error(f"Date validation error in YearlySummariesResponse: {e}", exc_info=True)
             # Raise a Pydantic validation error
             raise ValueError(f"Could not validate date components: {e}") from e
        return self

class MonthSummary(BaseModel):
    month: conint(ge=1, le=12) = Field(...)
    location: Optional[str] = Field(None, description="Assumed location for this month (usually same as year).")
    summary: ShortSummary = Field(...)

class MonthlySummariesResponse(BaseModel):
    summaries: List[MonthSummary] = Field(...)

class DaySummary(BaseModel):
    day: conint(ge=1, le=31) = Field(...)
    location: Optional[str] = Field(None, description="Assumed location for this day (usually same as year/month).")
    summary: ShortSummary = Field(...)

class DailySummariesResponse(BaseModel):
    summaries: List[DaySummary] = Field(...)

class HourActivity(BaseModel):
    hour: conint(ge=0, le=23) = Field(...)
    location: Optional[str] = Field(None, description="Assumed location during this hour (usually same as day).")
    activity: ShortSummary = Field(...)

class HourlyBreakdownResponse(BaseModel):
    activities: List[HourActivity] = Field(...)


# --- World Engine Models (Richer State) ---
# <<< Keep WorldReactionProfile as is >>>
class WorldReactionProfile(BaseModel):
    """Defines how the world reacts to the character's actions and manages NPCs."""
    # --- Existing Fields ---
    consequence_severity: Literal["mild", "moderate", "severe"] = Field("moderate", description="Severity of consequences to actions.")
    # social_responsiveness: Literal["hostile", "neutral", "friendly", "indifferent"] = Field("neutral", description="General reaction of NPCs.") # Modified below
    environmental_stability: Literal["stable", "dynamic", "chaotic"] = Field("dynamic", description="How often the environment changes unexpectedly.")
    coincidence_frequency: Literal["rare", "occasional", "frequent"] = Field("occasional", description="Frequency of convenient or inconvenient coincidences.")
    challenge_level: Literal["easy", "normal", "difficult"] = Field("normal", description="Difficulty of overcoming obstacles.")
    narrative_tone: Literal["comedic", "dramatic", "mundane", "suspenseful"] = Field("mundane", description="Overall tone of events.")
    opportunity_frequency: Literal["scarce", "normal", "abundant"] = Field("normal", description="How often new opportunities appear.")
    serendipity: Literal["low", "medium", "high"] = Field("medium", description="Likelihood of fortunate, unexpected discoveries.")
    world_awareness: Literal["invisible", "normal", "spotlight"] = Field("normal", description="How much attention the world pays to the character.")
    karmic_response: Literal["strong", "moderate", "none"] = Field("moderate", description="Connection between moral choices and outcomes.")

    # --- New/Modified NPC-related Fields ---
    social_responsiveness: Literal["hostile", "neutral", "friendly", "indifferent"] = Field("neutral", description="Default initial reaction of newly generated NPCs.")
    npc_density_modifier: float = Field(1.0, ge=0.0, description="Multiplier for how likely NPCs are to be present in locations (0.0 = empty, 1.0 = normal, 2.0 = crowded).")
    npc_proactivity_modifier: float = Field(0.5, ge=0.0, le=1.0, description="Likelihood NPCs initiate interactions (0.0 = never, 1.0 = often).") # Added for future use
    dialogue_quality_modifier: float = Field(1.0, ge=0.1, le=2.0, description="Multiplier affecting NPC dialogue coherence and length (0.5 = terse/simple, 1.0 = normal, 1.5 = verbose/complex).")
    # Consider adding more, e.g., npc_memory_depth_modifier, npc_goal_complexity_modifier etc. later

    @classmethod
    def create_profile(cls, profile_name: str = "balanced") -> "WorldReactionProfile":
        """Create a predefined world reaction profile."""
        # Define base settings for balanced
        base_settings = {
            "consequence_severity": "moderate", "social_responsiveness": "neutral",
            "environmental_stability": "dynamic", "coincidence_frequency": "occasional",
            "challenge_level": "normal", "narrative_tone": "mundane",
            "opportunity_frequency": "normal", "serendipity": "medium",
            "world_awareness": "normal", "karmic_response": "moderate",
            "npc_density_modifier": 1.0, "npc_proactivity_modifier": 0.5,
            "dialogue_quality_modifier": 1.0
        }

        # Define overrides for specific profiles
        profile_overrides = {
            "protagonist": {"social_responsiveness": "friendly", "coincidence_frequency": "frequent", "opportunity_frequency": "abundant", "serendipity": "high", "world_awareness": "spotlight", "karmic_response": "strong", "npc_proactivity_modifier": 0.7},
            "antagonist": {"social_responsiveness": "hostile", "challenge_level": "difficult", "narrative_tone": "dramatic", "opportunity_frequency": "scarce", "world_awareness": "spotlight", "npc_proactivity_modifier": 0.3},
            "comedic": {"consequence_severity": "mild", "coincidence_frequency": "frequent", "narrative_tone": "comedic", "serendipity": "high", "dialogue_quality_modifier": 1.2},
            "dramatic": {"consequence_severity": "severe", "environmental_stability": "chaotic", "narrative_tone": "dramatic", "challenge_level": "difficult", "npc_proactivity_modifier": 0.6},
            "realistic": {"coincidence_frequency": "rare", "karmic_response": "none", "world_awareness": "invisible", "serendipity": "low", "npc_density_modifier": 0.8, "dialogue_quality_modifier": 0.9},
            "serendipitous": {"coincidence_frequency": "frequent", "opportunity_frequency": "abundant", "serendipity": "high"},
            "challenging": {"consequence_severity": "severe", "challenge_level": "difficult", "opportunity_frequency": "scarce", "social_responsiveness": "indifferent"},
            "bustling_city": {"npc_density_modifier": 1.8, "social_responsiveness": "indifferent", "environmental_stability": "dynamic", "npc_proactivity_modifier": 0.4},
            "quiet_village": {"npc_density_modifier": 0.5, "social_responsiveness": "friendly", "environmental_stability": "stable", "narrative_tone": "mundane", "npc_proactivity_modifier": 0.6},
        }

        # Get the specific overrides for the requested profile
        selected_overrides = profile_overrides.get(profile_name.lower(), {})

        # Start with base settings and update with overrides
        final_settings = base_settings.copy()
        final_settings.update(selected_overrides)

        # Validate if the profile name exists, log if not found but still return merged base+override
        if profile_name.lower() != "balanced" and profile_name.lower() not in profile_overrides:
             logger.warning(f"Unknown profile name '{profile_name}'. Using 'balanced' profile with potential overrides if any matched: {selected_overrides}")

        return cls(**final_settings)

    def get_description(self) -> str:
        """Get a human-readable description of the profile."""
        descriptions = {
            "consequence_severity": {"mild": "Minimal consequences", "moderate": "Normal consequences", "severe": "Amplified consequences"},
            "social_responsiveness": {"hostile": "NPCs initially hostile", "neutral": "NPCs initially neutral", "friendly": "NPCs initially friendly", "indifferent": "NPCs initially indifferent"},
            "environmental_stability": {"stable": "Environment stable", "dynamic": "Environment dynamic", "chaotic": "Environment chaotic"},
            "coincidence_frequency": {"rare": "Rare coincidences", "occasional": "Occasional coincidences", "frequent": "Frequent coincidences"},
            "challenge_level": {"easy": "Easy challenges", "normal": "Normal challenges", "difficult": "Difficult challenges"},
            "narrative_tone": {"comedic": "Comedic tone", "dramatic": "Dramatic tone", "mundane": "Mundane tone", "suspenseful": "Suspenseful tone"},
            "opportunity_frequency": {"scarce": "Scarce opportunities", "normal": "Normal opportunities", "abundant": "Abundant opportunities"},
            "serendipity": {"low": "Low serendipity", "medium": "Medium serendipity", "high": "High serendipity"},
            "world_awareness": {"invisible": "World ignores actions", "normal": "Normal world awareness", "spotlight": "World focuses on actions"},
            "karmic_response": {"strong": "Strong karma", "moderate": "Moderate karma", "none": "No karma"},
            # Descriptions for new fields
            "npc_density_modifier": f"NPC Density: {self.npc_density_modifier:.1f}x Normal",
            "npc_proactivity_modifier": f"NPC Proactivity: {self.npc_proactivity_modifier:.1f}/1.0",
            "dialogue_quality_modifier": f"Dialogue Quality: {self.dialogue_quality_modifier:.1f}x Normal"
        }
        result = ["World Reacts With:"]
        for key, value in self.model_dump().items():
            if key in descriptions:
                 if isinstance(descriptions[key], dict): # Handle Literal descriptions
                     if value in descriptions[key]:
                          result.append(f"- {key.replace('_', ' ').title()}: {descriptions[key][value]}")
                 else: # Handle custom string descriptions for modifiers
                     result.append(f"- {descriptions[key]}") # Append the pre-formatted string
        return "\n".join(result)

# <<< Keep NPCEnvironmentStatus as is >>>
class NPCEnvironmentStatus(BaseModel):
    """Represents the state of a specific NPC within the immediate environment."""
    name: str = Field(..., description="Name of the NPC.")
    role: str = Field(..., description="Brief role or description (e.g., 'Barista', 'Customer', 'Passerby').")
    status: str = Field(..., description="Current observable status or activity (e.g., 'Idle', 'Working', 'Talking to Eleanor', 'Walking past').")

# <<< Keep ImmediateEnvironment as is (using List[NPCEnvironmentStatus]) >>>
class ImmediateEnvironment(BaseModel):
    """Represents the character's immediate surroundings and sensory input."""
    # Physical location
    current_location_name: str = Field(..., description="Specific place name (e.g., 'Joe's Cafe', 'Central Park - North Meadow', 'Apartment 3B')")
    location_type: str = Field(..., description="General category (e.g., 'Coffee Shop', 'Park', 'Residential Apartment', 'Office Building', 'Street Corner')")
    indoor_outdoor: Literal["Indoor", "Outdoor", "Mixed"] = Field(..., description="Is the location primarily indoors, outdoors, or a mix (like a covered patio)?")

    # Immediate physical conditions
    noise_level: Literal["Silent", "Quiet", "Moderate", "Loud", "Very Loud"] = Field(..., description="Ambient noise level.")
    lighting: str = Field(..., description="Description of light (e.g., 'Bright Fluorescent', 'Dim Natural Light', 'Warm Lamplight', 'Moonlit', 'Harsh Sunlight')")
    temperature_feeling: str = Field(..., description="How the temperature feels subjectively (e.g., 'Chilly', 'Pleasantly Warm', 'Comfortable', 'Stifling Hot', 'Freezing')")
    ambient_temperature_c: Optional[float] = Field(None, description="Approximate ambient temperature in Celsius if relevant/determinable.")
    humidity_level: Optional[Literal["Low", "Medium", "High"]] = Field(None, description="Approximate humidity level (e.g., Dry, Comfortable, Humid, Muggy).")
    air_quality: str = Field(..., description="Description of air quality/smell (e.g., 'Fresh and Clean', 'Stuffy', 'Smoky', 'Scent of Coffee and Pastries', 'Damp Earth after Rain').")

    # Social environment
    present_people: List[str] = Field(default_factory=list, description="General types/groups of people around (e.g., 'Commuters', 'Shoppers', 'Families with Children', 'Office Workers', 'Tourists', 'No one else').")
    specific_npcs_present: List[NPCEnvironmentStatus] = Field(default_factory=list, description="List of specific, named NPCs currently present with their role and status.")
    crowd_density: Literal["Empty", "Sparse", "Moderate", "Crowded", "Packed"] = Field(..., description="How crowded the location feels.")
    social_atmosphere: str = Field(..., description="The overall mood or vibe (e.g., 'Relaxed and Quiet', 'Tense and Silent', 'Busy and Energetic', 'Festive and Loud', 'Somber and Respectful').")
    ongoing_activities: List[str] = Field(default_factory=list, description="General activities people are engaged in (e.g., 'Dining', 'Working on Laptops', 'Commuting', 'Chatting Quietly', 'Playing Music', 'Arguing').")

    # Available options & Interactions
    nearby_objects: List[str] = Field(default_factory=list, description="List of notable, interactable objects nearby (e.g., 'Vending Machine', 'Newspaper Stand', 'Public Phone', 'Computer Terminal', 'Door', 'Window', 'Trash Can').")
    available_services: List[str] = Field(default_factory=list, description="Services available at this location (e.g., 'Wi-Fi', 'Counter Service', 'ATM', 'Information Desk', 'Public Transit Stop').")
    exit_options: List[str] = Field(default_factory=list, description="List of ways to leave the current immediate location (e.g., 'Main Door to Street', 'Back Alley', 'Stairwell Up', 'Subway Entrance', 'Window').")
    interaction_opportunities: List[str] = Field(default_factory=list, description="Specific people, objects, or elements inviting interaction (e.g., 'Barista taking orders', 'Help Desk', 'Dropped Wallet', 'Ringing Phone', 'Event Poster').")
    points_of_interest: List[str] = Field(default_factory=list, description="Specific interesting spots or features within the location (e.g., 'Window Seat overlooking street', 'Mural on the wall', 'Live Musician performing', 'Unusual architectural feature').")

    # Sensory information
    visible_features: List[str] = Field(default_factory=list, description="Notable visual details of the surroundings (e.g., 'Decor style', 'Architectural details', 'View outside window', 'Condition of furniture', 'Specific signs').")
    audible_sounds: List[str] = Field(default_factory=list, description="Specific distinct sounds that can be heard (e.g., 'Espresso Machine Hissing', 'Distant Sirens', 'Specific Music Genre Playing', 'Phone Ringing', 'Laughter', 'Footsteps').")
    noticeable_smells: List[str] = Field(default_factory=list, description="Specific distinct smells (e.g., 'Baking Bread', 'Cleaning Supplies', 'Rain', 'Exhaust Fumes', 'Someone's Perfume/Cologne').")

    # Practical considerations
    seating_availability: Literal["None", "Limited", "Ample"] = Field(..., description="Availability of places to sit.")
    food_drink_options: List[str] = Field(default_factory=list, description="Specific food or drinks available for purchase or consumption.")
    restroom_access: Literal["Available", "Unavailable", "Customer Only", "Unknown"] = Field(..., description="Bathroom availability and access.")

    # Dynamic elements
    recent_changes: List[str] = Field(default_factory=list, description="Things that just happened or changed (e.g., 'Someone just arrived/left', 'Announcement made over speaker', 'Lights flickered', 'Sudden silence').")
    ongoing_conversations: List[str] = Field(default_factory=list, description="Brief overheard snippets or topics of nearby conversations (if discernible).")
    attention_drawing_elements: List[str] = Field(default_factory=list, description="Things that currently stand out or demand attention.")

# <<< Keep WorldState as is >>>
class WorldState(BaseModel):
    """Represents the broader state of the world beyond immediate perception."""
    # Time and date
    current_time: str = Field(..., description="Time in HH:MM format (24-hour).", examples=["09:30", "17:05"])
    current_date: str = Field(..., description="Date in YYYY-MM-DD format.", examples=["2024-07-15"])
    day_phase: Literal["Morning", "Midday", "Afternoon", "Evening", "Night", "Late Night"] = Field(..., description="General time of day based on current_time.")

    # Location context
    city_name: str = Field(..., description="Name of the city.")
    country_name: str = Field(..., description="Name of the country.")
    region_name: str = Field(..., description="Name of the state, province, or region.")
    district_neighborhood: Optional[str] = Field(None, description="Specific district or neighborhood within the city, if applicable (e.g., 'SoHo', 'Financial District').")

    # Weather
    weather_condition: str = Field(..., description="Overall weather description (e.g., 'Sunny and Clear', 'Partly Cloudy', 'Overcast with Light Drizzle', 'Heavy Thunderstorm', 'Snow Flurries').")
    temperature_c: float = Field(..., description="Temperature in Celsius.")
    forecast: str = Field(..., description="Brief weather forecast for the next few hours (e.g., 'Chance of rain increasing this afternoon', 'Clearing overnight', 'Remaining sunny').")
    wind_description: Optional[str] = Field(None, description="Description of wind (e.g., 'Calm', 'Gentle Breeze from West', 'Strong Gusts up to 40 km/h').")
    precipitation_type: Optional[Literal["None", "Rain", "Snow", "Sleet", "Hail", "Drizzle", "Showers"]] = Field("None", description="Type of precipitation currently occurring, if any.")

    # Social and economic conditions
    social_climate: str = Field(..., description="General social mood of the city/region (e.g., 'Tense due to upcoming election', 'Festive for local holiday', 'Generally calm', 'Anxious about recent events').")
    economic_condition: str = Field(..., description="Overall economic health (e.g., 'Stable', 'Booming', 'Recession concerns', 'High unemployment', 'Affected by recent market changes').")
    political_climate: Optional[str] = Field(None, description="Prevailing political mood or significant political events/tensions.")

    # Current events
    major_events: List[str] = Field(default_factory=list, description="Significant ongoing or upcoming events impacting the area (e.g., 'City Marathon this weekend', 'Ongoing Transit Strike', 'Major Political Summit', 'Natural Disaster Recovery Efforts').")
    local_news: List[str] = Field(default_factory=list, description="Recent notable local news headlines or widely discussed topics.")

    # Infrastructure status
    transportation_status: str = Field(..., description="Status of public transit and road traffic (e.g., 'Normal schedule, light traffic', 'Major delays on subway lines', 'Heavy congestion downtown', 'Road closures due to event').")
    utility_status: str = Field(..., description="Status of essential utilities (e.g., 'All systems normal', 'Intermittent power outages reported in Sector 4', 'Water main break affecting downtown').")

    # Public information & Culture
    public_announcements: List[str] = Field(default_factory=list, description="Recent official communications relevant to the public (e.g., 'Emergency alert test scheduled', 'New public health advisory issued').")
    trending_topics: List[str] = Field(default_factory=list, description="What people are commonly talking about (social media trends, water cooler talk).")
    current_cultural_events: List[str] = Field(default_factory=list, description="Ongoing or upcoming cultural happenings (e.g., 'Art festival in the park', 'New museum exhibit opening', 'Popular band playing tonight').")
    technology_level_notes: Optional[str] = Field(None, description="Notes on prevalent technology or recent tech advancements/issues affecting daily life (e.g., 'Recent rollout of 6G network causing issues', 'AI assistants common', 'Cybersecurity alert active').")

    # Health and safety
    public_health_status: str = Field(..., description="Current public health situation (e.g., 'Seasonal flu activity high', 'No major outbreaks reported', 'Air quality alert active').")
    public_safety_status: str = Field(..., description="Current public safety situation (e.g., 'Normal police presence', 'Increased patrols due to recent incidents', 'Emergency services responding to nearby event').")


# --- Response Models Used by Engines/Simulacra ---

# <<< Keep WorldStateResponse, ImmediateEnvironmentResponse as is >>>
class WorldStateResponse(BaseModel):
    """Wrapper for returning WorldState, ensuring key name match if needed."""
    updated_world_state: WorldState

class ImmediateEnvironmentResponse(BaseModel):
    """Wrapper for returning ImmediateEnvironment."""
    updated_environment: ImmediateEnvironment

# --- MODIFIED: WorldStateChanges (Removed ConfigDict) ---
class WorldStateChanges(BaseModel):
    """Defines potential fields that might change in the WorldState. For schema generation."""
    # Define at least one optional field to ensure 'properties' is non-empty in the schema
    weather_condition: Optional[str] = Field(None, description="New weather description if changed.")
    temperature_c: Optional[float] = Field(None, description="New temperature in Celsius if changed.")
    social_climate: Optional[str] = Field(None, description="New social climate description if changed.")
    transportation_status: Optional[str] = Field(None, description="New transportation status if changed.")
    # Add more optional fields from WorldState here if they change frequently


# --- MODIFIED: WorldProcessUpdateResponse (Uses Optional for WorldStateChanges) ---
class WorldProcessUpdateResponse(BaseModel):
    """Defines the expected JSON structure after the world processes an action."""
    updated_environment: ImmediateEnvironment = Field(..., description="The complete state of the immediate environment AFTER the action.")
    # Use the WorldStateChanges model, make it Optional
    world_state_changes: Optional[WorldStateChanges] = Field(None, description="Dictionary containing ONLY the WorldState fields explicitly defined in WorldStateChanges that have changed. Null if no *defined* fields changed.")
    consequences: List[str] = Field(default_factory=list, description="List of strings describing direct consequences of the action.")
    observations: List[str] = Field(default_factory=list, description="List of strings describing new things the character perceives after the action.")

class EmotionAnalysisResponse(BaseModel):
    """Defines the structure for emotional analysis results."""
    primary_emotion: str = Field(..., description="The most dominant emotion felt now.")
    intensity: Literal['Low', 'Medium', 'High'] = Field(..., description="The intensity of the primary emotion.")
    secondary_emotion: Optional[str] = Field(None, description="A secondary emotion felt, if any.")
    emotional_update: str = Field(..., description="A concise summary sentence describing the character's new overall emotional state.")

class ActionDetails(BaseModel):
    """Optional details about how an action is performed."""
    # Make fields optional as not all apply to every action
    target: Optional[str] = Field(None, description="The target entity or location name (e.g., 'Bob', 'Library', 'Door').")
    utterance: Optional[str] = Field(None, description="The spoken words, if the action is 'talk'.")
    target_location: Optional[str] = Field(None, description="The destination, if the action is 'move'.")
    item: Optional[str] = Field(None, description="The item being interacted with (e.g., 'coffee cup', 'book').")
    manner: Optional[str] = Field(None, description="How the action is performed (e.g., 'Quickly', 'Cautiously', 'Loudly').")
    duration: Optional[str] = Field(None, description="Duration of the action (e.g., 'Briefly', 'For a few minutes').")

class ActionDecisionResponse(BaseModel):
    """Structured response for the Simulacra's action decision."""
    thought_process: str = Field(description="The reasoning behind choosing this action.")
    action: AllowedActionVerbs = Field(description="The chosen action verb from the allowed list.") # <<< Type remains AllowedActionVerbs
    action_details: Optional[ActionDetails] = Field(default=None, description="Specific details for the action (target, utterance, etc.).")
class DayResponse(BaseModel): # Used for Reflection output
    """Defines the structure for character reflection output."""
    reflect: str = Field(..., description="The character's detailed internal reflection text.")

# --- Helper Functions for Validation ---
# <<< Keep validate_model_instance as is >>>
def validate_model_instance(model_class: Type[T], data: Dict[str, Any], operation_description: str) -> Optional[T]:
    """Validates data against a Pydantic model, logs errors, and returns the instance or None."""
    try:
        instance = model_class.model_validate(data)
        logger.debug(f"Successfully validated {operation_description} against {model_class.__name__}.")
        return instance
    except ValidationError as e:
        logger.error(f"Pydantic validation failed during '{operation_description}' for {model_class.__name__}: {e}")
        # Log the problematic data, careful with size/sensitivity
        try:
            logger.error(f"Problematic data: {json.dumps(data, indent=2)}")
        except TypeError:
            logger.error(f"Problematic data (could not dump as JSON): {data}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during validation for '{operation_description}': {e}", exc_info=True)
        return None
