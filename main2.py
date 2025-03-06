import asyncio
import datetime
import json
# At the top of your file, update the logging configuration
import logging
import os
import re
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

import requests
import yaml
# Google Generative AI API
from google import genai
from google.genai import types
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure file logging
file_handler = RotatingFileHandler(
    "logs/simulation.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure root logger to use file handler
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)

# Set Google Generative AI logger to log to file only
genai_logger = logging.getLogger('google_genai')
genai_logger.setLevel(logging.INFO)
genai_logger.propagate = False  # Don't propagate to root logger
genai_logger.addHandler(file_handler)

# Set our application logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # still using openai key name for legacy reasons
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it with your OpenAI API key.")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it with your Google API key.")

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

class WorldReactionProfile(BaseModel):
    """Defines how the world reacts to the character's actions."""
    
    consequence_severity: Literal["mild", "moderate", "severe"] = "moderate"
    social_responsiveness: Literal["hostile", "neutral", "friendly", "indifferent"] = "neutral"
    environmental_stability: Literal["stable", "dynamic", "chaotic"] = "dynamic"
    coincidence_frequency: Literal["rare", "occasional", "frequent"] = "occasional"
    challenge_level: Literal["easy", "normal", "difficult"] = "normal"
    narrative_tone: Literal["comedic", "dramatic", "mundane", "suspenseful"] = "mundane"
    opportunity_frequency: Literal["scarce", "normal", "abundant"] = "normal"
    serendipity: Literal["low", "medium", "high"] = "medium"
    world_awareness: Literal["invisible", "normal", "spotlight"] = "normal"
    karmic_response: Literal["strong", "moderate", "none"] = "moderate"
    
    @classmethod
    def create_profile(cls, profile_name: str = "balanced") -> "WorldReactionProfile":
        """Create a predefined world reaction profile."""
        profiles = {
            "balanced": {}, # Default values
            "protagonist": {
                "social_responsiveness": "friendly",
                "coincidence_frequency": "frequent",
                "opportunity_frequency": "abundant",
                "serendipity": "high",
                "world_awareness": "spotlight",
                "karmic_response": "strong"
            },
            "antagonist": {
                "social_responsiveness": "hostile",
                "challenge_level": "difficult",
                "narrative_tone": "dramatic",
                "opportunity_frequency": "scarce",
                "world_awareness": "spotlight"
            },
            "comedic": {
                "consequence_severity": "mild",
                "coincidence_frequency": "frequent",
                "narrative_tone": "comedic",
                "serendipity": "high"
            },
            "dramatic": {
                "consequence_severity": "severe",
                "environmental_stability": "chaotic",
                "narrative_tone": "dramatic",
                "challenge_level": "difficult"
            },
            "realistic": {
                "coincidence_frequency": "rare",
                "karmic_response": "none",
                "world_awareness": "invisible",
                "serendipity": "low"
            },
            "serendipitous": {
                "coincidence_frequency": "frequent",
                "opportunity_frequency": "abundant",
                "serendipity": "high"
            },
            "challenging": {
                "consequence_severity": "severe",
                "challenge_level": "difficult",
                "opportunity_frequency": "scarce"
            }
        }
        
        if profile_name in profiles:
            return cls(**profiles[profile_name])
        else:
            logger.warning(f"Unknown profile '{profile_name}'. Using 'balanced' profile.")
            return cls()
    
    def get_description(self) -> str:
        """Get a human-readable description of the profile."""
        descriptions = {
            "consequence_severity": {
                "mild": "Actions have minimal consequences",
                "moderate": "Actions have normal, expected consequences",
                "severe": "Actions have amplified consequences"
            },
            "social_responsiveness": {
                "hostile": "People tend to be unfriendly or antagonistic",
                "neutral": "People react normally based on circumstances",
                "friendly": "People tend to be helpful and accommodating",
                "indifferent": "People largely ignore the character"
            },
            "environmental_stability": {
                "stable": "Environment changes little, predictable",
                "dynamic": "Environment changes at a normal, realistic pace",
                "chaotic": "Environment frequently changes in unexpected ways"
            },
            "coincidence_frequency": {
                "rare": "Few coincidences, highly realistic cause-and-effect",
                "occasional": "Normal level of coincidences",
                "frequent": "Many coincidences (meeting just the right person, etc.)"
            },
            "challenge_level": {
                "easy": "Obstacles are simpler than expected",
                "normal": "Obstacles require appropriate effort",
                "difficult": "Obstacles require exceptional effort"
            },
            "narrative_tone": {
                "comedic": "Humorous situations tend to arise",
                "dramatic": "Emotionally significant events occur",
                "mundane": "Everyday, ordinary events predominate",
                "suspenseful": "Tense, uncertain situations develop"
            },
            "opportunity_frequency": {
                "scarce": "Few new opportunities present themselves",
                "normal": "Realistic number of opportunities",
                "abundant": "Many opportunities appear"
            },
            "serendipity": {
                "low": "Rarely stumble upon helpful things",
                "medium": "Occasionally find useful things by chance",
                "high": "Frequently make fortunate discoveries"
            },
            "world_awareness": {
                "invisible": "Character's actions go largely unnoticed",
                "normal": "Appropriate recognition of actions",
                "spotlight": "Character's actions receive unusual attention"
            },
            "karmic_response": {
                "strong": "Good/bad actions quickly lead to rewards/consequences",
                "moderate": "Some connection between moral choices and outcomes",
                "none": "No special connection between moral choices and outcomes"
            }
        }
        
        result = []
        for key, value in self.model_dump().items():
            if key in descriptions and value in descriptions[key]:
                result.append(f"{key.replace('_', ' ').title()}: {descriptions[key][value]}")
        
        return "\n".join(result)

class ImmediateEnvironment(BaseModel):
    # Physical location
    current_location_name: str  # Specific place (café, park, office)
    location_type: str  # Category of location
    indoor_outdoor: str  # Whether inside or outside
    
    # Immediate physical conditions
    noise_level: str  # How loud/quiet it is
    lighting: str  # Brightness, natural/artificial
    temperature_feeling: str  # How it feels (may differ from weather)
    air_quality: str  # Fresh, stuffy, smoky, etc.
    
    # Social environment
    present_people: List[str]  # Types of people around
    crowd_density: str  # How crowded it is
    social_atmosphere: str  # Mood of people around
    ongoing_activities: List[str]  # What others are doing
    
    # Available options
    nearby_objects: List[str]  # Things that can be interacted with
    available_services: List[str]  # Services that can be used
    exit_options: List[str]  # Ways to leave current location
    interaction_opportunities: List[str]  # People to talk to, activities to join
    
    # Sensory information
    visible_features: List[str]  # Notable things that can be seen
    audible_sounds: List[str]  # What can be heard
    noticeable_smells: List[str]  # Olfactory information
    
    # Practical considerations
    seating_availability: str  # Places to sit
    food_drink_options: List[str]  # Available refreshments
    restroom_access: str  # Bathroom availability
    
    # Dynamic elements
    recent_changes: List[str]  # What just changed in the environment
    ongoing_conversations: List[str]  # Topics being discussed nearby
    attention_drawing_elements: List[str]  # Things that stand out

class WorldState(BaseModel):
    # Time and date
    current_time: str
    current_date: str
    
    # Location context
    city_name: str
    country_name: str
    region_name: str
    
    # Weather
    weather_condition: str
    temperature: str
    forecast: str
    
    # Social and economic conditions
    social_climate: str  # General social mood, tensions, celebrations
    economic_condition: str  # Economic health, job market, etc.
    
    # Current events
    major_events: List[str]  # Significant happenings in the area
    local_news: List[str]  # Recent news items
    
    # Infrastructure status
    transportation_status: str  # Public transit, traffic conditions
    utility_status: str  # Power, water, internet
    
    # Public information
    public_announcements: List[str]  # Official communications
    trending_topics: List[str]  # What people are talking about
    
    # Cultural context
    current_cultural_events: List[str]  # Festivals, performances, exhibitions
    sports_events: List[str]  # Games, matches, tournaments
    
    # Health and safety
    public_health_status: str  # Disease outbreaks, health advisories
    public_safety_status: str  # Crime levels, safety concerns

class WorldStateResponse(BaseModel):
    updated_world_state: WorldState

class ImmediateEnvironmentResponse(BaseModel):
    updated_environment: ImmediateEnvironment

class WorldUpdateResponse(BaseModel):
    time: str
    location: str
    weather: str
    environment: str
    events: str
    social: str
    economic: str
    consequences: str
    observations: str

class EmotionAnalysisResponse(BaseModel):
    primary_emotion: str
    intensity: Literal['Low', 'Medium', 'High']
    secondary_emotion: str
    emotional_update: str

class ActionDetails(BaseModel):
    manner: Optional[str] = None
    duration: Optional[str] = None

class ActionDecisionResponse(BaseModel):
    thought_process: str
    action: str
    action_details: ActionDetails

class DayResponse(BaseModel):
    reflect: str

class PromptManager:
    """Manages all prompts used in the simulation."""
    
    @staticmethod
    def analyze_information_prompt(information: str, context: Optional[Dict] = None) -> str:
        context_str = json.dumps(context, indent=2) if context else "No context provided."
        return f"""
        Information to analyze:
        {information}

        Context (previous world state or other relevant info):
        {context_str}

        Based on this, update the world state. Focus on making realistic and plausible changes to the world.
        Return the updated world state as a JSON object.
        """
    
    @staticmethod
    def initialize_world_state_prompt(news_results: str, config: Dict) -> str:
        return f"""
        News results:
        {news_results}

        World configuration:
        {json.dumps(config, indent=2)}

        Based on this information, create a comprehensive world state with the following elements:
        - current_time: The current time in 24-hour format
        - current_date: The current date in YYYY-MM-DD format
        - city_name: The name of the city
        - country_name: The name of the country
        - region_name: The name of the region or state
        - weather_condition: Current weather (sunny, cloudy, rainy, etc.)
        - temperature: Current temperature with units
        - forecast: Brief weather forecast for next 24 hours
        - social_climate: General social mood and atmosphere
        - economic_condition: Current economic situation
        - major_events: List of significant events happening in the area
        - local_news: List of recent local news items
        - transportation_status: Status of public transit and traffic
        - utility_status: Status of power, water, internet services
        - public_announcements: List of official announcements
        - trending_topics: List of topics people are discussing
        - current_cultural_events: List of ongoing cultural events
        - sports_events: List of sports events
        - public_health_status: Current public health situation
        - public_safety_status: Current safety and security situation

        Create a realistic and detailed world state based on the news and configuration.
        """
    
    @staticmethod
    def initialize_immediate_environment_prompt(world_state: Dict, location: str) -> str:
        return f"""
        World State:
        {json.dumps(world_state, indent=2)}

        Current Location: {location}

        Based on this information, create a detailed immediate environment for this location with the following elements:
        - current_location_name: Specific name of the current location
        - location_type: Type of location (restaurant, park, office, etc.)
        - indoor_outdoor: Whether the location is indoor or outdoor
        - noise_level: How loud or quiet it is
        - lighting: Lighting conditions
        - temperature_feeling: How the temperature feels
        - air_quality: Quality of the air
        - present_people: Types of people present
        - crowd_density: How crowded the location is
        - social_atmosphere: Social mood of the location
        - ongoing_activities: Activities happening around
        - nearby_objects: Objects that can be interacted with
        - available_services: Services available at this location
        - exit_options: Ways to leave this location
        - interaction_opportunities: Opportunities for interaction
        - visible_features: Notable visible features
        - audible_sounds: Sounds that can be heard
        - noticeable_smells: Smells that can be detected
        - seating_availability: Availability of seating
        - food_drink_options: Available food and drinks
        - restroom_access: Access to restrooms
        - recent_changes: Recent changes to the environment
        - ongoing_conversations: Topics being discussed nearby
        - attention_drawing_elements: Things that draw attention

        Create a realistic and detailed immediate environment that would be consistent with the world state and location.
        """
    
    @staticmethod
    def process_update_prompt(world_state: Dict, immediate_environment: Dict, 
                            simulacra_action: Dict, reaction_profile: WorldReactionProfile) -> str:
        """Create a prompt for processing an update based on the simulacra's action."""
        
        # Create guidance based on the reaction profile
        profile_guidance = f"""
        WORLD REACTION PROFILE:
        
        1. Consequence Severity: {reaction_profile.consequence_severity.upper()}
        - {"Actions have minimal consequences" if reaction_profile.consequence_severity == "mild" else
        "Actions have normal, expected consequences" if reaction_profile.consequence_severity == "moderate" else
        "Actions have amplified consequences"}
        
        2. Social Responsiveness: {reaction_profile.social_responsiveness.upper()}
        - {"People tend to be unfriendly or antagonistic" if reaction_profile.social_responsiveness == "hostile" else
        "People react normally based on circumstances" if reaction_profile.social_responsiveness == "neutral" else
        "People tend to be helpful and accommodating" if reaction_profile.social_responsiveness == "friendly" else
        "People largely ignore the character"}
        
        3. Environmental Stability: {reaction_profile.environmental_stability.upper()}
        - {"Environment changes little, predictable" if reaction_profile.environmental_stability == "stable" else
        "Environment changes at a normal, realistic pace" if reaction_profile.environmental_stability == "dynamic" else
        "Environment frequently changes in unexpected ways"}
        
        4. Coincidence Frequency: {reaction_profile.coincidence_frequency.upper()}
        - {"Few coincidences, highly realistic cause-and-effect" if reaction_profile.coincidence_frequency == "rare" else
        "Normal level of coincidences" if reaction_profile.coincidence_frequency == "occasional" else
        "Many coincidences (meeting just the right person, etc.)"}
        
        5. Challenge Level: {reaction_profile.challenge_level.upper()}
        - {"Obstacles are simpler than expected" if reaction_profile.challenge_level == "easy" else
        "Obstacles require appropriate effort" if reaction_profile.challenge_level == "normal" else
        "Obstacles require exceptional effort"}
        
        6. Narrative Tone: {reaction_profile.narrative_tone.upper()}
        - {"Humorous situations tend to arise" if reaction_profile.narrative_tone == "comedic" else
        "Emotionally significant events occur" if reaction_profile.narrative_tone == "dramatic" else
        "Everyday, ordinary events predominate" if reaction_profile.narrative_tone == "mundane" else
        "Tense, uncertain situations develop"}
        
        7. Opportunity Frequency: {reaction_profile.opportunity_frequency.upper()}
        - {"Few new opportunities present themselves" if reaction_profile.opportunity_frequency == "scarce" else
        "Realistic number of opportunities" if reaction_profile.opportunity_frequency == "normal" else
        "Many opportunities appear"}
        
        8. Serendipity: {reaction_profile.serendipity.upper()}
        - {"Rarely stumble upon helpful things" if reaction_profile.serendipity == "low" else
        "Occasionally find useful things by chance" if reaction_profile.serendipity == "medium" else
        "Frequently make fortunate discoveries"}
        
        9. World Awareness: {reaction_profile.world_awareness.upper()}
        - {"Character's actions go largely unnoticed" if reaction_profile.world_awareness == "invisible" else
        "Appropriate recognition of actions" if reaction_profile.world_awareness == "normal" else
        "Character's actions receive unusual attention"}
        
        10. Karmic Response: {reaction_profile.karmic_response.upper()}
        - {"Good/bad actions quickly lead to rewards/consequences" if reaction_profile.karmic_response == "strong" else
        "Some connection between moral choices and outcomes" if reaction_profile.karmic_response == "moderate" else
        "No special connection between moral choices and outcomes"}
        """

        agent_input = f"""
        Current World State:
        {json.dumps(world_state, indent=2)}

        Current Immediate Environment:
        {json.dumps(immediate_environment, indent=2)}

        Simulacra action:
        {json.dumps(simulacra_action, indent=2)}

        {profile_guidance}

        Analyze how the immediate environment should react to this action, considering:
        1. Physical laws and plausibility
        2. Environmental responses
        3. Social responses from people present
        4. Any changes to the surroundings
        5. New opportunities or limitations created by the action

        Focus primarily on updating the immediate environment, as this is what's directly affected by the simulacra's action.
        Only update the world state if the action would realistically have broader implications.
        
        For consequences and observations, provide a LIST of specific, detailed statements.
        Each consequence or observation should be a complete sentence describing one specific effect or thing noticed.
        """
        
        prompt_json_output = "\nRespond in the following JSON format: " + json.dumps(
            {
                "updated_environment": "The updated immediate environment after the action",
                "world_state_changes": "Any changes to the world state (if applicable)",
                "consequences": ["List of specific consequences of the action, each as a complete sentence"],
                "observations": ["List of specific things the simulacra would observe, each as a complete sentence"]
            }
        )
        return agent_input + prompt_json_output
    
    @staticmethod
    def reflect_on_situation_prompt(observations: str, persona_state: Dict) -> str:
        reflection_prompt = f"""
        {observations}

        Consider your personality as {persona_state['name']}, a {persona_state['age']}-year-old {persona_state['occupation']} who is {', '.join(persona_state['personality_traits'])},
        your current physical state ({persona_state['current_state']['physical']}),
        emotional state ({persona_state['current_state']['emotional']}),
        and mental state ({persona_state['current_state']['mental']}).

        Think about your goals: {', '.join(persona_state['goals'])}

        Consider your short-term memories of {', '.join(persona_state['memory']['short_term'])} and long-term experiences of {', '.join(persona_state['memory']['long_term'])}.
        """
        
        prompt_json_output = "\nRespond in the following JSON format: " + json.dumps(
            {
                "reflect": "A string reflecting on the situation (string)"
            }
        )
        return reflection_prompt + prompt_json_output
    
    @staticmethod
    def analyze_emotions_prompt(situation: str, current_emotional_state: str) -> str:
        prompt_template = f"""
Situation:
{situation}

Current emotional state:
{current_emotional_state}
        """
        
        prompt_json_day = "\nRespond in the following JSON format: " + json.dumps(
            {
                "primary_emotion": "The most dominant emotion (string)",
                "intensity": "Intensity of the primary emotion, choose from 'Low', 'Medium', or 'High' (string)",
                "secondary_emotion": "The secondary emotion (string)",
                "emotional_update": "A concise summary of the overall emotional state change (string)"
            }
        )
        return prompt_template + prompt_json_day
    
    @staticmethod
    def decide_action_prompt(reflection: str, emotional_analysis: Dict, goals: List[str], immediate_environment: Dict) -> str:
        prompt_template = f"""
        Reflection:
        {reflection}

        Emotional Analysis:
        {json.dumps(emotional_analysis, indent=2)}

        Considering your goals:
        {json.dumps(goals, indent=2)}

        Your immediate environment:
        {json.dumps(immediate_environment, indent=2)}

        What action will you take? Consider your personality traits, current state, available options in your environment, and what would be most consistent with who you are.
        """
        
        prompt_json_output = "\nRespond in the following JSON format: " + json.dumps(
            {
                "thought_process": "Your internal reasoning for the action (string)",
                "action": "The action you decide to take (string)",
                "action_details": "Specific details about how you'll perform the action (object, optional)"
            }
        )
        return prompt_template + prompt_json_output

class LLMService:
    """Centralized service for LLM interactions."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
    
    async def generate_content(self, 
                              prompt: str, 
                              model_name: str = "gemini-2.0-flash",
                              system_instruction: str = "",
                              response_model: Type[T] = None,
                              temperature: float = 0.0) -> Dict:
        """Generate content using the Gemini API with proper error handling."""
        response_text = "No response"
        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    response_mime_type='application/json',
                    response_schema=response_model
                )
            )
            
            response_text = response.text
            
            # Validate and parse JSON response using Pydantic model if provided
            if response_model:
                validated_response = response_model.model_validate_json(response_text)
                return validated_response.model_dump()
            
            # If no model provided, try to parse as JSON
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError from Gemini response: {e}, response text: {response_text}")
            return {"error": f"Could not decode JSON: {str(e)}", "raw_response": response_text}
        except Exception as e:
            logger.error(f"Error in LLM service: {e}")
            return {"error": str(e)}

class WorldEngine:
    """Manages the environment and physical laws of the simulated world."""

    def __init__(self, world_config_path: str = "world_config.yaml", load_state: bool = True, 
                console: Console = None, reaction_profile: Union[str, Dict, WorldReactionProfile] = "balanced"):
        """
        Initialize the WorldEngine.
        
        Args:
            world_config_path: Path to the world configuration file
            load_state: Whether to load existing state
            console: Rich console for output
            reaction_profile: How the world reacts to the character's actions
                            Can be a profile name, a dict of settings, or a WorldReactionProfile
        """
        # Initialize console
        self.console = console or Console()
        self.world_config_path = world_config_path
        self.state_path = "world_state.json"
        self.history = []
        self.world_state = None
        self.immediate_environment = None
        self.llm_service = LLMService()
        
        # Set reaction profile
        if isinstance(reaction_profile, str):
            self.reaction_profile = WorldReactionProfile.create_profile(reaction_profile)
        elif isinstance(reaction_profile, dict):
            self.reaction_profile = WorldReactionProfile(**reaction_profile)
        elif isinstance(reaction_profile, WorldReactionProfile):
            self.reaction_profile = reaction_profile
        else:
            logger.warning(f"Invalid reaction profile type. Using 'balanced' profile.")
            self.reaction_profile = WorldReactionProfile.create_profile("balanced")
        
        logger.info(f"World Engine initialized with reaction profile: {self.reaction_profile.model_dump()}")

        # Check if this is a continuation or new simulation
        if load_state and os.path.exists(self.state_path):
            logger.info("Loading existing world state...")
            self.load_state()

    def _search_news(self, query: str) -> str:
        """Search for news using DuckDuckGo directly."""
        try:
            search_url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "pretty": 1,
                "no_redirect": 1,
                "t": "api" # for api calls
            }
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0'} # need user agent
            response = requests.get(search_url, params=params, headers=headers)
            response.raise_for_status() # Raise an exception for HTTP errors

            data = response.json()
            if data and 'RelatedTopics' in data:
                news_items = []
                for topic in data['RelatedTopics']:
                    if isinstance(topic, dict) and 'Text' in topic and 'FirstURL' in topic:
                        news_items.append(f"- {topic['Text']} (Source: {topic['FirstURL']})")
                return "\n".join(news_items)
            return "No relevant news found."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from DuckDuckGo: {e}")
            return "Error fetching news."

    async def _gather_comprehensive_news(self, location: Dict) -> str:
        """Gather comprehensive news about a location from multiple queries."""
        city = location.get("city", "Seattle")
        country = location.get("country", "USA")
        
        # Create multiple search queries for different aspects
        queries = [
            f"current news {city} {country}",
            f"weather forecast {city} {country}",
            f"events in {city} this week",
            f"economic news {city}",
            f"social issues {city}",
            f"transportation {city} status",
            f"cultural events {city}",
            f"sports news {city}"
        ]
        
        # Gather news for each query
        news_results = []
        for query in queries:
            category = query.split()[0].capitalize()
            results = self._search_news(query)
            if results and results != "No relevant news found." and results != "Error fetching news.":
                news_results.append(f"--- {category} News ---\n{results}\n")
        
        return "\n".join(news_results)

    # Update the _determine_plausible_location method to use the LLM's knowledge
    async def _determine_plausible_location(self, config: Dict) -> str:
        """Use the LLM to determine a plausible location based on the world config and persona."""
        city = config.get("location", {}).get("city", "Seattle")
        country = config.get("location", {}).get("country", "USA")
        
        # Get persona information if available
        persona_path = "simulacra_state.json"
        persona = {}
        
        if os.path.exists(persona_path):
            try:
                with open(persona_path, 'r') as file:
                    data = json.load(file)
                    if "persona" in data:
                        persona = data["persona"]
                        self.console.print(f"[green]Loaded existing persona: {persona.get('name', 'Unknown')}[/green]")
            except Exception as e:
                logger.error(f"Error loading persona: {e}")
        
        # If no persona exists, generate one based on the location
        if not persona:
            self.console.print(f"[yellow]No existing persona found. Generating a persona based on {city}, {country}...[/yellow]")
            persona = await self._generate_persona_for_location(city, country)
            
            # Save the generated persona
            try:
                with open(persona_path, 'w') as file:
                    json.dump({"persona": persona, "history": []}, file, indent=2)
                self.console.print(f"[green]Generated and saved new persona: {persona.get('name', 'Unknown')}[/green]")
            except Exception as e:
                logger.error(f"Error saving generated persona: {e}")
        
        # Create a prompt for the LLM to determine a plausible location
        prompt = f"""
        Based on the following information, determine a realistic and specific location where the character might be at the start of the simulation:
        
        City: {city}
        Country: {country}
        
        Character information:
        Name: {persona.get('name', 'Unknown')}
        Age: {persona.get('age', 'Unknown')}
        Occupation: {persona.get('occupation', 'Unknown')}
        Personality traits: {', '.join(persona.get('personality_traits', ['Unknown']))}
        Goals: {', '.join(persona.get('goals', ['Unknown']))}
        Current state: {persona.get('current_state', {}).get('physical', 'Unknown')}
        """
        
        prompt += """
        Consider the character's occupation, age, and personality to determine where they might realistically be.
        Choose a specific location (like a particular coffee shop, park, office, etc.) that would exist in this city.
        The location should be realistic and specific, not generic.
        
        Return only the name of the location, nothing else.
        """
        
        self.console.print(f"[yellow]Determining plausible location in {city}...[/yellow]")
        
        try:
            # Use the LLM to determine a plausible location
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Determine a realistic and specific location for the character based on their profile and the city.",
                temperature=0.7  # Slightly higher temperature for creativity
            )
            
            # Extract the location from the response
            if isinstance(response, dict) and "text" in response:
                location = response["text"].strip()
            elif isinstance(response, str):
                location = response.strip()
            else:
                location = str(response).strip()
            
            # Clean up the response
            location = location.strip('"\'').strip()
            
            # If the response is too long or contains explanations, extract just the location name
            if len(location.split()) > 10 or ":" in location or "." in location:
                # Try to extract just the location name
                lines = location.split('\n')
                for line in lines:
                    if line.strip() and len(line.split()) < 10 and ":" not in line and "." not in line:
                        location = line.strip()
                        break
            
            self.console.print(f"[green]Determined location:[/green] {location}")
            return location
        except Exception as e:
            logger.error(f"Error determining plausible location: {e}")
            # Fallback to a generic location
            fallback_location = f"Coffee shop in {city}"
            self.console.print(f"[red]Error determining location, using fallback:[/red] {fallback_location}")
            return fallback_location

    async def _generate_persona_for_location(self, city: str, country: str) -> Dict:
        """Generate a persona that would plausibly be in the given location."""
        
        prompt = f"""
        Create a realistic persona for a character in {city}, {country}.
        
        Consider the demographics, culture, and lifestyle of this location to create a believable character.
        
        Return the persona as a JSON object with the following structure:
        {{
            "name": "Character's name",
            "age": age as number,
            "occupation": "Character's job",
            "personality_traits": ["trait1", "trait2", "trait3", "trait4"],
            "goals": ["goal1", "goal2"],
            "current_state": {{
                "physical": "Description of physical state",
                "emotional": "Description of emotional state",
                "mental": "Description of mental state"
            }},
            "memory": {{
                "short_term": ["recent memory 1", "recent memory 2"],
                "long_term": ["long-term memory 1", "long-term memory 2"]
            }}
        }}
        
        Make the character interesting but realistic for this location. Consider local industries, cultural context, and daily life.
        """
        
        system_instruction = f"Create a realistic persona for a character living in {city}, {country} as a structured JSON object."
        
        try:
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.7
            )
            
            # If response is a list with one item (as shown in your example)
            if isinstance(response, list) and len(response) > 0:
                persona = response[0]
            # If response is already a dictionary
            elif isinstance(response, dict):
                persona = response
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")
            
            self.console.print(f"[green]Successfully generated persona for {persona['name']}[/green]")
            return persona
            
        except Exception as e:
            logger.error(f"Error generating persona: {e}")
            # Return a default persona
            default_persona = {
                "name": f"Local Resident of {city}",
                "age": 30,
                "occupation": "Professional",
                "personality_traits": ["adaptable", "observant", "practical", "curious"],
                "goals": ["navigate daily life", "pursue personal interests"],
                "current_state": {
                    "physical": "Well-rested",
                    "emotional": "Neutral",
                    "mental": "Alert and aware"
                },
                "memory": {
                    "short_term": [f"Recent activities in {city}"],
                    "long_term": [f"Life experiences in {country}"]
                }
            }
            self.console.print(f"[red]Using default persona due to error: {e}[/red]")
            return default_persona
        
    def get_current_datetime(self):
        """Get the current date and time."""
        now = datetime.datetime.now()
        return {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A")
        }

    async def initialize_new_world(self):
        """Initialize a new world state with both macro and micro levels using real data."""
        logger.info("Initializing new world...")
        config = {}
        try:
            with open(self.world_config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"World config file not found at {self.world_config_path}. Using default settings.")
            config = {}

        try:
            location = config.get("location", {"city": "NYC", "country": "USA"})
            logger.info(f"Initializing world in {location.get('city', 'unknown location')}")
            
            # Get current date and time
            current_datetime = self.get_current_datetime()
            
            # Gather comprehensive news - real current events
            self.console.print(f"[yellow]Gathering news for {location.get('city')}...[/yellow]")
            news_results = await self._gather_comprehensive_news(location)
            
            # Initialize world state (macro level) with real data
            prompt = PromptManager.initialize_world_state_prompt(news_results, config)
            
            # Add current date and time to the prompt
            prompt += f"\n\nCurrent date: {current_datetime['date']}\nCurrent time: {current_datetime['time']}\nDay of week: {current_datetime['day_of_week']}"
            
            self.console.print("[yellow]Generating world state based on current events...[/yellow]")
            world_state_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction='Create a comprehensive world state based on the provided information, current news, and real-world data.',
                response_model=WorldStateResponse
            )
            
            if "updated_world_state" in world_state_response:
                self.world_state = world_state_response["updated_world_state"]
                # Ensure the time and date are current
                self.world_state["current_time"] = current_datetime["time"]
                self.world_state["current_date"] = current_datetime["date"]
            else:
                # Fallback if LLM fails
                self.world_state = self._create_default_world_state(config)
            
            # Determine a plausible location based on persona using the LLM
            starting_location = await self._determine_plausible_location(config)
            
            # Initialize immediate environment (micro level)
            self.console.print(f"[yellow]Creating environment for {starting_location}...[/yellow]")
            prompt = PromptManager.initialize_immediate_environment_prompt(self.world_state, starting_location)
            environment_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction='Create a detailed immediate environment based on the world state and location.',
                response_model=ImmediateEnvironmentResponse
            )
            
            if "updated_environment" in environment_response:
                self.immediate_environment = environment_response["updated_environment"]
            else:
                # Fallback if LLM fails - now uses LLM for fallback too
                self.immediate_environment = await self._create_default_immediate_environment(starting_location)
            
            # Get persona information if available
            persona = {}
            persona_path = "simulacra_state.json"
            if os.path.exists(persona_path):
                try:
                    with open(persona_path, 'r') as file:
                        data = json.load(file)
                        if "persona" in data:
                            persona = data["persona"]
                except Exception as e:
                    logger.error(f"Error loading persona: {e}")
            
            # Generate narrative context
            narrative_context = await self._generate_narrative_context(
                persona, 
                self.world_state, 
                self.immediate_environment.get('current_location_name', starting_location)
            )

            # Save the initial state
            self.save_state()

            logger.info("New world state and immediate environment initialized")
            return {
                "world_state": self.world_state,
                "immediate_environment": self.immediate_environment,
                "narrative_context": narrative_context,
                "observations": [
                    f"You're in {self.immediate_environment.get('current_location_name', starting_location)}.",
                    f"It's {self.world_state.get('current_time', current_datetime['time'])} on {self.world_state.get('current_date', current_datetime['date'])}.",
                    f"The weather is {self.world_state.get('weather_condition', 'normal')}.",
                    "You notice your surroundings and gather your thoughts."
                ]
            }
        except Exception as e:
            logger.error(f"Error initializing new world: {e}")
            # Create basic defaults if everything fails
            self.world_state = self._create_default_world_state(config)
            self.immediate_environment = self._create_default_immediate_environment("Coffee shop")
            self.save_state()
            return {
                "world_state": self.world_state,
                "immediate_environment": self.immediate_environment,
                "narrative_context": "You arrived at a coffee shop after a busy morning. You have several things on your mind and are trying to figure out your next steps.",
                "observations": [
                    f"You're in a coffee shop.",
                    f"It's {self.world_state.get('current_time', 'daytime')} on {self.world_state.get('current_date', 'today')}.",
                    f"The weather is {self.world_state.get('weather_condition', 'normal')}.",
                    "You notice your surroundings and gather your thoughts."
                ]
            }

    def _create_default_world_state(self, config: Dict) -> Dict:
        """Create a default world state if LLM initialization fails."""
        now = datetime.datetime.now()
        location = config.get("location", {"city": "NYC", "country": "USA"})
        
        return {
            "current_time": now.strftime("%H:%M"),
            "current_date": now.strftime("%Y-%m-%d"),
            "city_name": location.get("city", "NYC"),
            "country_name": location.get("country", "USA"),
            "region_name": location.get("region", "Northeast"),
            "weather_condition": "Partly cloudy",
            "temperature": "68°F (20°C)",
            "forecast": "Similar conditions expected for the next 24 hours",
            "social_climate": "Generally calm with typical urban activity",
            "economic_condition": "Stable with normal business activity",
            "major_events": ["No major events currently"],
            "local_news": ["Standard local news coverage"],
            "transportation_status": "Normal operation of public transit and typical traffic patterns",
            "utility_status": "All utilities functioning normally",
            "public_announcements": ["No significant public announcements"],
            "trending_topics": ["Local sports", "Weather", "Weekend activities"],
            "current_cultural_events": ["Regular museum exhibitions", "Some local music performances"],
            "sports_events": ["Regular season games for local teams"],
            "public_health_status": "No significant health concerns or advisories",
            "public_safety_status": "Normal safety conditions with typical police presence"
        }

    async def _create_default_immediate_environment(self, location_name: str) -> Dict:
        """Create a default immediate environment using the LLM if the standard initialization fails."""
        try:
            self.console.print(f"[yellow]Creating fallback environment for {location_name} using LLM...[/yellow]")
            
            # Determine a basic location type first
            location_type_prompt = f"""
            What type of location is "{location_name}"? 
            Examples: Coffee shop, Restaurant, Park, Office building, Library, Museum, etc.
            
            Return only the location type, nothing else.
            """
            
            location_type_response = await self.llm_service.generate_content(
                prompt=location_type_prompt,
                system_instruction="Determine the type of location based on its name.",
                temperature=0.3
            )
            
            # Extract location type
            if isinstance(location_type_response, list) and len(location_type_response) > 0:
                location_type = location_type_response[0]
            elif isinstance(location_type_response, dict) and "text" in location_type_response:
                location_type = location_type_response["text"].strip()
            elif isinstance(location_type_response, str):
                location_type = location_type_response.strip()
            else:
                location_type = "Public space"
            
            # Clean up location type
            location_type = location_type.strip('"\'').strip()
            
            # Now generate a complete environment based on the location name and type
            environment_prompt = f"""
            Create a detailed immediate environment for a location named "{location_name}" which is a {location_type}.
            
            Return the environment as a JSON object with the following structure:
            {{
                "current_location_name": "{location_name}",
                "location_type": "{location_type}",
                "indoor_outdoor": "Indoor or Outdoor",
                "noise_level": "Description of noise level",
                "lighting": "Description of lighting",
                "temperature_feeling": "How the temperature feels",
                "air_quality": "Description of air quality",
                "present_people": ["Types of people present"],
                "crowd_density": "Description of crowd density",
                "social_atmosphere": "Description of social atmosphere",
                "ongoing_activities": ["Activities happening around"],
                "nearby_objects": ["Objects that can be interacted with"],
                "available_services": ["Services available at this location"],
                "exit_options": ["Ways to leave this location"],
                "interaction_opportunities": ["Opportunities for interaction"],
                "visible_features": ["Notable visible features"],
                "audible_sounds": ["Sounds that can be heard"],
                "noticeable_smells": ["Smells that can be detected"],
                "seating_availability": "Availability of seating",
                "food_drink_options": ["Available food and drinks"],
                "restroom_access": "Access to restrooms",
                "recent_changes": ["Recent changes to the environment"],
                "ongoing_conversations": ["Topics being discussed nearby"],
                "attention_drawing_elements": ["Things that draw attention"]
            }}
            
            Make the environment realistic and detailed for this type of location.
            """
            
            environment_response = await self.llm_service.generate_content(
                prompt=environment_prompt,
                system_instruction=f"Create a detailed environment for a {location_type} named {location_name}.",
                temperature=0.7
            )
            
            # Extract the environment
            if isinstance(environment_response, list) and len(environment_response) > 0:
                environment = environment_response[0]
            elif isinstance(environment_response, dict):
                if "updated_environment" in environment_response:
                    environment = environment_response["updated_environment"]
                else:
                    environment = environment_response
            else:
                raise ValueError(f"Unexpected response format: {type(environment_response)}")
            
            self.console.print(f"[green]Successfully generated environment for {location_name} ({location_type})[/green]")
            return environment
            
        except Exception as e:
            logger.error(f"Error creating environment with LLM: {e}")
            self.console.print(f"[red]Error creating environment with LLM: {e}. Using hardcoded fallback.[/red]")
            
            # Ultimate fallback - hardcoded environment
            return {
                "current_location_name": location_name,
                "location_type": "Public space",
                "indoor_outdoor": "Indoor",
                "noise_level": "Moderate with conversation",
                "lighting": "Adequate lighting",
                "temperature_feeling": "Comfortable",
                "air_quality": "Fresh",
                "present_people": ["Various people"],
                "crowd_density": "Moderately busy",
                "social_atmosphere": "Neutral",
                "ongoing_activities": ["People going about their business"],
                "nearby_objects": ["Furniture", "Fixtures"],
                "available_services": ["Basic amenities"],
                "exit_options": ["Main entrance/exit"],
                "interaction_opportunities": ["People nearby"],
                "visible_features": ["Standard features for this type of location"],
                "audible_sounds": ["Ambient noise", "Conversations"],
                "noticeable_smells": ["Neutral smells"],
                "seating_availability": "Some seating available",
                "food_drink_options": ["Basic options if applicable"],
                "restroom_access": "Standard access",
                "recent_changes": ["Nothing notable has changed recently"],
                "ongoing_conversations": ["General conversations"],
                "attention_drawing_elements": ["Nothing particularly notable"]
            }

    async def process_update(self, simulacra_action: Dict[str, Any], simulacra_persona: Dict = None) -> Dict[str, Any]:
        """Process an action from the simulacra and update both world state and immediate environment."""

        # Save the action to history
        self.history.append({
            "timestamp": self.world_state.get("current_time", datetime.now().strftime("%H:%M")),
            "date": self.world_state.get("current_date", datetime.now().strftime("%Y-%m-%d")),
            "action": simulacra_action
        })

        self.console.print(f"\n[bold yellow]Processing action:[/bold yellow] {simulacra_action.get('action', 'Unknown action')}")
        
        # Include reaction profile in the prompt
        prompt = PromptManager.process_update_prompt(
            self.world_state, 
            self.immediate_environment, 
            simulacra_action,
            self.reaction_profile
        )
        
        system_instruction = 'Update the environment based on an action taken by a simulated character, following the specified world reaction profile.'
        
        try:
            self.console.print("[bold]Generating world response...[/bold]")
            update_response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                response_model=None  # Using a custom response format
            )

            # Update immediate environment
            if "updated_environment" in update_response:
                self.immediate_environment = update_response["updated_environment"]
                self.console.print("[green]Environment updated successfully[/green]")
            
            # Update world state if there are broader changes
            if "world_state_changes" in update_response and update_response["world_state_changes"]:
                # Apply only the changes to the world state
                world_changes = update_response["world_state_changes"]
                for key, value in world_changes.items():
                    if key in self.world_state:
                        self.world_state[key] = value
                self.console.print("[blue]World state updated with broader changes[/blue]")
            
            # Ensure consequences and observations are properly formatted as lists
            consequences = update_response.get("consequences", "No notable consequences.")
            observations = update_response.get("observations", "Nothing notable observed.")
            
            # Convert to list if they're strings
            if isinstance(consequences, str):
                consequences = [consequences]
                self.console.print("[yellow]Converted consequences from string to list[/yellow]")
            elif not isinstance(consequences, list):
                consequences = ["Unexpected response format for consequences."]
                self.console.print("[red]Unexpected format for consequences[/red]")
                
            if isinstance(observations, str):
                observations = [observations]
                self.console.print("[yellow]Converted observations from string to list[/yellow]")
            elif not isinstance(observations, list):
                observations = ["Unexpected response format for observations."]
                self.console.print("[red]Unexpected format for observations[/red]")
            
            # Generate an updated narrative context if persona is provided
            narrative_update = ""
            if simulacra_persona:
                narrative_update = await self._generate_updated_narrative(
                    simulacra_persona,
                    [action["action"] for action in self.history[-5:] if "action" in action],  # Last 5 actions
                    self.world_state,
                    self.immediate_environment,
                    consequences,
                    observations
                )
            
            # Save the updated state
            self.save_state()
            self.console.print("[green]World state saved successfully[/green]")

            return {
                "world_state": self.world_state,
                "immediate_environment": self.immediate_environment,
                "consequences": consequences,
                "observations": observations,
                "narrative_update": narrative_update
            }

        except Exception as e:
            logger.error(f"Error processing update: {e}")
            self.console.print(f"[bold red]Error processing update:[/bold red] {str(e)}")
            return {
                "error": str(e),
                "world_state": self.world_state,
                "immediate_environment": self.immediate_environment,
                "consequences": ["No changes due to error"],
                "observations": ["The environment continues as before"],
                "narrative_update": "The story continues without significant developments."
            }

    def save_state(self):
        """Save current world state and immediate environment to a file."""
        try:
            with open(self.state_path, 'w') as file:
                json.dump({
                    "world_state": self.world_state,
                    "immediate_environment": self.immediate_environment,
                    "history": self.history,
                    "reaction_profile": self.reaction_profile.model_dump()
                }, file, indent=2)
            logger.info(f"World state and immediate environment saved to {self.state_path}")
        except Exception as e:
            logger.error(f"Error saving world state: {e}")

    def load_state(self):
        """Load world state and immediate environment from a file."""
        try:
            with open(self.state_path, 'r') as file:
                state_data = json.load(file)
                
                # Validate loaded data against models
                if "world_state" in state_data:
                    WorldState.model_validate(state_data["world_state"])
                    self.world_state = state_data["world_state"]
                
                if "immediate_environment" in state_data:
                    ImmediateEnvironment.model_validate(state_data["immediate_environment"])
                    self.immediate_environment = state_data["immediate_environment"]
                
                if "history" in state_data:
                    self.history = state_data["history"]
                    
                if "reaction_profile" in state_data:
                    try:
                        self.reaction_profile = WorldReactionProfile(**state_data["reaction_profile"])
                        logger.info(f"Loaded reaction profile: {self.reaction_profile.model_dump()}")
                    except Exception as e:
                        logger.error(f"Error loading reaction profile: {e}. Using default profile.")
                        self.reaction_profile = WorldReactionProfile.create_profile("balanced")
                else:
                    logger.info("No reaction profile found in state file. Using default profile.")
                    self.reaction_profile = WorldReactionProfile.create_profile("balanced")
                        
            logger.info(f"World state and immediate environment loaded from {self.state_path}")
        except FileNotFoundError:
            logger.warning(f"No world state file found at {self.state_path}. Starting with a new world.")
        except json.JSONDecodeError:
            logger.error(f"Corrupted world state file at {self.state_path}. Starting with a new world.")
        except Exception as e:
            logger.error(f"Error loading world state from {self.state_path}: {e}. Starting with a new world.")

    def get_world_summary(self) -> str:
        """Generate a concise summary of the current world state, highlighting key events."""
        if not self.world_state:
            return "[bold red]World state not initialized.[/bold red]"
        
        # Extract key information
        time_date = f"[bold cyan]{self.world_state.get('current_time', 'unknown time')} on {self.world_state.get('current_date', 'unknown date')}[/bold cyan]"
        location = f"[bold green]{self.world_state.get('city_name', 'unknown city')}, {self.world_state.get('region_name', 'unknown region')}[/bold green]"
        weather = f"[bold yellow]{self.world_state.get('weather_condition', 'unknown weather')}, {self.world_state.get('temperature', '')}[/bold yellow]"
        
        # Check for major events
        major_events = self.world_state.get('major_events', [])
        if not major_events or major_events == ["No major events currently"]:
            event_str = "[italic]No significant events.[/italic]"
        else:
            event_str = f"[bold magenta]Events:[/bold magenta] {', '.join(major_events[:2])}"
        
        # Check for any unusual conditions
        conditions = []
        if "unstable" in self.world_state.get('economic_condition', '').lower() or "recession" in self.world_state.get('economic_condition', '').lower():
            conditions.append("[bold red]Economic instability[/bold red]")
        if "tension" in self.world_state.get('social_climate', '').lower() or "unrest" in self.world_state.get('social_climate', '').lower():
            conditions.append("[bold red]Social tensions[/bold red]")
        if "storm" in self.world_state.get('weather_condition', '').lower() or "severe" in self.world_state.get('weather_condition', '').lower():
            conditions.append("[bold red]Severe weather[/bold red]")
        if "outage" in self.world_state.get('utility_status', '').lower() or "disruption" in self.world_state.get('utility_status', '').lower():
            conditions.append("[bold red]Utility disruptions[/bold red]")
        if "delay" in self.world_state.get('transportation_status', '').lower() or "closure" in self.world_state.get('transportation_status', '').lower():
            conditions.append("[bold red]Transportation issues[/bold red]")
        if "warning" in self.world_state.get('public_health_status', '').lower() or "outbreak" in self.world_state.get('public_health_status', '').lower():
            conditions.append("[bold red]Health concerns[/bold red]")
        
        if conditions:
            condition_str = f"[bold orange3]Notable conditions:[/bold orange3] {', '.join(conditions)}"
        else:
            condition_str = "[green]Normal conditions overall.[/green]"
        
        # Combine into summary
        summary = f"[bold blue]World Summary:[/bold blue] {time_date} in {location}. {weather}. {event_str} {condition_str}"
        return summary

    def get_environment_summary(self) -> str:
        """Generate a concise summary of the immediate environment, highlighting key elements."""
        if not self.immediate_environment:
            return "[bold red]Immediate environment not initialized.[/bold red]"
        
        # Extract key information
        location = f"[bold cyan]{self.immediate_environment.get('current_location_name', 'unknown location')} ({self.immediate_environment.get('location_type', 'unknown type')})[/bold cyan]"
        setting = f"[bold green]{self.immediate_environment.get('indoor_outdoor', 'unknown setting')}[/bold green]"
        atmosphere = f"[bold yellow]{self.immediate_environment.get('social_atmosphere', 'unknown atmosphere')}[/bold yellow]"
        
        # People and activities
        people = self.immediate_environment.get('present_people', [])
        if people:
            people_str = f"[bold magenta]People:[/bold magenta] {', '.join(people[:2])}"
        else:
            people_str = "[italic]No people around.[/italic]"
        
        activities = self.immediate_environment.get('ongoing_activities', [])
        if activities:
            activities_str = f"[bold blue]Activities:[/bold blue] {', '.join(activities[:2])}"
        else:
            activities_str = "[italic]No notable activities.[/italic]"
        
        # Attention-grabbing elements
        attention = self.immediate_environment.get('attention_drawing_elements', [])
        if attention:
            attention_str = f"[bold orange3]Notable:[/bold orange3] {', '.join(attention[:2])}"
        else:
            attention_str = "[italic]Nothing particularly notable.[/italic]"
        
        # Recent changes
        changes = self.immediate_environment.get('recent_changes', [])
        if changes and changes != ["Nothing notable has changed recently"]:
            changes_str = f"[bold red]Recent changes:[/bold red] {', '.join(changes[:2])}"
        else:
            changes_str = ""
        
        # Combine into summary
        summary = f"[bold purple]Environment Summary:[/bold purple] {location}, {setting}. {atmosphere} atmosphere. {people_str} {activities_str} {attention_str} {changes_str}".strip()
        return summary
    
    async def _generate_narrative_context(self, persona: Dict, world_state: Dict, location: str) -> str:
        """Generate a narrative context explaining how the character arrived at their current situation."""
        
        # Create a prompt for the LLM to generate a narrative context
        prompt = f"""
        Based on the following information, create a brief narrative context (2-3 paragraphs) explaining how the character arrived at their current situation:
        
        Character information:
        Name: {persona.get('name', 'Unknown')}
        Age: {persona.get('age', 'Unknown')}
        Occupation: {persona.get('occupation', 'Unknown')}
        Personality traits: {', '.join(persona.get('personality_traits', ['Unknown']))}
        Goals: {', '.join(persona.get('goals', ['Unknown']))}
        Current physical state: {persona.get('current_state', {}).get('physical', 'Unknown')}
        Current emotional state: {persona.get('current_state', {}).get('emotional', 'Unknown')}
        Current mental state: {persona.get('current_state', {}).get('mental', 'Unknown')}
        Short-term memories: {', '.join(persona.get('memory', {}).get('short_term', ['Unknown']))}
        Long-term memories: {', '.join(persona.get('memory', {}).get('long_term', ['Unknown']))}
        
        World information:
        Current time: {world_state.get('current_time', 'Unknown')}
        Current date: {world_state.get('current_date', 'Unknown')}
        City: {world_state.get('city_name', 'Unknown')}
        Weather: {world_state.get('weather_condition', 'Unknown')}
        Social climate: {world_state.get('social_climate', 'Unknown')}
        Major events: {', '.join(world_state.get('major_events', ['None']))}
        
        Current location: {location}
        
        The narrative should explain:
        1. Why the character is at this specific location
        2. What they were doing earlier today
        3. What their immediate concerns or thoughts are
        4. How their current emotional and physical state came to be
        
        The narrative should be realistic, specific to this character, and connect to their goals and personality.
        """
        
        self.console.print(f"[yellow]Generating narrative context...[/yellow]")
        
        try:
            # Use the LLM to generate a narrative context
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Create a realistic narrative context for the character's current situation.",
                temperature=0.7  # Slightly higher temperature for creativity
            )
            
            # Extract the narrative from the response
            if isinstance(response, dict) and "text" in response:
                narrative = response["text"].strip()
            elif isinstance(response, str):
                narrative = response.strip()
            else:
                narrative = str(response).strip()
            
            self.console.print(f"[green]Narrative context generated successfully[/green]")
            return narrative
        except Exception as e:
            logger.error(f"Error generating narrative context: {e}")
            # Fallback to a generic narrative
            fallback_narrative = f"{persona.get('name', 'The character')} arrived at {location} after a busy morning. They have several things on their mind, particularly {', '.join(persona.get('goals', ['their goals']))}. They're feeling {persona.get('current_state', {}).get('emotional', 'mixed emotions')} as they navigate their day."
            self.console.print(f"[red]Error generating narrative context, using fallback[/red]")
            return fallback_narrative
        
    async def _generate_updated_narrative(self, persona: Dict, previous_actions: List[Dict], 
                                        world_state: Dict, immediate_environment: Dict, 
                                        consequences: List[str], observations: List[str]) -> str:
        """Generate an updated narrative context based on recent actions and events."""
        
        # Extract the most recent actions (up to 3)
        recent_actions = previous_actions[-3:] if previous_actions else []
        actions_text = ""
        for action in recent_actions:
            actions_text += f"- {action.get('action', 'Unknown action')}\n"
            if 'action_details' in action and action['action_details']:
                actions_text += f"  Details: {json.dumps(action['action_details'])}\n"
        
        if not actions_text:
            actions_text = "No recent actions recorded."
        
        # Create a prompt for the LLM to generate an updated narrative
        prompt = f"""
        Based on the following information, create a brief narrative update (1-2 paragraphs) that continues the character's story:
        
        Character information:
        Name: {persona.get('name', 'Unknown')}
        Current physical state: {persona.get('current_state', {}).get('physical', 'Unknown')}
        Current emotional state: {persona.get('current_state', {}).get('emotional', 'Unknown')}
        Current mental state: {persona.get('current_state', {}).get('mental', 'Unknown')}
        Goals: {', '.join(persona.get('goals', ['Unknown']))}
        
        Current location: {immediate_environment.get('current_location_name', 'Unknown location')}
        Current time: {world_state.get('current_time', 'Unknown time')}
        
        Recent actions taken by the character:
        {actions_text}
        
        Recent consequences of these actions:
        {', '.join(consequences) if consequences else 'No notable consequences.'}
        
        Recent observations by the character:
        {', '.join(observations) if observations else 'Nothing notable observed.'}
        
        The narrative update should:
        1. Reflect the character's progress toward their goals
        2. Acknowledge any changes in their emotional or physical state
        3. Incorporate the consequences of their recent actions
        4. Set up potential next steps or challenges
        5. Maintain continuity with their overall story
        
        The narrative should be realistic, specific to this character, and feel like a continuation of their ongoing story.
        """
        
        self.console.print(f"[yellow]Generating narrative update...[/yellow]")
        
        try:
            # Use the LLM to generate a narrative update
            response = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction="Create a realistic narrative update that continues the character's story.",
                temperature=0.7  # Slightly higher temperature for creativity
            )
            
            # Extract the narrative from the response
            if isinstance(response, dict) and "text" in response:
                narrative = response["text"].strip()
            elif isinstance(response, str):
                narrative = response.strip()
            else:
                narrative = str(response).strip()
            
            self.console.print(f"[green]Narrative update generated successfully[/green]")
            return narrative
        except Exception as e:
            logger.error(f"Error generating narrative update: {e}")
            # Fallback to a generic narrative update
            fallback_narrative = f"{persona.get('name', 'The character')} continues their day at {immediate_environment.get('current_location_name', 'their location')}. They're still focused on {', '.join(persona.get('goals', ['their goals']))} as they navigate the next steps."
            self.console.print(f"[red]Error generating narrative update, using fallback[/red]")
            return fallback_narrative

class Simulacra:
    """Represents a simulated human with personality, goals, and behaviors."""

    def __init__(self, persona_path: Optional[str] = None, console: Console = None):
        # Initialize console
        self.console = console or Console()
        # Initialize persona
        if persona_path and os.path.exists(persona_path):
            with open(persona_path, 'r') as file:
                self.persona = json.load(file)
            logger.info(f"Loaded persona from {persona_path}")
        else:
            self.persona = {
                "name": "Alex Chen",
                "age": 34,
                "occupation": "Software Engineer",
                "personality_traits": ["analytical", "introverted", "creative", "detail-oriented"],
                "goals": ["complete work project before deadline", "find better work-life balance"],
                "current_state": {
                    "physical": "Slightly tired, had coffee 1 hour ago",
                    "emotional": "Mildly stressed about project deadline",
                    "mental": "Focused but distracted occasionally"
                },
                "memory": {
                    "short_term": ["Meeting with team this morning", "Email from boss about deadline"],
                    "long_term": ["Computer Science degree", "5 years at current company"]
                }
            }

        self.history = []
        self.state_path = "simulacra_state.json"
        self.llm_service = LLMService()

    async def _reflect_on_situation(self, observations: str, immediate_environment: Dict, persona_state: Optional[Dict] = None) -> str:
        """Reflect on the current situation based on observations using Gemini API."""
        if persona_state is None:
            persona_state = self.persona

        # Include immediate environment in the reflection
        enhanced_observations = f"""
        Observations:
        {observations}
        
        Your immediate environment:
        {json.dumps(immediate_environment, indent=2)}
        """

        prompt = PromptManager.reflect_on_situation_prompt(enhanced_observations, persona_state)
        system_instruction = 'Reflect on the current situation based on these observations and your environment:'
        
        try:
            reflection_text_dict = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                response_model=DayResponse
            )
            return reflection_text_dict.get('reflect', "I'm trying to understand what's happening around me.")
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return "I'm trying to understand what's happening around me."

    async def _analyze_emotions(self, situation: str, current_emotional_state: str) -> Dict:
        """Analyze emotional response to a situation using Gemini API."""
        prompt = PromptManager.analyze_emotions_prompt(situation, current_emotional_state)
        system_instruction = "Analyze the emotional tone of the following situation and the character's current emotional state."
        
        try:
            emotion_analysis_dict = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                response_model=EmotionAnalysisResponse
            )
            return emotion_analysis_dict
        except Exception as e:
            logger.error(f"Error analyzing emotions: {e}")
            return {
                "primary_emotion": "confused",
                "intensity": "Medium",
                "secondary_emotion": "uncertain",
                "emotional_update": "I'm feeling a bit confused by what's happening."
            }

    async def _decide_action(self, reflection: str, emotional_analysis: Dict, goals: List[str], immediate_environment: Dict) -> Dict:
        """Decide on an action based on reflection, emotional analysis, and immediate environment."""
        prompt = PromptManager.decide_action_prompt(reflection, emotional_analysis, goals, immediate_environment)
        system_instruction = 'Decide on an action to take based on your reflection, emotional analysis, and environment.'
        
        try:
            action_decision_dict = await self.llm_service.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                response_model=ActionDecisionResponse
            )
            return action_decision_dict
        except Exception as e:
            logger.error(f"Error deciding action: {e}")
            return {
                "thought_process": "I need to take a moment to think about my next steps.",
                "action": "Pause and consider options",
                "action_details": {"manner": "thoughtful", "duration": "brief"}
            }

    def save_state(self):
        """Save current simulacra state to a file."""
        try:
            with open(self.state_path, 'w') as file:
                json.dump({
                    "persona": self.persona,
                    "history": self.history
                }, file, indent=2)
            logger.info(f"Successfully saved simulacra state to {self.state_path}")
        except Exception as e:
            logger.error(f"Error saving simulacra state: {e}")

    async def process_perception(self, world_update: Dict[str, Any]) -> Dict[str, Any]:
        """Process perceptions from the world and decide on actions."""

        # Extract the relevant parts from the world update
        world_state = world_update.get("world_state", {})
        immediate_environment = world_update.get("immediate_environment", {})
        observations = world_update.get("observations", [])
        
        # Print simulacra's current state before processing
        self.console.print(Panel(self.get_simulacra_summary(), 
                            title="[bold cyan]SIMULACRA CURRENT STATE[/bold cyan]", 
                            border_style="cyan"))
        
        # Save the perception to history
        self.history.append({
            "timestamp": world_state.get("current_time", "unknown"),
            "date": world_state.get("current_date", "unknown"),
            "perception": observations
        })

        observations_str = json.dumps(observations, indent=2)
        self.console.print(f"\n[bold yellow]Processing observations:[/bold yellow]", observations_str)

        # Use the immediate environment in reflection and decision making
        self.console.print("\n[bold green]Reflecting on situation...[/bold green]")
        reflection_text = await self._reflect_on_situation(observations_str, immediate_environment)
        self.console.print(f"[italic green]Reflection:[/italic green] {reflection_text}\n")
        
        self.console.print("[bold blue]Analyzing emotions...[/bold blue]")
        emotional_analysis = await self._analyze_emotions(reflection_text, self.persona["current_state"]["emotional"])
        self.console.print(f"[italic blue]Emotional analysis:[/italic blue] {json.dumps(emotional_analysis, indent=2)}\n")
        
        self.console.print("[bold magenta]Deciding on action...[/bold magenta]")
        action_decision = await self._decide_action(
            reflection_text, 
            emotional_analysis, 
            self.persona["goals"],
            immediate_environment
        )
        self.console.print(f"[italic magenta]Action decision:[/italic magenta] {json.dumps(action_decision, indent=2)}\n")

        # Update the persona's emotional state based on the analysis
        if "emotional_update" in emotional_analysis:
            previous_emotional = self.persona["current_state"]["emotional"]
            self.persona["current_state"]["emotional"] = emotional_analysis["emotional_update"]
            self.console.print(f"[bold orange3]Emotional state updated:[/bold orange3] {previous_emotional} → {self.persona['current_state']['emotional']}")

        simulacra_response = {
            "thought_process": action_decision.get("thought_process", "Processing observations..."),
            "emotional_update": emotional_analysis.get("emotional_update", "Maintaining emotional state."),
            "action": action_decision.get("action", "Observe further."),
            "action_details": action_decision.get("action_details", {}),
            "updated_state": self.persona["current_state"]
        }

        # Print final response
        response_panel = f"[bold]Thought process:[/bold] {simulacra_response['thought_process']}\n\n"
        response_panel += f"[bold]Emotional update:[/bold] {simulacra_response['emotional_update']}\n\n"
        response_panel += f"[bold]Action:[/bold] {simulacra_response['action']}"
        
        if simulacra_response.get("action_details"):
            response_panel += f"\n\n[bold]Action details:[/bold] {json.dumps(simulacra_response['action_details'], indent=2)}"
        
        self.console.print(Panel(response_panel, 
                            title="[bold red]SIMULACRA RESPONSE[/bold red]", 
                            border_style="red"))

        return simulacra_response

    def get_simulacra_summary(self) -> str:
        """Generate a concise summary of the simulacra's current state."""
        name = self.persona.get("name", "Unknown")
        age = self.persona.get("age", "Unknown")
        occupation = self.persona.get("occupation", "Unknown")
        
        # Get current state information
        physical = self.persona.get("current_state", {}).get("physical", "Unknown physical state")
        emotional = self.persona.get("current_state", {}).get("emotional", "Unknown emotional state")
        mental = self.persona.get("current_state", {}).get("mental", "Unknown mental state")
        
        # Get goals
        goals = self.persona.get("goals", [])
        goals_str = ", ".join(goals) if goals else "No specific goals"
        
        # Get recent memories
        short_term = self.persona.get("memory", {}).get("short_term", [])
        recent_memories = ", ".join(short_term[:2]) if short_term else "No recent memories"
        
        # Create summary with rich formatting
        summary = f"[bold blue]Simulacra:[/bold blue] {name}, {age}, {occupation}\n"
        summary += f"[bold green]Physical state:[/bold green] {physical}\n"
        summary += f"[bold yellow]Emotional state:[/bold yellow] {emotional}\n"
        summary += f"[bold magenta]Mental state:[/bold magenta] {mental}\n"
        summary += f"[bold cyan]Current goals:[/bold cyan] {goals_str}\n"
        summary += f"[bold orange3]Recent memories:[/bold orange3] {recent_memories}"
        
        return summary

async def run_simulation(cycles: int = 3, world_config_path: str = "world_config.yaml",
                       persona_path: Optional[str] = None, new_simulation: bool = False,
                       console: Console = None, reaction_profile: Union[str, Dict, WorldReactionProfile] = "balanced"):
    """
    Run the simulation for a specified number of cycles.
    
    Args:
        cycles: Number of simulation cycles to run
        world_config_path: Path to the world configuration file
        persona_path: Path to the persona file
        new_simulation: Whether to start a new simulation
        console: Rich console for output
        reaction_profile: How the world reacts to the character's actions
                         Can be a profile name, a dict of settings, or a WorldReactionProfile
    """
    
    # If no console is provided, create a new one
    if console is None:
        console = Console()

    # Initialize world and simulacra
    world = WorldEngine(
        world_config_path=world_config_path, 
        load_state=not new_simulation, 
        console=console,
        reaction_profile=reaction_profile
    )
    simulacra = Simulacra(persona_path=persona_path, console=console)

    # Display reaction profile
    if isinstance(reaction_profile, str):
        profile_name = reaction_profile.capitalize()
    elif isinstance(reaction_profile, dict):
        profile_name = "Custom"
    elif isinstance(reaction_profile, WorldReactionProfile):
        profile_name = "Custom"
    else:
        profile_name = "Balanced"
    
    profile_description = world.reaction_profile.get_description()
    
    console.print(Panel(
        f"[bold]SIMULATION START[/bold]\n\n[bold cyan]World Reaction Profile:[/bold cyan] [bold]{profile_name}[/bold]\n\n{profile_description}", 
        title="[bold green]SIMULACRA[/bold green]", 
        border_style="green",
        width=100
    ))

    # If it's a new simulation or world state is not loaded, initialize it
    if new_simulation or world.world_state is None or world.immediate_environment is None:
        console.print("[yellow]Initializing new world state and immediate environment...[/yellow]")
        world_data = await world.initialize_new_world()
        
        # Display the initial narrative context
        if "narrative_context" in world_data:
            console.print(Panel(
                Markdown(world_data["narrative_context"]),
                title="[bold blue]NARRATIVE CONTEXT[/bold blue]",
                border_style="blue",
                width=100
            ))
        
        # Print initial summaries
        console.print("\n" + world.get_world_summary())
        console.print("\n" + world.get_environment_summary())
    else:
        world_data = {
            "world_state": world.world_state,
            "immediate_environment": world.immediate_environment,
            "observations": [
                f"You're in {world.immediate_environment.get('current_location_name', 'a location')}.",
                f"It's {world.world_state.get('current_time', 'daytime')} on {world.world_state.get('current_date', 'today')}.",
                f"The weather is {world.world_state.get('weather_condition', 'normal')}.",
                "You notice your surroundings and gather your thoughts."
            ]
        }
        
        # Print initial summaries for existing world
        console.print("\n" + world.get_world_summary())
        console.print("\n" + world.get_environment_summary())

    # Initial world perception
    world_update = world_data
    narrative_context = world_data.get("narrative_context", "")

    for cycle in range(cycles):
        console.rule(f"[bold cyan]CYCLE {cycle+1}[/bold cyan]")

        # Simulacra perceives and acts
        simulacra_response = await simulacra.process_perception(world_update)

        # World processes the action
        console.print(Panel("[bold]WORLD PROCESSING[/bold]", border_style="blue"))
        world_update = await world.process_update(simulacra_response, simulacra.persona)

        # Display the narrative update if available
        if "narrative_update" in world_update and world_update["narrative_update"]:
            console.print(Panel(
                Markdown(world_update["narrative_update"]),
                title=f"[bold blue]NARRATIVE UPDATE - CYCLE {cycle+1}[/bold blue]",
                border_style="blue",
                width=100
            ))
            # Update the overall narrative context
            narrative_context = world_update["narrative_update"]

        # Print summaries after update
        console.print("\n" + world.get_world_summary())
        console.print("\n" + world.get_environment_summary())

        # Create a table for consequences and observations
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Consequences", style="green")
        table.add_column("Observations", style="yellow")
        
        # Get consequences and observations
        consequences = world_update.get("consequences", [])
        observations = world_update.get("observations", [])
        
        # Determine the maximum number of rows needed
        max_rows = max(len(consequences), len(observations), 1)
        
        # Add rows to the table
        for i in range(min(max_rows, 3)):  # Show at most 3 rows
            cons_text = consequences[i] if i < len(consequences) else ""
            obs_text = observations[i] if i < len(observations) else ""
            table.add_row(cons_text, obs_text)
            
        # If there are more than 3 consequences or observations, add a summary row
        if len(consequences) > 3 or len(observations) > 3:
            cons_more = f"... and {len(consequences) - 3} more" if len(consequences) > 3 else ""
            obs_more = f"... and {len(observations) - 3} more" if len(observations) > 3 else ""
            table.add_row(cons_more, obs_more)
            
        # If there are no consequences or observations, add a message
        if not consequences and not observations:
            table.add_row("No notable consequences.", "Nothing notable observed.")
            
        console.print(Panel(table, title="[bold purple]RESULTS[/bold purple]", border_style="purple"))

        # Small delay for readability
        time.sleep(1)

    # Save final states
    world.save_state()
    simulacra.save_state()

    # Display final narrative summary
    if narrative_context:
        console.print(Panel(
            Markdown(f"**Final Narrative Summary:**\n\n{narrative_context}"),
            title="[bold blue]STORY CONCLUSION[/bold blue]",
            border_style="blue",
            width=100
        ))

    console.print(Panel("[bold]SIMULATION END[/bold]", title="[bold green]SIMULACRA[/bold green]", border_style="green"))

if __name__ == "__main__":
    # Create a default world_config.yaml if it doesn't exist
    if not os.path.exists("world_config.yaml"):
        default_config = {
            "location": {
                "city": "NYC",
                "country": "USA",
                "region": "Northeast"
            },
            "environment": {
                "setting": "Urban downtown",
                "density": "High",
                "noise_level": "Moderate"
            },
            "starting_location": "unknown",
            "simulation_parameters": {
                "initial_time": "auto",
                "time_flow_rate": 1.0,
                "detail_level": "high"
            }
        }
        with open("world_config.yaml", 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
        print("Created default world_config.yaml")

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run a simulacra simulation')
    parser.add_argument('--cycles', type=int, default=3, help='Number of simulation cycles')
    parser.add_argument('--new', action='store_true', help='Start a new simulation')
    parser.add_argument('--config', type=str, default='world_config.yaml', help='Path to world config file')
    parser.add_argument('--persona', type=str, default=None, help='Path to persona file')
    parser.add_argument('--profile', type=str, default='balanced', 
                        choices=['balanced', 'protagonist', 'antagonist', 'comedic', 'dramatic', 
                                'realistic', 'serendipitous', 'challenging'],
                        help='World reaction profile')
    args = parser.parse_args()

    # Create a console
    console = Console()
    
    # Run the simulation with the console
    asyncio.run(run_simulation(
        cycles=args.cycles, 
        world_config_path=args.config, 
        persona_path=args.persona, 
        new_simulation=args.new, 
        console=console,
        reaction_profile=args.profile
    ))