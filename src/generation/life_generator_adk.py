import asyncio
import json
import logging
import uuid # Added import for uuid
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import google_search
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError, model_validator
# Removed LoopAgent and SequentialAgent as we are abandoning LoopAgent
from src.agents import create_search_llm_agent
from src.config import APP_NAME
from src.generation.llm_service import LLMService
from src.generation.models import (
    InitialRelationshipsResponse, Person,
    PersonaDetailsResponse,
)
# Rich imports
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.rule import Rule
from rich.tree import Tree
import calendar


logger = logging.getLogger(__name__)
console = Console()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIFE_SUMMARY_DIR = os.path.join(PROJECT_ROOT, "data", "life_summaries")

T_PydanticModel = TypeVar('T_PydanticModel', bound='BaseModel')

# --- Pydantic model for the input to the PersonaGeneratorAgent_ADK ---
class PersonaInitialInputSchema(BaseModel):
    world_type: str
    world_description: str
    gender_preference: Optional[str] = None

# --- Pydantic model for single year output from ADK iteration agent ---
class SingleYearDataFromADK(BaseModel):
    year: int
    location: str
    summary: str
    news_context_used: Optional[str] = None

class SingleMonthDataFromADK(BaseModel):
    month: int
    location: str
    summary: str
    news_context_used: Optional[str] = None

class SingleDayDataFromADK(BaseModel):
    day: int
    location: str
    summary: str
    news_context_used: Optional[str] = None

    @model_validator(mode='after')
    def check_day_valid(self) -> 'SingleDayDataFromADK':
        if not (1 <= self.day <= 31): # Basic check
            raise ValueError(f"Day {self.day} is outside the general valid range (1-31).")
        return self

class SingleHourDataFromADK(BaseModel):
    hour: int
    location: str
    activity: str
    # news_context_used: Optional[str] = None # News context is from daily for hourly

    @model_validator(mode='after')
    def check_hour_valid(self) -> 'SingleHourDataFromADK':
        if not (0 <= self.hour <= 23):
            raise ValueError(f"Hour {self.hour} is outside the valid range (0-23).")
        return self

class DailyHourlyBreakdownADKResponse(BaseModel):
    """Pydantic model for the ADK hourly agent to return a full day's activities."""
    activities: List[SingleHourDataFromADK]

# --- Pydantic models for ONE-SHOT ADK agent outputs (lists of the above) ---
class YearlySummariesADKResponse(BaseModel):
    birth_month: Optional[int] = None # Still capture this from the first year
    birth_day: Optional[int] = None   # Still capture this from the first year
    summaries: List[SingleYearDataFromADK]

    @model_validator(mode='after')
    def check_day_valid_for_month(self) -> 'YearlySummariesADKResponse':
        if self.birth_month is not None and self.birth_day is not None:
            try:
                ref_year = 2001 # Any non-leap year for general validation
                if not (1 <= self.birth_month <= 12): # Validate month first
                    raise ValueError(f"Birth month {self.birth_month} is invalid.")
                days_in_month = calendar.monthrange(ref_year, self.birth_month)[1]
                if not (1 <= self.birth_day <= days_in_month):
                    raise ValueError(f"Birth day {self.birth_day} is invalid for month {self.birth_month} (max {days_in_month}).")
            except Exception as e:
                raise ValueError(f"Could not validate date components: {e}") from e
        return self

# --- Utility function to extract location ---
def extract_location_from_text(text: str, default: str = "Unknown Location") -> str:
    patterns = [
        r'(?:lived|resided|based|worked|studied|grew up)\s+in\s+([\w\s,-]+)',
        r'moved\s+to\s+([\w\s,-]+)',
        r'born\s+in\s+([\w\s,-]+)',
        r'location\s*[:=]\s*([\w\s,-]+)',
        r'settled\s+in\s+([\w\s,-]+)'
    ]
    found_locations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            location = match.strip().split(', where')[0].split(', which')[0].split(' during')[0].split(' in the year')[0].strip()
            location = re.sub(r'^[.,;\s]+|[.,;\s]+$', '', location)
            if location and len(location) > 1:
                if location.lower() not in ["the city", "a town", "the country", "home", "school", "university", "college"]:
                    found_locations.append(location)
    if found_locations:
        primary_location = found_locations[-1]
        logger.debug(f"Extracted location '{primary_location}' from text.")
        return primary_location
    else:
        logger.warning(f"Could not extract specific location from text snippet: '{text[:100]}...' Using default.")
        return default

# --- Helper Function to call LLM (via LLMService) and validate ---
async def _call_llm_and_get_validated_data(
    llm_service: LLMService,
    prompt: str,
    response_model: Type[T_PydanticModel],
    operation_description: str
) -> Optional[Dict]:
    logger.debug(f"Calling LLM (via LLMService) for '{operation_description}'...")
    try:
        response_dict = await llm_service.generate_content(
            prompt=prompt,
            response_model=response_model
        )
        if not response_dict:
            logger.error(f"LLMService call for '{operation_description}' returned None or empty.")
            return None
        if "error" in response_dict:
            error_msg = response_dict["error"]
            logger.error(f"LLMService returned error during '{operation_description}': {error_msg}")
            if "raw_response" in response_dict:
                 logger.error(f"Raw response was: {response_dict['raw_response']}")
            return None
        logger.debug(f"Successfully received validated data for '{operation_description}'.")
        return response_dict
    except Exception as e:
        logger.error(f"Exception calling LLMService or processing result for '{operation_description}': {e}", exc_info=True)
        return None

# --- Fictional News Generation Helper (ADK context) ---
async def _generate_fictional_news_llm_adk(
    llm_service: LLMService,
    time_period_str: str,
    location: Optional[str],
    world_type: str,
    world_description: str,
    sim_persona_details_json_str: Optional[str] = None
) -> str:
    """Generates plausible fictional news snippets using LLMService for a given period and world."""
    fallback_news = f"No specific fictional news generated for {time_period_str} in {location or 'this region'}."
    if not llm_service:
        logger.warning("LLMService not available for ADK fictional news generation.")
        return fallback_news

    persona_context_str = f"Consider the general context of a persona described by the following JSON: {sim_persona_details_json_str}" if sim_persona_details_json_str else ""

    prompt = f"""
You are a creative news writer for a fictional world.
World Type: {world_type}
World Description: {world_description}
Current Time Period: {time_period_str}
Current Location: {location or 'Not specified, assume general relevance to the world description.'}
{persona_context_str}

Task: Generate 1-3 concise, plausible, and distinct fictional news headlines or very short news summaries (1 sentence each) that would be relevant for someone living in this world, at this location, during this time period.
These news items should feel like they could appear in a local or regional news feed.
Focus on events that are:
- Consistent with the World Type and Description.
- Plausible for the specified time period and location.
- Varied in nature (e.g., local politics, community events, technological advancements (if scifi), cultural happenings, minor unusual occurrences, economic news, environmental observations).
- NOT directly about the specific persona, but general news they might encounter.

Respond ONLY with the news items, each on a new line. Do not include any other conversational text or labels.
If you generate multiple, separate them with a newline.
"""
    try:
        logger.info(f"Requesting LLM (ADK context) for fictional news for {time_period_str} in {location or world_type}...")
        response_text = await asyncio.to_thread(llm_service.generate_content_text, prompt=prompt, temperature=0.8) # Slightly more creative
        if response_text and not response_text.startswith("Error:"):
            cleaned_news = re.sub(r"^(News:|Headlines?:?\s*)","", response_text.strip(), flags=re.IGNORECASE).strip()
            if cleaned_news:
                logger.info(f"LLM (ADK context) generated fictional news for {time_period_str}: {cleaned_news}")
                return cleaned_news
        logger.error(f"LLM (ADK context) failed to generate or returned empty/error for fictional news for {time_period_str}. Response: {response_text}")
        return fallback_news
    except Exception as e:
        logger.error(f"Error generating fictional news via LLM (ADK context) for {time_period_str}: {e}", exc_info=True)
        return fallback_news

# --- ADK Google Search Helper ---
async def perform_adk_search_via_components(
    search_agent: LlmAgent,
    session_service: InMemorySessionService,
    search_query: str
) -> Optional[str]:
    if not search_agent or not session_service:
        logger.error("ADK components (search_agent, session_service) not provided for search helper.")
        return None

    search_helper_app_name = f"{APP_NAME}_SearchHelperSession_{uuid.uuid4().hex[:6]}"
    search_runner = Runner(agent=search_agent, session_service=session_service, app_name=search_helper_app_name)
    search_session_id = (await session_service.create_session(
        app_name=search_helper_app_name,
        user_id="search_helper_user"
    )).id
    logger.info(f"Performing ADK Google Search for: '{search_query}' (using temp session: {search_session_id})")

    trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=search_query)])
    formatted_search_results_string: Optional[str] = None

    try:
        async for event in search_runner.run_async(
            user_id="search_helper_user",
            session_id=search_session_id,
            new_message=trigger_content # Ensure new_message is used
        ):
            if event.is_final_response() and event.content:
                if event.content.parts:
                    raw_search_text = event.content.parts[0].text
                    # Attempt to format it nicely if it looks like a list of results
                    # This is a simple heuristic; ADK search tool might return structured data or just text
                    if raw_search_text and isinstance(raw_search_text, str) and ("\n-" in raw_search_text or "\n*" in raw_search_text or "Search results:" in raw_search_text.lower()):
                        formatted_search_results_string = raw_search_text
                        console.print(Panel(
                            formatted_search_results_string,
                            title=f"Search Results for: '{search_query}'",
                            border_style="blue",
                            expand=False # Keep it concise
                        ))
                    elif raw_search_text: # Otherwise, just use the raw text if it exists
                        formatted_search_results_string = raw_search_text
                        # Optionally print raw text if it's not empty but doesn't fit the "list" heuristic
                        # console.print(Panel(formatted_search_results_string, title=f"Raw Search Output for: '{search_query}'", border_style="dim blue", expand=False))
                    else:
                        formatted_search_results_string = "Search returned no text content."

                    logger.debug(f"ADK Google Search (helper) received final text: {str(formatted_search_results_string)[:200]}...")
                break
        return formatted_search_results_string
    except Exception as e:
        logger.error(f"Error during ADK Google Search (helper) for '{search_query}': {e}", exc_info=True)
        return None
    finally:
        try:
            await session_service.delete_session(session_id=search_session_id, app_name=search_helper_app_name, user_id="search_helper_user")
            logger.debug(f"Deleted temporary search session: {search_session_id}")
        except Exception as e_del:
            logger.warning(f"Could not delete temporary search session {search_session_id}: {e_del}")

# --- ADK Sub-Agent Definitions ---

def create_persona_generator_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="PersonaGeneratorAgent_ADK",
        model=model_name,
        description="Generates a detailed random fictional persona based on a comprehensive input prompt.",
        instruction="""You are an expert character creator.
You will receive a detailed prompt containing:
- World Type
- World Description
- Gender Preference

Your primary task is to use this information to generate a persona that is deeply consistent and plausible within THAT SPECIFIC world.
The gender preference from the prompt should be strictly followed. If "gender_preference" is null or "any", you may choose any gender appropriate for the world.
Ensure the persona's details (occupation, background, location, etc.) are consistent with the provided world description from the prompt.
Age should be an integer between 1 and 120 (typically 18-65 for an adult).
Generate a plausible birthdate (YYYY-MM-DD) consistent with the generated age and the world type (e.g., future year for SciFi, past year for Fantasy/Historical).

**Required Fields (Match the PersonaDetailsResponse schema precisely):**
Name, Age (integer), Gender, Occupation, Current_location (City, State/Country appropriate for the world), Personality_Traits (list, 3-6 adjectives), Birthplace (City, State/Country appropriate for the world), Birthdate (YYYY-MM-DD), Birthtime (optional HH:MM), Education (optional), Physical_Appearance (brief description), Hobbies, Skills, Languages, Health_Status, Family_Background, Life_Goals, Notable_Achievements, Fears, Strengths, Weaknesses, Ethnicity, Religion (optional), Political_Views (optional), Favorite_Foods, Favorite_Music, Favorite_Books, Favorite_Movies, Pet_Peeves, Dreams, Past_Traumas (optional).

The output MUST be a single JSON object that directly matches the PersonaDetailsResponse schema.
Do NOT wrap the response in any other keys (e.g., do not use a "persona" key at the root of your JSON output).
The JSON output should start directly with the fields of PersonaDetailsResponse (e.g., "Name": "...", "Age": ..., etc.).
""",
        # input_schema=PersonaInitialInputSchema, # Input is now via prompt text
        output_schema=PersonaDetailsResponse,
        output_key="persona_details_json"
    )

def create_initial_relationships_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="InitialRelationshipsAgent_ADK",
        model=model_name,
        description="Establishes initial family structure based on a prompt containing persona details.",
        instruction="""You are a family tree specialist.
You will receive a prompt containing persona details as a JSON string.
Based on these details, establish a plausible immediate family structure (parents, siblings). For each person, include their 'name', 'relationship' to the main character, and brief 'details' (e.g., occupation, age relative to character, key personality trait) consistent with the persona's background and world.

**Output Format (Respond ONLY with valid JSON matching the InitialRelationshipsResponse schema):**
`{{"parents": [{{"name": "...", "relationship": "...", "details": "..."}}], "siblings": [{{"name": "...", "relationship": "...", "details": "..."}}]}}`
If no siblings, provide an empty list for "siblings". Parents list should typically have 1 or 2 entries.
""",
        output_schema=InitialRelationshipsResponse,
        output_key="initial_relationships_json"
    )

def create_yearly_iteration_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    # This agent now generates ALL yearly summaries in one call
    return LlmAgent(
        name="YearlyIterationAgent_ADK",
        model=model_name,
        description="Generates summaries for a range of years based on a comprehensive input prompt.",
        instruction="""You are a biographer.
You will receive a detailed prompt containing:
- Persona details (JSON string)
- Initial family relationships (JSON string)
- Birth year and the last year to generate summaries for.
- World Type and World Description.
- News/External Context for the relevant years (JSON string or "No external context used.").
- Instructions on whether to use real-world events or invent fictional ones.

**Instructions:**
- For each year from the birth year to the last year specified in the prompt, provide a rich, narrative summary of the persona's life. This summary should detail:
    - **Major Life Events:** Significant personal milestones (e.g., education, relationships, family changes, personal achievements, setbacks, health issues).
    - **Professional/Occupational Developments:** Career progression, job changes, skill acquisition, significant projects, or periods of unemployment, relevant to their age and the world.
    - **Impact of External Context:** Weave in the provided "News/External Context" for the year. Show how these events might have directly impacted the persona's life, decisions, or outlook, or how they served as a backdrop to their personal story. If no direct impact, describe how the persona might have perceived or reacted to these events.
- Include the persona's primary location for that year and key life events (personal, professional, relationships), consistent with all provided context.
- If generating birth year summary, include plausible birth month and day.
- For each year's summary object in the "summaries" list, include a "news_context_used" field. This field should contain the specific news string for THAT year, extracted from the "News/External Context" JSON provided in your input prompt. If no specific context was available for a year in the input, use a placeholder like "No specific news context provided for this year."

**Output Format:** Respond ONLY with JSON:
`{{"birth_month": Optional[int], "birth_day": Optional[int], "summaries": [{{"year": int, "location": str, "summary": str, "news_context_used": str}}]}}`
""",
        output_schema=YearlySummariesADKResponse, # Output is now a list of summaries
        output_key="yearly_summaries_list_json"  # New output key
    )

def create_monthly_iteration_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="MonthlyIterationAgent_ADK",
        model=model_name,
        description="Generates a summary for a single month based on a comprehensive input prompt.",
        instruction="""You are a detailed chronicler.
You will receive a detailed prompt containing all necessary information:
- Persona details (JSON string)
- World context (Type and Description)
- Target year and month for the summary.
- Yearly summary for the target year (JSON string).
- Location for the year.
- News context for the target month (if available, as a string).
- Instructions on whether to use real context or invent fictional events.

Task: Based on ALL the information provided in the input prompt, generate a rich, narrative summary for the specified month of the specified year. This summary should detail:
- **Key Personal Developments:** Significant changes in routine, mood, relationships, health, or personal projects.
- **Notable Events or Activities:** Specific occurrences, outings, or experiences that stood out during the month.
- **Connection to Yearly/World Context:** Briefly link the month's activities or mood to the broader yearly summary and any provided news/external context for the month. Show how these larger factors might have subtly influenced the persona's daily life or thoughts during this month.
Include the persona's location (can usually be inferred from the year's location or specified in the prompt).

Respond ONLY with valid JSON matching SingleMonthDataFromADK:
`{{"month": int, "location": str, "summary": str, "news_context_used": str}}`
The "month" field MUST be the target month from the prompt. Location should be consistent with yearly location unless specified. The "summary" should be a comprehensive paragraph.
The "news_context_used" field should contain the news string that was provided in the prompt for this month, or a statement like "No external context used." if none was provided or applicable.
""",
        output_schema=SingleMonthDataFromADK,
        output_key="current_month_data_json"
    )

def create_daily_iteration_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="DailyIterationAgent_ADK",
        model=model_name,
        description="Generates a summary for a single day based on a comprehensive input prompt.",
        instruction="""You are a meticulous diarist.
You will receive a detailed prompt containing all necessary information:
- Persona details (JSON string)
- World context (Type and Description)
- Target date (year, month, day) for the summary.
- Monthly summary for the target month (JSON string).
- Yearly summary for the target year (JSON string).
- Location for the month/year.
- News context for the target day (if available, as a string).
- Instructions on whether to use real context or invent fictional events.

Task: Based on ALL the information provided in the input prompt, generate a detailed, narrative summary for the specified day. This summary should describe:
- **Main Activities & Events:** What did the persona primarily do? Were there any significant interactions, tasks, or occurrences?
- **Mood and Reflections (Optional but encouraged):** Briefly touch upon the persona's likely mood or any brief reflections they might have had, consistent with their personality and the day's events.
- **Integration of Context:** Subtly weave in elements from the monthly summary and any provided daily news/external context. How did the day fit into the month's flow? Did any external events affect their plans or thoughts?
Include the persona's location (can usually be inferred from the month's location or specified in the prompt).

Respond ONLY with valid JSON matching SingleDayDataFromADK:
`{{"day": int, "location": str, "summary": str, "news_context_used": str}}`
The "day" field MUST be the target day from the prompt. Location should be consistent. The "summary" should be a comprehensive paragraph.
The "news_context_used" field should contain the news string that was provided in the prompt for this day, or a statement like "No external context used." if none was provided or applicable.
""",
        output_schema=SingleDayDataFromADK,
        output_key="current_day_data_json"
    )

def create_hourly_iteration_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="HourlyIterationAgent_ADK",
        model=model_name,
        description="Generates a full day's hourly activities (0-23) based on a comprehensive input prompt.",
        instruction="""You are an activity logger.
You will receive a detailed prompt containing all necessary information:
- Persona details (JSON string)
- World context (Type and Description)
- Target date (year, month, day) for which to generate a full 24-hour breakdown.
- Daily summary for the target day (JSON string).
- Monthly and Yearly summaries for context.
- Location for the day.
- Immediate Local News/Events Context for this DAY (if available, as a string, freshly searched).
- Instructions on whether to use real context or invent fictional events/activities.

Task: Based on ALL the information provided in the input prompt, generate a plausible primary activity for **each hour from 00:00 to 23:00** for the specified day.
For each hour, the activity description should be specific. Include the persona's location for that hour (can often be inferred from the daily location).
Consider how *immediate, small-scale local conditions and spontaneous occurrences* might affect the persona's activities or attention throughout the day. Examples include:
    - **Environmental Factors:** Sudden changes in weather (e.g., a quick downpour, unexpected sunshine, a gust of wind), noticeable sounds (e.g., nearby construction, distant music, a car alarm), strong or unusual smells (e.g., food from a street vendor, blooming flowers, smoke), changes in lighting (e.g., a street light flickering on, the sun setting).
    - **Logistical & Navigational Issues:** Minor public transportation delays (e.g., a bus running a few minutes late), an unexpected short detour due to road maintenance, a longer-than-expected queue for a planned activity (like coffee), a specific item they intended to interact with being temporarily unavailable (e.g., ATM out of order).
    - **Social Micro-Interactions & Observations:** Briefly bumping into an acquaintance, a short, mundane exchange with a shopkeeper or barista, being asked for simple directions, observing a minor public spectacle (e.g., a street performer setting up, a dog chasing a squirrel, a brief, harmless commotion).
    - **Personal Needs & Impulses:** Suddenly feeling hungry or thirsty and deciding to grab a quick snack or drink, needing to use a restroom, an impulse to check their phone for messages, a fleeting thought or memory triggered by something in the environment, a moment of people-watching.
    - **Minor Opportunities & Obstacles:** Noticing an interesting item in a shop window, a street vendor with an appealing snack, a minor spill on the sidewalk to avoid, a dropped item they might notice or briefly consider picking up.
This Immediate Local News/Events Context for the DAY should be interpreted for its *most direct and immediate local impact* on the hours of the day, if any. For example, news of a nearby traffic accident could explain transportation delays or unusual traffic noise during relevant hours.

Respond ONLY with valid JSON matching DailyHourlyBreakdownADKResponse: `{{"activities": [{{"hour": int, "location": str, "activity": str}}, ...]}}`
The "activities" list should contain an entry for each hour from 0 to 23.
""",
        output_schema=DailyHourlyBreakdownADKResponse, # Changed to the new schema
        output_key="current_hour_data_json" # Output key remains the same, but content will be a list
    )

# --- Fallback Python-based agent function (if ADK agent fails for persona) ---
async def agent_generate_random_persona_fallback(
    llm_service: LLMService,
    world_type: str,
    world_description: str,
    gender_preference: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    logger.info(f"Fallback Agent: Generating random persona for world type '{world_type}'...")
    gender_instruction = f"- The persona's gender MUST be '{gender_preference}'. Ensure the name chosen is appropriate for this gender." if gender_preference and gender_preference != "any" else "- You may choose any gender appropriate for the world."
    prompt = f"""
Create a detailed and plausible random fictional persona profile suitable for the following world:
World Type: {world_type}
World Description: {world_description}
Instructions: Age 1-120 (typically 18-65 for an adult). Generate plausible birthdate (YYYY-MM-DD). {gender_instruction} Ensure all details are consistent with the world.
Required Fields (Match the PersonaDetailsResponse schema precisely): Name, Age (integer), Gender, Occupation, Current_location (City, State/Country appropriate for the world), Personality_Traits (list, 3-6 adjectives), Birthplace (City, State/Country appropriate for the world), Birthdate (YYYY-MM-DD), Birthtime (optional HH:MM), Education (optional), Physical_Appearance (brief description), Hobbies, Skills, Languages, Health_Status, Family_Background, Life_Goals, Notable_Achievements, Fears, Strengths, Weaknesses, Ethnicity, Religion (optional), Political_Views (optional), Favorite_Foods, Favorite_Music, Favorite_Books, Favorite_Movies, Pet_Peeves, Dreams, Past_Traumas (optional).
Respond ONLY with valid JSON.
"""
    validated_data = await _call_llm_and_get_validated_data(llm_service, prompt, PersonaDetailsResponse, "fallback random persona generation")
    if validated_data:
        logger.info(f"Fallback Agent: Successfully generated persona: {validated_data.get('Name', 'Unknown')}")
        return validated_data
    else:
        logger.error("Fallback Agent: Failed to generate or validate random persona.")
        return None

# --- Orchestration (Python-driven, using ADK Agents for tasks) ---
async def run_adk_orchestrated_life_generation(
    session_service: InMemorySessionService,
    search_llm_agent: Optional[LlmAgent],
    llm_service: LLMService, # Changed from llm_service_for_validation
    initial_user_input: Dict[str, Any],
    persona_details_override: Optional[Dict],
    generation_params: Dict[str, Any],
    session_id_for_workflow: str,
    session_app_name: str,
    session_user_id: str,
    allow_real_context: bool,
) -> Optional[Dict[str, Any]]:
    console.print(Rule("Starting ADK-Orchestrated Life Generation", style="bold magenta"))
    life_summary: Dict[str, Any] = {
        "persona_details": None, "initial_relationships": None,
        "yearly_summaries": {}, "monthly_summaries": {},
        "daily_summaries": {}, "hourly_breakdowns": {},
        "generation_info": generation_params.get("generation_info", {})
    }

    generated_persona_data: Optional[Dict[str, Any]] = None
    persona_details_json_str: Optional[str] = None

    if persona_details_override:
        generated_persona_data = persona_details_override
        try:
            generated_persona_data = PersonaDetailsResponse.model_validate(generated_persona_data).model_dump()
            persona_details_json_str = json.dumps(generated_persona_data)
        except (ValidationError, TypeError) as e:
            logger.error(f"Overridden persona data failed validation or serialization: {e}")
            generated_persona_data = None
            persona_details_json_str = None
        logger.info("Using overridden persona details.")
    else:
        console.print(Rule("ADK Standalone Agent: Persona Generation", style="yellow"))
        persona_agent = create_persona_generator_adk_agent()
        persona_runner = Runner(
            agent=persona_agent,
            session_service=session_service,
            app_name=session_app_name
        )
        # Construct the prompt for persona generation
        persona_prompt_text = f"""
World Type: {initial_user_input.get("world_type", "N/A")}
World Description: {initial_user_input.get("world_description", "N/A")}
Gender Preference: {initial_user_input.get("gender_preference", "male")}

Task: Create a detailed fictional persona based on the above information.
Follow the detailed instructions provided to the PersonaGeneratorAgent_ADK regarding output format and required fields.
Respond ONLY with valid JSON matching the PersonaDetailsResponse schema.
"""
        persona_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=persona_prompt_text)])

        async for event in persona_runner.run_async(
            session_id=session_id_for_workflow,
            new_message=persona_trigger_content,
            user_id=session_user_id
        ):
            logger.debug(f"ADK Persona Event: Author={event.author}, Final={event.is_final_response()}, Content={str(event.content)[:200]}, Actions={event.actions}")
            if event.is_final_response():
                logger.info("PersonaGeneratorAgent_ADK has finished processing events.")
                break
            if event.error_message:
                logger.error(f"Error during PersonaGeneratorAgent_ADK execution: {event.error_message} (Code: {event.error_code})")
                break

        retrieved_session_after_persona = await session_service.get_session(
            session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id
        )

        if retrieved_session_after_persona and retrieved_session_after_persona.state:
            final_session_state_after_persona = retrieved_session_after_persona.state
            logger.info("Retrieved session state after PersonaGeneratorAgent_ADK.")
            # console.print(Rule("Contents of Session State (After Persona Agent)", style="bold purple"))
            # console.print(pretty_repr(final_session_state_after_persona)) # Too verbose
            generated_persona_data = final_session_state_after_persona.get(persona_agent.output_key)

            if generated_persona_data and isinstance(generated_persona_data, dict):
                try:
                    generated_persona_data = PersonaDetailsResponse.model_validate(generated_persona_data).model_dump()
                    # console.print(Panel(pretty_repr(generated_persona_data), title="ADK Persona (from Session State)", expand=False)) # Too verbose
                    persona_details_json_str = json.dumps(generated_persona_data)
                except (ValidationError, TypeError) as e:
                    logger.error(f"ADK Persona (from Session State) re-validation/serialization error: {e}. Data: {str(generated_persona_data)[:500]}")
                    generated_persona_data = None
                    persona_details_json_str = None
            else:
                logger.error(f"Persona data (key: '{persona_agent.output_key}') is MISSING or not a dict from session state. Found: {type(generated_persona_data)}")
                generated_persona_data = None
                persona_details_json_str = None
        else:
            logger.error(f"Could not retrieve session or session state after PersonaGeneratorAgent_ADK execution.")
            generated_persona_data = None
            persona_details_json_str = None

    relationships_data_dict = None
    if generated_persona_data and persona_details_json_str:
        console.print(Rule("ADK Standalone Agent: Initial Relationships", style="yellow"))
        relationships_agent = create_initial_relationships_adk_agent()
        relationships_runner = Runner(
            agent=relationships_agent, session_service=session_service, app_name=session_app_name
        )

        relationships_prompt_text = f"""
Persona Details (JSON):
{persona_details_json_str}

Task: Based on the provided persona details, establish a plausible immediate family structure (parents, siblings).
Follow the detailed instructions provided to the InitialRelationshipsAgent_ADK regarding output format (InitialRelationshipsResponse schema).
Respond ONLY with valid JSON.
"""
        relationships_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=relationships_prompt_text)])

        if persona_details_json_str: # Only run if persona_details_json_str is valid
            async for event in relationships_runner.run_async(
                session_id=session_id_for_workflow, user_id=session_user_id, new_message=relationships_trigger_content # Ensure new_message is used
            ):
                logger.debug(f"ADK Relationships Event: Author={event.author}, Final={event.is_final_response()}, Content={str(event.content)[:200]}, Actions={event.actions}")
                if event.is_final_response():
                    logger.info("InitialRelationshipsAgent_ADK has finished processing events.")
                    break
                if event.error_message:
                    logger.error(f"Error during InitialRelationshipsAgent_ADK execution: {event.error_message} (Code: {event.error_code})")
                    break

            retrieved_session_after_relationships = await session_service.get_session(
                session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id
            )
            if retrieved_session_after_relationships and retrieved_session_after_relationships.state:
                final_session_state_after_relationships = retrieved_session_after_relationships.state
                logger.info("Retrieved session state after InitialRelationshipsAgent_ADK.")
                # console.print(Rule("Contents of Session State (After Relationships Agent)", style="bold purple"))
                # console.print(pretty_repr(final_session_state_after_relationships)) # Too verbose
                relationships_data_dict = final_session_state_after_relationships.get(relationships_agent.output_key)
            else:
                 logger.error(f"Could not retrieve session or session state after InitialRelationshipsAgent_ADK execution.")
    elif generated_persona_data and not persona_details_json_str:
        logger.error("Persona data was generated but failed to serialize to JSON string. Skipping relationships generation.")
    else:
        logger.warning("Skipping relationships generation because persona generation failed or produced no valid data.")

    if relationships_data_dict and isinstance(relationships_data_dict, dict):
        try:
            life_summary["initial_relationships"] = InitialRelationshipsResponse.model_validate(relationships_data_dict).model_dump()
            logger.info("Successfully validated relationships from session state.")
            # console.print(Panel(pretty_repr(life_summary["initial_relationships"]), title="Processed Relationships (from Session State)", expand=False, border_style="green")) # Too verbose
        except (ValidationError, TypeError) as e:
            logger.error(f"ADK Relationships (from Session State) validation error: {e}. Data: {str(relationships_data_dict)[:500]}")
            life_summary["initial_relationships"] = None
    elif generated_persona_data:
        logger.warning("Relationships agent did not produce output or output was not a dict, though persona was generated.")
        life_summary["initial_relationships"] = None

    if not generated_persona_data:
        if not persona_details_override:
            logger.warning("ADK Persona generation failed, trying LLMService fallback...")
            generated_persona_data = await agent_generate_random_persona_fallback(
                llm_service, initial_user_input.get("world_type"), # Use passed llm_service
                initial_user_input.get("world_description"), initial_user_input.get("gender_preference")
            )
            if generated_persona_data:
                persona_details_json_str = json.dumps(generated_persona_data)
                if not life_summary.get("initial_relationships") and persona_details_json_str:
                    console.print(Rule("ADK Standalone Agent: Initial Relationships (for Fallback Persona)", style="orange1"))
                    relationships_agent_fallback = create_initial_relationships_adk_agent()
                    relationships_runner_fallback = Runner(
                        agent=relationships_agent_fallback, session_service=session_service, app_name=session_app_name
                    )

                    fallback_relationships_prompt_text = f"""
Persona Details (JSON):
{persona_details_json_str}

Task: Based on the provided persona details (this is for a fallback persona), establish a plausible immediate family structure.
Follow the detailed instructions provided to the InitialRelationshipsAgent_ADK regarding output format.
Respond ONLY with valid JSON.
"""
                    fallback_rel_trigger = genai_types.UserContent(parts=[genai_types.Part(text=fallback_relationships_prompt_text)])
                    if persona_details_json_str:
                        async for event in relationships_runner_fallback.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=fallback_rel_trigger): # Ensure new_message
                            if event.is_final_response(): break
                            if event.error_message: logger.error(f"Error during relationships for fallback: {event.error_message}"); break

                        retrieved_session_after_fallback_rel = await session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
                        if retrieved_session_after_fallback_rel and retrieved_session_after_fallback_rel.state:
                            relationships_output_dict_fallback = retrieved_session_after_fallback_rel.state.get(relationships_agent_fallback.output_key)
                            if relationships_output_dict_fallback and isinstance(relationships_output_dict_fallback, dict):
                                try:
                                    life_summary["initial_relationships"] = InitialRelationshipsResponse.model_validate(relationships_output_dict_fallback).model_dump()
                                except Exception as e_rel_fallback: logger.error(f"Error processing relationships for fallback: {e_rel_fallback}")

    if not generated_persona_data:
        logger.error("All persona generation attempts failed.")
        return None
    life_summary["persona_details"] = generated_persona_data

    simulated_current_dt = generation_params["generation_timestamp"]
    birthdate_str_from_persona = generated_persona_data.get("Birthdate")

    try:
        birthdate_obj = datetime.strptime(str(birthdate_str_from_persona), "%Y-%m-%d").date()
        birth_year = birthdate_obj.year
        birth_month_initial = birthdate_obj.month
        birth_day_initial = birthdate_obj.day

        actual_age = simulated_current_dt.year - birth_year - \
                     ((simulated_current_dt.month, simulated_current_dt.day) < (birth_month_initial, birth_day_initial))
        end_year_for_generation = birth_year + actual_age

        life_summary["generation_info"].update({
            "birth_year": birth_year,
            "birth_month": birth_month_initial,
            "birth_day": birth_day_initial,
            "actual_age_at_generation": actual_age,
            "end_year_for_generation": end_year_for_generation
        })
        logger.info(f"Orchestrator Date Setup: Birth {birthdate_obj}, Actual Age {actual_age}, End Year {end_year_for_generation}")
    except (ValueError, TypeError) as e:
        logger.error(f"Error processing birthdate from persona ('{birthdate_str_from_persona}'): {e}. Cannot proceed.")
        return life_summary

    initial_relationships_str = json.dumps(life_summary["initial_relationships"] or {})
    persona_details_json_str_for_loop = json.dumps(generated_persona_data)

    console.print(Rule("Generating Yearly Summaries (One-Shot ADK Call)", style="bold yellow"))
    news_context_by_year_with_str_keys: Dict[str, str] = {} # Changed to Dict[str, str]

    console.print(Panel("Pre-fetching/generating news context for all relevant years...", title="Yearly News Context Phase", style="dim"))
    for year_to_search in range(birth_year, end_year_for_generation + 1):
        year_str = str(year_to_search)
        if allow_real_context and search_llm_agent:
            search_query = f"major world events {year_to_search}"
            search_results_string = await perform_adk_search_via_components(
                search_llm_agent, session_service, search_query
            )
            if search_results_string:
                news_context_by_year_with_str_keys[year_str] = search_results_string
            else:
                news_context_by_year_with_str_keys[year_str] = "No specific external events found or search failed."
        else:
            logger.info(f"Generating fictional news for year {year_to_search} (ADK Orchestration)...")
            fictional_news = await _generate_fictional_news_llm_adk(
                llm_service, f"the year {year_to_search}",
                f"the world of {initial_user_input.get('world_type')}", # General location for yearly
                initial_user_input.get("world_type"), initial_user_input.get("world_description"),
                persona_details_json_str_for_loop
            )
            news_context_by_year_with_str_keys[year_str] = fictional_news
            logger.info(f"Fictional news for ADK year {year_to_search}: {fictional_news[:200]}...")

    world_details_json = json.dumps({
        "world_type": initial_user_input.get("world_type"),
        "world_description": initial_user_input.get("world_description")
    })

    yearly_iteration_agent = create_yearly_iteration_adk_agent()
    yearly_iteration_runner = Runner(agent=yearly_iteration_agent, session_service=session_service, app_name=session_app_name)

    years_to_generate_list = list(range(birth_year, end_year_for_generation + 1))
    num_years_to_generate = len(years_to_generate_list)

    if num_years_to_generate > 0:
        # console.print(Rule(f"Generating {num_years_to_generate} Years ({birth_year}-{end_year_for_generation})", style="cyan")) # Covered by main rule

        context_instruction_for_yearly = ""
        if allow_real_context:
            context_instruction_for_yearly = "Use the provided News Context (which may contain real-world events if applicable to the world type) and your internal knowledge for these years."
        else:
            context_instruction_for_yearly = f"Use the provided Fictional News Context (snippets like: '{str(list(news_context_by_year_with_str_keys.values())[0])[:100].replace('\n',' / ')}...') and your internal knowledge of the fictional world to detail the persona's life. The Fictional News Context was generated to be plausible for this world."

        yearly_prompt_text = f"""
Persona Details (JSON): {persona_details_json_str_for_loop}
Initial family: {initial_relationships_str}
Born in {birth_year}. Summaries needed up to end of {end_year_for_generation}.
World Details: {world_details_json}
News/External Context (or 'No external context used.'):
{json.dumps(news_context_by_year_with_str_keys)}

Contextual Instruction: {context_instruction_for_yearly}

Task: Generate yearly summaries for the persona from {birth_year} to {end_year_for_generation}.
Follow the detailed instructions provided to the YearlyIterationAgent_ADK regarding output format (YearlySummariesADKResponse schema),
including birth month/day if it's the birth year summary.
Respond ONLY with valid JSON.
"""
        yearly_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=yearly_prompt_text)])

        if not await session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id):
            logger.error("Session lost before yearly one-shot call. Aborting yearly summaries.")
        else:
            yearly_output_dict: Optional[Dict] = None
            async for event in yearly_iteration_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=yearly_trigger_content): # Ensure new_message is used
                if event.is_final_response():
                    logger.info(f"YearlyIterationAgent_ADK finished.")
                    break
                if event.error_message:
                    logger.error(f"Error during YearlyIterationAgent_ADK execution: {event.error_message}")
                    break

            session_after_yearly = await session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
            if session_after_yearly and session_after_yearly.state:
                yearly_output_dict = session_after_yearly.state.get(yearly_iteration_agent.output_key)
                if yearly_output_dict and isinstance(yearly_output_dict, dict):
                    try:
                        validated_yearly_data_response = YearlySummariesADKResponse.model_validate(yearly_output_dict)
                        # Process the list of summaries
                        for year_data_item in validated_yearly_data_response.summaries:
                            year_num = year_data_item.year
                            life_summary["yearly_summaries"][year_num] = year_data_item.model_dump() # Store the whole SingleYearDataFromADK object
                            logger.info(f"Processed data for year: {year_num}")
                        # Capture birth month/day from the response if provided
                        life_summary["generation_info"]["birth_month"] = validated_yearly_data_response.birth_month or life_summary["generation_info"].get("birth_month", 1)
                        life_summary["generation_info"]["birth_day"] = validated_yearly_data_response.birth_day or life_summary["generation_info"].get("birth_day", 1)
                        logger.info(f"Birth month/day from YearlySummariesADKResponse: {life_summary['generation_info']['birth_month']}/{life_summary['generation_info']['birth_day']}")
                    except ValidationError as ve:
                        logger.error(f"Validation error for YearlySummariesADKResponse output: {ve}. Data: {yearly_output_dict}")
                else:
                    logger.error(f"YearlyIterationAgent_ADK did not produce valid output. Found: {yearly_output_dict}")
            else:
                logger.error(f"Could not retrieve session state after YearlyIterationAgent_ADK execution.")

        # if life_summary["yearly_summaries"]: # Check if any summaries were processed
            # console.print(Rule("Processing Accumulated Yearly Summaries", style="green")) # Covered by main step rule
            # console.print(Panel(pretty_repr(life_summary["yearly_summaries"]), title="ADK Yearly Summaries (Accumulated)", expand=False)) # Too verbose
    else:
        logger.info("No years to generate for yearly summaries.")

    console.print(Rule("Generating Monthly Summaries (Relevant Months)", style="bold yellow"))
    sim_curr_year = life_summary["generation_info"]["current_year"]
    sim_curr_month = life_summary["generation_info"]["current_month"]
    months_to_process_for_loop: List[Tuple[int, int]] = []
    current_sim_date_obj_m = date(sim_curr_year, sim_curr_month, 1)
    months_to_process_for_loop.append((current_sim_date_obj_m.year, current_sim_date_obj_m.month))
    prev_month_date_obj_m = current_sim_date_obj_m - timedelta(days=1)
    prev_month_date_obj_m = prev_month_date_obj_m.replace(day=1)

    birth_month_for_compare = life_summary["generation_info"].get("birth_month", 1)
    if not (isinstance(birth_month_for_compare, int) and 1 <= birth_month_for_compare <= 12):
        birth_month_for_compare = 1
        logger.warning(f"Invalid birth_month '{life_summary['generation_info'].get('birth_month')}' found. Using 1 for date comparison.")

    if date(prev_month_date_obj_m.year, prev_month_date_obj_m.month, 1) >= date(birth_year, birth_month_for_compare, 1):
        months_to_process_for_loop.append((prev_month_date_obj_m.year, prev_month_date_obj_m.month))

    monthly_iteration_agent = create_monthly_iteration_adk_agent()
    monthly_iteration_runner = Runner(agent=monthly_iteration_agent, session_service=session_service, app_name=session_app_name)

    for target_year_for_months, target_month_for_loop_start in months_to_process_for_loop:
        # console.print(Rule(f"Generating Month {target_year_for_months}-{target_month_for_loop_start:02d}", style="cyan")) # Covered by main rule
        yearly_summary_for_target_year_dict = life_summary["yearly_summaries"].get(target_year_for_months)
        if not yearly_summary_for_target_year_dict:
            logger.warning(f"No yearly summary for {target_year_for_months}, skipping monthly generation.")
            continue
        yearly_summary_for_target_year_json = json.dumps(yearly_summary_for_target_year_dict)
        news_context_for_this_month_key = f"{target_year_for_months}-{target_month_for_loop_start:02d}"
        news_context_for_this_month_str = "No external context used."

        if allow_real_context and search_llm_agent:
            s_results_string = await perform_adk_search_via_components(
                search_llm_agent, session_service,
                f"major events {yearly_summary_for_target_year_dict.get('location', '')} {news_context_for_this_month_key}"
            )
            if s_results_string: news_context_for_this_month_str = s_results_string
        else:
            logger.info(f"Generating fictional news for month {news_context_for_this_month_key} (ADK Orchestration)...")
            fictional_news_monthly = await _generate_fictional_news_llm_adk(
                llm_service, f"the month of {news_context_for_this_month_key}",
                yearly_summary_for_target_year_dict.get('location'),
                initial_user_input.get("world_type"), initial_user_input.get("world_description"),
                persona_details_json_str_for_loop
            )
            news_context_for_this_month_str = fictional_news_monthly
            logger.info(f"Fictional news for ADK month {news_context_for_this_month_key}: {fictional_news_monthly[:200]}...")

        current_session_check_m = await session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if not current_session_check_m:
            logger.error(f"Session lost before monthly iteration for {target_year_for_months}-{target_month_for_loop_start}. Aborting.")
            break

        monthly_context_instruction = ""
        if allow_real_context:
            monthly_context_instruction = "Use the provided News Context and your internal knowledge of real-world events for this month."
        else:
            monthly_context_instruction = f"Use the provided Fictional News Context (snippets like: '{news_context_for_this_month_str[:100].replace('\n',' / ')}...') and your internal knowledge of the fictional world to detail the persona's life for this month. The Fictional News Context was generated to be plausible for this world."

        monthly_prompt_text = f"""
Persona Details (JSON): {persona_details_json_str_for_loop}
World Details: {world_details_json}
Target Year: {target_year_for_months}, Target Month: {target_month_for_loop_start}
Yearly Summary for {target_year_for_months} (JSON): {yearly_summary_for_target_year_json}
Location for Year {target_year_for_months}: {yearly_summary_for_target_year_dict.get('location', 'Unknown')}
News Context for {news_context_for_this_month_key}: "{news_context_for_this_month_str}"

Contextual Instruction: {monthly_context_instruction}

Task: Generate a monthly summary for month {target_month_for_loop_start} of year {target_year_for_months}.
Follow the detailed instructions provided to the MonthlyIterationAgent_ADK regarding output format (SingleMonthDataFromADK schema).
The 'month' field in the output must be {target_month_for_loop_start}.
The 'news_context_used' field should be "{news_context_for_this_month_str}".
Respond ONLY with valid JSON.
"""
        monthly_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=monthly_prompt_text)])

        async for event in monthly_iteration_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=monthly_trigger_content): # Ensure new_message
            if event.is_final_response(): break
            if event.error_message: logger.error(f"Error in MonthlyIterationAgent: {event.error_message}"); break

        session_after_iteration_m = await session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if session_after_iteration_m and session_after_iteration_m.state:
            iteration_output_dict_m = session_after_iteration_m.state.get(monthly_iteration_agent.output_key)
            if iteration_output_dict_m and isinstance(iteration_output_dict_m, dict):
                try:
                    month_data = SingleMonthDataFromADK.model_validate(iteration_output_dict_m).model_dump()
                    life_summary["monthly_summaries"].setdefault(target_year_for_months, {})[month_data["month"]] = month_data
                    # console.print(Panel(pretty_repr(month_data), title=f"ADK Monthly: {target_year_for_months}-{month_data['month']:02d}", expand=False)) # Too verbose
                except ValidationError as ve:
                    logger.error(f"Validation error for month {target_year_for_months}-{target_month_for_loop_start} output: {ve}. Data: {iteration_output_dict_m}")
            else:
                logger.error(f"Monthly iteration for {target_year_for_months}-{target_month_for_loop_start} did not produce valid output.")
        else:
            logger.error(f"Could not retrieve session state after monthly iteration for {target_year_for_months}-{target_month_for_loop_start}.")
            break

    console.print(Rule("Generating Daily Summaries (Last 7 Simulated Days)", style="bold yellow"))
    sim_curr_day = life_summary["generation_info"]["current_day"]
    days_to_process_for_loop_d: List[date] = []
    character_current_sim_date_d = date(sim_curr_year, sim_curr_month, sim_curr_day)
    for i in range(7):
        day_to_add = character_current_sim_date_d - timedelta(days=i)
        if day_to_add >= birthdate_obj: days_to_process_for_loop_d.append(day_to_add)
    days_to_process_for_loop_d.reverse()

    daily_iteration_agent = create_daily_iteration_adk_agent()
    daily_iteration_runner = Runner(agent=daily_iteration_agent, session_service=session_service, app_name=session_app_name)

    for target_date_for_daily in days_to_process_for_loop_d:
        # console.print(Rule(f"Generating Day {target_date_for_daily.isoformat()}", style="cyan")) # Covered by main rule
        yr_d, m_d, d_d = target_date_for_daily.year, target_date_for_daily.month, target_date_for_daily.day
        monthly_summary_d_dict = life_summary["monthly_summaries"].get(yr_d, {}).get(m_d)
        if not monthly_summary_d_dict:
            logger.warning(f"No monthly summary for {yr_d}-{m_d}, skipping daily generation for {target_date_for_daily.isoformat()}.")
            continue
        monthly_summary_d_json = json.dumps(monthly_summary_d_dict)
        news_key_d = target_date_for_daily.isoformat()
        news_str_d = "No external context."

        if allow_real_context and search_llm_agent:
            s_res_d_string = await perform_adk_search_via_components(
                search_llm_agent, session_service,
                f"local events and news {monthly_summary_d_dict.get('location','')} {news_key_d}"
            )
            if s_res_d_string: news_str_d = s_res_d_string
        else:
            logger.info(f"Generating fictional news for day {news_key_d} (ADK Orchestration)...")
            fictional_news_daily = await _generate_fictional_news_llm_adk(
                llm_service, f"the day {news_key_d}",
                monthly_summary_d_dict.get('location'),
                initial_user_input.get("world_type"), initial_user_input.get("world_description"),
                persona_details_json_str_for_loop
            )
            news_str_d = fictional_news_daily
            logger.info(f"Fictional news for ADK day {news_key_d}: {fictional_news_daily[:200]}...")

        current_session_check_d = await session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if not current_session_check_d:
            logger.error(f"Session lost before daily iteration for {target_date_for_daily.isoformat()}. Aborting.")
            break

        daily_context_instruction = ""
        if allow_real_context:
            daily_context_instruction = "Use the provided News Context and your internal knowledge of real-world events for this day."
        else:
            daily_context_instruction = f"Use the provided Fictional News Context (snippets like: '{news_str_d[:100].replace('\n',' / ')}...') and your internal knowledge of the fictional world to detail the persona's life for this day. The Fictional News Context was generated to be plausible for this world."

        yearly_summary_for_daily_context_json = json.dumps(life_summary["yearly_summaries"].get(yr_d, {}))

        daily_prompt_text = f"""
Persona Details (JSON): {persona_details_json_str_for_loop}
World Details: {world_details_json}
Target Date: {target_date_for_daily.isoformat()} (Year: {yr_d}, Month: {m_d}, Day: {d_d})
Monthly Summary for {yr_d}-{m_d:02d} (JSON): {monthly_summary_d_json}
Yearly Summary for {yr_d} (JSON): {yearly_summary_for_daily_context_json}
Location for Month {yr_d}-{m_d:02d}: {monthly_summary_d_dict.get('location', 'Unknown')}
News Context for {news_key_d}: "{news_str_d}"

Contextual Instruction: {daily_context_instruction}

Task: Generate a daily summary for {target_date_for_daily.isoformat()}.
Follow the detailed instructions provided to the DailyIterationAgent_ADK regarding output format (SingleDayDataFromADK schema).
The 'day' field in the output must be {d_d}.
The 'news_context_used' field should be "{news_str_d}".
Respond ONLY with valid JSON.
"""
        daily_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=daily_prompt_text)])

        async for event in daily_iteration_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=daily_trigger_content): # Ensure new_message
            if event.is_final_response(): break
            if event.error_message: logger.error(f"Error in DailyIterationAgent for {target_date_for_daily.isoformat()}: {event.error_message}"); break

        session_after_iteration_d = await session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if session_after_iteration_d and session_after_iteration_d.state:
            iteration_output_dict_d = session_after_iteration_d.state.get(daily_iteration_agent.output_key)
            if iteration_output_dict_d and isinstance(iteration_output_dict_d, dict):
                try:
                    day_data = SingleDayDataFromADK.model_validate(iteration_output_dict_d).model_dump()
                    life_summary["daily_summaries"].setdefault(yr_d, {}).setdefault(m_d, {})[day_data["day"]] = day_data
                    # console.print(Panel(pretty_repr(day_data), title=f"ADK Daily: {target_date_for_daily.isoformat()}", expand=False)) # Too verbose
                except ValidationError as ve:
                    logger.error(f"Validation error for day {target_date_for_daily.isoformat()} output: {ve}. Data: {iteration_output_dict_d}")
            else:
                logger.error(f"Daily iteration for {target_date_for_daily.isoformat()} did not produce valid output.")
        else:
            logger.error(f"Could not retrieve session state after daily iteration for {target_date_for_daily.isoformat()}.")
            break

    console.print(Rule("Generating Hourly Breakdowns (Simulated Today & Yesterday)", style="bold yellow"))
    sim_curr_hour = life_summary["generation_info"]["current_hour"]
    days_for_hourly_loop_h: List[date] = [character_current_sim_date_d]
    yesterday_sim_date_h = character_current_sim_date_d - timedelta(days=1)
    if yesterday_sim_date_h >= birthdate_obj: days_for_hourly_loop_h.append(yesterday_sim_date_h)

    hourly_iteration_agent = create_hourly_iteration_adk_agent()
    hourly_iteration_runner = Runner(agent=hourly_iteration_agent, session_service=session_service, app_name=session_app_name)

    for target_date_for_hourly in days_for_hourly_loop_h:
        # console.print(Rule(f"Generating Hourly for Day {target_date_for_hourly.isoformat()}", style="cyan")) # Covered by main rule
        yr_h, m_h, d_h = target_date_for_hourly.year, target_date_for_hourly.month, target_date_for_hourly.day
        daily_summary_h = life_summary["daily_summaries"].get(yr_h, {}).get(m_h, {}).get(d_h)
        if not daily_summary_h:
            logger.warning(f"No daily summary for {target_date_for_hourly.isoformat()}, skipping hourly generation.")
            continue
        daily_summary_h_json = json.dumps(daily_summary_h)
        
        # Fetch specific news for this day to be used for hourly generation
        hourly_specific_news_str = "No external hourly context used."
        if target_date_for_hourly == character_current_sim_date_d: # Only fetch/generate for "today"
            if allow_real_context and search_llm_agent:
                hourly_search_query = f"immediate local news or events affecting {daily_summary_h.get('location', 'this area')} on {target_date_for_hourly.strftime('%B %d, %Y')}"
                logger.info(f"Performing ADK Google Search for daily context for hourly generation (day {target_date_for_hourly.isoformat()}): '{hourly_search_query}'")
                s_results_hourly_string = await perform_adk_search_via_components(
                    search_llm_agent, session_service, hourly_search_query
                )
                if s_results_hourly_string:
                    hourly_specific_news_str = s_results_hourly_string
                else:
                    hourly_specific_news_str = "No specific hourly-relevant events found from daily search."
            else:
                logger.info(f"Generating fictional news for hourly context for day {target_date_for_hourly.isoformat()} (ADK Orchestration)...")
                fictional_news_hourly_day = await _generate_fictional_news_llm_adk(
                    llm_service, f"local happenings on the day {target_date_for_hourly.isoformat()}",
                    daily_summary_h.get('location'),
                    initial_user_input.get("world_type"), initial_user_input.get("world_description"),
                    persona_details_json_str_for_loop
                )
                hourly_specific_news_str = fictional_news_hourly_day
                logger.info(f"Fictional news for ADK hourly context (day {target_date_for_hourly.isoformat()}): {fictional_news_hourly_day[:200]}...")

        # Single call for the entire day's hourly breakdown
        if not await session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id): # Check session before call
            logger.error(f"Session lost before hourly generation for day {target_date_for_hourly.isoformat()}. Aborting day.")
            continue # Skip to next day if session is lost

        hourly_context_instruction_for_day = ""
        if allow_real_context:
            hourly_context_instruction_for_day = "Consider the provided local news context for the day. Invent plausible activities for each hour."
        else:
            hourly_context_instruction_for_day = f"Use the provided Fictional Local Context for the day (snippets like: '{hourly_specific_news_str[:100].replace('\n',' / ')}...') and your internal knowledge of the fictional world to detail the persona's hourly activities. The Fictional Local Context was generated to be plausible for this world."

        monthly_summary_for_hourly_context_json = json.dumps(life_summary["monthly_summaries"].get(yr_h, {}).get(m_h, {}))
        yearly_summary_for_hourly_context_json = json.dumps(life_summary["yearly_summaries"].get(yr_h, {}))

        hourly_prompt_text_for_day = f"""
Persona Details (JSON): {persona_details_json_str_for_loop}
World Details: {world_details_json}
Target Date for Full Hourly Breakdown: {target_date_for_hourly.isoformat()}
Daily Summary for {target_date_for_hourly.isoformat()} (JSON): {daily_summary_h_json}
Monthly Summary for {yr_h}-{m_h:02d} (JSON): {monthly_summary_for_hourly_context_json}
Yearly Summary for {yr_h} (JSON): {yearly_summary_for_hourly_context_json}
Location for Day {target_date_for_hourly.isoformat()}: {daily_summary_h.get('location', 'Unknown')}
Immediate Local News/Events Context for this DAY ({target_date_for_hourly.isoformat()}): "{hourly_specific_news_str}"

Contextual Instruction for the Day: {hourly_context_instruction_for_day}

Task: Generate a plausible primary activity for **each hour from 00:00 to 23:00** for day {target_date_for_hourly.isoformat()}.
Follow the detailed instructions provided to the HourlyIterationAgent_ADK regarding output format (DailyHourlyBreakdownADKResponse schema), ensuring the 'activities' list contains entries for all 24 hours.
Respond ONLY with valid JSON.
"""
        hourly_trigger_content_for_day = genai_types.UserContent(parts=[genai_types.Part(text=hourly_prompt_text_for_day)])

        all_day_hourly_activities_dict: Optional[Dict] = None
        async for event in hourly_iteration_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=hourly_trigger_content_for_day):
            if event.is_final_response(): break
            if event.error_message: logger.error(f"Error in HourlyIterationAgent for day {target_date_for_hourly.isoformat()}: {event.error_message}"); break

        session_after_hourly_day = await session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if session_after_hourly_day and session_after_hourly_day.state:
            all_day_hourly_activities_dict = session_after_hourly_day.state.get(hourly_iteration_agent.output_key)

        hourly_activities_for_day_accumulated: Dict[int, Dict] = {}
        if all_day_hourly_activities_dict and isinstance(all_day_hourly_activities_dict, dict):
            try:
                validated_daily_hourly_data = DailyHourlyBreakdownADKResponse.model_validate(all_day_hourly_activities_dict)
                
                # Determine how many hours to actually store based on whether it's the current simulated day
                max_h_to_store = sim_curr_hour + 1 if target_date_for_hourly == character_current_sim_date_d else 24

                for hour_data_model in validated_daily_hourly_data.activities:
                    if hour_data_model.hour < max_h_to_store: # Truncate if it's the current day
                        hourly_activities_for_day_accumulated[hour_data_model.hour] = hour_data_model.model_dump()
                    elif hour_data_model.hour >= 24 : # Safety check for invalid hour from LLM
                        logger.warning(f"Hourly agent returned activity for invalid hour {hour_data_model.hour} for day {target_date_for_hourly.isoformat()}. Skipping.")

            except ValidationError as ve:
                logger.error(f"Validation error for full day hourly output for {target_date_for_hourly.isoformat()}: {ve}. Data: {all_day_hourly_activities_dict}")
        else:
            logger.error(f"Hourly generation for day {target_date_for_hourly.isoformat()} did not produce valid output or output was not a dict.")

        if hourly_activities_for_day_accumulated:
            life_summary["hourly_breakdowns"].setdefault(yr_h, {}).setdefault(m_h, {})[d_h] = {
                "activities": hourly_activities_for_day_accumulated,
                "news_context_for_hour_generation": hourly_specific_news_str # Store the news context used for generating these hours
            }
            # console.print(Panel(pretty_repr(hourly_activities_for_day_accumulated), title=f"ADK Hourly: {target_date_for_hourly.isoformat()}", expand=False)) # Too verbose
        else:
            logger.warning(f"No hourly activities accumulated for {target_date_for_hourly.isoformat()}.")

    console.print(Rule("ADK Orchestration Complete", style="bold green"))

    # --- Final Output: Build and Print Summary Tree (similar to life_generator.py) ---
    if life_summary.get("persona_details"):
        summary_tree = Tree(f"[bold blue]Life Summary for {life_summary['persona_details'].get('Name', 'Unknown ADK Persona')}[/bold blue] (ADK Generated)")
        
        # Persona Details
        persona_node = summary_tree.add(f"[bold green]Persona Details[/bold green]")
        for key, value in life_summary["persona_details"].items():
            persona_node.add(f"{key}: {str(value)[:200]}") # Truncate long values for display
        
        # Birth Information
        birth_info_node = summary_tree.add(f"[bold green]Birth Information[/bold green]")
        gen_info = life_summary.get("generation_info", {})
        birth_info_node.add(f"Year: {gen_info.get('birth_year')}")
        birth_info_node.add(f"Month: {gen_info.get('birth_month')}")
        birth_info_node.add(f"Day: {gen_info.get('birth_day')}")
        birth_info_node.add(f"Actual Age at Generation: {gen_info.get('actual_age_at_generation')}")
        
        # Relationships
        relationships_node = summary_tree.add(f"[bold green]Initial Relationships[/bold green]")
        if life_summary.get("initial_relationships"):
            parents = life_summary["initial_relationships"].get("parents", [])
            siblings = life_summary["initial_relationships"].get("siblings", [])
            if parents:
                for parent in parents:
                    relationships_node.add(f"Parent: {parent.get('name', 'Unknown')} ({parent.get('relationship', '')}) - {str(parent.get('details',''))[:100]}")
            else:
                relationships_node.add("No parents listed.")
            if siblings:
                for sibling in siblings:
                    relationships_node.add(f"Sibling: {sibling.get('name', 'Unknown')} ({sibling.get('relationship', '')}) - {str(sibling.get('details',''))[:100]}")
            else:
                relationships_node.add("No siblings listed.")
        else:
            relationships_node.add("No relationship data.")

        # Yearly Summaries
        yearly_summaries_node = summary_tree.add(f"[bold green]Yearly Summaries[/bold green]")
        if life_summary.get("yearly_summaries"):
            for year, data in sorted(life_summary.get("yearly_summaries", {}).items()):
                news_used = data.get('news_context_used', data.get('news', 'N/A')) # Check both keys for backward compatibility
                yearly_summaries_node.add(f"[bold yellow]{year}[/bold yellow] ({data.get('location','N/A')}): {str(data.get('summary', 'No summary'))[:150]}... (News: {str(news_used)[:50]}...)")
        else:
            yearly_summaries_node.add("No yearly summaries generated.")

        # Monthly Summaries
        monthly_summaries_node = summary_tree.add(f"[bold green]Monthly Summaries[/bold green]")
        if life_summary.get("monthly_summaries"):
            for year, months in sorted(life_summary.get("monthly_summaries", {}).items()):
                year_node = monthly_summaries_node.add(f"[bold yellow]{year}[/bold yellow]")
                for month, data in sorted(months.items()):
                    year_node.add(f"[bold cyan]{month:02d}[/bold cyan] ({data.get('location','N/A')}): {str(data.get('summary', 'No summary'))[:120]}... (News: {str(data.get('news_context_used','N/A'))[:50]}...)")
        else:
            monthly_summaries_node.add("No monthly summaries generated.")
            
        # Daily Summaries
        daily_summaries_node = summary_tree.add(f"[bold green]Daily Summaries[/bold green]")
        if life_summary.get("daily_summaries"):
            for year, months in sorted(life_summary.get("daily_summaries", {}).items()):
                year_node = daily_summaries_node.add(f"[bold yellow]{year}[/bold yellow]")
                for month, days in sorted(months.items()):
                    month_node = year_node.add(f"[bold cyan]{month:02d}[/bold cyan]")
                    for day, data in sorted(days.items()):
                        month_node.add(f"[bold magenta]{day:02d}[/bold magenta] ({data.get('location','N/A')}): {str(data.get('summary', 'No summary'))[:100]}... (News: {str(data.get('news_context_used','N/A'))[:50]}...)")
        else:
            daily_summaries_node.add("No daily summaries generated.")

        # Hourly Breakdowns
        hourly_breakdowns_node = summary_tree.add(f"[bold green]Hourly Breakdowns[/bold green]")
        if life_summary.get("hourly_breakdowns"):
            for year, months in sorted(life_summary.get("hourly_breakdowns", {}).items()):
                year_node = hourly_breakdowns_node.add(f"[bold yellow]{year}[/bold yellow]")
                for month, days in sorted(months.items()):
                    month_node = year_node.add(f"[bold cyan]{month:02d}[/bold cyan]")
                    for day, data in sorted(days.items()):
                        day_node = month_node.add(f"[bold magenta]{day:02d}[/bold magenta] (News for day's hours: {str(data.get('news_context_for_hour_generation','N/A'))[:50]}...)")
                        activities = data.get("activities", {})
                        if activities:
                            for hour, activity_data in sorted(activities.items()):
                                day_node.add(f"[dim]{hour:02d}:00[/dim] ({activity_data.get('location','N/A')}): {str(activity_data.get('activity', 'Unknown Activity'))[:80]}")
                        else:
                            day_node.add("No hourly activities recorded.")
        else:
            hourly_breakdowns_node.add("No hourly breakdowns generated.")
            
        console.print(summary_tree)

    return life_summary

# --- Main Entry Point ---
async def generate_new_simulacra_background(
    sim_id: str,
    world_instance_uuid: Optional[str],
    world_type: str,
    world_description: str,
    allow_real_context: bool,
    gender_preference: Optional[str] = "male"
) -> Optional[Dict[str, Any]]:
    initial_allow_real_context = allow_real_context
    logger.info(f"Starting ADK-based Simulacra background generation for sim: {sim_id}...")

    # Instantiate services per call for better isolation and testability
    llm_service_instance: LLMService
    session_service_instance: InMemorySessionService
    search_llm_agent_instance: Optional[LlmAgent] = None

    try:
        llm_service_instance = LLMService(api_key=os.getenv("GOOGLE_API_KEY"))
        logger.info("LLMService instance initialized for this generation run.")
    except Exception as e:
        logger.critical(f"Error initializing LLMService: {e}.", exc_info=True)
        return None

    try:
        session_service_instance = InMemorySessionService()
        logger.info("InMemorySessionService instance initialized for this generation run.")
    except Exception as e:
        logger.critical(f"Error initializing InMemorySessionService: {e}.", exc_info=True)
        return None

    if allow_real_context:
        try:
            search_llm_agent_instance = create_search_llm_agent() # This uses google_search tool
            logger.info("SearchLLMAgent instance (with google_search tool) initialized for this run.")
        except Exception as e_search_agent:
            logger.error(f"Failed to create SearchLLMAgent instance: {e_search_agent}. Real context searches will be disabled.", exc_info=True)
            allow_real_context = False # Disable if agent creation fails
            search_llm_agent_instance = None

    main_workflow_session_id: Optional[str] = None
    app_name_for_session = f"{APP_NAME}_LifeGenWorkflowInstance_{uuid.uuid4().hex[:6]}"
    user_id_for_session = f"workflow_user_{sim_id}"

    if session_service_instance:
        try:
            created_session = await session_service_instance.create_session(
                app_name=app_name_for_session, user_id=user_id_for_session
            )
            main_workflow_session_id = created_session.id
            logger.info(f"Main workflow session created: {main_workflow_session_id} with app_name='{app_name_for_session}' and user_id='{user_id_for_session}'")
        except Exception as e_sess:
            logger.error(f"Failed to create main workflow session: {e_sess}", exc_info=True)
            return None
    else:
        logger.error("SessionService instance not available for workflow session creation.")
        return None

    persona_initial_input = {
        "world_type": world_type,
        "world_description": world_description,
        "gender_preference": gender_preference,
    }

    generation_start_time = datetime.now(timezone.utc)
    logger.info(f"Initial reference timestamp for generation: {generation_start_time.isoformat()}")

    generation_params = {
        "generation_timestamp": generation_start_time,
        "generation_info": {
            "generated_at_realworld": generation_start_time.isoformat(),
            "current_year": generation_start_time.year,
            "current_month": generation_start_time.month,
            "current_day": generation_start_time.day,
            "current_hour": generation_start_time.hour,
        }
    }

    life_data = await run_adk_orchestrated_life_generation(
        session_service=session_service_instance,
        search_llm_agent=search_llm_agent_instance,
        llm_service=llm_service_instance,
        initial_user_input=persona_initial_input,
        persona_details_override=None,
        generation_params=generation_params,
        session_id_for_workflow=main_workflow_session_id,
        session_app_name=app_name_for_session,
        session_user_id=user_id_for_session,
        allow_real_context=allow_real_context
    )

    if not life_data:
        logger.error("ADK-orchestrated life summary generation failed critically.")
        return None

    life_data["sim_id"] = sim_id
    life_data["world_instance_uuid"] = world_instance_uuid
    final_gen_info = life_data.get("generation_info", {})
    final_gen_info["real_context_search_attempted"] = initial_allow_real_context
    final_gen_info["real_context_search_enabled"] = allow_real_context
    final_gen_info["generated_at_realworld"] = generation_start_time.isoformat()

    persona_name_safe = re.sub(r'[^\w\-]+', '_', life_data.get("persona_details", {}).get('Name', 'UnknownADK'))[:30]
    # Corrected filename format: life_summary_CharacterName_WorldUUID.json
    output_filename = f"life_summary_{persona_name_safe}_{world_instance_uuid}.json"
    output_path = os.path.join(LIFE_SUMMARY_DIR, output_filename)

    console.print(Rule(f"Saving ADK results to [green]{output_path}[/green]", style="bold green"))
    try:
        os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(life_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"ADK Results saved successfully to {output_path}.")
    except Exception as e_save:
        logger.error(f"Error saving ADK results to {output_path}: {e_save}", exc_info=True)

    return life_data
