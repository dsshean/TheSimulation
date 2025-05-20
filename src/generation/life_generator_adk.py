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
    birth_month: Optional[int] = None
    birth_day: Optional[int] = None

    @model_validator(mode='after')
    def check_day_valid_for_month(self) -> 'SingleYearDataFromADK':
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
    search_session_id = session_service.create_session(
        app_name=search_helper_app_name, 
        user_id="search_helper_user"
    ).id
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
                    formatted_search_results_string = event.content.parts[0].text
                    logger.debug(f"ADK Google Search (helper) received final text: {formatted_search_results_string[:200]}...")
                break 
        return formatted_search_results_string
    except Exception as e:
        logger.error(f"Error during ADK Google Search (helper) for '{search_query}': {e}", exc_info=True)
        return None
    finally:
        try:
            session_service.delete_session(session_id=search_session_id, app_name=search_helper_app_name, user_id="search_helper_user") 
            logger.debug(f"Deleted temporary search session: {search_session_id}")
        except Exception as e_del:
            logger.warning(f"Could not delete temporary search session {search_session_id}: {e_del}")

# --- ADK Sub-Agent Definitions ---

def create_persona_generator_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="PersonaGeneratorAgent_ADK",
        model=model_name,
        description="Generates a detailed random fictional persona.",
        instruction="""You are an expert character creator.
Your input will be a JSON string containing "world_type", "world_description", and "gender_preference".

Your primary task is to use the provided "world_type" and "world_description" from the input JSON to generate a persona that is deeply consistent and plausible within THAT SPECIFIC world.
The "gender_preference" from the input JSON should also be strictly followed. If "gender_preference" is null or "any", you may choose any gender appropriate for the world.

Create a detailed fictional persona. Ensure the persona's details (occupation, background, location, etc.) are consistent with the provided world description.
Age should be an integer between 1 and 120 (typically 18-65 for an adult).
Generate a plausible birthdate (YYYY-MM-DD) consistent with the generated age and the world type (e.g., future year for SciFi, past year for Fantasy/Historical).

**Required Fields (Match the PersonaDetailsResponse schema precisely):**
Name, Age (integer), Gender, Occupation, Current_location (City, State/Country appropriate for the world), Personality_Traits (list, 3-6 adjectives), Birthplace (City, State/Country appropriate for the world), Birthdate (YYYY-MM-DD), Birthtime (optional HH:MM), Education (optional), Physical_Appearance (brief description), Hobbies, Skills, Languages, Health_Status, Family_Background, Life_Goals, Notable_Achievements, Fears, Strengths, Weaknesses, Ethnicity, Religion (optional), Political_Views (optional), Favorite_Foods, Favorite_Music, Favorite_Books, Favorite_Movies, Pet_Peeves, Dreams, Past_Traumas (optional).

The output MUST be a single JSON object that directly matches the PersonaDetailsResponse schema.
Do NOT wrap the response in any other keys (e.g., do not use a "persona" key at the root of your JSON output).
The JSON output should start directly with the fields of PersonaDetailsResponse (e.g., "Name": "...", "Age": ..., etc.).
""",
        input_schema=PersonaInitialInputSchema,
        output_schema=PersonaDetailsResponse, 
        output_key="persona_details_json"
    )

def create_initial_relationships_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="InitialRelationshipsAgent_ADK",
        model=model_name,
        description="Establishes initial family structure.",
        instruction="""Based on the input persona details: {persona_details_json}
Establish a plausible immediate family structure (parents, siblings). For each person, include their 'name', 'relationship' to the main character, and brief 'details' (e.g., occupation, age relative to character, key personality trait) consistent with the persona's background and world.

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
        description="Generates summaries for a range of years.",
        instruction="""You are a biographer.
You will receive the following data from the session state:
- `persona_details_json`: JSON string of the persona.
- `initial_relationships_json`: JSON string of initial family.
- `world_details_json`: JSON string with "world_type" and "world_description".
- `news_context_by_year_json`: JSON string of a dictionary mapping years (as strings) to news context strings. (Pre-fetched by orchestrator)

Task: Generate summaries for ALL years listed in `years_to_generate_list_int`.
1. Parse all inputs.
2. For each year in `years_to_generate_list_int`:
    - Determine the persona's age in that year.
    - Parse `news_context_by_year_json` (this is a JSON string of a dictionary where keys are year strings and values are news strings for that year).
    - For the current year you are processing, retrieve the news string from this parsed dictionary. This string will either be actual news or a message like "No specific external events found or search failed." if the pre-fetch was unsuccessful. Use this retrieved news string as the value for the `news_context_used` field in your output for this year.
    - Based on the persona's details, relationships, world context, and news, generate a summary of major events for that year.
    - Include the persona's primary location for that year (be specific, e.g., "City, State/Country").
    - If the year is the `birth_year_int`, include plausible `birth_month` (1-12) and `birth_day` (1-31, valid for month) in the output for that specific year's object.

Respond ONLY with valid JSON matching the YearlySummariesADKResponse schema:
`{{"birth_month": Optional[int], "birth_day": Optional[int], "summaries": [{{"year": int, "location": str, "summary": str, "news_context_used": str, "birth_month": Optional[int], "birth_day": Optional[int]}}]}}`
The `birth_month` and `birth_day` fields at the top level should be the values for the birth year. The `summaries` list must contain an object for EACH year requested in `years_to_generate_list_int`.
""",
        # Input schema could be a new model expecting years_to_generate_list_int, but for now, it's passed via session.
        output_schema=YearlySummariesADKResponse, # Output is now a list of summaries
        output_key="yearly_summaries_list_json"  # New output key
    )

def create_monthly_iteration_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="MonthlyIterationAgent_ADK",
        model=model_name,
        description="Generates a summary for a single month.",
        instruction="""You are a detailed chronicler.
You will receive from session state:
- `persona_details_json`, `world_details_json`, `initial_relationships_json`
- `target_year_int`: The year for which months are being generated.
- `yearly_summary_for_target_year_json`: JSON string of the SingleYearDataFromADK for `target_year_int`.
- `loop_iteration_count`: 0-indexed month (0 for January, 1 for February, etc.).
- `news_context_by_month_key_json`: JSON dict mapping "YYYY-MM" keys to news strings.
- `allow_real_context_bool`.

Task: Generate a summary for ONE month.
1. `current_processing_month = loop_iteration_count + 1`.
2. Parse inputs. Get `yearly_summary_text` and `yearly_location` from `yearly_summary_for_target_year_json`.
3. Get `news_for_current_month`: From `news_context_by_month_key_json` for key `target_year_int`-`current_processing_month`. If no news or not allowed, use "Focus on personal development and fictional world events."

Based on all info, provide a detailed summary for `current_processing_month` of `target_year_int`. The summary should cover significant personal events or developments.
Respond ONLY with valid JSON matching SingleMonthDataFromADK:
`{{"month": int, "location": str, "summary": str, "news_context_used": str}}`
The "month" field MUST be `current_processing_month`. Location should be consistent with yearly location unless specified. The "summary" should be a comprehensive paragraph.
""",
        output_schema=SingleMonthDataFromADK,
        output_key="current_month_data_json"
    )

def create_daily_iteration_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="DailyIterationAgent_ADK",
        model=model_name,
        description="Generates a summary for a single day.",
        instruction="""You are a meticulous diarist.
You will receive from session state:
- `persona_details_json`, `world_details_json`, `initial_relationships_json`
- `target_year_int`, `target_month_int`.
- `monthly_summary_for_target_month_json`: JSON string of SingleMonthDataFromADK for the target month.
- `loop_iteration_count`: 0-indexed day of the month (0 for day 1, 1 for day 2, etc.).
- `news_context_by_day_key_json`: JSON dict mapping "YYYY-MM-DD" keys to news strings.
- `allow_real_context_bool`.

Task: Generate a summary for ONE day.
1. `current_processing_day = loop_iteration_count + 1`.
2. Parse inputs. Get `monthly_summary_text` and `monthly_location` from `monthly_summary_for_target_month_json`.
3. Get `news_for_current_day`: From `news_context_by_day_key_json` for key `target_year_int`-`target_month_int`-`current_processing_day`. If no news or not allowed, use "Focus on personal routines and fictional local events."

Based on all info, provide a detailed summary for `current_processing_day` of `target_month_int`/`target_year_int`. The summary should cover key activities or events of the day.
Respond ONLY with valid JSON matching SingleDayDataFromADK:
`{{"day": int, "location": str, "summary": str, "news_context_used": str}}`
The "day" field MUST be `current_processing_day`. Location should be consistent. The "summary" should be a comprehensive paragraph.
""",
        output_schema=SingleDayDataFromADK,
        output_key="current_day_data_json"
    )

def create_hourly_iteration_adk_agent(model_name: str = "gemini-1.5-flash-latest") -> LlmAgent:
    return LlmAgent(
        name="HourlyIterationAgent_ADK",
        model=model_name,
        description="Generates an activity for a single hour.",
        instruction="""You are an activity logger.
You will receive from session state:
- `persona_details_json`, `world_details_json`, `initial_relationships_json`
- `target_year_int`, `target_month_int`, `target_day_int`.
- `daily_summary_for_target_day_json`: JSON string of SingleDayDataFromADK for the target day.
- `loop_iteration_count`: 0-indexed hour of the day (0 for 00:00-00:59, 1 for 01:00-01:59, etc.).
- `news_context_for_this_hourly_day_str`: String containing news context for the target day (this is the news context from the daily summary).

Task: Generate an activity for ONE hour.
1. `current_processing_hour = loop_iteration_count`.
2. Parse inputs. Get `daily_summary_text` and `daily_location` from `daily_summary_for_target_day_json`.
3. Consider the `news_context_for_this_hourly_day_str` to inform the activities if relevant.

Based on all info, describe the primary activity for `current_processing_hour` on `target_day_int`/`target_month_int`/`target_year_int`. The activity description should be specific.
Respond ONLY with valid JSON matching SingleHourDataFromADK: `{{"hour": int, "location": str, "activity": str}}`
The "hour" field MUST be `current_processing_hour`. Location should be consistent. The "activity" should be a descriptive sentence.
""",
        output_schema=SingleHourDataFromADK,
        output_key="current_hour_data_json"
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
        pipeline_input_text = json.dumps(initial_user_input)
        pipeline_input_content = genai_types.UserContent(parts=[genai_types.Part(text=pipeline_input_text)])

        async for event in persona_runner.run_async(
            session_id=session_id_for_workflow,
            new_message=pipeline_input_content, # Ensure new_message is used
            user_id=session_user_id
        ):
            logger.debug(f"ADK Persona Event: Author={event.author}, Final={event.is_final_response()}, Content={str(event.content)[:200]}, Actions={event.actions}")
            if event.is_final_response():
                logger.info("PersonaGeneratorAgent_ADK has finished processing events.")
                break
            if event.error_message:
                logger.error(f"Error during PersonaGeneratorAgent_ADK execution: {event.error_message} (Code: {event.error_code})")
                break

        retrieved_session_after_persona = session_service.get_session(
            session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id
        )
        
        if retrieved_session_after_persona and retrieved_session_after_persona.state:
            final_session_state_after_persona = retrieved_session_after_persona.state
            logger.info("Retrieved session state after PersonaGeneratorAgent_ADK.")
            console.print(Rule("Contents of Session State (After Persona Agent)", style="bold purple"))
            console.print(pretty_repr(final_session_state_after_persona))
            generated_persona_data = final_session_state_after_persona.get(persona_agent.output_key)
            
            if generated_persona_data and isinstance(generated_persona_data, dict):
                try:
                    generated_persona_data = PersonaDetailsResponse.model_validate(generated_persona_data).model_dump()
                    console.print(Panel(pretty_repr(generated_persona_data), title="ADK Persona (from Session State)", expand=False))
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
        current_session_for_rel = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if current_session_for_rel:
            current_session_for_rel.state[persona_agent.output_key] = persona_details_json_str 
            logger.info(f"Primed session state with persona_details_json (string) for relationships agent using key '{persona_agent.output_key}'.")
        else:
            logger.error(f"Session {session_id_for_workflow} not found to prime state for relationships agent.")

        if current_session_for_rel:
            relationships_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=persona_details_json_str)])
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

            retrieved_session_after_relationships = session_service.get_session(
                session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id
            )
            if retrieved_session_after_relationships and retrieved_session_after_relationships.state:
                final_session_state_after_relationships = retrieved_session_after_relationships.state
                logger.info("Retrieved session state after InitialRelationshipsAgent_ADK.")
                console.print(Rule("Contents of Session State (After Relationships Agent)", style="bold purple"))
                console.print(pretty_repr(final_session_state_after_relationships))
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
            console.print(Panel(pretty_repr(life_summary["initial_relationships"]), title="Processed Relationships (from Session State)", expand=False, border_style="green"))
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
                    current_session_for_fallback_rel = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
                    if current_session_for_fallback_rel:
                        current_session_for_fallback_rel.state[persona_agent.output_key] = persona_details_json_str 
                        logger.info(f"Primed session state with FALLBACK persona_details_json for relationships agent.")
                        fallback_rel_trigger = genai_types.UserContent(parts=[genai_types.Part(text=persona_details_json_str)])
                        async for event in relationships_runner_fallback.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=fallback_rel_trigger): # Ensure new_message
                            if event.is_final_response(): break
                            if event.error_message: logger.error(f"Error during relationships for fallback: {event.error_message}"); break
                        
                        retrieved_session_after_fallback_rel = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
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
    if allow_real_context and search_llm_agent:
        console.print(Panel("Pre-fetching news context for all relevant years...", title="News Search Phase", style="dim"))
        for year_to_search in range(birth_year, end_year_for_generation + 1):
            search_query = f"major world events {year_to_search}"
            search_results_string = await perform_adk_search_via_components(
                search_llm_agent, session_service, search_query
            )
            if search_results_string:
                news_context_by_year_with_str_keys[str(year_to_search)] = search_results_string # Use string key
            else:
                news_context_by_year_with_str_keys[str(year_to_search)] = "No specific external events found or search failed." # Use string key
    # news_context_by_year_json = json.dumps(news_context_by_year_with_str_keys) # This is not needed if passing the dict directly

    world_details_json = json.dumps({
        "world_type": initial_user_input.get("world_type"),
        "world_description": initial_user_input.get("world_description")
    })

    yearly_iteration_agent = create_yearly_iteration_adk_agent()
    yearly_iteration_runner = Runner(agent=yearly_iteration_agent, session_service=session_service, app_name=session_app_name)
    
    years_to_generate_list = list(range(birth_year, end_year_for_generation + 1))
    num_years_to_generate = len(years_to_generate_list)

    if num_years_to_generate > 0:
        console.print(Rule(f"Generating {num_years_to_generate} Years ({birth_year}-{end_year_for_generation})", style="cyan"))
        current_session_for_yearly = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if not current_session_for_yearly:
            logger.error("Session lost before yearly one-shot call. Aborting yearly summaries.")
        else:
            current_session_for_yearly.state.update({
                "persona_details_json": persona_details_json_str_for_loop,
                "initial_relationships_json": initial_relationships_str,
                "world_details_json": world_details_json,
                # "birth_year_int": birth_year,
                # "years_to_generate_list_int": json.dumps(years_to_generate_list), # Pass the list of years
                "news_context_by_year_json": news_context_by_year_with_str_keys, # Pass the dict with string keys
                # "allow_real_context_bool": allow_real_context,
                # No need for list_of_all_yearly_data_json as it's one shot
            })
            yearly_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=f"Generate life summaries for years {birth_year} through {end_year_for_generation}. This is the news during those years: {json.dumps(news_context_by_year_with_str_keys)}")]) # Ensure new_message is used
            yearly_output_dict: Optional[Dict] = None
            async for event in yearly_iteration_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=yearly_trigger_content): # Ensure new_message
                if event.is_final_response():
                    logger.info(f"YearlyIterationAgent_ADK finished.")
                    break
                if event.error_message:
                    logger.error(f"Error during YearlyIterationAgent_ADK execution: {event.error_message}")
                    break
            
            session_after_yearly = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
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

        if life_summary["yearly_summaries"]: # Check if any summaries were processed
            console.print(Rule("Processing Accumulated Yearly Summaries", style="green"))
            console.print(Panel(pretty_repr(life_summary["yearly_summaries"]), title="ADK Yearly Summaries (Accumulated)", expand=False))
    else:
        logger.info("No years to generate for yearly summaries.")

    console.print(Rule("Generating Monthly Summaries (Manual Loop)", style="bold yellow"))
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
        console.print(Rule(f"Generating Month {target_year_for_months}-{target_month_for_loop_start:02d}", style="cyan"))
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
        
        current_loop_session_m = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if not current_loop_session_m: 
            logger.error(f"Session lost before monthly iteration for {target_year_for_months}-{target_month_for_loop_start}. Aborting.")
            break
        current_loop_session_m.state.update({
            "persona_details_json": persona_details_json_str_for_loop, 
            "initial_relationships_json": initial_relationships_str,
            "world_details_json": world_details_json, 
            "target_year_int": target_year_for_months,
            "yearly_summary_for_target_year_json": yearly_summary_for_target_year_json,
            "loop_iteration_count": target_month_for_loop_start - 1,
            "news_context_by_month_key_json": json.dumps({news_context_for_this_month_key: news_context_for_this_month_str}),
            "allow_real_context_bool": allow_real_context
        })
        monthly_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=f"Generate month {target_year_for_months}-{target_month_for_loop_start:02d}")])
        async for event in monthly_iteration_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=monthly_trigger_content): # Ensure new_message
            if event.is_final_response(): break
            if event.error_message: logger.error(f"Error in MonthlyIterationAgent: {event.error_message}"); break
        
        session_after_iteration_m = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if session_after_iteration_m and session_after_iteration_m.state:
            iteration_output_dict_m = session_after_iteration_m.state.get(monthly_iteration_agent.output_key)
            if iteration_output_dict_m and isinstance(iteration_output_dict_m, dict):
                try:
                    month_data = SingleMonthDataFromADK.model_validate(iteration_output_dict_m).model_dump()
                    life_summary["monthly_summaries"].setdefault(target_year_for_months, {})[month_data["month"]] = month_data
                    console.print(Panel(pretty_repr(month_data), title=f"ADK Monthly: {target_year_for_months}-{month_data['month']:02d}", expand=False))
                except ValidationError as ve:
                    logger.error(f"Validation error for month {target_year_for_months}-{target_month_for_loop_start} output: {ve}. Data: {iteration_output_dict_m}")
            else:
                logger.error(f"Monthly iteration for {target_year_for_months}-{target_month_for_loop_start} did not produce valid output.")
        else:
            logger.error(f"Could not retrieve session state after monthly iteration for {target_year_for_months}-{target_month_for_loop_start}.")
            break

    console.print(Rule("Generating Daily Summaries (Manual Loop)", style="bold yellow"))
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
        console.print(Rule(f"Generating Day {target_date_for_daily.isoformat()}", style="cyan"))
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
                f"local events {monthly_summary_d_dict.get('location','')} {news_key_d}"
            )
            if s_res_d_string: news_str_d = s_res_d_string
        
        current_loop_session_d = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if not current_loop_session_d: 
            logger.error(f"Session lost before daily iteration for {target_date_for_daily.isoformat()}. Aborting.")
            break
        current_loop_session_d.state.update({
            "persona_details_json": persona_details_json_str_for_loop, 
            "initial_relationships_json": initial_relationships_str,
            "world_details_json": world_details_json, 
            "target_year_int": yr_d, "target_month_int": m_d,
            "monthly_summary_for_target_month_json": monthly_summary_d_json, 
            "loop_iteration_count": d_d - 1,
            "news_context_by_day_key_json": json.dumps({news_key_d: news_str_d}),
            "allow_real_context_bool": allow_real_context
        })
        daily_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=f"Generate day {target_date_for_daily.isoformat()}")])
        async for event in daily_iteration_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=daily_trigger_content): # Ensure new_message
            if event.is_final_response(): break
            if event.error_message: logger.error(f"Error in DailyIterationAgent for {target_date_for_daily.isoformat()}: {event.error_message}"); break

        session_after_iteration_d = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
        if session_after_iteration_d and session_after_iteration_d.state:
            iteration_output_dict_d = session_after_iteration_d.state.get(daily_iteration_agent.output_key)
            if iteration_output_dict_d and isinstance(iteration_output_dict_d, dict):
                try:
                    day_data = SingleDayDataFromADK.model_validate(iteration_output_dict_d).model_dump()
                    life_summary["daily_summaries"].setdefault(yr_d, {}).setdefault(m_d, {})[day_data["day"]] = day_data
                    console.print(Panel(pretty_repr(day_data), title=f"ADK Daily: {target_date_for_daily.isoformat()}", expand=False))
                except ValidationError as ve:
                    logger.error(f"Validation error for day {target_date_for_daily.isoformat()} output: {ve}. Data: {iteration_output_dict_d}")
            else:
                logger.error(f"Daily iteration for {target_date_for_daily.isoformat()} did not produce valid output.")
        else:
            logger.error(f"Could not retrieve session state after daily iteration for {target_date_for_daily.isoformat()}.")
            break

    console.print(Rule("Generating Hourly Breakdowns (Manual Loop)", style="bold yellow"))
    sim_curr_hour = life_summary["generation_info"]["current_hour"]
    days_for_hourly_loop_h: List[date] = [character_current_sim_date_d]
    yesterday_sim_date_h = character_current_sim_date_d - timedelta(days=1)
    if yesterday_sim_date_h >= birthdate_obj: days_for_hourly_loop_h.append(yesterday_sim_date_h)

    hourly_iteration_agent = create_hourly_iteration_adk_agent()
    hourly_iteration_runner = Runner(agent=hourly_iteration_agent, session_service=session_service, app_name=session_app_name)

    for target_date_for_hourly in days_for_hourly_loop_h:
        console.print(Rule(f"Generating Hourly for Day {target_date_for_hourly.isoformat()}", style="cyan"))
        yr_h, m_h, d_h = target_date_for_hourly.year, target_date_for_hourly.month, target_date_for_hourly.day
        daily_summary_h = life_summary["daily_summaries"].get(yr_h, {}).get(m_h, {}).get(d_h)
        if not daily_summary_h: 
            logger.warning(f"No daily summary for {target_date_for_hourly.isoformat()}, skipping hourly generation.")
            continue
        daily_summary_h_json = json.dumps(daily_summary_h)
        news_for_this_hourly_day_str = daily_summary_h.get("news_context_used", "No external context available for this day.") # Get news from daily summary

        max_h_for_this_day = sim_curr_hour + 1 if target_date_for_hourly == character_current_sim_date_d else 24
        
        hourly_activities_for_day_accumulated: Dict[int, Dict] = {}
        for hour_iteration in range(max_h_for_this_day): 
            current_loop_session_h = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
            if not current_loop_session_h: 
                logger.error(f"Session lost before hourly iteration {hour_iteration} for {target_date_for_hourly.isoformat()}. Aborting day.")
                break
            current_loop_session_h.state.update({
                "persona_details_json": persona_details_json_str_for_loop, 
                "initial_relationships_json": initial_relationships_str,
                "world_details_json": world_details_json, 
                "target_year_int": yr_h, "target_month_int": m_h, "target_day_int": d_h,
                "daily_summary_for_target_day_json": daily_summary_h_json, 
                "loop_iteration_count": hour_iteration,
                "news_context_for_this_hourly_day_str": news_for_this_hourly_day_str # Pass daily news to hourly agent
            })
            hourly_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=f"Generate hour {hour_iteration} for {target_date_for_hourly.isoformat()}")])
            async for event in hourly_iteration_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id, new_message=hourly_trigger_content): # Ensure new_message
                if event.is_final_response(): break
                if event.error_message: logger.error(f"Error in HourlyIterationAgent for hour {hour_iteration} of {target_date_for_hourly.isoformat()}: {event.error_message}"); break
            
            session_after_iteration_h = session_service.get_session(session_id=session_id_for_workflow, app_name=session_app_name, user_id=session_user_id)
            if session_after_iteration_h and session_after_iteration_h.state:
                iteration_output_dict_h = session_after_iteration_h.state.get(hourly_iteration_agent.output_key)
                if iteration_output_dict_h and isinstance(iteration_output_dict_h, dict):
                    try:
                        hour_data = SingleHourDataFromADK.model_validate(iteration_output_dict_h).model_dump()
                        hourly_activities_for_day_accumulated[hour_data["hour"]] = hour_data
                    except ValidationError as ve:
                        logger.error(f"Validation error for hour {hour_iteration} of {target_date_for_hourly.isoformat()} output: {ve}. Data: {iteration_output_dict_h}")
                else:
                    logger.error(f"Hourly iteration {hour_iteration} for {target_date_for_hourly.isoformat()} did not produce valid output.")
            else:
                logger.error(f"Could not retrieve session state after hourly iteration {hour_iteration} for {target_date_for_hourly.isoformat()}.")
                break 
        
        if hourly_activities_for_day_accumulated:
            life_summary["hourly_breakdowns"].setdefault(yr_h, {}).setdefault(m_h, {})[d_h] = {
                "activities": hourly_activities_for_day_accumulated, 
                "news": news_for_this_hourly_day_str # Store the daily news context with the hourly breakdown
            }
            console.print(Panel(pretty_repr(hourly_activities_for_day_accumulated), title=f"ADK Hourly: {target_date_for_hourly.isoformat()}", expand=False))
        else:
            logger.warning(f"No hourly activities accumulated for {target_date_for_hourly.isoformat()}.")
        
    console.print(Rule("ADK Orchestration Complete (All Levels Implemented)", style="bold green"))
    return life_summary

# --- Main Entry Point ---
async def generate_new_simulacra_background(
    sim_id: str,
    world_instance_uuid: Optional[str],
    world_type: str,
    world_description: str,
    allow_real_context: bool,
    gender_preference: Optional[str] = "any"
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
            search_llm_agent_instance = create_search_llm_agent() 
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
            created_session = session_service_instance.create_session(
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
    output_filename = f"life_summary_ADK_{persona_name_safe}_{sim_id}_{world_instance_uuid}.json"
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
