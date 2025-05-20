# c:\Users\dshea\Desktop\TheSimulation\src\generation\life_generator_adk.py
import asyncio
import json
import logging
import uuid # Added import for uuid
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from google.adk.agents import LlmAgent # ADK Workflow (LoopAgent, SequentialAgent are for more advanced ADK-native workflows)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import google_search # ADK Google Search tool
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError, model_validator # Keep Pydantic for validation
from google.adk.agents import LoopAgent, SequentialAgent # ADK Workflow
from src.agents import create_search_llm_agent # Your existing search agent creator
from src.config import APP_NAME
from src.generation.llm_service import LLMService # Still needed for direct LLM calls if any
from src.generation.models import ( # Keep Pydantic models
    DailySummariesResponse,
    HourlyBreakdownResponse,
    InitialRelationshipsResponse, Person, 
    MonthlySummariesResponse,
    PersonaDetailsResponse,
    YearlySummariesResponse, # This might be less used if loop outputs single year dicts
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
    # age_range_min: Optional[int] = None # Future: if you want to pass age range
    # age_range_max: Optional[int] = None # Future: if you want to pass age range
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
        # Basic check, assumes month context is handled by the calling agent/loop setup
        if not (1 <= self.day <= 31):
            raise ValueError(f"Day {self.day} is outside the general valid range (1-31).")
        return self

class SingleHourDataFromADK(BaseModel):
    hour: int
    location: str
    activity: str
    # News context is likely too granular for hourly, will be part of daily context

    @model_validator(mode='after')
    def check_hour_valid(self) -> 'SingleHourDataFromADK':
        if not (0 <= self.hour <= 23):
            raise ValueError(f"Hour {self.hour} is outside the valid range (0-23).")
        return self

# --- Module-level "Shared" ADK Components ---
# These are initialized once per call to generate_new_simulacra_background
# and are passed around or accessed by helper functions within that call's scope.
# They are not true globals across multiple independent calls to the main entry point.
# _runner_instance: Optional[Runner] = None # Runner will be created locally per agent
_session_service_instance: Optional[InMemorySessionService] = None
_search_llm_agent_instance: Optional[LlmAgent] = None # The agent with google_search tool
_llm_service_instance: Optional[LLMService] = None # For Pydantic validation or fallback

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
    search_agent: LlmAgent, # The actual search agent instance
    session_service: InMemorySessionService, # The _session_service_instance
    search_query: str,
    max_results: int = 3
) -> Optional[List[Dict[str, str]]]:
    """
    Performs a search using the provided search_agent and session_service.
    Creates a new session for each search call for isolation.
    """
    if not search_agent or not session_service:
        logger.error("ADK components (search_agent, session_service) not provided for search helper.")
        return None

    # Create a Runner specifically for this search agent
    search_runner = Runner(agent=search_agent, session_service=session_service, app_name=f"{APP_NAME}_SearchRunner")

    # Create a new, temporary session specifically for this search call
    search_session_id = session_service.create_session(
        app_name=f"{APP_NAME}_SearchHelperSession_{uuid.uuid4().hex[:6]}", # Unique app name for session
        user_id="search_helper_user"
    ).id
    logger.info(f"Performing ADK Google Search for: '{search_query}' (using temp session: {search_session_id})")
    
    trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
    search_results_list: List[Dict[str, str]] = []

    try:
        async for event in search_runner.run_async(
            user_id="life_generator_search_user_HELPER", # Can be a generic user for this helper
            session_id=search_session_id, 
            new_message=trigger_content
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.function_response and part.function_response.name == "google_search":
                        response_data = dict(part.function_response.response)
                        if "results" in response_data and isinstance(response_data["results"], list):
                            for item in response_data["results"]:
                                if isinstance(item, dict):
                                    search_results_list.append({str(k): str(v) for k, v in item.items()})
                            logger.debug(f"ADK Google Search (helper) returned {len(search_results_list)} results.")
                        break 
                if search_results_list:
                    break
        
        return search_results_list[:max_results] if search_results_list else None
    except Exception as e:
        logger.error(f"Error during ADK Google Search (helper) for '{search_query}': {e}", exc_info=True)
        return None
    finally:
        # Clean up the temporary search session (optional, but good practice for InMemorySessionService)
        try:
            session_service.delete_session(search_session_id)
            logger.debug(f"Deleted temporary search session: {search_session_id}")
        except Exception as e_del:
            logger.warning(f"Could not delete temporary search session {search_session_id}: {e_del}")


# --- ADK Sub-Agent Definitions ---

def create_persona_generator_adk_agent(model_name: str = "gemini-2.0-flash") -> LlmAgent:
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
        # response_mime_type="application/json",
        output_key="persona_details_json"
    )

def create_initial_relationships_adk_agent(model_name: str = "gemini-2.0-flash") -> LlmAgent:
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
        # response_mime_type="application/json",
        output_key="initial_relationships_json" # Output will be stored in state['initial_relationships_json']
    )

# This agent is the body of the LoopAgent for yearly summaries
def create_yearly_iteration_adk_agent(model_name: str = "gemini-2.0-flash") -> LlmAgent:
    return LlmAgent(
        name="YearlyIterationAgent_ADK",
        model=model_name,
        description="Generates a summary for a single year based on loop iteration and context.",
        instruction="""You are a biographer.
You will receive the following data from the session state:
- `persona_details_json`: JSON string of the persona.
- `initial_relationships_json`: JSON string of initial family.
- `world_details_json`: JSON string with "world_type" and "world_description".
- `birth_year_int`: The integer birth year of the persona.
- `loop_iteration_count`: The current 0-indexed iteration of the loop.
- `news_context_by_year_json`: JSON string of a dictionary mapping years (as strings) to news context strings.
- `allow_real_context_bool`: Boolean indicating if real-world news should be considered.
- `list_of_all_yearly_data_json`: JSON string of a list containing summary dicts from PREVIOUS iterations.

Your task is to generate a summary for ONE year.
1. Calculate `current_processing_year = birth_year_int + loop_iteration_count`.
2. Parse `persona_details_json`, `initial_relationships_json`, `world_details_json`, `news_context_by_year_json`, and `list_of_all_yearly_data_json`.
3. Get `previous_year_summary_text`: If `list_of_all_yearly_data_json` is not empty, take the 'summary' from the LAST element. Otherwise, "This is the first year of life."
4. Get `news_for_current_year`: From `news_context_by_year_json`, get the news for `current_processing_year`. If `allow_real_context_bool` is false or no news, use "No specific external events noted for this year; focus on personal development and fictional world events."

Based on all this information, provide a summary of major events for the `current_processing_year`.
Include the persona's primary location for that year (be specific, e.g., "City, State/Country") and key life events (personal, professional, relationships). The summary should be detailed enough to cover significant happenings.
If `current_processing_year == birth_year_int`, include plausible `birth_month` (1-12) and `birth_day` (1-31, valid for month).

Respond ONLY with valid JSON matching the SingleYearDataFromADK schema:
`{{"year": int, "location": str, "summary": str, "news_context_used": str, "birth_month": Optional[int], "birth_day": Optional[int]}}`
The "year" field in the JSON MUST be `current_processing_year`. The "summary" should be a comprehensive paragraph.
""",
        # response_mime_type="application/json",
        output_key="current_year_data_json" # Output of one iteration
    )

def create_monthly_iteration_adk_agent(model_name: str = "gemini-2.0-flash") -> LlmAgent:
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
        # response_mime_type="application/json",
        output_key="current_month_data_json"
    )

def create_daily_iteration_adk_agent(model_name: str = "gemini-2.0-flash") -> LlmAgent:
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
        # response_mime_type="application/json",
        output_key="current_day_data_json"
    )

def create_hourly_iteration_adk_agent(model_name: str = "gemini-2.0-flash") -> LlmAgent:
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

Task: Generate an activity for ONE hour.
1. `current_processing_hour = loop_iteration_count`.
2. Parse inputs. Get `daily_summary_text` and `daily_location` from `daily_summary_for_target_day_json`.

Based on all info, describe the primary activity for `current_processing_hour` on `target_day_int`/`target_month_int`/`target_year_int`. The activity description should be specific.
Respond ONLY with valid JSON matching SingleHourDataFromADK: `{{"hour": int, "location": str, "activity": str}}`
The "hour" field MUST be `current_processing_hour`. Location should be consistent. The "activity" should be a descriptive sentence.
""",
        # response_mime_type="application/json",
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
    # runner: Runner, # Runner will be created locally per main agent execution
    session_service: InMemorySessionService, # Passed in
    search_llm_agent: Optional[LlmAgent], # Passed in, the one with google_search tool
    llm_service_for_validation: LLMService, # Passed in, for Pydantic if needed
    initial_user_input: Dict[str, Any],
    persona_details_override: Optional[Dict],
    generation_params: Dict[str, Any],
    session_id_for_workflow: str,
    session_app_name: str, # Added: app_name used to create the session
    session_user_id: str,  # Added: user_id used to create the session
    allow_real_context: bool,
) -> Optional[Dict[str, Any]]:
    console.print(Rule("Starting ADK-Orchestrated Life Generation", style="bold magenta"))
    life_summary: Dict[str, Any] = {
        "persona_details": None, "initial_relationships": None,
        "yearly_summaries": {}, "monthly_summaries": {},
        "daily_summaries": {}, "hourly_breakdowns": {},
        "generation_info": generation_params.get("generation_info", {}) # Pre-populated with timestamps
    }

    generated_persona_data: Optional[Dict[str, Any]] = None
    persona_details_json_str: Optional[str] = None

    if persona_details_override:
        generated_persona_data = persona_details_override
        persona_details_json_str = json.dumps(generated_persona_data)
        logger.info("Using overridden persona details.")
    else:
        # 1. Initial Setup Pipeline (Persona + Relationships) using SequentialAgent
        console.print(Rule("ADK SequentialAgent: Persona & Initial Relationships", style="bold yellow"))
        persona_agent = create_persona_generator_adk_agent()
        relationships_agent = create_initial_relationships_adk_agent()

        initial_setup_pipeline = SequentialAgent(
            name="InitialLifeSetupPipeline_ADK",
            sub_agents=[persona_agent, relationships_agent],
            description="Generates persona and initial relationships sequentially."
        )

        # Create a Runner for this specific pipeline
        pipeline_runner = Runner(
            agent=initial_setup_pipeline,
            session_service=session_service,
            app_name=session_app_name # Use the app_name associated with the session
        )

        pipeline_input_text = json.dumps(initial_user_input) # world_type, world_description, gender_preference
        pipeline_input_content = genai_types.UserContent(parts=[genai_types.Part(text=pipeline_input_text)])
        # pipeline_input_content = pipeline_input_text
        async for event in pipeline_runner.run_async(
            session_id=session_id_for_workflow,
            new_message=pipeline_input_content,
            user_id=session_user_id # Use the user_id associated with the session
        ):
            logger.debug(f"ADK Pipeline Event: Author={event.author}, Final={event.is_final_response()}, Content={str(event.content)[:200]}, Actions={event.actions}")
            if event.is_final_response():
                logger.info("InitialLifeSetupPipeline_ADK has finished processing events.")
                break
            if event.error_message:
                logger.error(f"Error during InitialLifeSetupPipeline_ADK execution: {event.error_message} (Code: {event.error_code})")
                break
        
        # After the pipeline has run, retrieve the session to get the final accumulated state
        final_session_state_after_pipeline: Optional[Dict[str, Any]] = None
        retrieved_session_after_pipeline = session_service.get_session(
            session_id=session_id_for_workflow,
            app_name=session_app_name, # Use the correct app_name
            user_id=session_user_id    # Use the correct user_id
        )
        
        if retrieved_session_after_pipeline and retrieved_session_after_pipeline.state:
            final_session_state_after_pipeline = retrieved_session_after_pipeline.state
            logger.info("Retrieved final session state after InitialLifeSetupPipeline_ADK.")
        else:
            logger.error(f"Could not retrieve session or session state after InitialLifeSetupPipeline_ADK execution. Session ID: {session_id_for_workflow}, App: {session_app_name}, User: {session_user_id}")
            # Fallback or error handling might be needed here if state is crucial

        if final_session_state_after_pipeline:
            persona_details_json_str = final_session_state_after_pipeline.get(persona_agent.output_key)
            relationships_json_str = final_session_state_after_pipeline.get(relationships_agent.output_key)

            if persona_details_json_str:
                try:
                    raw_output_dict = json.loads(persona_details_json_str)
                    if "persona" in raw_output_dict and isinstance(raw_output_dict["persona"], dict):
                        persona_data_to_validate = raw_output_dict["persona"]
                        generated_persona_data = PersonaDetailsResponse.model_validate(persona_data_to_validate).model_dump()
                    else:
                        generated_persona_data = PersonaDetailsResponse.model_validate_json(persona_details_json_str).model_dump()
                    console.print(Panel(pretty_repr(generated_persona_data), title="ADK Persona (from Final Session State)", expand=False))
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    logger.error(f"ADK Persona (from Final Session State) output validation error: {e}. Response: {persona_details_json_str[:500]}")
                    generated_persona_data = None
            if relationships_json_str:
                try:
                    life_summary["initial_relationships"] = InitialRelationshipsResponse.model_validate_json(relationships_json_str).model_dump()
                    console.print(Panel(pretty_repr(life_summary["initial_relationships"]), title="ADK Relationships (from Final Session State)", expand=False))
                except (ValidationError, json.JSONDecodeError) as e:
                    logger.error(f"ADK Relationships (from Final Session State) output error: {e}. Response: {relationships_json_str[:500]}")
            elif generated_persona_data: 
                logger.warning("Relationships part of pipeline did not produce output (from final session state), though persona was generated.")

    if not generated_persona_data: # Fallback if ADK agent/pipeline failed or was skipped due to no override
        if not persona_details_override: # Only try fallback if we didn't have an override
            logger.warning("ADK Persona generation via pipeline failed or produced invalid data, trying fallback...")
            generated_persona_data = await agent_generate_random_persona_fallback(
                llm_service_for_validation,
                initial_user_input.get("world_type"),
                initial_user_input.get("world_description"),
                initial_user_input.get("gender_preference")
            )
            if generated_persona_data:
                try:
                    persona_details_json_str = json.dumps(generated_persona_data) # For relationship generation if needed
                except TypeError:
                     logger.error("Fallback persona data could not be serialized to JSON.")
                     persona_details_json_str = None

    if not generated_persona_data:
        logger.error("All persona generation attempts failed.")
        return None
    life_summary["persona_details"] = generated_persona_data

    # --- Date/Time Setup based on Generated Persona ---
    simulated_current_dt = generation_params["generation_timestamp"]
    birthdate_str_from_persona = generated_persona_data.get("Birthdate")
    
    try:
        birthdate_obj = datetime.strptime(birthdate_str_from_persona, "%Y-%m-%d").date()
        birth_year = birthdate_obj.year
        birth_month = birthdate_obj.month # Will be updated by yearly agent if it provides one
        birth_day = birthdate_obj.day   # Will be updated by yearly agent
        
        actual_age = simulated_current_dt.year - birth_year - \
                     ((simulated_current_dt.month, simulated_current_dt.day) < (birth_month, birth_day))
        end_year_for_generation = birth_year + actual_age
        
        life_summary["generation_info"].update({
            "birth_year": birth_year, "birth_month": birth_month, "birth_day": birth_day,
            "actual_age_at_generation": actual_age, "end_year_for_generation": end_year_for_generation
        })
        logger.info(f"Orchestrator Date Setup: Birth {birthdate_obj}, Actual Age {actual_age}, End Year {end_year_for_generation}")
    except (ValueError, TypeError) as e:
        logger.error(f"Error processing birthdate from persona ('{birthdate_str_from_persona}'): {e}. Cannot proceed.")
        return life_summary

    # 2. Initial Relationships (if not generated by pipeline, e.g., due to override or fallback for persona)
    if not life_summary.get("initial_relationships") and persona_details_json_str:
        console.print(Rule("ADK Agent: Initial Relationships (Standalone)", style="bold yellow"))
        relationships_adk_agent_standalone = create_initial_relationships_adk_agent() # Re-create or use existing definition
        # The instruction for relationships_agent expects {persona_details_json} in state.
        # For a standalone call, we need to ensure the session state is primed or pass it directly.
        # Simpler: update session state before calling.
        current_session = session_service.get_session(session_id=session_id_for_workflow)
        current_session.state.update({persona_agent.output_key: persona_details_json_str}) # Use the defined output_key
        session_service.update_session(current_session)

        # Create a Runner for this standalone agent
        standalone_relationships_runner = Runner(
            agent=relationships_adk_agent_standalone,
            session_service=session_service,
            app_name=session_app_name # Use session's app_name
        )
        async for event in standalone_relationships_runner.run_async(
            session_id=session_id_for_workflow,
            user_id=session_user_id # Use session's user_id
        ):
            if event.is_final_response():
                if hasattr(event, 'state') and event.state is not None:
                    response_text = event.state.get(relationships_adk_agent_standalone.output_key) # type: ignore
                    if response_text:
                        try:
                            life_summary["initial_relationships"] = InitialRelationshipsResponse.model_validate_json(response_text).model_dump()
                        except (ValidationError, json.JSONDecodeError) as e:
                            logger.error(f"ADK Relationships Agent (standalone) output error: {e}. Response: {response_text[:500]}")
                else:
                    logger.warning("Standalone RelationshipsAgent_ADK final event did not have state or state was None.")
                break

    initial_relationships_str = json.dumps(life_summary["initial_relationships"] or {})

    # 3. Yearly Summaries (using ADK LoopAgent)
    console.print(Rule("ADK LoopAgent: Yearly Summaries", style="bold yellow"))

    # 3a. Pre-fetch all news context if allowed
    news_context_by_year: Dict[int, str] = {}
    if allow_real_context and search_llm_agent:
        console.print(Panel("Pre-fetching news context for all relevant years...", title="News Search Phase", style="dim"))
        for year_to_search in range(birth_year, end_year_for_generation + 1):
            search_query = f"major world events {year_to_search}"
            search_results = await perform_adk_search_via_components(
                search_llm_agent, session_service, search_query, max_results=3
            )
            if search_results:
                formatted_results = [f"- {r.get('title', 'N/A')}: {r.get('snippet', 'N/A')[:100]}..." for r in search_results]
                news_context_by_year[year_to_search] = "\n".join(formatted_results)
                # console.print(f"News for {year_to_search}: Found {len(search_results)} items.")
            else:
                news_context_by_year[year_to_search] = "No specific external events found or search failed."
                # console.print(f"News for {year_to_search}: None found.")
    news_context_by_year_json = json.dumps(news_context_by_year)

    # 3b. Prepare initial state for the LoopAgent
    world_details_json = json.dumps({
        "world_type": initial_user_input.get("world_type"),
        "world_description": initial_user_input.get("world_description")
    })

    # Get current session and update its state
    loop_session = session_service.get_session(session_id=session_id_for_workflow)
    if not loop_session: # Should not happen if session creation was successful
        logger.error(f"Cannot retrieve session {session_id_for_workflow} for LoopAgent setup.")
        return life_summary # type: ignore

    loop_session.state.update({
        "persona_details_json": persona_details_json_str, # From step 1
        "initial_relationships_json": initial_relationships_str, # From step 2
        "world_details_json": world_details_json,
        "birth_year_int": birth_year,
        "news_context_by_year_json": news_context_by_year_json,
        "allow_real_context_bool": allow_real_context,
        "list_of_all_yearly_data_json": json.dumps([]) # Initialize as empty list string for the loop body agent
    })
    session_service.update_session(loop_session) # Save updated state

    # 3c. Define and run the LoopAgent
    yearly_iteration_agent = create_yearly_iteration_adk_agent()
    num_years_to_generate = end_year_for_generation - birth_year + 1

    if num_years_to_generate > 0:
        yearly_summary_loop_agent = LoopAgent(
            name="YearlySummaryLoop_ADK",
            loop_body_agent=yearly_iteration_agent,
            max_iterations=num_years_to_generate,
            # The loop body agent's output_key is "current_year_data_json"
            # LoopAgent will collect these into a list under this key:
            output_key_loop_results="list_of_all_yearly_data_json",
            description="Generates yearly summaries iteratively."
        )

        yearly_loop_runner = Runner(
            agent=yearly_summary_loop_agent,
            session_service=session_service,
            app_name=session_app_name # Use session's app_name
        )

        async for event in yearly_loop_runner.run_async(
            session_id=session_id_for_workflow,
            user_id=session_user_id # Use session's user_id
        ):
            if event.is_final_response():
                logger.info("YearlySummaryLoop_ADK finished.")
                if hasattr(event, 'state') and event.state is not None:
                    final_loop_state = event.state
                    if final_loop_state: # Redundant if event.state is not None, but harmless
                        raw_yearly_data_list_json = final_loop_state.get("list_of_all_yearly_data_json")
                        if raw_yearly_data_list_json:
                            try:
                                parsed_yearly_data_list = []
                                if isinstance(raw_yearly_data_list_json, str): 
                                    temp_list = json.loads(raw_yearly_data_list_json)
                                    for item_str in temp_list:
                                         parsed_yearly_data_list.append(SingleYearDataFromADK.model_validate_json(item_str).model_dump())
                                elif isinstance(raw_yearly_data_list_json, list): 
                                    for item_data in raw_yearly_data_list_json:
                                        if isinstance(item_data, str):
                                            parsed_yearly_data_list.append(SingleYearDataFromADK.model_validate_json(item_data).model_dump())
                                        elif isinstance(item_data, dict): 
                                            parsed_yearly_data_list.append(SingleYearDataFromADK.model_validate(item_data).model_dump())

                                for year_data in parsed_yearly_data_list:
                                    year_num = year_data["year"]
                                    life_summary["yearly_summaries"][year_num] = {
                                        "summary": year_data["summary"],
                                        "location": year_data["location"],
                                        "news": year_data.get("news_context_used", news_context_by_year.get(year_num, "No external context used."))
                                    }
                                    if year_num == birth_year:
                                        bm = year_data.get("birth_month")
                                        bd = year_data.get("birth_day")
                                        if bm is not None and bd is not None:
                                            life_summary["generation_info"]["birth_month"] = bm
                                            life_summary["generation_info"]["birth_day"] = bd
                                            logger.info(f"Birth month/day from ADK LoopAgent (Year {year_num}): {bm}/{bd}")
                                console.print(Panel(pretty_repr(life_summary["yearly_summaries"]), title="ADK Yearly Summaries (from Loop)", expand=False))
                            except (ValidationError, json.JSONDecodeError, TypeError) as e:
                                logger.error(f"Error processing yearly summaries from LoopAgent: {e}. Data: {str(raw_yearly_data_list_json)[:500]}")
                else:
                    logger.warning("YearlySummaryLoop_ADK final event did not have state or state was None.")
                break
    else:
        logger.info("No years to generate for yearly summaries (num_years_to_generate <= 0).")

    # 4. Monthly Summaries (ADK LoopAgent) - For relevant recent months
    console.print(Rule("ADK LoopAgent: Monthly Summaries", style="bold yellow"))
    sim_curr_year = life_summary["generation_info"]["current_year"]
    sim_curr_month = life_summary["generation_info"]["current_month"]

    # Determine months to generate for (e.g., current simulated month and previous one)
    # This logic can be adjusted based on desired depth.
    months_to_process_for_loop: List[Tuple[int, int]] = [] # (year, month_num)
    current_sim_date_obj = date(sim_curr_year, sim_curr_month, 1)
    months_to_process_for_loop.append((current_sim_date_obj.year, current_sim_date_obj.month))
    
    prev_month_date_obj = current_sim_date_obj - timedelta(days=1) # Go to end of prev month
    prev_month_date_obj = prev_month_date_obj.replace(day=1) # Go to start of prev month
    
    # Ensure previous month is not before birth year/month
    if date(prev_month_date_obj.year, prev_month_date_obj.month, 1) >= date(birth_year, life_summary["generation_info"]["birth_month"], 1):
        months_to_process_for_loop.append((prev_month_date_obj.year, prev_month_date_obj.month))

    monthly_iteration_agent = create_monthly_iteration_adk_agent()

    for target_year_for_months, target_month_for_loop_start in months_to_process_for_loop:
        yearly_summary_for_target_year = life_summary["yearly_summaries"].get(target_year_for_months)
        if not yearly_summary_for_target_year:
            logger.warning(f"No yearly summary for {target_year_for_months}, skipping monthly generation for it.")
            continue
        
        yearly_summary_for_target_year_json = json.dumps(yearly_summary_for_target_year) # Pass the whole dict

        # Max iterations: if it's the current simulated year, go up to current month. Otherwise, all 12.
        # For this targeted approach, we are doing one month at a time, so max_iterations is 1,
        # but the loop_iteration_count for the agent will be the month index (0-11).
        # The LoopAgent itself will run for each month in months_to_process_for_loop.
        # Let's refine: LoopAgent iterates N times. We want to generate for specific months.
        # Simpler: Loop for each (target_year, target_month) pair.
        # The ADK LoopAgent is better for contiguous iterations.
        # For sparse (e.g. "this month" and "last month"), direct calls might be simpler.
        # Let's stick to LoopAgent for now, assuming we might want a contiguous block in future.
        # For now, we'll run a "loop" of 1 iteration for each specific month we target.

        console.print(f"-- Monthly Loop for: {target_year_for_months}-{target_month_for_loop_start:02d} --")
        
        news_context_for_this_month_key = f"{target_year_for_months}-{target_month_for_loop_start:02d}"
        news_context_for_this_month_str = "No external context used."
        if allow_real_context and search_llm_agent:
            search_query = f"major events {yearly_summary_for_target_year.get('location', '')} {news_context_for_this_month_key}"
            s_results = await perform_adk_search_via_components(search_llm_agent, session_service, search_query, 1)
            if s_results:
                news_context_for_this_month_str = f"- {s_results[0].get('title', 'N/A')}: {s_results[0].get('snippet', 'N/A')[:100]}..."

        current_session = session_service.get_session(session_id=session_id_for_workflow)
        current_session.state.update({
            "persona_details_json": persona_details_json_str,
            "initial_relationships_json": initial_relationships_str,
            "world_details_json": world_details_json,
            "target_year_int": target_year_for_months,
            "yearly_summary_for_target_year_json": yearly_summary_for_target_year_json,
            "loop_iteration_count": target_month_for_loop_start - 1, # 0-indexed month
            "news_context_by_month_key_json": json.dumps({news_context_for_this_month_key: news_context_for_this_month_str}),
            "allow_real_context_bool": allow_real_context,
            "list_of_all_monthly_data_json": json.dumps([]) # For this "single month" loop
        })
        session_service.update_session(current_session)

        monthly_loop = LoopAgent(
            name=f"MonthlySummaryLoop_ADK_{target_year_for_months}_{target_month_for_loop_start}",
            loop_body_agent=monthly_iteration_agent,
            max_iterations=1, # We are processing one specific month
            output_key_loop_results="list_of_all_monthly_data_json"
        )

        monthly_loop_runner = Runner(
            agent=monthly_loop,
            session_service=session_service,
            app_name=session_app_name # Use session's app_name
        )

        async for event in monthly_loop_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id): # Use session's user_id
            if event.is_final_response():
                if hasattr(event, 'state') and event.state is not None:
                    raw_monthly_data_list = event.state.get("list_of_all_monthly_data_json")
                    if raw_monthly_data_list and isinstance(raw_monthly_data_list, list) and raw_monthly_data_list:
                        try:
                            month_data_json_str = raw_monthly_data_list[0] 
                            month_data = SingleMonthDataFromADK.model_validate_json(month_data_json_str).model_dump()
                            month_num_actual = month_data["month"]
                            life_summary["monthly_summaries"].setdefault(target_year_for_months, {})[month_num_actual] = month_data
                            console.print(Panel(pretty_repr(month_data), title=f"ADK Monthly Summary: {target_year_for_months}-{month_num_actual:02d}", expand=False))
                        except (ValidationError, json.JSONDecodeError, IndexError) as e:
                            logger.error(f"Error processing monthly summary for {target_year_for_months}-{target_month_for_loop_start}: {e}")
                else:
                    logger.warning(f"MonthlySummaryLoop_ADK for {target_year_for_months}-{target_month_for_loop_start} final event did not have state or state was None.")
                break

    # 5. Daily Summaries (ADK LoopAgent) - For relevant recent days
    console.print(Rule("ADK LoopAgent: Daily Summaries", style="bold yellow"))
    sim_curr_day = life_summary["generation_info"]["current_day"]
    days_to_process_for_loop: List[date] = [] # List of date objects
    character_current_sim_date = date(sim_curr_year, sim_curr_month, sim_curr_day)
    for i in range(7): # Last 7 days
        day_to_add = character_current_sim_date - timedelta(days=i)
        if day_to_add >= birthdate_obj: # Ensure not before birth
            days_to_process_for_loop.append(day_to_add)
    days_to_process_for_loop.reverse() # Process chronologically

    daily_iteration_agent = create_daily_iteration_adk_agent()

    for target_date_for_daily in days_to_process_for_loop:
        target_year_d, target_month_d, target_day_d = target_date_for_daily.year, target_date_for_daily.month, target_date_for_daily.day
        monthly_summary_for_target_month = life_summary["monthly_summaries"].get(target_year_d, {}).get(target_month_d)
        if not monthly_summary_for_target_month:
            logger.warning(f"No monthly summary for {target_year_d}-{target_month_d}, skipping daily for {target_date_for_daily}.")
            continue
        monthly_summary_for_target_month_json = json.dumps(monthly_summary_for_target_month)

        console.print(f"-- Daily Loop for: {target_date_for_daily.isoformat()} --")
        news_context_for_this_day_key = target_date_for_daily.isoformat()
        news_context_for_this_day_str = "No external context used."
        if allow_real_context and search_llm_agent:
            search_query = f"local events {monthly_summary_for_target_month.get('location','')} {news_context_for_this_day_key}"
            s_results = await perform_adk_search_via_components(search_llm_agent, session_service, search_query, 1)
            if s_results:
                news_context_for_this_day_str = f"- {s_results[0].get('title', 'N/A')}: {s_results[0].get('snippet', 'N/A')[:100]}..."

        current_session = session_service.get_session(session_id=session_id_for_workflow)
        current_session.state.update({
            "persona_details_json": persona_details_json_str, "initial_relationships_json": initial_relationships_str, "world_details_json": world_details_json,
            "target_year_int": target_year_d, "target_month_int": target_month_d,
            "monthly_summary_for_target_month_json": monthly_summary_for_target_month_json,
            "loop_iteration_count": target_day_d - 1, # 0-indexed day
            "news_context_by_day_key_json": json.dumps({news_context_for_this_day_key: news_context_for_this_day_str}),
            "allow_real_context_bool": allow_real_context,
            "list_of_all_daily_data_json": json.dumps([])
        })
        session_service.update_session(current_session)

        daily_loop = LoopAgent(name=f"DailyLoop_ADK_{target_date_for_daily.isoformat()}", loop_body_agent=daily_iteration_agent, max_iterations=1, output_key_loop_results="list_of_all_daily_data_json")

        daily_loop_runner = Runner(
            agent=daily_loop,
            session_service=session_service,
            app_name=session_app_name # Use session's app_name
        )

        async for event in daily_loop_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id): # Use session's user_id
            if event.is_final_response():
                if hasattr(event, 'state') and event.state is not None:
                    raw_daily_data_list = event.state.get("list_of_all_daily_data_json")
                    if raw_daily_data_list and isinstance(raw_daily_data_list, list) and raw_daily_data_list:
                        try:
                            day_data_json_str = raw_daily_data_list[0]
                            day_data = SingleDayDataFromADK.model_validate_json(day_data_json_str).model_dump()
                            day_num_actual = day_data["day"]
                            life_summary["daily_summaries"].setdefault(target_year_d, {}).setdefault(target_month_d, {})[day_num_actual] = day_data
                            console.print(Panel(pretty_repr(day_data), title=f"ADK Daily Summary: {target_date_for_daily.isoformat()}", expand=False))
                        except (ValidationError, json.JSONDecodeError, IndexError) as e:
                            logger.error(f"Error processing daily summary for {target_date_for_daily.isoformat()}: {e}")
                else:
                    logger.warning(f"DailyLoop_ADK for {target_date_for_daily.isoformat()} final event did not have state or state was None.")
                break

    # 6. Hourly Breakdowns (ADK LoopAgent) - For relevant recent hours (e.g., today and yesterday)
    console.print(Rule("ADK LoopAgent: Hourly Breakdowns", style="bold yellow"))
    sim_curr_hour = life_summary["generation_info"]["current_hour"]
    days_for_hourly_loop: List[date] = []
    days_for_hourly_loop.append(character_current_sim_date) # Today
    yesterday_sim_date = character_current_sim_date - timedelta(days=1)
    if yesterday_sim_date >= birthdate_obj:
        days_for_hourly_loop.append(yesterday_sim_date) # Yesterday

    hourly_iteration_agent = create_hourly_iteration_adk_agent()

    for target_date_for_hourly in days_for_hourly_loop:
        target_year_h, target_month_h, target_day_h = target_date_for_hourly.year, target_date_for_hourly.month, target_date_for_hourly.day
        daily_summary_for_target_day = life_summary["daily_summaries"].get(target_year_h, {}).get(target_month_h, {}).get(target_day_h)
        if not daily_summary_for_target_day:
            logger.warning(f"No daily summary for {target_date_for_hourly.isoformat()}, skipping hourly generation.")
            continue
        daily_summary_for_target_day_json = json.dumps(daily_summary_for_target_day)

        console.print(f"-- Hourly Loop for: {target_date_for_hourly.isoformat()} --")
        max_hours_for_day = sim_curr_hour + 1 if target_date_for_hourly == character_current_sim_date else 24

        current_session = session_service.get_session(session_id=session_id_for_workflow)
        current_session.state.update({
            "persona_details_json": persona_details_json_str, "initial_relationships_json": initial_relationships_str, "world_details_json": world_details_json,
            "target_year_int": target_year_h, "target_month_int": target_month_h, "target_day_int": target_day_h,
            "daily_summary_for_target_day_json": daily_summary_for_target_day_json,
            # loop_iteration_count will be set by LoopAgent
            "list_of_all_hourly_data_json": json.dumps([])
        })
        session_service.update_session(current_session)

        hourly_loop = LoopAgent(name=f"HourlyLoop_ADK_{target_date_for_hourly.isoformat()}", loop_body_agent=hourly_iteration_agent, max_iterations=max_hours_for_day, output_key_loop_results="list_of_all_hourly_data_json")

        hourly_loop_runner = Runner(
            agent=hourly_loop,
            session_service=session_service,
            app_name=session_app_name # Use session's app_name
        )

        async for event in hourly_loop_runner.run_async(session_id=session_id_for_workflow, user_id=session_user_id): # Use session's user_id
            if event.is_final_response():
                if hasattr(event, 'state') and event.state is not None:
                    raw_hourly_data_list = event.state.get("list_of_all_hourly_data_json")
                    if raw_hourly_data_list:
                        hourly_activities_for_day = {}
                        try:
                            parsed_hourly_list = []
                            if isinstance(raw_hourly_data_list, str): temp_list = json.loads(raw_hourly_data_list)
                            else: temp_list = raw_hourly_data_list

                            for item_str_or_dict in temp_list:
                                if isinstance(item_str_or_dict, str): item_dict = SingleHourDataFromADK.model_validate_json(item_str_or_dict).model_dump()
                                else: item_dict = SingleHourDataFromADK.model_validate(item_str_or_dict).model_dump()
                                hourly_activities_for_day[item_dict["hour"]] = item_dict
                            
                            life_summary["hourly_breakdowns"].setdefault(target_year_h, {}).setdefault(target_month_h, {})[target_day_h] = {"activities": hourly_activities_for_day, "news": daily_summary_for_target_day.get("news_context_used", "N/A")}
                            console.print(Panel(pretty_repr(hourly_activities_for_day), title=f"ADK Hourly Activities: {target_date_for_hourly.isoformat()}", expand=False))
                        except (ValidationError, json.JSONDecodeError, TypeError) as e:
                            logger.error(f"Error processing hourly activities for {target_date_for_hourly.isoformat()}: {e}. Data: {str(raw_hourly_data_list)[:200]}")
                else:
                    logger.warning(f"HourlyLoop_ADK for {target_date_for_hourly.isoformat()} final event did not have state or state was None.")
                break
        
    console.print(Rule("ADK Orchestration Complete (All Levels Implemented)", style="bold green"))
    return life_summary

# --- Main Entry Point ---
async def generate_new_simulacra_background(
    sim_id: str,
    world_instance_uuid: Optional[str],
    world_type: str,
    world_description: str,
    allow_real_context: bool,
    gender_preference: Optional[str] = "any" # Added gender_preference
) -> Optional[Dict[str, Any]]:
    """
    Generates a new persona and their life summary using ADK agents orchestrated by Python.
    """
    # Use global-like module variables for ADK components for this run
    global _session_service_instance, _search_llm_agent_instance, _llm_service_instance # _runner_instance removed

    initial_allow_real_context = allow_real_context # Store the original value
    logger.info(f"Starting ADK-based Simulacra background generation for sim: {sim_id}...")

    # 1. Initialize LLMService (for fallback and Pydantic validation if needed)
    if not _llm_service_instance:
        try:
            _llm_service_instance = LLMService(api_key=os.getenv("GOOGLE_API_KEY"))
            logger.info("LLMService instance initialized.")
        except Exception as e:
            logger.critical(f"Error initializing LLMService: {e}.", exc_info=True)
            return None

    # 2. Initialize ADK SessionService
    if not _session_service_instance:
        _session_service_instance = InMemorySessionService()
        logger.info("InMemorySessionService instance initialized.")

    # 3. Initialize Search LLM Agent (if real context allowed and not already init)
    if allow_real_context and not _search_llm_agent_instance:
        try:
            _search_llm_agent_instance = create_search_llm_agent() 
            logger.info("SearchLLMAgent instance (with google_search tool) initialized.")
        except Exception as e_search_agent: # If search agent creation fails, disable real context
            logger.error(f"Failed to create SearchLLMAgent instance: {e_search_agent}. Real context searches will be disabled.", exc_info=True)
            allow_real_context = False # Disable if agent creation fails
            _search_llm_agent_instance = None # Ensure it's None

    # 4. ADK Runner is no longer initialized globally here.
    # It will be created within run_adk_orchestrated_life_generation for each main agent.

    # 5. Create a main session for this entire life generation workflow
    main_workflow_session_id: Optional[str] = None
    if _session_service_instance:
        try:
            # Store app_name and user_id used for session creation
            app_name_for_session = f"{APP_NAME}_LifeGenWorkflowInstance_{uuid.uuid4().hex[:6]}"
            user_id_for_session = f"workflow_user_{sim_id}"
            
            created_session = _session_service_instance.create_session(
                app_name=app_name_for_session, user_id=user_id_for_session
            )
            main_workflow_session_id = created_session.id
            logger.info(f"Main workflow session created: {main_workflow_session_id}")
            
            # DEBUG: Confirm session retrieval
            retrieved = _session_service_instance.get_session(
                session_id=main_workflow_session_id,
                app_name=app_name_for_session, # Pass the app_name used
                user_id=user_id_for_session     # Pass the user_id used
            )
            if not retrieved:
                logger.error(f"CRITICAL DEBUG: Workflow session {main_workflow_session_id} NOT RETRIEVABLE from SessionService (id: {id(_session_service_instance)}) Store: {list(_session_service_instance._sessions.keys())}")
            else:
                logger.debug(f"DEBUG: Workflow session {main_workflow_session_id} confirmed in SessionService instance.")
        except Exception as e_sess:
            logger.error(f"Failed to create main workflow session: {e_sess}", exc_info=True)
            return None
    else:
        logger.error("SessionService instance not available for workflow session creation.")
        return None

    # --- Prepare initial inputs and parameters ---
    persona_initial_input = {
        "world_type": world_type,
        "world_description": world_description,
        "gender_preference": gender_preference, # Use the parameter
        # "age_range_min": age_range[0], # For future agent prompt update
        # "age_range_max": age_range[1]  # For future agent prompt update
    }
    
    # --- Determine the correct 'generation_timestamp' based on world type ---
    # This logic is ported from life_generator.py for more nuanced timestamp setting.
    # First, we need a preliminary persona to get birthdate and age, if not overridden.
    # This is a bit chicken-and-egg. The ADK orchestrator will get the *final* persona.
    # For now, we'll set a preliminary generation_start_time. The orchestrator
    # will then use the *actual* persona's birthdate to refine calculations.
    # The `generation_params["generation_timestamp"]` will be the key reference.

    # If a persona_override were available here, we could use its birthdate.
    # Since it's not (it's passed to the orchestrator), we'll use a placeholder.
    # The orchestrator `run_adk_orchestrated_life_generation` will handle the
    # definitive calculation once the persona (and thus birthdate) is known.

    # For the purpose of what `generate_new_simulacra_background` passes to the orchestrator,
    # `datetime.now(timezone.utc)` is a reasonable starting point for `generation_timestamp`.
    # The *internal* logic of `run_adk_orchestrated_life_generation` correctly uses
    # the persona's birthdate against this `generation_timestamp` to determine age and history length.

    # The more complex logic from life_generator.py to determine `generation_start_time`
    # was based on having the persona *before* calling the main sequential generator.
    # In the ADK model, persona is generated *as part* of the sequence.
    # So, we pass `datetime.now(timezone.utc)` as the initial reference "current time".
    # The `run_adk_orchestrated_life_generation` function will then use the generated
    # persona's birthdate against this reference to calculate the character's actual age
    # and the span of their life to simulate.

    generation_start_time = datetime.now(timezone.utc)
    logger.info(f"Initial reference timestamp for generation: {generation_start_time.isoformat()}")

    generation_params = {
        "generation_timestamp": generation_start_time, # This is the "simulated now"
        "generation_info": { # Pre-populate with current real time of generation
            "generated_at_realworld": generation_start_time.isoformat(),
            # These will be updated by the orchestrator based on the persona's actual birthdate
            # relative to generation_timestamp
            "current_year": generation_start_time.year, # These will be updated based on persona
            "current_month": generation_start_time.month,
            "current_day": generation_start_time.day,
            "current_hour": generation_start_time.hour,
        }
    }

    # --- Call the Orchestrator ---
    life_data = await run_adk_orchestrated_life_generation(
        session_service=_session_service_instance, # type: ignore
        search_llm_agent=_search_llm_agent_instance, # type: ignore
        llm_service_for_validation=_llm_service_instance,
        initial_user_input=persona_initial_input,
        persona_details_override=None, # Let the ADK agent generate it
        generation_params=generation_params,
        session_id_for_workflow=main_workflow_session_id, # type: ignore
        session_app_name=app_name_for_session, # Pass the app_name used for session creation
        session_user_id=user_id_for_session,   # Pass the user_id used for session creation
        allow_real_context=allow_real_context
    )

    if not life_data:
        logger.error("ADK-orchestrated life summary generation failed critically.")
        return None

    # --- Finalize Metadata and Save ---
    life_data["sim_id"] = sim_id
    life_data["world_instance_uuid"] = world_instance_uuid
    # Ensure generation_info is correctly populated with final values
    final_gen_info = life_data.get("generation_info", {})
    final_gen_info["real_context_search_attempted"] = initial_allow_real_context # Use the stored initial value
    final_gen_info["real_context_search_enabled"] = allow_real_context # Potentially modified if search agent failed
    final_gen_info["generated_at_realworld"] = generation_start_time.isoformat() # Time this script ran
    # "generated_at" within generation_info should reflect the character's "current time"
    # which is set by generation_params["generation_timestamp"]

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
