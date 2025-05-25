# In c:\Users\dshea\Desktop\TheSimulation\src\simulation_utils.py

import json
import logging
import random
import re
from datetime import datetime, timezone  # Added for get_time_string_for_prompt
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # For Python 3.9+
except ImportError:
    ZoneInfo, ZoneInfoNotFoundError = None, None # Fallback for older Python
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from geopy.geocoders import Nominatim
# from google.adk.runners import Runner  # For type hinting
from google.genai import types as genai_types  # For type hinting
from rich import box
from google.adk.tools import agent_tool # For AgentTool type hint
from google.adk.agents.invocation_context import InvocationContext # For type hint
from google.adk.tools import ToolContext # Added for AgentTool.run_async
from rich.columns import Columns  # Added for two-column layout
from rich.panel import Panel  # For generate_table
from rich.table import Table  # For generate_table
from rich.text import Text  # For styled text in table
from timezonefinder import TimezoneFinder

# Import constants from the config module
from .config import APP_NAME  # Added APP_NAME for geopy user_agent
from .config import (  # PROB_INTERJECT_AS_NARRATIVE removed; SIMULACRA_KEY is imported from config and will now point to "simulacra_profiles"; Added SIMULACRA_KEY
    ACTIVE_SIMULACRA_IDS_KEY, LOCATION_DETAILS_KEY, LOCATION_KEY, MODEL_NAME,
    PROB_INTERJECT_AS_SELF_REFLECTION, SIMULACRA_KEY, USER_ID, WORLD_STATE_KEY,
    WORLD_TEMPLATE_DETAILS_KEY)
from .loop_utils import \
    get_nested  # Assuming get_nested remains in loop_utils or is moved here

logger = logging.getLogger(__name__)

def _update_state_value(target_state: Dict[str, Any], key_path: str, value: Any, logger_instance: logging.Logger) -> bool:
    """
    Safely updates a nested value in the state dictionary.
    Requires a logger instance to be passed.
    """
    try:
        keys = key_path.split('.')
        target = target_state
        for i, key in enumerate(keys[:-1]):
            if not isinstance(target, dict):
                logger_instance.error(f"Invalid path '{key_path}': Segment '{keys[i-1]}' is not a dictionary.")
                return False
            if key not in target or not isinstance(target[key], dict):
                logger_instance.warning(f"Path segment '{key}' not found or not dict in '{key_path}'. Creating.")
                target[key] = {}
            target = target[key]

        final_key = keys[-1]
        if not isinstance(target, dict):
             logger_instance.error(f"Invalid path '{key_path}': Segment before final key '{final_key}' is not a dictionary.")
             return False

        target[final_key] = value
        logger_instance.info(f"[StateUpdate] Applied: {key_path} = {value}")
        return True
    except Exception as e:
        logger_instance.error(f"Error updating state for path '{key_path}' with value '{value}': {e}", exc_info=True)
        return False

def generate_table(current_state: Dict[str, Any], event_bus_qsize: int, narration_qsize: int) -> Columns:
    """
    Generates a two-column Rich layout for live display based on the provided current_state.
    """
    sim_time_str = f"{current_state.get('world_time', 0.0):.2f}s"
    overall_title = f"Simulation State @ {sim_time_str}"

    # --- Table 1: General Info, System, World Feeds, Pending Events ---
    table1 = Table(title="[bold cyan]World & System[/]", show_header=False, box=box.MINIMAL, padding=(0,1), expand=True)
    table1.add_column("Parameter", style="dim", no_wrap=True, ratio=1)
    table1.add_column("Value", overflow="fold", no_wrap=False, ratio=3)

    table1.add_row("World Time", sim_time_str)
    table1.add_row("World UUID", str(get_nested(current_state, 'world_instance_uuid', default='N/A')))
    world_desc = get_nested(current_state, WORLD_TEMPLATE_DETAILS_KEY, 'description', default='N/A')
    table1.add_row("World Desc", world_desc[:35] + ("..." if len(world_desc) > 35 else ""))

    table1.add_row(Text("--- System ---", style="bold blue"), Text("---", style="bold blue"))
    table1.add_row("Event Bus Size", str(event_bus_qsize))
    table1.add_row("Narration Q Size", str(narration_qsize))

    # --- Table 2: Simulacra, Location Objects & NPCs ---
    table2 = Table(title="[bold yellow]Agents & Location[/]", show_header=False, box=box.MINIMAL, padding=(0,1), expand=True)
    table2.add_column("Parameter", style="dim", no_wrap=True, ratio=1)
    table2.add_column("Value", overflow="fold", no_wrap=False, ratio=3)
    
    active_sim_ids = current_state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
    sim_limit = 3
    primary_actor_location_id: Optional[str] = None # To store the location of the first active sim

    for i, sim_id in enumerate(active_sim_ids):
        if i == 0: # Get location of the first active simulacra for object display
            primary_actor_location_id = get_nested(current_state, SIMULACRA_KEY, sim_id, "location")

        if i >= sim_limit and sim_limit > 0 : # Add check for sim_limit > 0
            table2.add_row(f"... ({len(active_sim_ids) - sim_limit} more)", "...")
            break
        sim_state_data = get_nested(current_state, SIMULACRA_KEY, sim_id, default={})
        table2.add_row(Text(f"--- Sim: {get_nested(sim_state_data, 'persona_details', 'Name', default=sim_id)} ---", style="bold magenta"), Text("---", style="bold magenta"))
        table2.add_row(f"  Status", get_nested(sim_state_data, 'status', default="Unknown"))
        table2.add_row(f"  Location", get_nested(sim_state_data, 'location', default="Unknown"))
        sim_goal = get_nested(sim_state_data, 'goal', default="Unknown")
        table2.add_row(f"  Goal", sim_goal[:35] + ("..." if len(sim_goal) > 35 else ""))
        table2.add_row(f"  Action End", f"{get_nested(sim_state_data, 'current_action_end_time', default=0.0):.2f}s" if get_nested(sim_state_data, 'status')=='busy' else "N/A")
        last_obs = get_nested(sim_state_data, 'last_observation', default="None")
        table2.add_row(f"  Last Obs.", last_obs[:35] + ("..." if len(last_obs) > 35 else ""))
        action_desc = get_nested(sim_state_data, 'current_action_description', default="N/A")
        table2.add_row(f"  Curr. Action", action_desc[:35] + ("..." if len(action_desc) > 35 else ""))
        interrupt_prob = get_nested(sim_state_data, 'current_interrupt_probability')
        if interrupt_prob is not None:
            table2.add_row(f"  Dyn.Int.Prob", f"{interrupt_prob:.2%}")
        else:
            # Check if busy and potentially on cooldown
            if get_nested(sim_state_data, 'status') == 'busy':
                last_interjection_time = get_nested(sim_state_data, "last_interjection_sim_time", 0.0)
                current_sim_time_table = get_nested(current_state, "world_time", 0.0) # Get current time for check
                # INTERJECTION_COOLDOWN_SIM_SECONDS would need to be imported from config
                # For simplicity, we'll just assume if it's None and busy, it might be cooldown or too short
                table2.add_row(f"  Dyn.Int.Prob", "[dim]N/A (Cooldown/Short)[/dim]")
            # else: if not busy, don't show the line

    # --- Ephemeral Objects in Primary Actor's Location ---
    location_name_display = primary_actor_location_id if primary_actor_location_id else "Unknown Location"
    table2.add_row(Text(f"--- Objects in {location_name_display} ---", style="bold cyan"), Text("---", style="bold cyan"))
    
    ephemeral_objects_in_loc: List[Dict[str, Any]] = []
    if primary_actor_location_id:
        ephemeral_objects_in_loc = get_nested(
            current_state,
            WORLD_STATE_KEY,
            LOCATION_DETAILS_KEY,
            primary_actor_location_id,
            "ephemeral_objects", # Key where ephemeral objects are stored
            default=[]
        )
    
    table2.add_row(f"  (Ephemeral)", f"({len(ephemeral_objects_in_loc)} total)")
    if ephemeral_objects_in_loc:
        object_display_limit = 5 # How many ephemeral objects to show
        for i, obj_data in enumerate(ephemeral_objects_in_loc):
            if i >= object_display_limit:
                table2.add_row(f"    ... ({len(ephemeral_objects_in_loc) - object_display_limit} more)", "")
                break
            obj_name = obj_data.get("name", "N/A")
            obj_id = obj_data.get("id", "N/A")
            obj_desc = obj_data.get('description', '')
            table2.add_row(f"    {obj_name} ({obj_id})", obj_desc[:35] + ("..." if len(obj_desc) > 35 else ""))
    
    # --- Ephemeral NPCs in Primary Actor's Location ---
    table2.add_row(Text(f"--- NPCs in {location_name_display} ---", style="bold green"), Text("---", style="bold green"))
    ephemeral_npcs_in_loc: List[Dict[str, Any]] = []
    if primary_actor_location_id:
        ephemeral_npcs_in_loc = get_nested(
            current_state,
            WORLD_STATE_KEY,
            LOCATION_DETAILS_KEY,
            primary_actor_location_id,
            "ephemeral_npcs", # Key where ephemeral NPCs are stored
            default=[]
        )
    table2.add_row(f"  (Ephemeral)", f"({len(ephemeral_npcs_in_loc)} total)")
    if ephemeral_npcs_in_loc:
        npc_display_limit = 3 # How many ephemeral NPCs to show
        for i, npc_data in enumerate(ephemeral_npcs_in_loc):
            if i >= npc_display_limit:
                table2.add_row(f"    ... ({len(ephemeral_npcs_in_loc) - npc_display_limit} more)", "")
                break
            npc_name = npc_data.get("name", "N/A")
            npc_id = npc_data.get("id", "N/A") 
            npc_desc = npc_data.get('description', '')
            table2.add_row(f"    {npc_name} ({npc_id})", npc_desc[:35] + ("..." if len(npc_desc) > 35 else ""))

    narrative_log_entries = get_nested(current_state, 'narrative_log', default=[])[-6:]
    truncated_log_entries = []
    max_log_line_length = 35 # Reduced from 70
    for entry in narrative_log_entries:
        if len(entry) > max_log_line_length:
            truncated_log_entries.append(entry[:max_log_line_length - 3] + "...")
        else:
            truncated_log_entries.append(entry)
    log_display = "\n".join(truncated_log_entries)
    table1.add_row("Narrative Log", log_display)

    weather_feed = get_nested(current_state, 'world_feeds', 'weather', 'condition', default='N/A')
    news_updates = get_nested(current_state, 'world_feeds', 'news_updates', default=[])
    pop_culture_updates = get_nested(current_state, 'world_feeds', 'pop_culture_updates', default=[])
    news_headlines_display = [item.get('headline', 'N/A') for item in news_updates[:3]]
    pop_culture_headline_display = pop_culture_updates[0].get('headline', 'N/A') if pop_culture_updates else 'N/A'
    table1.add_row(Text("--- World Feeds ---", style="bold yellow"), Text("---", style="bold yellow"))
    table1.add_row("  Weather", weather_feed)
    for i, headline in enumerate(news_headlines_display):
        table1.add_row(f"  News {i+1}", headline[:35] + "..." if len(headline) > 35 else headline)
    table1.add_row(f"  Pop Culture", pop_culture_headline_display[:35] + "..." if len(pop_culture_headline_display) > 35 else pop_culture_headline_display)

    # --- Pending Simulation Events ---
    pending_events = get_nested(current_state, 'pending_simulation_events', default=[])
    table1.add_row(Text("--- Pending Events ---", style="bold orange_red1"), Text("---", style="bold orange_red1"))
    table1.add_row("  (Scheduled)", f"({len(pending_events)} total)")

    if pending_events:
        # Sort events by trigger time for better readability
        sorted_pending_events = sorted(pending_events, key=lambda x: x.get('trigger_sim_time', float('inf')))
        event_display_limit = 3 # How many pending events to show
        for i, event_data in enumerate(sorted_pending_events):
            if i >= event_display_limit:
                table1.add_row(f"    ... ({len(pending_events) - event_display_limit} more)", "")
                break
            event_type = event_data.get("event_type", "Unknown Event")
            trigger_time = event_data.get("trigger_sim_time", 0.0)
            target_agent = event_data.get("target_agent_id", "World")
            details_snippet = str(event_data.get("details", {}))[:35] + "..." # Snippet of details
            table1.add_row(f"    {event_type} for {target_agent}", f"at {trigger_time:.1f}s ({details_snippet})")

    # Combine tables into Columns
    return Columns([table1, table2], title=overall_title, expand=True, padding=1)

async def _fetch_and_summarize_real_feed(
    category_for_logging: str,
    search_query: str,
    summarization_prompt_template: str, # Should contain {search_results} placeholder
    output_format_note: str, # This is part of the summarization prompt now
    search_agent_tool: Optional[agent_tool.AgentTool], # Changed from runner/session
    adk_context_for_tool_run: Optional[InvocationContext], # Context for running the tool
    llm_for_summarization: genai.GenerativeModel,
    logger_instance: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Helper to perform search and then summarize the results for world feeds."""
    raw_search_results_text = ""
    search_tool_used_successfully = False
    
    if not (search_agent_tool and adk_context_for_tool_run):
        logger_instance.warning(f"[{category_for_logging}] Search agent tool or ADK context unavailable. Cannot fetch real feed.")
        return None

    logger_instance.info(f"[{category_for_logging}] Attempting REAL search with query: '{search_query}'")
    
    try:
        # The SearchLLMAgent is expected to take the query as text input.
        # AgentTool.run_async returns the final content of the wrapped agent.
        # We assume SearchLLMAgent's final output is the summarized search result text.
        # If SearchLLMAgent had an output_schema, this would be a parsed object.

        # 1. Create ToolContext from the InvocationContext passed to this function
        tool_ctx = ToolContext(invocation_context=adk_context_for_tool_run)
        
        # 2. Prepare the arguments dictionary for the tool.
        #    For an LlmAgent without a specific input_schema (like your SearchLLMAgent),
        #    AgentTool expects the query string under the 'request' key.
        tool_args_dict = {'request': search_query}

        # 3. Call run_async with keyword arguments
        tool_output = await search_agent_tool.run_async(
            args=tool_args_dict,
            tool_context=tool_ctx
        )
        if isinstance(tool_output, str):
            raw_search_results_text = tool_output
            search_tool_used_successfully = True
        elif isinstance(tool_output, dict) and "answer" in tool_output: # If SearchLLMAgent returns structured
            raw_search_results_text = tool_output.get("answer", "")
            if not raw_search_results_text and "search_results" in tool_output: # Fallback to raw results
                raw_search_results_text = json.dumps(tool_output["search_results"])
            search_tool_used_successfully = True
        else:
            logger_instance.warning(f"[{category_for_logging}] Search agent tool returned unexpected output type: {type(tool_output)}")

    except Exception as e_tool_run:
        logger_instance.error(f"[{category_for_logging}] Error running search agent tool: {e_tool_run}", exc_info=True)
        return None

    if search_tool_used_successfully and raw_search_results_text.strip():
        logger_instance.info(f"[{category_for_logging}] REAL search returned: {raw_search_results_text.strip()[:400]}")
        summarization_prompt = summarization_prompt_template.format(search_results=raw_search_results_text.strip()[:1000]) + f"\n{output_format_note}"
        # The response_obj here is from the summarization LLM, not the search tool directly.
        # The search tool's output (raw_search_results_text) is fed into this summarization.
        summarization_response = await llm_for_summarization.generate_content_async(summarization_prompt)
        
        # Parse the summarization LLM's response
        if summarization_response and summarization_response.text:
            try:
                # Attempt to parse the JSON from the summarization LLM
                parsed_summary = json.loads(re.sub(r'^```json\s*|\s*```$', '', summarization_response.text.strip(), flags=re.MULTILINE))
                return parsed_summary # Return the parsed dictionary
            except json.JSONDecodeError:
                logger_instance.error(f"[{category_for_logging}] Failed to decode JSON from summarization LLM: {summarization_response.text.strip()}")
                # Fallback: return the raw text if JSON parsing fails, wrapped in a dict
                return {"summary_text": summarization_response.text.strip(), "error": "Summarization JSON decode failed"}
        else:
            logger_instance.warning(f"[{category_for_logging}] Summarization LLM returned no text.")
            return None
    
    logger_instance.warning(f"[{category_for_logging}] REAL search did not yield usable results. Raw: {raw_search_results_text[:200]}")
    return None

async def generate_simulated_world_feed_content(
    current_sim_state: Dict[str, Any],
    category: str,
    simulation_time: float,
    location_context: str,
    world_mood: str,
    logger_instance: logging.Logger,
    # Parameters for real feeds using AgentTool
    search_agent_tool_for_real_feeds: Optional[agent_tool.AgentTool] = None,
    adk_context_for_tool_run: Optional[InvocationContext] = None # Context from the calling ADK agent
) -> Dict[str, Any]:
    """Generates world feed content, conditionally using real search."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        llm_for_summarization = genai.GenerativeModel(MODEL_NAME)
        prompt_text = f"Current simulation time: {simulation_time:.0f} seconds. Location context: {location_context}. World Mood: {world_mood}.\n"
        output_format_note = " Respond ONLY with a JSON object matching the specified format." # Added space
        response_obj = None

        world_type = get_nested(current_sim_state, WORLD_TEMPLATE_DETAILS_KEY, 'world_type', default="fictional")
        sub_genre = get_nested(current_sim_state, WORLD_TEMPLATE_DETAILS_KEY, 'sub_genre', default="turn_based")
        use_real_feeds = world_type == "real" and sub_genre == "realtime"

        search_tool_used_successfully = False 

        if category == "weather":
            if use_real_feeds:
                search_query = f"What is the current weather in {location_context}?"
                summarization_template = "Based on this weather information: '{search_results}'\nExtract the current weather condition, temperature in Celsius (as an integer), and a short forecast. Format: {{{{ \"condition\": \"str\", \"temperature_celsius\": int, \"forecast_short\": \"str\" }}}}\nIf temperature is in Fahrenheit, convert it to Celsius. If exact data is missing, make a plausible estimation."
                parsed_summary_data = await _fetch_and_summarize_real_feed(
                    category_for_logging="WorldInfoGatherer_Weather", search_query=search_query,
                    summarization_prompt_template=summarization_template, output_format_note=output_format_note,
                    search_agent_tool=search_agent_tool_for_real_feeds, adk_context_for_tool_run=adk_context_for_tool_run,
                    llm_for_summarization=llm_for_summarization,
                    logger_instance=logger_instance
                )
                if not parsed_summary_data: # Fallback if real search failed or components missing
                    logger_instance.warning(f"[WorldInfoGatherer] REAL weather search for '{location_context}' did not yield usable results or components missing. Falling back to simulation.")
                    prompt_text += f"Generate a plausible, brief weather report for {location_context}. Format: {{{{ \"condition\": \"str\", \"temperature_celsius\": int, \"forecast_short\": \"str\" }}}}\n{output_format_note}"
            else: 
                # This branch is for when use_real_feeds is False
                prompt_text += f"Generate a plausible, brief weather report for {location_context}. Format: {{{{ \"condition\": \"str\", \"temperature_celsius\": int, \"forecast_short\": \"str\" }}}}\n{output_format_note}"
        
        elif category in ["world_news", "regional_news", "local_news", "pop_culture"]:
            if use_real_feeds:
                search_query = ""
                if category == "world_news":
                    search_query = "What are the latest top world news headlines (e.g., politics, social issues, environment, major international events)?"
                elif category == "regional_news":
                    country = get_nested(current_sim_state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'country', default="").strip()
                    search_query = f"What are the latest top national news headlines for {country}?" if country else f"What are the latest top regional news headlines for {location_context}?"
                elif category == "local_news":
                    city = get_nested(current_sim_state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'city', default="").strip()
                    state_province = get_nested(current_sim_state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'state', default="").strip()
                    local_search_term = f"{city}, {state_province}" if city and state_province else city or "current location"
                    search_query = f"What are the top latest local news headlines for {local_search_term}?"
                elif category == "pop_culture":
                    country = get_nested(current_sim_state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'country', default="").strip()
                    pop_culture_region = f"{country} " if country else ""
                    search_query = f"What are the top latest {pop_culture_region} pop culture trends and entertainment news headlines (e.g., movies, music, viral trends)?"

                summarization_template = f"Based on these search results for {category}: '{{search_results}}'\nProvide a single, very concise headline and a one-sentence summary. Format: {{{{ \"headline\": \"str\", \"summary\": \"str\" }}}}"
                parsed_summary_data = await _fetch_and_summarize_real_feed(
                    category_for_logging=f"WorldInfoGatherer_{category}", search_query=search_query,
                    summarization_prompt_template=summarization_template, output_format_note=output_format_note,
                    search_agent_tool=search_agent_tool_for_real_feeds, adk_context_for_tool_run=adk_context_for_tool_run,
                    llm_for_summarization=llm_for_summarization,
                    logger_instance=logger_instance
                )
                if not parsed_summary_data: # Fallback
                    logger_instance.warning(f"[WorldInfoGatherer] REAL search for '{category}' did not yield usable results or components missing. Falling back to simulation.")
                    prompt_text += f"Generate a plausible, concise {category.replace('_', ' ')} headline and summary. Format: {{{{ \"headline\": \"str\", \"summary\": \"str\" }}}}\n{output_format_note}"
            else: 
                # This branch is for when use_real_feeds is False
                prompt_text += f"Generate a plausible, concise {category.replace('_', ' ')} headline and summary. Format: {{{{ \"headline\": \"str\", \"summary\": \"str\" }}}}\n{output_format_note}"
        else: 
            return {"error": "Unknown category"}

        # If parsed_summary_data exists from real search, use it directly
        if 'parsed_summary_data' in locals() and parsed_summary_data:
            parsed_summary_data["timestamp"] = simulation_time
            parsed_summary_data["source_category"] = category
            return parsed_summary_data
        
        # Otherwise, generate content using LLM (fallback or non-realtime path)
        if not response_obj: # Ensure response_obj is only set if not using parsed_summary_data
            response_obj = await model.generate_content_async(prompt_text)
        response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_obj.text.strip() if response_obj and response_obj.text else "{}", flags=re.MULTILINE)
        
        try:
            data = json.loads(response_text_clean)
            data["timestamp"] = simulation_time
            data["source_category"] = category
            return data
        except json.JSONDecodeError:
            logger_instance.error(f"Failed to decode JSON for {category} from LLM: {response_text_clean}")
            return {"error": f"JSON decode error for {category}", "raw_response": response_text_clean, "timestamp": simulation_time, "source_category": category}
    except Exception as e:
        logger_instance.error(f"Error generating LLM world feed for {category}: {e}", exc_info=True)
        return {"error": f"LLM generation error for {category}", "timestamp": simulation_time, "source_category": category}

def get_time_string_for_prompt(
    state: Dict[str, Any],
    sim_elapsed_time_seconds: Optional[float] = None
) -> str:
    """
    Gets the appropriate time string for agent prompts based on the global state.
    If in "real/realtime" mode, returns the current real-world localized time.
    Otherwise, returns the provided sim_elapsed_time_seconds formatted as elapsed time.
    Requires sim_elapsed_time_seconds if not in real/realtime mode.
    """
    world_template_details_time = state.get(WORLD_TEMPLATE_DETAILS_KEY, {})
    sim_world_type_time = world_template_details_time.get("world_type")
    sim_sub_genre_time = world_template_details_time.get("sub_genre")

    # Initialize geolocator and timezonefinder once if possible, or handle potential re-init
    # For simplicity here, we'll init them inside the 'realtime' block.
    # In a high-frequency scenario, you might initialize them once globally or pass them in.

    if sim_world_type_time == "real" and sim_sub_genre_time == "realtime":
        now_utc = datetime.now(timezone.utc)
        overall_location_dict_time = world_template_details_time.get(LOCATION_KEY, {})
        city_name_raw = overall_location_dict_time.get('city', 'Unknown City')
        country_name_raw = overall_location_dict_time.get('country', '')
        location_context_for_time_str = f"{city_name_raw}{', ' + country_name_raw if country_name_raw else ''}"

        iana_tz_str = None
        if city_name_raw != 'Unknown City':
            location_query_for_geocoding = f"{city_name_raw}, {country_name_raw}".strip(", ")
            try:
                geolocator = Nominatim(user_agent=APP_NAME) # APP_NAME from your config
                location_geo = geolocator.geocode(location_query_for_geocoding, timeout=5) # Reduced timeout

                if location_geo:
                    tf = TimezoneFinder()
                    iana_tz_str = tf.timezone_at(lng=location_geo.longitude, lat=location_geo.latitude)
                    if iana_tz_str:
                        logger.info(f"Determined IANA timezone for '{location_query_for_geocoding}': {iana_tz_str}")
                    else:
                        logger.warning(f"TimezoneFinder could not determine timezone for {location_query_for_geocoding} at ({location_geo.latitude}, {location_geo.longitude}).")
                else:
                    logger.warning(f"Could not geocode location: '{location_query_for_geocoding}'")
            except Exception as e_geo: # Catch broader exceptions from geopy/timezonefinder
                logger.error(f"Error during geocoding/timezone finding for '{location_query_for_geocoding}': {e_geo}")

        if ZoneInfo and iana_tz_str:
            try:
                city_tz = ZoneInfo(iana_tz_str)
                now_local = now_utc.astimezone(city_tz)
                return f"{now_local.strftime('%I:%M %p on %A, %B %d, %Y')} (Local time for {location_context_for_time_str})"
            except ZoneInfoNotFoundError:
                logger.warning(f"Dynamically found IANA timezone '{iana_tz_str}' for city '{city_name_raw}' not found by zoneinfo. Falling back to UTC.")
            except Exception as e_tz:
                logger.error(f"Error converting time for city '{city_name_raw}' with timezone '{iana_tz_str}': {e_tz}. Falling back to UTC.")
        elif ZoneInfo and city_name_raw != 'Unknown City' and not iana_tz_str:
             logger.warning(f"Could not dynamically determine IANA timezone for '{city_name_raw}'. Falling back to UTC.")
        elif not ZoneInfo:
            logger.warning("zoneinfo module not available (requires Python 3.9+). Falling back to UTC with context for time.")

        # Fallback if dynamic lookup fails, ZoneInfo not available, or city is "Unknown City"
        return f"{now_utc.strftime('%I:%M %p on %A, %B %d, %Y')} (UTC). The current local time in {location_context_for_time_str} should be inferred."

    elif sim_elapsed_time_seconds is not None:
        return f"{sim_elapsed_time_seconds:.1f}s elapsed"
    return "Time unknown (elapsed not provided for non-realtime or state missing)"

def get_random_style_combination(
    include_general=True,
    num_general=1,
    include_lighting=True,
    num_lighting=1,
    include_color=True,
    num_color=1,
    include_technique=True,
    num_technique=1,
    include_composition=True, # New category for composition
    num_composition=1,
    include_atmosphere=True,  # New category for atmosphere/mood
    num_atmosphere=1
):
    """
    Generates a random combination of photographic styles.

    Args:
        # ... (existing args) ...
        include_general (bool): Whether to include general photographic styles.
        num_general (int): Number of general styles to sample (if included).
        include_lighting (bool): Whether to include lighting/mood styles.
        num_lighting (int): Number of lighting/mood styles to sample (if included).
        include_color (bool): Whether to include color/tone styles.
        num_color (int): Number of color/tone styles to sample (if included).
        include_technique (bool): Whether to include camera technique styles.
        num_technique (int): Number of camera technique styles to sample (if included).
        include_composition (bool): Whether to include compositional styles.
        num_composition (int): Number of compositional styles to sample.
        include_atmosphere (bool): Whether to include atmospheric/emotional styles.
        num_atmosphere (int): Number of atmospheric styles to sample.

    Returns:
        str: A comma-separated string of randomly selected styles.
             Returns an empty string if no categories are included or no styles are sampled.
    """

    # Define the lists of styles by category
    general_styles = [
        "Documentary Photography", "Street Photography", "Fine Art Photography",
        "Environmental Portraiture", "Minimalist Photography", "Abstract Photography", "Photojournalism",
        "Conceptual Photography", "Urban Photography", "Landscape Photography",
        "Still Life Photography", "Fashion Photography", "Architectural Photography"
    ]

    lighting_styles = [
        "Cinematic Lighting", "Soft Natural Light", "High Key", "Low Key",
        "Golden Hour Photography", "Blue Hour Photography", "Dramatic Lighting",
        "Rim Lighting", "Backlit", "Chiaroscuro", "Studio Lighting", "Available Light"
    ]

    color_styles = [
        "Monochromatic (Black and White)", "Vibrant and Saturated", "Muted Tones",
        "Sepia Tone", "High Contrast Color", "Pastel Colors", "Duotone", "Cross-processed look",
        "Natural Color Palette", "Warm Tones", "Cool Tones"
    ]

    # Note: Bokeh/Shallow Depth are often implied by your prompt's requirement
    # for a blurred background. Deep Depth is the opposite.
    # Include these if you want to explicitly reinforce or add variety.
    technique_styles = [
        "Bokeh-rich", "Shallow Depth of Field", "Deep Depth of Field", "Long Exposure", "Motion Blur",
        "Panning Shot", "High-Speed Photography", "Tilt-Shift Effect", "Lens Flare (subtle)",
        "Wide-Angle Perspective", "Telephoto Compression", "Macro Detail", "Clean and Sharp"
    ]

    compositional_styles = [
        "Rule of Thirds", "Leading Lines", "Symmetrical Composition", "Asymmetrical Balance",
        "Frame within a Frame", "Dynamic Symmetry", "Golden Ratio", "Negative Space Emphasis",
        "Pattern and Repetition", "Centered Subject", "Off-center Subject"
    ]

    atmospheric_styles = [
        "Ethereal Mood", "Dreamlike Atmosphere", "Gritty Realism", "Nostalgic Feel",
        "Serene and Calm", "Dynamic and Energetic", "Mysterious Ambiance", "Whimsical Charm",
        "Dramatic and Intense", "Melancholic Tone", "Uplifting and Bright", "Crisp Morning Air",
        "Humid Haze", "Foggy Overlay"
    ]

    selected_styles = []
    style_categories_used = 0 # To track how many categories contributed

    # Sample from each category based on the parameters
    if include_general and num_general > 0:
        # Ensure we don't try to sample more than available styles
        k = min(num_general, len(general_styles))
        selected_styles.extend(random.sample(general_styles, k))

    if include_lighting and num_lighting > 0:
        k = min(num_lighting, len(lighting_styles))
        selected_styles.extend(random.sample(lighting_styles, k))
        if k > 0: style_categories_used +=1

    if include_color and num_color > 0:
        k = min(num_color, len(color_styles))
        selected_styles.extend(random.sample(color_styles, k))
        if k > 0: style_categories_used +=1

    if include_technique and include_technique > 0:
        k = min(num_technique, len(technique_styles))
        selected_styles.extend(random.sample(technique_styles, k))
        if k > 0: style_categories_used +=1

    if include_composition and num_composition > 0:
        k = min(num_composition, len(compositional_styles))
        selected_styles.extend(random.sample(compositional_styles, k))
        if k > 0: style_categories_used +=1

    if include_atmosphere and num_atmosphere > 0:
        k = min(num_atmosphere, len(atmospheric_styles))
        selected_styles.extend(random.sample(atmospheric_styles, k))
        if k > 0: style_categories_used +=1

    # Shuffle the final list to mix the order of styles
    random.shuffle(selected_styles)
    
    # Log the selected styles for debugging/monitoring
    # Ensure logger is defined in the scope where this function is defined,
    # or pass it as an argument if necessary.
    # For now, assuming 'logger' is accessible (e.g., module-level logger).
    if selected_styles:
        logger.info(f"Selected {len(selected_styles)} styles from {style_categories_used} categories: {', '.join(selected_styles)}")

    # Join the selected styles into a comma-separated string
    return ", ".join(selected_styles)

def get_random_style_combination(
    include_general=True,
    num_general=1,
    include_lighting=True,
    num_lighting=1,
    include_color=True,
    num_color=1,
    include_technique=True,
    num_technique=1,
    include_composition=True, # New category for composition
    num_composition=1,
    include_atmosphere=True,  # New category for atmosphere/mood
    num_atmosphere=1
):
    """
    Generates a random combination of photographic styles.
    """

    # Define the lists of styles by category
    general_styles = [
        "Documentary Photography", "Street Photography", "Fine Art Photography",
        "Environmental Portraiture", "Minimalist Photography", "Abstract Photography", "Photojournalism",
        "Conceptual Photography", "Urban Photography", "Landscape Photography",
        "Still Life Photography", "Fashion Photography", "Architectural Photography"
    ]

    lighting_styles = [
        "Cinematic Lighting", "Soft Natural Light", "High Key", "Low Key",
        "Golden Hour Photography", "Blue Hour Photography", "Dramatic Lighting",
        "Rim Lighting", "Backlit", "Chiaroscuro", "Studio Lighting", "Available Light"
    ]

    color_styles = [
        "Monochromatic (Black and White)", "Vibrant and Saturated", "Muted Tones",
        "Sepia Tone", "High Contrast Color", "Pastel Colors", "Duotone", "Cross-processed look",
        "Natural Color Palette", "Warm Tones", "Cool Tones"
    ]

    technique_styles = [
        "Bokeh-rich", "Shallow Depth of Field", "Deep Depth of Field", "Long Exposure", "Motion Blur",
        "Panning Shot", "High-Speed Photography", "Tilt-Shift Effect", "Lens Flare (subtle)",
        "Wide-Angle Perspective", "Telephoto Compression", "Macro Detail", "Clean and Sharp"
    ]

    compositional_styles = [
        "Rule of Thirds", "Leading Lines", "Symmetrical Composition", "Asymmetrical Balance",
        "Frame within a Frame", "Dynamic Symmetry", "Golden Ratio", "Negative Space Emphasis",
        "Pattern and Repetition", "Centered Subject", "Off-center Subject"
    ]

    atmospheric_styles = [
        "Ethereal Mood", "Dreamlike Atmosphere", "Gritty Realism", "Nostalgic Feel",
        "Serene and Calm", "Dynamic and Energetic", "Mysterious Ambiance", "Whimsical Charm",
        "Dramatic and Intense", "Melancholic Tone", "Uplifting and Bright", "Crisp Morning Air",
        "Humid Haze", "Foggy Overlay"
    ]

    selected_styles = []
    style_categories_used = 0 

    if include_general and num_general > 0:
        k = min(num_general, len(general_styles))
        selected_styles.extend(random.sample(general_styles, k))
    if include_lighting and num_lighting > 0:
        k = min(num_lighting, len(lighting_styles))
        selected_styles.extend(random.sample(lighting_styles, k))
        if k > 0: style_categories_used +=1
    if include_color and num_color > 0:
        k = min(num_color, len(color_styles))
        selected_styles.extend(random.sample(color_styles, k))
        if k > 0: style_categories_used +=1
    if include_technique and num_technique > 0: # Corrected variable name
        k = min(num_technique, len(technique_styles))
        selected_styles.extend(random.sample(technique_styles, k))
        if k > 0: style_categories_used +=1
    if include_composition and num_composition > 0:
        k = min(num_composition, len(compositional_styles))
        selected_styles.extend(random.sample(compositional_styles, k))
        if k > 0: style_categories_used +=1
    if include_atmosphere and num_atmosphere > 0:
        k = min(num_atmosphere, len(atmospheric_styles))
        selected_styles.extend(random.sample(atmospheric_styles, k))
        if k > 0: style_categories_used +=1

    random.shuffle(selected_styles)
    
    if selected_styles:
        logger.info(f"Selected {len(selected_styles)} styles from {style_categories_used} categories: {', '.join(selected_styles)}")

    return ", ".join(selected_styles)
