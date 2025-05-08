# In c:\Users\dshea\Desktop\TheSimulation\src\simulation_utils.py

import json
import logging
import random
import re
from typing import Any, Dict, Optional, List

import google.generativeai as genai
from google.adk.runners import Runner # For type hinting
from google.genai import types as genai_types # For type hinting

from rich.table import Table # For generate_table
from rich.panel import Panel # For generate_table
from rich.text import Text # For styled text in table

# Import constants from the config module
from .config import (
    MODEL_NAME, PROB_INTERJECT_AS_SELF_REFLECTION, # PROB_INTERJECT_AS_NARRATIVE removed
    WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, ACTIVE_SIMULACRA_IDS_KEY, USER_ID,
    # Added these constants as they are used for fetching ephemeral objects/NPCs
    WORLD_STATE_KEY, LOCATION_DETAILS_KEY, SIMULACRA_KEY
)
from .loop_utils import get_nested # Assuming get_nested remains in loop_utils or is moved here

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

def generate_table(current_state: Dict[str, Any], event_bus_qsize: int, narration_qsize: int) -> Table:
    """
    Generates the Rich table for live display based on the provided current_state.
    """
    table = Table(title=f"Simulation State @ {current_state.get('world_time', 0.0):.2f}s",
                  show_header=True, header_style="bold magenta", box=None,
                  padding=(0, 1), expand=True)
    table.add_column("Parameter", style="dim", no_wrap=True)
    table.add_column("Value", overflow="fold", no_wrap=False)

    table.add_row("World Time", f"{current_state.get('world_time', 0.0):.2f}s")
    table.add_row("World UUID", str(get_nested(current_state, 'world_instance_uuid', default='N/A')))
    world_desc = get_nested(current_state, WORLD_TEMPLATE_DETAILS_KEY, 'description', default='N/A')
    table.add_row("World Desc", world_desc[:80] + ("..." if len(world_desc) > 80 else ""))

    active_sim_ids = current_state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
    sim_limit = 3
    primary_actor_location_id: Optional[str] = None # To store the location of the first active sim

    for i, sim_id in enumerate(active_sim_ids):
        if i == 0: # Get location of the first active simulacra for object display
            primary_actor_location_id = get_nested(current_state, SIMULACRA_KEY, sim_id, "location")

        if i >= sim_limit:
            table.add_row(f"... ({len(active_sim_ids) - sim_limit} more)", "...")
            break
        sim_state_data = current_state.get(SIMULACRA_KEY, {}).get(sim_id, {})
        table.add_row(f"--- Sim: {get_nested(sim_state_data, 'name', default=sim_id)} ---", "---")
        table.add_row(f"  Status", get_nested(sim_state_data, 'status', default="Unknown"))
        table.add_row(f"  Location", get_nested(sim_state_data, 'location', default="Unknown"))
        sim_goal = get_nested(sim_state_data, 'goal', default="Unknown")
        table.add_row(f"  Goal", sim_goal[:60] + ("..." if len(sim_goal) > 60 else ""))
        table.add_row(f"  Action End", f"{get_nested(sim_state_data, 'current_action_end_time', default=0.0):.2f}s" if get_nested(sim_state_data, 'status')=='busy' else "N/A")
        last_obs = get_nested(sim_state_data, 'last_observation', default="None")
        table.add_row(f"  Last Obs.", last_obs[:80] + ("..." if len(last_obs) > 80 else ""))
        action_desc = get_nested(sim_state_data, 'current_action_description', default="N/A")
        table.add_row(f"  Curr. Action", action_desc[:70] + ("..." if len(action_desc) > 70 else ""))
        interrupt_prob = get_nested(sim_state_data, 'current_interrupt_probability')
        if interrupt_prob is not None:
            table.add_row(f"  Dyn.Int.Prob", f"{interrupt_prob:.2%}")
        else:
            # Check if busy and potentially on cooldown
            if get_nested(sim_state_data, 'status') == 'busy':
                last_interjection_time = get_nested(sim_state_data, "last_interjection_sim_time", 0.0)
                current_sim_time_table = get_nested(current_state, "world_time", 0.0) # Get current time for check
                # INTERJECTION_COOLDOWN_SIM_SECONDS would need to be imported from config
                # For simplicity, we'll just assume if it's None and busy, it might be cooldown or too short
                table.add_row(f"  Dyn.Int.Prob", "[dim]N/A (Cooldown/Short)[/dim]")
            # else: if not busy, don't show the line

    # --- Ephemeral Objects in Primary Actor's Location ---
    location_name_display = primary_actor_location_id if primary_actor_location_id else "Unknown Location"
    table.add_row(Text(f"--- Objects in {location_name_display} ---", style="bold cyan"), Text("---", style="bold cyan"))
    
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
    
    table.add_row(f"  (Ephemeral)", f"({len(ephemeral_objects_in_loc)} total)")
    if ephemeral_objects_in_loc:
        object_display_limit = 5 # How many ephemeral objects to show
        for i, obj_data in enumerate(ephemeral_objects_in_loc):
            if i >= object_display_limit:
                table.add_row(f"    ... ({len(ephemeral_objects_in_loc) - object_display_limit} more)", "")
                break
            obj_name = obj_data.get("name", "N/A")
            obj_id = obj_data.get("id", "N/A")
            obj_desc = obj_data.get('description', '')
            table.add_row(f"    {obj_name} ({obj_id})", obj_desc[:60] + ("..." if len(obj_desc) > 60 else ""))
    
    # --- Ephemeral NPCs in Primary Actor's Location ---
    table.add_row(Text(f"--- NPCs in {location_name_display} ---", style="bold green"), Text("---", style="bold green"))
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
    table.add_row(f"  (Ephemeral)", f"({len(ephemeral_npcs_in_loc)} total)")
    if ephemeral_npcs_in_loc:
        npc_display_limit = 3 # How many ephemeral NPCs to show
        for i, npc_data in enumerate(ephemeral_npcs_in_loc):
            if i >= npc_display_limit:
                table.add_row(f"    ... ({len(ephemeral_npcs_in_loc) - npc_display_limit} more)", "")
                break
            npc_name = npc_data.get("name", "N/A")
            npc_id = npc_data.get("id", "N/A") 
            npc_desc = npc_data.get('description', '')
            table.add_row(f"    {npc_name} ({npc_id})", npc_desc[:60] + ("..." if len(npc_desc) > 60 else ""))


    table.add_row("--- System ---", "---")
    table.add_row("Event Bus Size", str(event_bus_qsize))
    table.add_row("Narration Q Size", str(narration_qsize))

    narrative_log_entries = get_nested(current_state, 'narrative_log', default=[])[-6:]
    truncated_log_entries = []
    max_log_line_length = 70
    for entry in narrative_log_entries:
        if len(entry) > max_log_line_length:
            truncated_log_entries.append(entry[:max_log_line_length - 3] + "...")
        else:
            truncated_log_entries.append(entry)
    log_display = "\n".join(truncated_log_entries)
    table.add_row("Narrative Log", log_display)

    weather_feed = get_nested(current_state, 'world_feeds', 'weather', 'condition', default='N/A')
    news_updates = get_nested(current_state, 'world_feeds', 'news_updates', default=[])
    pop_culture_updates = get_nested(current_state, 'world_feeds', 'pop_culture_updates', default=[])
    news_headlines_display = [item.get('headline', 'N/A') for item in news_updates[:3]]
    pop_culture_headline_display = pop_culture_updates[0].get('headline', 'N/A') if pop_culture_updates else 'N/A'
    table.add_row("--- World Feeds ---", "---")
    table.add_row("  Weather", weather_feed)
    for i, headline in enumerate(news_headlines_display):
        table.add_row(f"  News {i+1}", headline[:70] + "..." if len(headline) > 70 else headline)
    table.add_row(f"  Pop Culture", pop_culture_headline_display[:70] + "..." if len(pop_culture_headline_display) > 70 else pop_culture_headline_display)
    return table

async def generate_simulated_world_feed_content(
    current_sim_state: Dict[str, Any],
    category: str,
    simulation_time: float,
    location_context: str,
    world_mood: str,
    global_search_agent_runner: Optional[Runner],
    search_agent_session_id: Optional[str],
    user_id_for_search: str,
    logger_instance: logging.Logger
) -> Dict[str, Any]:
    """Generates world feed content, conditionally using real search."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        llm_for_summarization = genai.GenerativeModel(MODEL_NAME)
        prompt_text = f"Current simulation time: {simulation_time:.0f} seconds. Location context: {location_context}. World Mood: {world_mood}.\n"
        output_format_note = "Respond ONLY with a JSON object matching the specified format."
        response_obj = None

        world_type = get_nested(current_sim_state, WORLD_TEMPLATE_DETAILS_KEY, 'world_type', default="fictional")
        sub_genre = get_nested(current_sim_state, WORLD_TEMPLATE_DETAILS_KEY, 'sub_genre', default="turn_based")
        use_real_feeds = world_type == "real" and sub_genre == "realtime"

        search_tool_used_successfully = False 
        raw_search_results_text = ""

        if category == "weather":
            if use_real_feeds and global_search_agent_runner and search_agent_session_id:
                search_query = f"What is the current weather in {location_context}?"
                logger_instance.info(f"[WorldInfoGatherer] Attempting REAL weather search for '{location_context}' with query: '{search_query}'")
                search_trigger_content = genai_types.Content(role='user', parts=[genai_types.Part(text=search_query)])
                async for event in global_search_agent_runner.run_async(user_id=user_id_for_search, session_id=search_agent_session_id, new_message=search_trigger_content): # type: ignore
                    logger_instance.debug(f"[WorldInfoGatherer_SearchEvent_Weather] Event ID: {getattr(event, 'id', 'N/A')}, Author: {getattr(event, 'author', 'N/A')}")
                    if event.is_final_response() and event.content and event.content.parts:
                        part = event.content.parts[0]
                        if hasattr(part, 'function_response') and part.function_response and hasattr(part.function_response, 'response'):
                            tool_response_data = dict(part.function_response.response)
                            raw_search_results_text = json.dumps(tool_response_data.get("results", tool_response_data))
                            search_tool_used_successfully = True
                            logger_instance.info(f"[WorldInfoGatherer_SearchEvent_Weather] Tool response: {raw_search_results_text[:200]}...")
                        elif part.text:
                            raw_search_results_text = part.text
                            if not ("tool_code" in raw_search_results_text and "google_search" in raw_search_results_text):
                                search_tool_used_successfully = True
                                logger_instance.info(f"[WorldInfoGatherer_SearchEvent_Weather] Text response: {raw_search_results_text[:200]}...")
                            else:
                                logger_instance.warning(f"[WorldInfoGatherer_SearchEvent_Weather] Agent returned tool_code: {raw_search_results_text[:200]}...")
                        break 
                
                if search_tool_used_successfully and raw_search_results_text.strip():
                    logger_instance.info(f"[WorldInfoGatherer] REAL weather search returned: {raw_search_results_text.strip()[:500]}")
                    summarization_prompt = f"Based on this weather information: '{raw_search_results_text.strip()[:1000]}...'\nExtract the current weather condition, temperature in Celsius (as an integer), and a short forecast. Format: {{\"condition\": \"str\", \"temperature_celsius\": int, \"forecast_short\": \"str\"}}\nIf temperature is in Fahrenheit, convert it to Celsius. If exact data is missing, make a plausible estimation. {output_format_note}"
                    response_obj = await llm_for_summarization.generate_content_async(summarization_prompt)
                else:
                    logger_instance.warning(f"[WorldInfoGatherer] REAL weather search for '{location_context}' did not yield usable results. Falling back. Raw: {raw_search_results_text[:200]}")
                    prompt_text += f"Generate a plausible, brief weather report for {location_context}. Format: {{\"condition\": \"str\", \"temperature_celsius\": int, \"forecast_short\": \"str\"}}\n{output_format_note}"
            else: 
                if use_real_feeds: logger_instance.warning(f"[WorldInfoGatherer] Intended REAL weather for '{location_context}' but search components unavailable. Falling back.")
                prompt_text += f"Generate a plausible, brief weather report for {location_context}. Format: {{\"condition\": \"str\", \"temperature_celsius\": int, \"forecast_short\": \"str\"}}\n{output_format_note}"
        
        elif category in ["world_news", "regional_news", "local_news", "pop_culture"]:
            if use_real_feeds and global_search_agent_runner and search_agent_session_id:
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

                logger_instance.info(f"[WorldInfoGatherer] Attempting REAL search for '{category}' with query: '{search_query}'")
                search_trigger_content = genai_types.Content(role='user', parts=[genai_types.Part(text=search_query)])
                
                async for event in global_search_agent_runner.run_async(user_id=user_id_for_search, session_id=search_agent_session_id, new_message=search_trigger_content): # type: ignore
                    logger_instance.debug(f"[WorldInfoGatherer_SearchEvent_{category}] Event ID: {getattr(event, 'id', 'N/A')}, Author: {getattr(event, 'author', 'N/A')}")
                    if event.is_final_response() and event.content and event.content.parts:
                        part = event.content.parts[0]
                        if hasattr(part, 'function_response') and part.function_response and hasattr(part.function_response, 'response'):
                            tool_response_data = dict(part.function_response.response)
                            raw_search_results_text = json.dumps(tool_response_data.get("results", tool_response_data))
                            search_tool_used_successfully = True
                            logger_instance.info(f"[WorldInfoGatherer_SearchEvent_{category}] Tool response: {raw_search_results_text[:200]}...")
                        elif part.text:
                            raw_search_results_text = part.text
                            if not ("tool_code" in raw_search_results_text and "google_search" in raw_search_results_text):
                                search_tool_used_successfully = True
                                logger_instance.info(f"[WorldInfoGatherer_SearchEvent_{category}] Text response: {raw_search_results_text[:200]}...")
                            else:
                                logger_instance.warning(f"[WorldInfoGatherer_SearchEvent_{category}] Agent returned tool_code: {raw_search_results_text[:200]}...")
                        break
                
                if search_tool_used_successfully and raw_search_results_text.strip():
                    logger_instance.info(f"[WorldInfoGatherer] Search for '{category}' returned: {raw_search_results_text.strip()[:500]}")
                    summarization_prompt = f"Based on these search results for {category}: '{raw_search_results_text.strip()[:1000]}...'\nProvide a single, very concise headline and a one-sentence summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
                    response_obj = await llm_for_summarization.generate_content_async(summarization_prompt)
                else:
                    logger_instance.warning(f"[WorldInfoGatherer] REAL search for '{category}' did not yield usable results. Falling back. Raw: {raw_search_results_text[:200]}")
                    prompt_text += f"Generate a plausible, concise {category.replace('_', ' ')} headline and summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
            else: 
                if use_real_feeds: logger_instance.warning(f"[WorldInfoGatherer] Intended REAL search for '{category}' but search components unavailable. Falling back.")
                prompt_text += f"Generate a plausible, concise {category.replace('_', ' ')} headline and summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
        else: 
            return {"error": "Unknown category"}

        if not (response_obj and response_obj.text):
            response_obj = await model.generate_content_async(prompt_text)

        response_text = response_obj.text.strip() if response_obj and response_obj.text else "{}"
        response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        
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
