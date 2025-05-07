# src/simulation_utils.py - Utility functions for the simulation

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

# Import constants from the config module
from .config import (
    MODEL_NAME, PROB_INTERJECT_AS_SELF_REFLECTION, PROB_INTERJECT_AS_NARRATIVE,
    WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, ACTIVE_SIMULACRA_IDS_KEY, USER_ID
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
    for i, sim_id in enumerate(active_sim_ids):
        if i >= sim_limit:
            table.add_row(f"... ({len(active_sim_ids) - sim_limit} more)", "...")
            break
        sim_state_data = current_state.get("simulacra", {}).get(sim_id, {})
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

    object_limit = 3
    objects_dict = get_nested(current_state, 'objects', default={})
    table.add_row("--- Objects ---", f"({len(objects_dict)} total)")
    for i, (obj_id, obj_state_data) in enumerate(objects_dict.items()):
         if i >= object_limit:
             table.add_row(f"... ({len(objects_dict) - object_limit} more)", "...")
             break
         obj_name = get_nested(obj_state_data, 'name', default=obj_id)
         obj_loc = get_nested(obj_state_data, 'location', default='Unknown')
         obj_power = get_nested(obj_state_data, 'power')
         obj_locked = get_nested(obj_state_data, 'locked')
         obj_status = get_nested(obj_state_data, 'status')
         obj_interactive = get_nested(obj_state_data, 'interactive')
         details = f"Loc: {obj_loc}"
         if obj_power is not None: details += f", Pwr: {obj_power}"
         if obj_locked is not None: details += f", Lck: {'Y' if obj_locked else 'N'}"
         if obj_status is not None: details += f", Sts: {obj_status}"
         if obj_interactive is not None: details += f", Int: {'Y' if obj_interactive else 'N'}"
         table.add_row(f"  {obj_name}", details)

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

async def generate_llm_interjection_detail(
    agent_name_for_prompt: str,
    agent_current_action_desc: str,
    interjection_category: str,
    world_mood: str,
    global_search_agent_runner: Optional[Runner],
    search_agent_session_id: Optional[str],
    user_id_for_search: str,
    logger_instance: logging.Logger
) -> str:
    """Generates a brief interjection detail using an LLM."""
    try:
        if PROB_INTERJECT_AS_SELF_REFLECTION + PROB_INTERJECT_AS_NARRATIVE > 1.01: # type: ignore
            logger_instance.warning(
                f"Sum of PROB_INTERJECT_AS_SELF_REFLECTION ({PROB_INTERJECT_AS_SELF_REFLECTION}) and " # type: ignore
                f"PROB_INTERJECT_AS_NARRATIVE ({PROB_INTERJECT_AS_NARRATIVE}) exceeds 1.0. " # type: ignore
                "World events might not be chosen.")

        model = genai.GenerativeModel(MODEL_NAME)
        prompt_text = ""
        if interjection_category == "narrative":
            prompt_text = f"""
Agent {agent_name_for_prompt} is currently: "{agent_current_action_desc}".
The general world mood is: "{world_mood}".
Invent a brief, personal, and distracting event for {agent_name_for_prompt}. This could be a sudden vivid memory, an unexpected personal thought, a brief message or call from a generic acquaintance (e.g., "a friend," "a colleague," "an old contact" - do not use specific names unless it's a generic title like "your boss"), or a minor bodily sensation.
The event should be something that would momentarily break their concentration.
Output ONLY the single, short, descriptive sentence of this event. Example: "A wave of nostalgia for a childhood memory washes over you." or "Your comm-link buzzes with an incoming call from an unknown number."
Keep it concise and impactful.
"""
        elif interjection_category == "world_event":
            if global_search_agent_runner and search_agent_session_id:
                try:
                    search_query = "latest brief world news update"
                    # Send the query directly
                    search_trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
                    raw_search_results_text = ""
                    search_tool_used_successfully = False

                    async for event in global_search_agent_runner.run_async(user_id=user_id_for_search, session_id=search_agent_session_id, new_message=search_trigger_content):
                        logger_instance.debug(f"[InterjectionSearch] Event ID: {getattr(event, 'id', 'N/A')}, Author: {getattr(event, 'author', 'N/A')}")
                        if event.is_final_response() and event.content and event.content.parts:
                            part = event.content.parts[0]
                            if hasattr(part, 'function_response') and part.function_response and hasattr(part.function_response, 'response'):
                                tool_response_data = dict(part.function_response.response)
                                raw_search_results_text = json.dumps(tool_response_data.get("results", tool_response_data))
                                search_tool_used_successfully = True
                                logger_instance.info(f"[InterjectionSearch] Tool response received: {raw_search_results_text[:200]}...")
                            elif part.text:
                                raw_search_results_text = part.text
                                # Check if it's NOT the agent just echoing tool_code
                                if not ("tool_code" in raw_search_results_text and "google_search" in raw_search_results_text):
                                    search_tool_used_successfully = True
                                    logger_instance.info(f"[InterjectionSearch] Text response received: {raw_search_results_text[:200]}...")
                                else:
                                    logger_instance.warning(f"[InterjectionSearch] Agent returned tool_code: {raw_search_results_text[:200]}...")
                            break # Process first final response

                    if search_tool_used_successfully and raw_search_results_text:
                        summarization_model = genai.GenerativeModel(MODEL_NAME)
                        summarization_prompt = f"Given this raw search result: '{raw_search_results_text[:500]}...'\nCreate a very short, impactful, one-sentence news flash suitable for a brief interjection for {agent_name_for_prompt}. Example: 'A news alert flashes on a nearby screen: Major international agreement reached.'"
                        summary_response = await summarization_model.generate_content_async(summarization_prompt)
                        if summary_response.text:
                            return summary_response.text.strip()
                    else:
                        logger_instance.warning("[InterjectionSearch] Search did not yield usable results. Falling back to LLM invention for interjection.")
                except Exception as search_interject_e:
                    logger_instance.error(f"Error using search for world_event interjection: {search_interject_e}", exc_info=True)
            
            # Fallback prompt if search fails or not used
            prompt_text = f"""
Agent {agent_name_for_prompt} is currently: "{agent_current_action_desc}".
The general world mood is: "{world_mood}".
Invent a brief, subtle, and distracting environmental event or a piece of background world news. This could be a flicker of lights, a distant sound, a change in temperature, a news snippet on a nearby screen, or a minor system alert.
The event should be something that would momentarily break their concentration.
Output ONLY the single, short, descriptive sentence of this event. Example: "The overhead lights flicker momentarily." or "A news bulletin flashes on a nearby screen: 'Local transport system experiencing minor delays.'"
Keep it concise and impactful.
"""
        else:
            return "A moment of quiet contemplation passes."

        response = await model.generate_content_async(prompt_text)
        return response.text.strip() if response.text else "You notice something out of the corner of your eye."
    except Exception as e:
        logger_instance.error(f"Error generating LLM interjection detail: {e}", exc_info=True)
        return "A fleeting distraction crosses your mind."

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

        search_tool_used_successfully = False # More accurately, did we get usable results
        raw_search_results_text = ""

        if category == "weather":
            if use_real_feeds and global_search_agent_runner and search_agent_session_id:
                search_query = f"What is the current weather in {location_context}?"
                logger_instance.info(f"[WorldInfoGatherer] Attempting REAL weather search for '{location_context}' with query: '{search_query}'")
                # Send the query directly
                # search_trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
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
            else: # Not using real feeds or components unavailable
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
                # Send the query directly
                search_trigger_content = genai_types.Content(role='user', parts=[genai_types.Part(text=search_query)])
                # search_trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
                
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
            else: # Not using real feeds or components unavailable
                if use_real_feeds: logger_instance.warning(f"[WorldInfoGatherer] Intended REAL search for '{category}' but search components unavailable. Falling back.")
                prompt_text += f"Generate a plausible, concise {category.replace('_', ' ')} headline and summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
        else: # Unknown category
            return {"error": "Unknown category"}

        # If search was attempted but failed to produce response_obj, or if not using real feeds
        if not (response_obj and response_obj.text):
            response_obj = await model.generate_content_async(prompt_text)

        response_text = response_obj.text.strip() if response_obj and response_obj.text else "{}"
        # Clean potential markdown ```json ... ```
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