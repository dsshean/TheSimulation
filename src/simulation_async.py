# src/simulation_async.py - Core Simulation Orchestrator
import asyncio
import glob
import json
import logging
import os
import random
import re
import sys
from types import SimpleNamespace 
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai  # For direct API config
# from atproto import Client, models  # Moved to core_tasks
# from google import genai as genai_image  # Moved to core_tasks
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types as genai_types
# from PIL import Image # No longer used directly here
from pydantic import ValidationError # BaseModel, Field, etc. are no longer needed here
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .socket_server import socket_server_task

console = Console() # Keep a global console for direct prints if needed by run_simulation
from .core_tasks import dynamic_interruption_task, narrative_image_generation_task

from .agents import create_narration_llm_agent  # Agent creation functions
from .agents import (create_search_llm_agent, create_simulacra_llm_agent,
                     create_world_engine_llm_agent, create_world_generator_llm_agent) # Added WorldGenerator
# Import from our new/refactored modules
from .config import (  # For run_simulation; For self-reflection; New constants for dynamic_interruption_task; PROB_INTERJECT_AS_NARRATIVE removed; from this import list; Import Bluesky and social post config; Import SIMULACRA_KEY
    ACTIVE_SIMULACRA_IDS_KEY, AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS,
    AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS, API_KEY, APP_NAME,
    BLUESKY_APP_PASSWORD, BLUESKY_HANDLE, CURRENT_LOCATION_KEY,
    DEFAULT_HOME_DESCRIPTION, DEFAULT_HOME_LOCATION_NAME, # DYNAMIC_INTERRUPTION constants moved to core_tasks import
    ENABLE_BLUESKY_POSTING, # ENABLE_NARRATIVE_IMAGE_GENERATION moved to core_tasks import
    HOME_LOCATION_KEY,
    IMAGE_GENERATION_INTERVAL_REAL_SECONDS, IMAGE_GENERATION_MODEL_NAME, MODEL_NAME, # Removed AGENT_MODEL_NAME alias
    IMAGE_GENERATION_OUTPUT_DIR, INTERJECTION_COOLDOWN_SIM_SECONDS,
    LIFE_SUMMARY_DIR, LOCATION_DETAILS_KEY, LOCATION_KEY,
    LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS, MAX_MEMORY_LOG_ENTRIES, # MODEL_NAME as AGENT_MODEL_NAME removed
    MAX_SIMULATION_TIME, MEMORY_LOG_CONTEXT_LENGTH,
    MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK,
    PROB_INTERJECT_AS_SELF_REFLECTION, RANDOM_SEED, SEARCH_AGENT_MODEL_NAME,
    SIMULACRA_KEY, SIMULACRA_PROFILES_KEY, SIMULATION_SPEED_FACTOR,
    SOCIAL_POST_HASHTAGS, SOCIAL_POST_TEXT_LIMIT, STATE_DIR, UPDATE_INTERVAL, # CONFIG_MODEL_NAME removed
    USER_ID, WORLD_STATE_KEY, WORLD_TEMPLATE_DETAILS_KEY)
from .core_tasks import time_manager_task, world_info_gatherer_task
from .loop_utils import (get_nested, load_json_file,
                         load_or_initialize_simulation, parse_json_output_last,
                         save_json_file)
from .perception_manager import PerceptionManager # Import the moved PerceptionManager
from .models import NarratorOutput, WorldGeneratorOutput, GeneratedLocationDetail  # Pydantic models for tasks in this file
from .models import SimulacraIntentResponse, WorldEngineResponse
from .simulation_utils import (  # Utility functions; generate_llm_interjection_detail REMOVED
    _update_state_value, generate_table, get_time_string_for_prompt, get_target_entity_state, # Re-added get_time_string_for_prompt, added get_target_entity_state
    _log_event) # get_random_style_combination is used by core_tasks
from .state_loader import parse_location_string  # Used in run_simulation

logger = logging.getLogger(__name__) # Use logger from main entry point setup

# --- Core Components (Module Scope) ---
# These are the "globals" that will be managed here and passed around or accessed directly by tasks in this file.
event_bus = asyncio.Queue()
narration_queue = asyncio.Queue()
perception_manager_global: Optional['PerceptionManager'] = None # Forward declaration
state: Dict[str, Any] = {} # Global state dictionary
event_logger_global: Optional[logging.Logger] = None # Global variable for the event logger

adk_session_service: Optional[InMemorySessionService] = None
adk_memory_service: Optional[InMemoryMemoryService] = None

world_engine_agent: Optional[LlmAgent] = None
world_engine_runner: Optional[Runner] = None
world_engine_session_id: Optional[str] = None

narration_agent_instance: Optional[LlmAgent] = None # Renamed
narration_runner: Optional[Runner] = None
narration_session_id: Optional[str] = None

simulacra_agents_map: Dict[str, LlmAgent] = {} # Renamed
simulacra_runners_map: Dict[str, Runner] = {}
simulacra_session_ids_map: Dict[str, str] = {}

search_llm_agent_instance: Optional[LlmAgent] = None # Renamed
search_agent_runner_instance: Optional[Runner] = None # Renamed
search_agent_session_id_val: Optional[str] = None # Renamed

world_generator_agent: Optional[LlmAgent] = None
world_generator_runner: Optional[Runner] = None
world_generator_session_id: Optional[str] = None

world_mood_global: str = "The familiar, everyday real world; starting the morning routine at home."
live_display_object: Optional[Live] = None

# Add these helper functions after the imports
async def safe_queue_get(queue, timeout=5.0, task_name="Unknown"):
    """Safely get from queue with timeout and retry logic."""
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"[{task_name}] Queue get() timed out after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"[{task_name}] Queue get() error: {e}")
        return None

async def safe_queue_put(queue, item, timeout=5.0, task_name="Unknown"):
    """Safely put to queue with timeout and retry logic."""
    try:
        await asyncio.wait_for(queue.put(item), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.warning(f"[{task_name}] Queue put() timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"[{task_name}] Queue put() error: {e}")
        return False

async def queue_health_monitor():
    """Monitor queue health and log warnings for potential deadlocks."""
    last_event_bus_size = 0
    last_narration_size = 0
    stalled_count = 0
    
    while True:
        try:
            current_eb_size = event_bus.qsize()
            current_nq_size = narration_queue.qsize()
            
            # Check for stalled queues (same size for multiple cycles)
            if current_eb_size > 0 and current_eb_size == last_event_bus_size:
                stalled_count += 1
                if stalled_count >= 3:  # 15 seconds of no progress
                    logger.warning(f"Event bus appears stalled: {current_eb_size} items for {stalled_count * 5}s")
            else:
                stalled_count = 0
                
            # Log warnings for large queues
            if current_eb_size > 10:
                logger.warning(f"Event bus queue large: {current_eb_size} items")
            if current_nq_size > 5:
                logger.warning(f"Narration queue large: {current_nq_size} items")
                
            last_event_bus_size = current_eb_size
            last_narration_size = current_nq_size
            
            await asyncio.sleep(5.0)
            
        except Exception as e:
            logger.error(f"Queue health monitor error: {e}")
            await asyncio.sleep(10.0)

def get_current_sim_time():
    return state.get("world_time", 0.0)

# --- Helper function to convert Pydantic models or SimpleNamespace to dict ---
def _to_dict(obj: Any, exclude: Optional[set[str]] = None) -> Dict[str, Any]:
    """Converts Pydantic models or SimpleNamespace objects to dictionaries."""
    if obj is None:
        return {}
    if hasattr(obj, 'model_dump'): # Pydantic model
        return obj.model_dump(exclude=exclude) if exclude else obj.model_dump()
    if isinstance(obj, SimpleNamespace):
        data = vars(obj)
        if exclude:
            return {k: v for k, v in data.items() if k not in exclude}
        return data
    if isinstance(obj, dict): # If it's already a dict
        if exclude:
            return {k: v for k, v in obj.items() if k not in exclude}
        return obj
    logger.warning(f"Unsupported type for _to_dict: {type(obj)}. Returning empty dict.")
    return {}

def _list_to_dicts_if_needed(list_of_items: List[Any]) -> List[Dict[str, Any]]:
    """Converts a list of items (Pydantic models, SimpleNamespace, or dicts) to a list of dicts."""
    if not isinstance(list_of_items, list): # Handle cases where it might not be a list due to LLM error
        return []
    return [
        _to_dict(item) for item in list_of_items
        # No need to check item type here, _to_dict handles Pydantic, SimpleNamespace, and dict
        # It will return {} for unsupported types within the list.
    ]

# --- Helper Function for ADK Agent Calls ---
async def _call_adk_agent_and_parse(
    runner: Runner,
    agent_instance: LlmAgent,
    agent_dedicated_session_id: str, # The agent's own persistent session ID
    user_id: str,
    trigger_content: genai_types.Content,
    expected_pydantic_model: type,
    agent_name_for_logging: str,
    logger_instance: logging.Logger,
) -> Optional[Any]:
    """
    Calls an ADK agent, processes its response, and parses it into the expected Pydantic model.
    Ensures only the trigger_content is sent to the LLM.
    Handles dynamic instruction updates if provided.
    """
    modified_instruction = False
    original_include_contents = agent_instance.include_contents # Save original value
    # The `always_clear_llm_contents_callback` in agents.py handles clearing history before model call.
    # So, we can use the agent's dedicated session ID directly.

    try:

        # No need to set runner.agent = agent_instance, as the passed `runner`
        # is already the dedicated runner for `agent_instance`.

        llm_response_data = None
        raw_text_from_llm = ""

        async for event_llm in runner.run_async(user_id=user_id, session_id=agent_dedicated_session_id, new_message=trigger_content):
            if event_llm.error_message:
                logger_instance.error(f"[{agent_name_for_logging}] LLM Error in session {agent_dedicated_session_id}: {event_llm.error_message}")
                return None 
            
            if event_llm.is_final_response() and event_llm.content:
                if isinstance(event_llm.content, expected_pydantic_model):
                    llm_response_data = event_llm.content
                    logger_instance.debug(f"[{agent_name_for_logging}] ADK successfully parsed {expected_pydantic_model.__name__} schema.")
                    break 
                elif event_llm.content.parts:
                    raw_text_from_llm = event_llm.content.parts[0].text.strip()
                    logger_instance.debug(f"[{agent_name_for_logging}] LLM Final Raw Content: {raw_text_from_llm[:200]}...")
                    
                    # Attempt to parse the raw text using the robust parser first.
                    # parse_json_output_last handles markdown fences and attempts json.loads.
                    parsed_dict_from_robust_parser = parse_json_output_last(raw_text_from_llm)
                    
                    if parsed_dict_from_robust_parser:
                        logger_instance.info(f"[{agent_name_for_logging}] Successfully parsed with robust parser into a dictionary.")
                        llm_response_data = parsed_dict_from_robust_parser
                        # try:
                        #     llm_response_data = expected_pydantic_model.model_validate(parsed_dict_from_robust_parser)
                        #     logger_instance.debug(f"[{agent_name_for_logging}] Successfully validated robust parse with Pydantic.")
                        #     # Successfully parsed and validated
                        # except ValidationError as ve:
                        #     logger_instance.error(f"[{agent_name_for_logging}] Pydantic validation failed for robust parse: {ve}. Raw: {raw_text_from_llm}")
                        #     llm_response_data = None # Ensure it's None if validation fails
                    else:
                        logger_instance.error(f"[{agent_name_for_logging}] Robust parser (parse_json_output_last) failed to extract JSON. Raw: {raw_text_from_llm}")
                        # llm_response_data remains None (its initial value from the top of the try block)
                        llm_response_data = None

                    # If all parsing attempts (ADK, robust, etc.) fail, llm_response_data will remain None.
                    if llm_response_data: # If any method succeeded
                        break # Exit loop after processing final response

                else:
                    print(f"[{agent_name_for_logging}] No content in final response, skipping.")
    finally:
        pass
        # Restore original instruction if it was modified
        # Restore original include_contents setting
    # If llm_response_data is a dictionary (from robust parsing), convert it to SimpleNamespace.
    # If it's already a Pydantic model instance (from ADK direct parse) or None, it will be returned as is.
    if isinstance(llm_response_data, dict):
        try:
            return SimpleNamespace(**llm_response_data)
        except TypeError as e: # Handles cases like non-string keys if they somehow occur
            logger_instance.error(f"[{agent_name_for_logging}] Failed to convert dict to SimpleNamespace: {e}. Returning dict. Dict: {llm_response_data}")
            # Fallback to returning the dictionary itself if SimpleNamespace conversion fails
    return llm_response_data

# --- ADK-Dependent Tasks (Remain in this file for global context access) ---

async def narration_task():
    """Listens for completed actions on the narration queue and generates stylized narrative."""
    logger.info("[NarrationTask] Task started.")
    if not narration_runner or not narration_agent_instance or not narration_session_id:
        logger.error("[NarrationTask] Narration ADK components (runner, agent, session_id) not initialized. Task cannot proceed.")
        return

    while True:
        action_event = None
        try:
            action_event = await safe_queue_get(narration_queue, timeout=10.0, task_name="NarrationTask")
            
            if action_event is None:
                continue

            actor_id = get_nested(action_event, "actor_id")
            intent = get_nested(action_event, "action")
            results = get_nested(action_event, "results", default={})
            outcome_desc = get_nested(action_event, "outcome_description", default="Something happened.")
            
            # Discovery details are passed for narrative context only - state is already updated
            discovered_objects_for_narration_ctx = action_event.get("discovered_objects_for_narration", [])
            discovered_npcs_for_narration_ctx = action_event.get("discovered_npcs_for_narration", [])
            discovered_connections_for_narration_ctx = action_event.get("discovered_connections_for_narration", [])

            completion_time = get_nested(action_event, "completion_time", default=state.get("world_time", 0.0))
            actor_location_at_action_time = get_nested(action_event, "actor_current_location_id")

            if not actor_id:
                logger.warning(f"[NarrationTask] Received narration event without actor_id: {action_event}")
                narration_queue.task_done()
                continue

            actor_name = get_nested(state, SIMULACRA_KEY, actor_id, "persona_details", "Name", default=actor_id)
            logger.debug(f"[NarrationTask] Using global world mood: '{world_mood_global}' for actor {actor_name}")

            # Build narrative context
            def clean_history_entry(entry: str) -> str:
                cleaned = re.sub(r'^\[T\d+\.\d+\]\s*', '', entry)
                cleaned = re.sub(r'\[\w+Agent(?:_sim_\w+)?\] said: ```json.*?```', '', cleaned, flags=re.DOTALL).strip()
                return cleaned
            
            raw_recent_narrative = state.get("narrative_log", [])[-5:]
            cleaned_recent_narrative = [clean_history_entry(entry) for entry in raw_recent_narrative if clean_history_entry(entry)]
            history_str = "\n".join(cleaned_recent_narrative)

            time_for_narrator_prompt = get_time_string_for_prompt(state, sim_elapsed_time_seconds=completion_time)
            weather_for_narrator_prompt = get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
            news_updates_for_narrator = get_nested(state, 'world_feeds', 'news_updates', default=[])
            news_snippet_for_narrator = " ".join([item.get('headline', '') for item in news_updates_for_narrator[:1]]) or "No significant news."

            logger.info(f"[NarrationTask] Generating narrative for {actor_name}'s action completion. Outcome: '{outcome_desc}'")

            # Convert trigger information to a dictionary
            trigger_data_narrator = {
                "actor_id": actor_id,
                "actor_name": actor_name,
                "original_intent": intent,
                "factual_outcome_description": outcome_desc,
                "state_changes_results_context": results,
                "discovered_objects_context": discovered_objects_for_narration_ctx,
                "discovered_npcs_context": discovered_npcs_for_narration_ctx,
                "discovered_connections_context": discovered_connections_for_narration_ctx,
                "recent_narrative_history_cleaned": history_str,
                "world_style_mood_context": world_mood_global,
                "world_time_context": time_for_narrator_prompt,
                "weather_context": weather_for_narrator_prompt,
                "news_context": news_snippet_for_narrator,
                "instruction": "Generate the narrative paragraph based on these details and your agent instructions."
            }
            trigger_text_narrator = json.dumps(trigger_data_narrator)

            trigger_content_narrator = genai_types.UserContent(parts=[genai_types.Part(text=trigger_text_narrator)])
            validated_narrator_output = await _call_adk_agent_and_parse(
                narration_runner, narration_agent_instance, narration_session_id, USER_ID,
                trigger_content_narrator, NarratorOutput, f"NarrationTask_{actor_name}", logger,
            )

            if validated_narrator_output:
                try:
                    actual_narrative_paragraph = getattr(validated_narrator_output, 'narrative', "Narration error or missing narrative field.")
                    logger.debug(f"[NarrationTask] Narrator output received: {actual_narrative_paragraph[:100]}")

                    cleaned_narrative_text = actual_narrative_paragraph
                    internal_agent_name_placeholder = f"[SimulacraLLM_{actor_id}]"
                    cleaned_narrative_text = cleaned_narrative_text.replace(internal_agent_name_placeholder, actor_name)

                    if cleaned_narrative_text and live_display_object:
                        live_display_object.console.print(Panel(cleaned_narrative_text, title=f"Narrator @ {completion_time:.1f}s", border_style="green", expand=False))
                    elif cleaned_narrative_text:
                        console.print(Panel(cleaned_narrative_text, title=f"Narrator @ {completion_time:.1f}s", border_style="green", expand=False))

                    final_narrative_entry = f"[T{completion_time:.1f}] {cleaned_narrative_text}"
                    state.setdefault("narrative_log", []).append(final_narrative_entry)
                    max_narrative_log = 50
                    if len(state["narrative_log"]) > max_narrative_log:
                        state["narrative_log"] = state["narrative_log"][-max_narrative_log:]

                    if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                        _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", cleaned_narrative_text, logger)
                    logger.info(f"[NarrationTask] Appended narrative for {actor_name}: {cleaned_narrative_text[:80]}...")

                    # NOTE: State is already updated by WorldEngine, this is just for narrative generation
                    logger.debug(f"[NarrationTask] Discovery details used for narrative context only (state already updated by WorldEngine)")

                    _log_event(
                        sim_time=completion_time,
                        agent_id="Narrator",
                        event_type="narration",
                        data={"narrative": actual_narrative_paragraph},
                        logger_instance=logger, event_logger_global=event_logger_global
                    )

                except ValidationError as e_val:
                    logger.error(f"[NarrationTask] Narrator output validation error: {e_val}")
                    cleaned_narrative_text = getattr(validated_narrator_output, 'narrative', "Error in narration processing.")
                    if cleaned_narrative_text and live_display_object:
                        live_display_object.console.print(Panel(cleaned_narrative_text, title=f"Narrator (Fallback) @ {completion_time:.1f}s", border_style="yellow", expand=False))
                    if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                        _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", cleaned_narrative_text, logger)

            narration_queue.task_done()

        except asyncio.CancelledError:
            logger.info("[NarrationTask] Task cancelled.")
            if action_event and narration_queue._unfinished_tasks > 0:
                try: 
                    narration_queue.task_done()
                except ValueError: 
                    pass
            break
        except Exception as e:
            logger.exception(f"[NarrationTask] Error processing event: {e}")
            if action_event and narration_queue._unfinished_tasks > 0:
                try: 
                    narration_queue.task_done()
                except ValueError: 
                    pass

async def world_engine_task_llm():
    """Listens for action requests, calls LLM to resolve, stores results, and triggers narration."""
    logger.info("[WorldEngineLLM] Task started.")
    if not world_engine_runner or not world_engine_agent or not world_engine_session_id:
        logger.error("[WorldEngineLLM] World Engine ADK components (runner, agent, session_id) not initialized. Task cannot proceed.")
        return

    while True:
        request_event = None
        actor_id = None 
        outcome_description = "Action failed due to internal error (pre-processing)."

        try:
            # Use safe queue access with timeout
            request_event = await safe_queue_get(event_bus, timeout=10.0, task_name="WorldEngineLLM")
            
            if request_event is None:
                # Timeout occurred, continue loop to check for cancellation
                continue
                
            if get_nested(request_event, "type") != "intent_declared":
                logger.debug(f"[WorldEngineLLM] Ignoring event type: {get_nested(request_event, 'type')}")
                if event_bus._unfinished_tasks > 0:
                    try:
                        event_bus.task_done()
                    except ValueError:
                        pass
                continue

            actor_id = get_nested(request_event, "actor_id")
            
            # Check if agent is in interaction mode
            if not actor_id or actor_id not in get_nested(state, SIMULACRA_KEY, default={}):
                logger.warning(f"[WorldEngineLLM] Received event with invalid or missing actor_id: {request_event}")
                event_bus.task_done()
                continue
            in_interaction_mode = get_nested(state, SIMULACRA_KEY, actor_id, "interaction_mode", default=False)
            if in_interaction_mode:
                logger.info(f"[WorldEngineLLM] Ignoring action request from {actor_id} as they are in interaction mode")
                event_bus.task_done()
                continue
                
            intent = get_nested(request_event, "intent")
            action_type = intent.get("action_type") if intent else None
            if not actor_id or not intent:
                logger.warning(f"[WorldEngineLLM] Received invalid action request event: {request_event}")
                event_bus.task_done()
                continue

            # --- BEGIN MOVED CLASSIFICATION LOGIC ---
            target_id_for_classification = intent.get("target_id")
            interaction_class = "environment" # Default

            if target_id_for_classification:
                # Check if target is another Simulacra
                if target_id_for_classification in get_nested(state, SIMULACRA_KEY, default={}):
                    interaction_class = "entity"
                else:
                    # Check if target is an interactive object
                    objects_list = get_nested(state, "objects", default=[])
                    for obj in objects_list:
                        if isinstance(obj, dict) and obj.get("id") == target_id_for_classification and obj.get("interactive", False):
                            interaction_class = "entity"
                            break
            # --- END MOVED CLASSIFICATION LOGIC ---

            logger.info(f"[WorldEngineLLM] Received '{interaction_class}' action request from {actor_id}: {intent}")
            
            actor_state_we = get_nested(state, SIMULACRA_KEY, actor_id, default={})
            actor_name = get_nested(actor_state_we, "persona_details", "Name", default=actor_id)
            current_sim_time = state.get("world_time", 0.0)
            actor_location_id = get_nested(actor_state_we, "location")
            location_state_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, default={})
            world_rules = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'rules', default={})
            
            # --- World Generation Check for MOVE action ---
            target_location_id_from_intent = intent.get("details") if action_type == "move" else None

            if action_type == "move" and target_location_id_from_intent:
                is_target_defined = target_location_id_from_intent in get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, default={})
                
                if not is_target_defined:
                    logger.info(f"[WorldEngineLLM] Target location '{target_location_id_from_intent}' for move by {actor_name} is undefined. Triggering WorldGenerator.")
                    if not world_generator_agent or not world_generator_runner or not world_generator_session_id:
                        logger.error("[WorldEngineLLM] WorldGenerator ADK components not initialized. Cannot generate new location.")
                        # Action will likely fail or be handled as invalid by standard WorldEngine logic
                    else:
                        # Get world context from state
                        world_template_details = state.get(WORLD_TEMPLATE_DETAILS_KEY, {})
                        current_world_type = world_template_details.get("world_type", "unknown")
                        current_sub_genre = world_template_details.get("sub_genre", "unknown")
                        trigger_data_wg = {
                            "location_id_to_define": target_location_id_from_intent,
                            "origin_location_id_context": actor_location_id,
                            "origin_location_description_context": get_nested(location_state_data, "description", "Unknown origin location."),
                            "world_details_context": {
                                "current_time": get_time_string_for_prompt(state, sim_elapsed_time_seconds=current_sim_time),
                                "weather": get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.'),
                                "mood": world_mood_global,
                                "world_type": current_world_type,
                                "sub_genre": current_sub_genre
                            },
                            "instruction": "Generate the location details based on your agent instructions and the provided JSON context. Infer the appropriate location type from the location ID, origin context, and world details."
                        }
                        wg_trigger_text = json.dumps(trigger_data_wg)
                        wg_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=wg_trigger_text)])
                        
                        # Original instruction for WorldGenerator is static for now, no dynamic replacements needed in this call
                        generated_world_data_ns = await _call_adk_agent_and_parse(
                            world_generator_runner, world_generator_agent, world_generator_session_id, USER_ID,
                            wg_trigger_content, WorldGeneratorOutput, f"WorldGenerator_{actor_name}", logger
                        )

                        if generated_world_data_ns:
                            generated_world_data = _to_dict(generated_world_data_ns) # Convert SimpleNamespace to dict
                            defined_loc = generated_world_data.get("defined_location")
                            additional_locs = generated_world_data.get("additional_related_locations", [])
                            origin_conn_update = generated_world_data.get("connection_update_for_origin")

                            if defined_loc and defined_loc.get("id") == target_location_id_from_intent:
                                # Ensure the new location connects back to the origin
                                origin_id_for_check = actor_location_id # The location the actor is coming FROM
                                connection_back_to_origin_exists = False
                                if origin_id_for_check and defined_loc.get("connected_locations"):
                                    for conn in defined_loc["connected_locations"]:
                                        if conn.get("to_location_id_hint") == origin_id_for_check:
                                            connection_back_to_origin_exists = True
                                            break
                                if origin_id_for_check and not connection_back_to_origin_exists:
                                    origin_loc_name_for_desc = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, origin_id_for_check, "name", default=origin_id_for_check)
                                    connection_to_origin = {
                                        "to_location_id_hint": origin_id_for_check,
                                        "description": f"The way back to {origin_loc_name_for_desc}."
                                    }
                                    defined_loc.setdefault("connected_locations", []).append(connection_to_origin)
                                    logger.info(f"[WorldEngineLLM] WorldGenerator: Added missing connection from new location '{defined_loc['id']}' back to origin '{origin_id_for_check}'.")

                                _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{defined_loc['id']}", defined_loc, logger)
                                logger.info(f"[WorldEngineLLM] WorldGenerator defined new location: {defined_loc['id']} ({defined_loc.get('name')})")
                                for add_loc in additional_locs:
                                    # Similar check for additional_locs if they should connect back to defined_loc could be added here if needed
                                    _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{add_loc['id']}", add_loc, logger)
                                    logger.info(f"[WorldEngineLLM] WorldGenerator added related location: {add_loc['id']} ({add_loc.get('name')})")
                                
                                if origin_conn_update and origin_conn_update.get("origin_id") == actor_location_id:
                                    conn_to_add = origin_conn_update.get("connection_to_add")
                                    if conn_to_add:
                                        current_origin_conns = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, "connected_locations", default=[])
                                        current_origin_conns.append(conn_to_add)
                                        _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_id}.connected_locations", current_origin_conns, logger)
                                        logger.info(f"[WorldEngineLLM] WorldGenerator updated connections for origin {actor_location_id} to include {conn_to_add.get('to_location_id_hint')}")
                                # The target location is now defined, WorldEngine can proceed with the move.
                                is_target_defined = True # Update flag
                            else:
                                logger.error(f"[WorldEngineLLM] WorldGenerator failed to define the target location '{target_location_id_from_intent}' correctly.")
            # --- End World Generation Check ---
            location_state_data["objects_present"] = location_state_data.get("ephemeral_objects", [])
            time_for_world_engine_prompt = get_time_string_for_prompt(state, sim_elapsed_time_seconds=current_sim_time) # type: ignore
            # Use the new helper function to get target_state_data
            target_id_we = get_nested(intent, "target_id")
            target_state_data = get_target_entity_state(state, target_id_we, actor_location_id) or {}

            # Convert trigger information to a dictionary
            trigger_data_we = {
                "actor_name_and_id": f"{actor_name} ({actor_id})",
                "current_location_id": actor_location_id,
                "intent": intent, # intent is already a dict
                "target_entity_state": target_state_data, # target_state_data is already a dict
                "target_entity_id_hint": target_id_we or 'N/A',
                "location_state": location_state_data, # location_state_data is already a dict
                "world_rules": world_rules, # world_rules is already a dict
                "world_time_context": time_for_world_engine_prompt,
                "weather_context": get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.'),
                "instruction": "Resolve this intent based on your agent instructions and the provided JSON context."
            }
            # Convert the dictionary to a compact JSON string
            trigger_text_we = json.dumps(trigger_data_we)

            logger.debug(f"[WorldEngineLLM] Sending prompt to LLM for {actor_id}'s intent ({action_type}).")

            trigger_content_we = genai_types.UserContent(parts=[genai_types.Part(text=trigger_text_we)])

            validated_data = await _call_adk_agent_and_parse(
                world_engine_runner, world_engine_agent, world_engine_session_id, USER_ID,
                trigger_content_we, WorldEngineResponse, f"WorldEngineLLM_{actor_id}", logger,
            )

            # validated_data can be Pydantic model, SimpleNamespace, or None
            parsed_resolution = _to_dict(validated_data) if validated_data else None
            if validated_data:
                outcome_description = getattr(validated_data, 'outcome_description', "Outcome not described.")
                if live_display_object and parsed_resolution:
                    live_display_object.console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                    try: live_display_object.console.print(json.dumps(parsed_resolution, indent=2))
                    except TypeError: live_display_object.console.print(str(parsed_resolution))
                _log_event(sim_time=current_sim_time, agent_id="WorldEngine", event_type="resolution", data=_to_dict(validated_data) or {}, logger_instance=logger, event_logger_global=event_logger_global)

            # Handle case where validated_data is still None after loop (e.g. LLM error or no response text)
            if 'validated_data' not in locals() or not validated_data:
                if action_type == "move" and target_location_id_from_intent and not is_target_defined:
                    outcome_description = f"{actor_name} attempted to move to '{target_location_id_from_intent}', but the way could not be materialized."
                else:
                    outcome_description = "Action failed: No response from World Engine LLM."
            elif getattr(validated_data, 'valid_action', False):
                completion_time = current_sim_time + getattr(validated_data, 'duration', 0.0)
                # --- BEGIN ADDITION: Handle scheduled_future_event ---
                sfe_dict = _to_dict(getattr(validated_data, 'scheduled_future_event', None)) # Convert to dict for consistent access
                if sfe_dict:
                    # The event should trigger relative to when the actor's action (speaking) completes.
                    event_trigger_time = completion_time + sfe_dict.get('estimated_delay_seconds', 0.0)

                    event_to_schedule = {
                        "event_type": sfe_dict.get('event_type', 'unknown_event'),
                        "target_agent_id": sfe_dict.get('target_agent_id'), 
                        "location_id": sfe_dict.get('location_id', 'unknown_location'),
                        "details": sfe_dict.get('details', {}), 
                        "trigger_sim_time": event_trigger_time, # Absolute simulation time for the event
                        "source_actor_id": actor_id # The one whose action generated this event
                    }
                    state.setdefault("pending_simulation_events", []).append(event_to_schedule)
                    state["pending_simulation_events"].sort(key=lambda x: x.get("trigger_sim_time", float('inf')))
                    logger.info(
                        f"[WorldEngineLLM] Scheduled future event '{event_to_schedule.get('event_type')}' "
                        f"for agent {event_to_schedule.get('target_agent_id')} at sim_time {event_trigger_time:.2f} "
                        f"triggered by {actor_id}."
                    )

                # --- CONSOLIDATE RESULTS FOR TIME MANAGER ---
                # Get base results from World Engine
                pending_results_dict = _to_dict(getattr(validated_data, 'results', {}))
                action_completion_results_dict = {}

                # Determine effective location for discoveries
                effective_location_id_for_discoveries = actor_location_id  # Default to current location

                if action_type == "move":
                    # For move actions, discoveries apply to the DESTINATION
                    destination_location_id = pending_results_dict.get(f"{SIMULACRA_KEY}.{actor_id}.location")
                    if destination_location_id:
                        effective_location_id_for_discoveries = destination_location_id
                        logger.info(f"[WorldEngineLLM] Move action: discoveries will apply to destination {destination_location_id}")
                        
                        # ENSURE BIDIRECTIONAL CONNECTION: Add connection back to origin location
                        discovered_connections_list = _list_to_dicts_if_needed(getattr(validated_data, 'discovered_connections', []))
                        
                        # Check if connection back to origin already exists
                        origin_connection_exists = any(
                            conn.get('to_location_id_hint') == actor_location_id 
                            for conn in discovered_connections_list
                        )
                        
                        if not origin_connection_exists and actor_location_id:
                            # Add connection back to where the actor came from
                            origin_location_name = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, "name", default=actor_location_id)
                            back_connection = {
                                "to_location_id_hint": actor_location_id,
                                "description": f"Back to {origin_location_name}."
                            }
                            discovered_connections_list.append(back_connection)
                            logger.info(f"[WorldEngineLLM] Added bidirectional connection from {destination_location_id} back to {actor_location_id}")
                        
                        # Update the validated_data with the enhanced connections
                        if hasattr(validated_data, 'discovered_connections'):
                            validated_data.discovered_connections = discovered_connections_list
                    
                    # Defer location change to completion if action has duration
                    if getattr(validated_data, 'duration', 0.0) > 0.1:
                        location_change_key = f"{SIMULACRA_KEY}.{actor_id}.location"
                        location_details_key = f"{SIMULACRA_KEY}.{actor_id}.location_details"
                        
                        if location_change_key in pending_results_dict:
                            action_completion_results_dict[location_change_key] = pending_results_dict.pop(location_change_key)
                            new_location_id = action_completion_results_dict[location_change_key]
                            new_location_name = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, new_location_id, "name", default=new_location_id)
                            action_completion_results_dict[location_details_key] = f"You are now in {new_location_name}."
                            logger.info(f"[WorldEngineLLM] Deferred location change for {actor_id} to action completion")

                # --- IMMEDIATE DISCOVERY UPDATES TO STATE ---
                results_dict = _to_dict(getattr(validated_data, 'results', {}))
                discovered_objects_list = _list_to_dicts_if_needed(results_dict.get('discovered_objects', []))
                # discovered_npcs_list = _list_to_dicts_if_needed(results_dict.get('discovered_npcs', []))
                discovered_connections_list = _list_to_dicts_if_needed(results_dict.get('discovered_connections', []))
                
                # IMMEDIATE APPLICATION FOR CURRENT LOCATION DISCOVERIES
                if action_type == "look_around" and effective_location_id_for_discoveries == actor_location_id:
                    # Apply discoveries immediately to state for look_around actions
                    if discovered_objects_list or True:  # Always update for look_around to potentially clear state
                        _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{effective_location_id_for_discoveries}.ephemeral_objects", discovered_objects_list, logger)
                        logger.info(f"[WorldEngineLLM] IMMEDIATELY applied ephemeral_objects at {effective_location_id_for_discoveries}: {len(discovered_objects_list)} items.")

                    # if discovered_npcs_list or True:  # Always update for look_around
                    #     _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{effective_location_id_for_discoveries}.ephemeral_npcs", discovered_npcs_list, logger)
                    #     logger.info(f"[WorldEngineLLM] IMMEDIATELY applied ephemeral_npcs at {effective_location_id_for_discoveries}: {len(discovered_npcs_list)} items.")

                    # **FIX: For look_around, MERGE discovered connections with existing ones**
                    if discovered_connections_list or True:  # Always update for look_around
                        # Get existing connections
                        existing_connections = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, effective_location_id_for_discoveries, "connected_locations", default=[])
                        
                        # Create a map of existing connections by target ID to avoid duplicates
                        existing_connection_targets = {
                            conn.get("to_location_id_hint"): conn 
                            for conn in existing_connections 
                            if isinstance(conn, dict) and conn.get("to_location_id_hint")
                        }
                        
                        # Merge discovered connections, keeping existing ones and adding new ones
                        merged_connections = list(existing_connections)
                        for new_conn in discovered_connections_list:
                            if isinstance(new_conn, dict) and new_conn.get("to_location_id_hint"):
                                target_id = new_conn["to_location_id_hint"]
                                if target_id not in existing_connection_targets:
                                    merged_connections.append(new_conn)
                                    logger.info(f"[WorldEngineLLM] Added new connection: {target_id} - {new_conn.get('description', 'No description')}")
                                else:
                                    # Optionally update description if new one is more detailed
                                    existing_desc = existing_connection_targets[target_id].get("description", "")
                                    new_desc = new_conn.get("description", "")
                                    if len(new_desc) > len(existing_desc):
                                        existing_connection_targets[target_id]["description"] = new_desc
                                        logger.info(f"[WorldEngineLLM] Updated connection description for {target_id}")
                        
                        _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{effective_location_id_for_discoveries}.connected_locations", merged_connections, logger)
                        logger.info(f"[WorldEngineLLM] IMMEDIATELY applied merged connected_locations at {effective_location_id_for_discoveries}: {len(merged_connections)} total connections.")
                else:
                    # For moves or other actions, defer to pending_results as before
                    if discovered_objects_list or intent.get("action_type") == "look_around":
                        obj_path = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{effective_location_id_for_discoveries}.ephemeral_objects"
                        pending_results_dict[obj_path] = discovered_objects_list
                        logger.info(f"[WorldEngineLLM] Queuing update for ephemeral_objects at {effective_location_id_for_discoveries}: {len(discovered_objects_list)} items.")

                    # if discovered_npcs_list or intent.get("action_type") == "look_around":
                    #     npc_path = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{effective_location_id_for_discoveries}.ephemeral_npcs"
                    #     pending_results_dict[npc_path] = discovered_npcs_list
                    #     logger.info(f"[WorldEngineLLM] Queuing update for ephemeral_npcs at {effective_location_id_for_discoveries}: {len(discovered_npcs_list)} items.")

                    if discovered_connections_list or intent.get("action_type") == "look_around":
                        conn_path = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{effective_location_id_for_discoveries}.connected_locations"
                        pending_results_dict[conn_path] = discovered_connections_list
                        logger.info(f"[WorldEngineLLM] Queuing update for connected_locations at {effective_location_id_for_discoveries}: {len(discovered_connections_list)} items.")
                # --- END IMMEDIATE DISCOVERY UPDATES ---

                narration_event = {
                    "type": "action_complete", "actor_id": actor_id, "action": intent,
                    "results_for_narration_context": pending_results_dict, # Pass the consolidated results
                    "outcome_description": getattr(validated_data, 'outcome_description', ""),
                    "completion_time": completion_time,
                    "current_action_description": f"Action: {intent.get('action_type', 'unknown')} - Details: {intent.get('details', 'N/A')[:100]}",
                    "actor_current_location_id": actor_location_id, 
                    "world_mood": world_mood_global, 
                    # Pass discovery lists explicitly from WorldEngine's validated_data for narration context
                    "discovered_objects_for_narration": _list_to_dicts_if_needed(getattr(validated_data, 'discovered_objects', [])),
                    "discovered_npcs_for_narration": _list_to_dicts_if_needed(getattr(validated_data, 'discovered_npcs', [])),
                    "discovered_connections_for_narration": _list_to_dicts_if_needed(getattr(validated_data, 'discovered_connections', [])),
                }
                # If the move was to a newly generated location, adjust outcome description slightly for narration
                if action_type == "move" and target_location_id_from_intent and \
                   (target_location_id_from_intent not in get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, default={}) # Should not happen if WG worked
                    or get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, target_location_id_from_intent, "name", default="").startswith("Newly Defined")): # Check if WG named it generically
                    loc_name_for_narration = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, target_location_id_from_intent, "name", default=target_location_id_from_intent)
                    narration_event["outcome_description"] = f"{actor_name} moved into the newly revealed area: {loc_name_for_narration} (ID: {target_location_id_from_intent})."

                # The logic for updating actor's location_details on move is now part of building pending_results_dict above.

                if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.status", "busy", logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.pending_results", pending_results_dict, logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.action_completion_results", action_completion_results_dict, logger) # Store completion results
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.current_action_end_time", completion_time, logger)
                    
                    # Refine action description for listening
                    action_desc_for_state = narration_event["current_action_description"]
                    if action_type == "wait" and "listened attentively to" in outcome_description:
                        action_desc_for_state = outcome_description # Use the more specific "listened attentively to..."
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.current_action_description", action_desc_for_state, logger)
                    
                    narration_success = await safe_queue_put(narration_queue, narration_event, timeout=5.0, task_name=f"WorldEngineLLM_{actor_id}")

                    if not narration_success:
                        logger.error(f"[WorldEngineLLM] Failed to queue narration for {actor_id}")
                        _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", outcome_description, logger)
                    else:
                        logger.info(f"[WorldEngineLLM] Action VALID for {actor_id}. Updated state immediately, set end time {completion_time:.1f}s")
                else:
                    logger.error(f"[WorldEngineLLM] Actor {actor_id} not found in state after valid action resolution.")
            else: 
                final_outcome_desc = getattr(validated_data, 'outcome_description', outcome_description) if validated_data else outcome_description
                logger.info(f"[WorldEngineLLM] Action INVALID for {actor_id}. Reason: {final_outcome_desc}")
                # --- Log World Engine Resolution Event (Failure) ---
                _log_event(
                    sim_time=current_sim_time,
                    agent_id="WorldEngine",
                    event_type="resolution",
                    data={"valid_action": False, "duration": 0.0, "results": {}, "outcome_description": final_outcome_desc},
                    logger_instance=logger, event_logger_global=event_logger_global
                )
                if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", final_outcome_desc, logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.status", "idle", logger)
                actor_name_for_log = get_nested(state, SIMULACRA_KEY, actor_id, "persona_details", "Name", default=actor_id)
                resolution_details = {"valid_action": False, "duration": 0.0, "results": {}, "outcome_description": final_outcome_desc}
                if live_display_object: 
                    live_display_object.console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                    try: live_display_object.console.print(json.dumps(resolution_details, indent=2))
                    except TypeError: live_display_object.console.print(str(resolution_details))
                state.setdefault("narrative_log", []).append(f"[T{current_sim_time:.1f}] {actor_name_for_log}'s action failed: {final_outcome_desc}")

        except asyncio.CancelledError:
            logger.info("[WorldEngineLLM] Task cancelled.")
            if request_event and event_bus._unfinished_tasks > 0:
                try: 
                    event_bus.task_done()
                except ValueError: 
                    pass
            break
        except Exception as e:
            logger.exception(f"[WorldEngineLLM] Error processing event for actor {actor_id}: {e}")
            if actor_id and actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                 _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.status", "idle", logger)
                 _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.pending_results", {}, logger)
                 _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", f"Action failed unexpectedly: {e}", logger)
            if request_event and event_bus._unfinished_tasks > 0: 
                try: 
                    event_bus.task_done()
                except ValueError: 
                    pass 
            await asyncio.sleep(1) 
        finally: 
            if request_event and event_bus._unfinished_tasks > 0:
                try: 
                    event_bus.task_done()
                except ValueError: 
                    logger.warning("[WorldEngineLLM] task_done() called too many times in finally.")
                except Exception as td_e: 
                    logger.error(f"[WorldEngineLLM] Error calling task_done() in finally: {td_e}")


def _build_simulacra_prompt(
    agent_id: str,
    agent_name: str,
    agent_state_data: Dict[str, Any], 
    current_sim_time: float,
    global_state_ref: Dict[str, Any], 
    perception_manager_instance: PerceptionManager, 
    status_message_for_prompt: str
) -> str:
    """Builds the detailed prompt for the Simulacra agent - reads from consolidated state."""
    
    # All agents now read from the same state source
    fresh_percepts = perception_manager_instance.get_percepts_for_simulacrum(agent_id)
    logger.debug(f"[{agent_name}] Built percepts from consolidated state: {json.dumps(fresh_percepts, sort_keys=True)[:250]}...")

    # Build perception summary
    perceptual_summary_for_prompt = "Perception system error or offline."
    audible_env_str = "  The environment is quiet."
    recently_departed_sim_str = "  No one recently departed from this location."
    
    if fresh_percepts and not fresh_percepts.get("error"):
        loc_desc_from_percepts = fresh_percepts.get("location_description", "An unknown place.")
        
        visible_sim_text_parts = [
            f"  - Simulacra: {s.get('name', s.get('id'))} (ID: {s.get('id')}, Status: {s.get('status', 'unknown')})"
            for s in fresh_percepts.get("visible_simulacra", [])
        ]
        visible_sim_str = "\n".join(visible_sim_text_parts) if visible_sim_text_parts else "  No other simulacra perceived."

        visible_static_obj_text_parts = [
            f"  - Static Object: {o.get('name', o.get('id'))} (ID: {o.get('id')}, Desc: {o.get('description', '')[:30]}...)"
            for o in fresh_percepts.get("visible_static_objects", [])
        ]
        visible_static_obj_str = "\n".join(visible_static_obj_text_parts) if visible_static_obj_text_parts else "  No static objects perceived."

        visible_eph_obj_text_parts = [
            f"  - Ephemeral Object: {o.get('name', o.get('id'))} (ID: {o.get('id')})"
            for o in fresh_percepts.get("visible_ephemeral_objects", [])
        ]
        visible_eph_obj_str = "\n".join(visible_eph_obj_text_parts) if visible_eph_obj_text_parts else "  No ephemeral objects perceived."
        
        visible_eph_npc_text_parts = [
            f"  - Ephemeral NPC: {n.get('name', n.get('id'))} (ID: {n.get('id')})"
            for n in fresh_percepts.get("visible_npcs", [])
        ]
        visible_eph_npc_str = "\n".join(visible_eph_npc_text_parts) if visible_eph_npc_text_parts else "  No ephemeral NPCs perceived."

        audible_events_text_parts = [
            f"  - Sound ({s.get('type', 'general')} from {s.get('source_id', 'unknown')}): {s.get('description', 'An indistinct sound.')}"
            for s in fresh_percepts.get("audible_events", [])
        ]
        audible_env_str = "\n".join(audible_events_text_parts) if audible_events_text_parts else "  The environment is quiet."

        recently_departed_sim_text_parts = [
            f"  - Recently Departed: {s.get('name', s.get('id'))} (ID: {s.get('id')}) went to location ID '{s.get('departed_to_location_id', 'Unknown')}'."
            for s in fresh_percepts.get("recently_departed_simulacra", [])
        ]
        recently_departed_sim_str = "\n".join(recently_departed_sim_text_parts) if recently_departed_sim_text_parts else "  No one recently departed from this location."

        perceptual_summary_for_prompt = (
            f"Official Location Description: \"{loc_desc_from_percepts}\"\n"
            f"Visible Entities:\n{visible_sim_str}\n{visible_static_obj_str}\n{visible_eph_obj_str}\n{visible_eph_npc_str}"
        )

    # Get location data from consolidated state
    agent_current_location_id = agent_state_data.get('location', DEFAULT_HOME_LOCATION_NAME)
    agent_personal_location_details = agent_state_data.get(LOCATION_DETAILS_KEY, "You are unsure of your exact surroundings.")
    current_location_name = get_nested(global_state_ref, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, agent_current_location_id, "name", default=agent_current_location_id)
    
    # Get connections from consolidated state (single source of truth)
    connected_locations = get_nested(global_state_ref, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, agent_current_location_id, "connected_locations", default=[])
    
    # De-duplicate connections
    unique_connections_map = {}
    if isinstance(connected_locations, list):
        for conn in connected_locations:
            if isinstance(conn, dict) and "to_location_id_hint" in conn and "description" in conn:
                hint = conn["to_location_id_hint"]
                if hint not in unique_connections_map:
                    unique_connections_map[hint] = conn
                else:
                    if len(conn["description"]) > len(unique_connections_map[hint]["description"]):
                        unique_connections_map[hint] = conn
    connected_locations = list(unique_connections_map.values())

    # Build rest of prompt from state
    raw_recent_narrative = global_state_ref.get("narrative_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
    cleaned_recent_narrative = [re.sub(r'^\[T\d+\.\d+\]\s*', '', entry).strip() for entry in raw_recent_narrative]
    history_str = "\n".join(cleaned_recent_narrative)
    weather_summary = get_nested(global_state_ref, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
    latest_news_headlines = [item.get('headline', '') for item in get_nested(global_state_ref, 'world_feeds', 'news_updates', default=[])[:2]]
    news_summary = " ".join(h for h in latest_news_headlines if h) or "No major news."

    prompt_text_parts = [
        f"**Current State Info for {agent_name} ({agent_id}):**",
        f"- Persona: {agent_state_data.get('persona_details', {})}",
        f"- You are currently at: {current_location_name} (ID: {agent_current_location_id or 'Unknown'}).",
        f"- Your Understanding of this Named Location: \"{agent_personal_location_details}\"",
        f"- Your Immediate Vicinity: Described by your 'Last Observation/Event' and 'Perceived Environment' below.",
        f"- Perceived Environment:\n{perceptual_summary_for_prompt}",
        f"- Recently Departed from this Location:\n{recently_departed_sim_str}",
        f"- Status: {agent_state_data.get('status', 'idle')} ({status_message_for_prompt})",
        f"- Current Weather: {weather_summary}",
        f"- Recent News Snippet: {news_summary}",
        f"- Current Goal: {agent_state_data.get('goal', 'Determine goal.')}",
        f"- Current Time: {get_time_string_for_prompt(global_state_ref, sim_elapsed_time_seconds=current_sim_time)}",
        f"- Last Observation/Event: {agent_state_data.get('last_observation', 'None.')}",
        f"- Audible Environment:\n{audible_env_str}",
        f"- Recent Narrative History:\n{history_str if history_str else 'None.'}",
        f"- Exits/Connections from this location: {json.dumps(connected_locations) if connected_locations else 'None observed.'}",
        "\n**General Instructions:**"
        "Follow your thinking process and provide your response ONLY in the specified JSON format."
    ]
    return "\n".join(prompt_text_parts)


async def simulacra_agent_task_llm(agent_id: str):
    """Asynchronous task for managing a single Simulacra LLM agent."""
    agent_name = get_nested(state, SIMULACRA_KEY, agent_id, "persona_details", "Name", default=agent_id)
    logger.info(f"[{agent_name}] LLM Agent task started.")

    # Get dedicated runner and session for this simulacrum
    sim_runner = simulacra_runners_map.get(agent_id)
    sim_session_id = simulacra_session_ids_map.get(agent_id)
    sim_agent = simulacra_agents_map.get(agent_id)

    if not sim_runner or not sim_session_id or not sim_agent:
        logger.error(f"[{agent_name}] ADK components (runner, session_id, or agent) not found for this simulacrum. Task cannot proceed.")
        return

    # Store the original instruction here, at the beginning of the function
    original_simulacra_agent_instruction = sim_agent.instruction
    
    try:
        sim_state_init = get_nested(state, SIMULACRA_KEY, agent_id, default={})
        if "last_interjection_sim_time" not in sim_state_init:
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.last_interjection_sim_time", 0.0, logger)
        if "current_interrupt_probability" not in sim_state_init: # Initialize if not present
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None, logger)
        # next_simple_timer_interjection_sim_time is no longer managed by this task
        
        if get_nested(state, SIMULACRA_KEY, agent_id, "status") == "idle":
            current_sim_state_init = get_nested(state, SIMULACRA_KEY, agent_id, default={})
            current_world_time_init = state.get("world_time", 0.0)

            initial_trigger_text = _build_simulacra_prompt(
                agent_id, agent_name, current_sim_state_init, current_world_time_init,
                state, perception_manager_global, "You wake up and are ready to act."
            )
            logger.debug(f"[{agent_name}] Sending initial context prompt as agent is idle.")
            

            initial_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=initial_trigger_text)])
            
            validated_intent = await _call_adk_agent_and_parse(
                sim_runner, sim_agent, sim_session_id, USER_ID,
                initial_trigger_content, SimulacraIntentResponse, f"Simulacra_{agent_name}_Initial", logger,
            )
            
            if validated_intent:
                if live_display_object:
                    # validated_intent can be Pydantic or SimpleNamespace
                    monologue_text = getattr(validated_intent, 'internal_monologue', "No monologue.")
                    live_display_object.console.print(Panel(monologue_text, title=f"{agent_name} Monologue @ {current_world_time_init:.1f}s", border_style="yellow", expand=False))
                    live_display_object.console.print(f"\n[{agent_name} Intent @ {current_world_time_init:.1f}s]")
                    live_display_object.console.print(json.dumps(_to_dict(validated_intent, exclude={'internal_monologue'}), indent=2))
                success = await safe_queue_put(
                    event_bus, 
                    {"type": "intent_declared", "actor_id": agent_id, "intent": _to_dict(validated_intent, exclude={'internal_monologue'})},
                    timeout=5.0,
                    task_name=f"Simulacra_{agent_name}_Subsequent"
                )
                if not success:
                    logger.error(f"[{agent_name}] Failed to queue subsequent intent")

                # Only update status if queue operation succeeded
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "thinking", logger)

        next_interjection_check_sim_time = state.get("world_time", 0.0) + AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS
        while True:
            await asyncio.sleep(AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS)
            current_sim_time_busy_loop = state.get("world_time", 0.0)
            agent_state_busy_loop = get_nested(state, SIMULACRA_KEY, agent_id, default={})
            current_status_busy_loop = agent_state_busy_loop.get("status")

            if current_status_busy_loop == "idle":
                logger.debug(f"[{agent_name}] Status is idle. Proceeding to plan next action.")
                
                prompt_text = _build_simulacra_prompt(
                    agent_id, agent_name, agent_state_busy_loop, current_sim_time_busy_loop,
                    state, perception_manager_global, "You should act now."
                )
                logger.debug(f"[{agent_name}] Sending subsequent prompt.")
                

                trigger_content_loop = genai_types.UserContent(parts=[genai_types.Part(text=prompt_text)])

                validated_intent = await _call_adk_agent_and_parse(
                    sim_runner, sim_agent, sim_session_id, USER_ID,
                    trigger_content_loop, SimulacraIntentResponse, f"Simulacra_{agent_name}_Subsequent", logger,
                )

                if validated_intent:
                    if live_display_object:
                        monologue_text_loop = getattr(validated_intent, 'internal_monologue', "No monologue.")
                        live_display_object.console.print(Panel(monologue_text_loop, title=f"{agent_name} Monologue @ {current_sim_time_busy_loop:.1f}s", border_style="yellow", expand=False))
                        live_display_object.console.print(f"\n[{agent_name} Intent @ {current_sim_time_busy_loop:.1f}s]")
                        live_display_object.console.print(json.dumps(_to_dict(validated_intent, exclude={'internal_monologue'}), indent=2))
                    
                    success = await safe_queue_put(
                        event_bus, 
                        {"type": "intent_declared", "actor_id": agent_id, "intent": _to_dict(validated_intent, exclude={'internal_monologue'})},
                        timeout=5.0,
                        task_name=f"Simulacra_{agent_name}_Subsequent"
                    )
                    if not success:
                        logger.error(f"[{agent_name}] Failed to queue subsequent intent")
                    
                    _log_event(
                        sim_time=current_sim_time_busy_loop, agent_id=agent_id, event_type="intent",
                        data=_to_dict(validated_intent, exclude={'internal_monologue'}),
                        logger_instance=logger, event_logger_global=event_logger_global
                    )
                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "thinking", logger)
                continue 

            # Busy Action Interjection Logic - RETAINED FOR SELF-REFLECTION ONLY
            if current_status_busy_loop == "busy" and current_sim_time_busy_loop >= next_interjection_check_sim_time:
                next_interjection_check_sim_time = current_sim_time_busy_loop + AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS
                remaining_duration = agent_state_busy_loop.get("current_action_end_time", 0.0) - current_sim_time_busy_loop
                last_interjection_time_busy = agent_state_busy_loop.get("last_interjection_sim_time", 0.0) 
                cooldown_passed_busy = (current_sim_time_busy_loop - last_interjection_time_busy) >= INTERJECTION_COOLDOWN_SIM_SECONDS

                if remaining_duration > LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS and cooldown_passed_busy:
                    logger.info(f"[{agent_name}] Busy with long task (rem: {remaining_duration:.1f}s). Checking for self-reflection.")
                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.last_interjection_sim_time", current_sim_time_busy_loop, logger)

                    if random.random() < PROB_INTERJECT_AS_SELF_REFLECTION: 
                        original_status_before_reflection = agent_state_busy_loop.get("status")
                        _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "reflecting", logger)
                        current_action_desc_for_prompt = agent_state_busy_loop.get("current_action_description", "your current task")
                        reflection_prompt_text = f"""You are {agent_name}. You are currently busy with: "{current_action_desc_for_prompt}".
This task is scheduled to continue for a while (until simulation time {agent_state_busy_loop.get("current_action_end_time", 0.0):.1f}).
It's time for a brief self-reflection. Your status is 'reflecting'.
Do you:
1. Wish to `continue_current_task` without interruption?
2. Feel the need to `initiate_change`? (e.g., take a break, address hunger, switch focus, react to boredom/monotony).
Your `internal_monologue` should explain your reasoning.
If you choose to continue, your `action_type` must be `continue_current_task`.
If you choose to `initiate_change`, provide the `action_type` and `details` for that change as usual.
Output ONLY the JSON: `{{"internal_monologue": "...", "action_type": "...", "target_id": "...", "details": "..."}}`"""
                        reflection_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=reflection_prompt_text)])
                        # For reflection, instruction replacements are not typically needed as the prompt is self-contained.
                        validated_reflection_intent = await _call_adk_agent_and_parse(
                            sim_runner, sim_agent, sim_session_id, USER_ID,
                            reflection_trigger_content, SimulacraIntentResponse, f"Simulacra_{agent_name}_Reflection", logger
                            # No original_instruction/replacements needed here as reflection prompt is specific
                        )

                        if validated_reflection_intent:
                            action_type_reflect = getattr(validated_reflection_intent, 'action_type', 'unknown')
                            monologue_reflect = getattr(validated_reflection_intent, 'internal_monologue', '')
                            if action_type_reflect == "continue_current_task":
                                logger.info(f"[{agent_name}] Reflection: Chose to continue. Monologue: {monologue_reflect[:50]}...")
                                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", original_status_before_reflection, logger)
                            else:
                                logger.info(f"[{agent_name}] Reflection: Chose to '{action_type_reflect}'. Monologue: {monologue_reflect[:50]}...")
                                success = await safe_queue_put(
                                    event_bus, 
                                    {"type": "intent_declared", "actor_id": agent_id, "intent": _to_dict(validated_reflection_intent, exclude={'internal_monologue'})},
                                    timeout=5.0,
                                    task_name=f"Simulacra_{agent_name}_Reflection"
                                )
                                if not success:
                                    logger.error(f"[{agent_name}] Failed to queue reflection intent")
                                
                                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None, logger)
                                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "thinking", logger)
                        else: # Parsing failed or LLM error
                            logger.error(f"[{agent_name}] Error processing reflection response. Staying busy.")
                            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", original_status_before_reflection, logger)

    except asyncio.CancelledError:
        logger.info(f"[{agent_name}] Task cancelled.")
    except Exception as e:
        logger.error(f"[{agent_name}] Error in agent task: {e}", exc_info=True)
        if agent_id in get_nested(state, SIMULACRA_KEY, default={}):
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
    finally:
        logger.info(f"[{agent_name}] Task finished.")

async def run_simulation(
    instance_uuid_arg: Optional[str] = None,
    location_override_arg: Optional[str] = None,
    mood_override_arg: Optional[str] = None,
    event_logger_instance: Optional[logging.Logger] = None # Added event_logger_instance parameter
    ):
    global adk_session_service, adk_memory_service
    global world_engine_agent, world_engine_runner, world_engine_session_id
    global narration_agent_instance, narration_runner, narration_session_id
    global world_generator_agent, world_generator_runner, world_generator_session_id # Added WorldGenerator globals
    global simulacra_agents_map, simulacra_runners_map, simulacra_session_ids_map # Keep this line
    global state, live_display_object
    global world_mood_global, search_llm_agent_instance, search_agent_runner_instance, search_agent_session_id_val, perception_manager_global
    global event_logger_global # Ensure we can assign to the global

    console.rule("[bold green]Starting Async Simulation[/]")

    event_logger_global = event_logger_instance # Assign the passed logger to our global variable

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        logger.info(f"Global random seed initialized to: {RANDOM_SEED}")

    adk_memory_service = InMemoryMemoryService()
    adk_session_service = InMemorySessionService()
    logger.info("ADK InMemoryMemoryService initialized.")

    if not API_KEY:
        console.print("[bold red]ERROR: GOOGLE_API_KEY environment variable not set.[/bold red]")
        sys.exit(1)
    try:
        genai.configure(api_key=API_KEY)
        logger.info("Google Generative AI configured.")
    except Exception as e:
        logger.critical(f"Failed to configure Google API: {e}", exc_info=True)
        console.print(f"[bold red]ERROR: Failed to configure Google API: {e}[/bold red]")
        sys.exit(1)

    console.print(Panel(f"[[bold yellow]{APP_NAME}[/]] - Initializing Simulation State...", title="Startup", border_style="blue"))
    logger.info("Starting simulation initialization.")
    
    loaded_state_data, state_file_path = load_or_initialize_simulation(instance_uuid_arg)
    if loaded_state_data is None:
        logger.critical("Failed to load or create simulation state. Cannot proceed.")
        console.print("[bold red]Error:[/bold red] Could not obtain simulation state.")
        sys.exit(1)
    state = loaded_state_data 
    world_instance_uuid = state.get("world_instance_uuid")

    if location_override_arg:
        try:
            logger.info(f"Applying location override: '{location_override_arg}'")
            parsed_override_loc = parse_location_string(location_override_arg)
            state.setdefault(WORLD_TEMPLATE_DETAILS_KEY, {}).setdefault(LOCATION_KEY, {})
            state[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]['city'] = parsed_override_loc.get('city')
            state[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]['state'] = parsed_override_loc.get('state')
            state[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]['country'] = parsed_override_loc.get('country')
            logger.info(f"World location overridden: {state[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]}")
            console.print(f"Location overridden to: [yellow]{location_override_arg}[/yellow]")
        except Exception as e:
            logger.error(f"Failed to apply location override: {e}", exc_info=True)

    if mood_override_arg:
        world_mood_global = mood_override_arg.strip()
        logger.info(f"Global world mood overridden to '{world_mood_global}'.")
        console.print(f"Global world mood set to: [yellow]{world_mood_global}[/yellow]")
        state.setdefault(WORLD_TEMPLATE_DETAILS_KEY, {})['mood'] = world_mood_global
    else: 
        world_mood_global = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'mood', default="The familiar, everyday real world; starting the morning routine at home.")

    # Initialize Perception Manager with the loaded state
    perception_manager_global = PerceptionManager(state)
    logger.info("PerceptionManager initialized.")

    # The block that previously populated state["simulacra"] from state["simulacra_profiles"]
    # has been removed. `load_or_initialize_simulation` now ensures `state[SIMULACRA_PROFILES_KEY]`
    # (accessed via SIMULACRA_KEY from config) is correctly populated with all runtime fields.
    # We will use SIMULACRA_KEY directly.

    final_active_sim_ids = state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
    if not final_active_sim_ids:
         logger.critical("No active simulacra available after state load. Cannot proceed.")
         console.print("[bold red]Error:[/bold red] No verified Simulacra available.")
         sys.exit(1)
    logger.info(f"Initialization complete. Instance {world_instance_uuid} ready with {len(final_active_sim_ids)} simulacra.")
    console.print(f"Running simulation with: {', '.join(final_active_sim_ids)}")


    # Create shared agents once
    # Assuming the first simulacra's details can be used for generic agent naming/mood context if needed,
    # or prompts are fully generic. The current prompts take actor details in the trigger.
    first_sim_id = final_active_sim_ids[0] if final_active_sim_ids else "default_sim"
    first_sim_profile = get_nested(state, SIMULACRA_KEY, first_sim_id, default={})
    first_persona_name = get_nested(first_sim_profile, "persona_details", "Name", default=first_sim_id)
    
    # Get world_type and sub_genre for agent creation
    world_template_details_for_agents = state.get(WORLD_TEMPLATE_DETAILS_KEY, {})
    current_world_type = world_template_details_for_agents.get("world_type", "unknown")
    current_sub_genre = world_template_details_for_agents.get("sub_genre", "unknown")

    # --- Initialize World Engine Agent, Runner, and Session ---
    world_engine_agent = create_world_engine_llm_agent(
        sim_id=first_sim_id, persona_name=first_persona_name,
        world_type=current_world_type, sub_genre=current_sub_genre
    )
    world_engine_runner = Runner(agent=world_engine_agent, app_name=APP_NAME + "_WorldEngine", session_service=adk_session_service)
    world_engine_session_id = f"world_engine_session_{world_instance_uuid}"
    await adk_session_service.create_session(app_name=world_engine_runner.app_name, user_id=USER_ID, session_id=world_engine_session_id, state={})
    logger.info(f"World Engine Runner and Session ({world_engine_session_id}) initialized.")

    # --- Initialize Narration Agent, Runner, and Session ---
    narration_agent_instance = create_narration_llm_agent(
        sim_id=first_sim_id, persona_name=first_persona_name,
        world_mood=world_mood_global, world_type=current_world_type, sub_genre=current_sub_genre
    )
    narration_runner = Runner(agent=narration_agent_instance, app_name=APP_NAME + "_Narrator", session_service=adk_session_service)
    narration_session_id = f"narration_session_{world_instance_uuid}"
   
    await adk_session_service.create_session(app_name=narration_runner.app_name, user_id=USER_ID, session_id=narration_session_id, state={})
    logger.info(f"Narration Runner and Session ({narration_session_id}) initialized.")

    # --- Initialize WorldGenerator Agent, Runner, and Session ---
    world_generator_agent = create_world_generator_llm_agent(
       
        world_mood=world_mood_global, world_type=current_world_type, sub_genre=current_sub_genre
    )
    world_generator_runner = Runner(agent=world_generator_agent, app_name=APP_NAME + "_WorldGenerator", session_service=adk_session_service)
    world_generator_session_id = f"world_generator_session_{world_instance_uuid}"
    await adk_session_service.create_session(app_name=world_generator_runner.app_name, user_id=USER_ID, session_id=world_generator_session_id, state={})
    logger.info(f"WorldGenerator Runner and Session ({world_generator_session_id}) initialized.")


    # --- Initialize Search Agent, Runner, and Session (already mostly dedicated) ---
    search_llm_agent_instance = create_search_llm_agent()
    search_agent_runner_instance = Runner(
        agent=search_llm_agent_instance, app_name=APP_NAME + "_Search",
        session_service=adk_session_service
    )
    search_agent_session_id_val = f"world_feed_search_session_{world_instance_uuid}"
    search_adk_session = await adk_session_service.create_session(
        app_name=search_agent_runner_instance.app_name, user_id=USER_ID,
        session_id=search_agent_session_id_val, state={}
    )
    if search_adk_session:
        logger.info(f"ADK Search Agent Session created: {search_adk_session.id}")
    else:
        logger.error(f"Failed to create ADK Search Agent Session: {search_agent_session_id_val}. World feeds may not function correctly.")
    logger.info(f"Dedicated Search Agent Runner initialized.")

   
    simulacra_agents_map.clear()
    simulacra_runners_map.clear()
    simulacra_session_ids_map.clear()
    for sim_id_val in final_active_sim_ids:
        sim_profile_data = get_nested(state, SIMULACRA_KEY, sim_id_val, default={})
        persona_name = get_nested(sim_profile_data, "persona_details", "Name", default=sim_id_val)
        sim_agent_instance = create_simulacra_llm_agent(sim_id_val, persona_name, world_mood=world_mood_global)
        simulacra_agents_map[sim_id_val] = sim_agent_instance
        sim_runner_instance = Runner(agent=sim_agent_instance, app_name=f"{APP_NAME}_Sim_{sim_id_val}", session_service=adk_session_service)
        simulacra_runners_map[sim_id_val] = sim_runner_instance
        sim_session_id = f"sim_agent_session_{sim_id_val}_{world_instance_uuid}"
        await adk_session_service.create_session(app_name=sim_runner_instance.app_name, user_id=USER_ID, session_id=sim_session_id, state={})
        simulacra_session_ids_map[sim_id_val] = sim_session_id
    logger.info(f"Created {len(simulacra_agents_map)} simulacra agents.")

    # Register system agents to prevent "unknown agent" errors
    simulacra_agents_map["WorldEngineLLMAgent"] = world_engine_agent
    simulacra_agents_map["NarrationLLMAgent"] = narration_agent_instance  # Note: Changed from NarrationAgent to NarrationLLMAgent
    simulacra_agents_map["WorldGeneratorLLMAgent"] = world_generator_agent # Register WorldGenerator
    if search_llm_agent_instance:
        simulacra_agents_map["SearchAgent"] = search_llm_agent_instance
        simulacra_agents_map["SearchLLMAgent"] = search_llm_agent_instance  # Add alternative name

    # Also register simulacra with their LLM-prefixed names
    for sim_id in final_active_sim_ids:
        prefixed_sim_id = f"SimulacraLLM_{sim_id}"
        # Register with prefixed ID if not already present
        if prefixed_sim_id not in simulacra_agents_map and sim_id in simulacra_agents_map:
            simulacra_agents_map[prefixed_sim_id] = simulacra_agents_map[sim_id]

    logger.info(f"Added system agents to agent map: WorldEngineLLMAgent, NarrationLLMAgent, SearchAgent")
    logger.info(f"Registered {len(simulacra_agents_map)} total agents (including prefixed variants)")

    tasks = []
    final_state_path = os.path.join(STATE_DIR, f"simulation_state_{world_instance_uuid}.json")

    try:
        def get_current_table_for_live():
            eb_qsize = event_bus.qsize() if event_bus else 0
            nq_qsize = narration_queue.qsize() if narration_queue else 0
            return generate_table(state, eb_qsize, nq_qsize)

        with Live(get_current_table_for_live(), console=console, refresh_per_second=1.0/UPDATE_INTERVAL, vertical_overflow="visible") as live:
            live_display_object = live
            tasks.append(asyncio.create_task(
                socket_server_task(
                    state=state,
                    narration_queue=narration_queue,
                    world_mood=world_mood_global,
                    simulation_time_getter=get_current_sim_time,
                    live_display_object_ref=live # Pass the live object
                ), 
                name="SocketServer"
            ))
            tasks.append(asyncio.create_task(time_manager_task(
                current_state=state, 
                event_bus_qsize_func=lambda: event_bus.qsize(), 
                narration_qsize_func=lambda: narration_queue.qsize(), 
                live_display=live, 
                logger_instance=logger
            ), name="TimeManager"))
            # tasks.append(asyncio.create_task(interaction_dispatcher_task(state, event_bus, logger), name="InteractionDispatcher")) # REMOVED
            tasks.append(asyncio.create_task(world_info_gatherer_task(state, world_mood_global, search_agent_runner_instance, search_agent_session_id_val, logger), name="WorldInfoGatherer"))
            tasks.append(asyncio.create_task(dynamic_interruption_task(
                current_state=state,
                world_mood=world_mood_global,
                logger_instance=logger,
                event_logger_instance=event_logger_global
            ), name="DynamicInterruptionTask"))
            
            tasks.append(asyncio.create_task(narration_task(), name="NarrationTask"))
            tasks.append(asyncio.create_task(world_engine_task_llm(), name="WorldEngine"))
            for sim_id_val_task in final_active_sim_ids: 
                tasks.append(asyncio.create_task(simulacra_agent_task_llm(agent_id=sim_id_val_task), name=f"Simulacra_{sim_id_val_task}"))

            tasks.append(asyncio.create_task(queue_health_monitor(), name="QueueHealthMonitor"))

            if not tasks:
                 logger.error("No tasks were created. Simulation cannot run.")
                 console.print("[bold red]Error: No simulation tasks started.[/bold red]")
            else:
                logger.info(f"Started {len(tasks)} tasks.")
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try: 
                        task.result()
                    except asyncio.CancelledError: 
                        logger.info(f"Task {task.get_name()} was cancelled.")
                    except Exception as task_exc: 
                        logger.error(f"Task {task.get_name()} raised: {task_exc}", exc_info=task_exc)
                logger.info("One main task completed/failed. Initiating shutdown.")

    except Exception as e:
        logger.exception(f"Error during simulation setup or execution: {e}")
        console.print(f"[bold red]Unexpected error during simulation run: {e}[/bold red]")
    finally:
        logger.info("Cancelling remaining tasks...")
        if 'tasks' in locals() and tasks: 
            for task in tasks:
                if not task.done(): 
                    task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                 if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                     task_name = tasks[i].get_name() if i < len(tasks) else f"Task_{i}" 
                     logger.error(f"Error during task cleanup for {task_name}: {result}", exc_info=result)
        else:
            logger.warning("No tasks list found or empty during cleanup.")
        logger.info("All tasks cancelled or finished.")

        final_uuid_to_save = state.get("world_instance_uuid") 
        if final_uuid_to_save and final_state_path: # final_state_path should be defined if final_uuid_to_save is valid
            logger.info("Saving final simulation state.")
            try:
                if not isinstance(state.get("world_time"), (int, float)):
                     logger.warning(f"Final world_time is not a number ({type(state.get('world_time'))}). Saving as 0.0.")
                     state["world_time"] = 0.0
                save_json_file(final_state_path, state) # Use final_state_path
                logger.info(f"Final simulation state saved to {final_state_path}")
                console.print(f"Final state saved to {final_state_path}")
            except Exception as save_e:
                 logger.error(f"Failed to save final state to {final_state_path}: {save_e}", exc_info=True)
                 console.print(f"[red]Error saving final state: {save_e}[/red]")
        elif not final_uuid_to_save: # Only log error if UUID was the issue
             logger.error("Cannot save final state: world_instance_uuid is not defined in module state.")
             console.print("[bold red]Error: Cannot save final state (UUID unknown).[/bold red]")

        console.print("\nFinal State Table:")
        if state:
            eb_qsize_final = event_bus.qsize() if event_bus else 0
            nq_qsize_final = narration_queue.qsize() if narration_queue else 0
            console.print(generate_table(state, eb_qsize_final, nq_qsize_final))
        else:
            console.print("[yellow]State dictionary is empty.[/yellow]")
        console.rule("[bold green]Simulation Shutdown Complete[/]")
