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
                     create_world_engine_llm_agent) # Removed WorldGenerator
# Import from our new/refactored modules
from .config import (  # For run_simulation; For self-reflection; New constants for dynamic_interruption_task; PROB_INTERJECT_AS_NARRATIVE removed; from this import list; Import Bluesky and social post config; Import SIMULACRA_KEY
    ACTIVE_SIMULACRA_IDS_KEY, AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS,
    AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS, API_KEY, APP_NAME,
    DEFAULT_HOME_LOCATION_NAME, CURRENT_LOCATION_KEY, # DYNAMIC_INTERRUPTION constants moved to core_tasks import; Added CURRENT_LOCATION_KEY
    INTERJECTION_COOLDOWN_SIM_SECONDS,
    LOCATION_DETAILS_KEY, LOCATION_KEY,
    LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS, 
    MEMORY_LOG_CONTEXT_LENGTH,
    PROB_INTERJECT_AS_SELF_REFLECTION, RANDOM_SEED, 
    SIMULACRA_KEY, STATE_DIR, UPDATE_INTERVAL, # CONFIG_MODEL_NAME removed
    USER_ID, WORLD_STATE_KEY, WORLD_TEMPLATE_DETAILS_KEY)
from .core_tasks import time_manager_task, world_info_gatherer_task
from .loop_utils import (get_nested, load_json_file,
                         load_or_initialize_simulation, parse_json_output_last,
                         save_json_file)
from .perception_manager import PerceptionManager # Import the moved PerceptionManager
from .models import NarratorOutput, GeneratedLocationDetail  # Removed WorldGeneratorOutput
from .models import SimulacraIntentResponse, WorldEngineResponse
from .simulation_utils import (  # Utility functions; generate_llm_interjection_detail REMOVED
    _update_state_value, generate_table, get_time_string_for_prompt, get_target_entity_state, # Re-added get_time_string_for_prompt, added get_target_entity_state
    _log_event, handle_action_interruption) # get_random_style_combination is used by core_tasks
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

# Remove world_generator globals
# world_generator_agent: Optional[LlmAgent] = None
# world_generator_runner: Optional[Runner] = None
# world_generator_session_id: Optional[str] = None

world_mood_global: str = "The familiar, everyday real world; starting the morning routine at home."
live_display_object: Optional[Live] = None

# Add these helper functions after the imports
async def safe_queue_get(queue, timeout=5.0, task_name="Unknown"):
    """Safely get from queue with timeout and retry logic."""
    try:
        queue_name = "EVENT_BUS" if queue == event_bus else "NARRATION_QUEUE" if queue == narration_queue else "UNKNOWN_QUEUE"
        
        item = await asyncio.wait_for(queue.get(), timeout=timeout)
        
        # QUEUE LOGGING - Log what's coming out of the queue
        logger.info(f"[QUEUE] {queue_name} -> {task_name}: {json.dumps(item, indent=2)}")
        
        return item
    except asyncio.TimeoutError:
        logger.warning(f"[QUEUE] {queue_name} GET TIMEOUT after {timeout}s for {task_name}")
        return None
    except Exception as e:
        logger.error(f"[QUEUE] {queue_name} GET ERROR for {task_name}: {e}")
        return None

async def safe_queue_put(queue, item, timeout=5.0, task_name="Unknown"):
    """Safely put to queue with timeout and retry logic."""
    try:
        # QUEUE LOGGING - Log what's going into the queue
        queue_name = "EVENT_BUS" if queue == event_bus else "NARRATION_QUEUE" if queue == narration_queue else "UNKNOWN_QUEUE"
        logger.info(f"[QUEUE] {queue_name} <- {task_name}: {json.dumps(item, indent=2)}")
        
        await asyncio.wait_for(queue.put(item), timeout=timeout)
        logger.info(f"[QUEUE] {queue_name} PUT SUCCESS from {task_name}")
        return True
    except asyncio.TimeoutError:
        logger.warning(f"[QUEUE] {queue_name} PUT TIMEOUT after {timeout}s from {task_name}")
        return False
    except Exception as e:
        logger.error(f"[QUEUE] {queue_name} PUT ERROR from {task_name}: {e}")
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

# ENHANCED CONNECTION MERGING FOR ALL ACTIONS
def merge_connections_safely(existing_connections, new_connections, location_id, logger):
    """Safely merge connections, preserving existing ones and adding new unique ones."""
    if not isinstance(existing_connections, list):
        existing_connections = []
    if not isinstance(new_connections, list):
        new_connections = []
    
    # Create a map of existing connections by target ID
    existing_map = {}
    for conn in existing_connections:
        if isinstance(conn, dict) and conn.get("to_location_id_hint"):
            target_id = conn["to_location_id_hint"]
            existing_map[target_id] = conn
    
    # Merge connections
    merged = list(existing_connections)
    added_count = 0
    
    for new_conn in new_connections:
        if isinstance(new_conn, dict) and new_conn.get("to_location_id_hint"):
            target_id = new_conn["to_location_id_hint"]
            if target_id not in existing_map:
                merged.append(new_conn)
                added_count += 1
                logger.debug(f"Added new connection from {location_id}: {target_id}")
            else:
                # Update description if new one is more detailed
                existing_desc = existing_map[target_id].get("description", "")
                new_desc = new_conn.get("description", "")
                if len(new_desc) > len(existing_desc):
                    existing_map[target_id]["description"] = new_desc
                    logger.debug(f"Updated connection description for {location_id} -> {target_id}")
    
    logger.info(f"Connection merge for {location_id}: {len(existing_connections)} existing + {added_count} new = {len(merged)} total")
    return merged

# Add after the existing merge_connections_safely function
def ensure_bidirectional_connections(state, location_id, logger):
    """Ensure all connections from a location have corresponding back-connections."""
    location_connections = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, location_id, "connected_locations", default=[])
    
    for conn in location_connections:
        target_id = conn.get("to_location_id_hint")
        if target_id and target_id in get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, default={}):
            # Check if target has connection back to this location
            target_connections = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, target_id, "connected_locations", default=[])
            
            has_back_connection = any(
                back_conn.get("to_location_id_hint") == location_id 
                for back_conn in target_connections
            )
            
            if not has_back_connection:
                location_name = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, location_id, "name", default=location_id)
                back_connection = {
                    "to_location_id_hint": location_id,
                    "description": f"Back to {location_name}."
                }
                target_connections.append(back_connection)
                _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{target_id}.connected_locations", target_connections, logger)
                logger.info(f"Added missing bidirectional connection: {target_id} -> {location_id}")

def update_location_connections(state, location_id, new_connections, logger):
    """Centralized function to update location connections with deduplication and bidirectional linking."""
    existing_connections = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, location_id, "connected_locations", default=[])
    
    # Merge connections safely
    merged_connections = merge_connections_safely(existing_connections, new_connections, location_id, logger)
    
    # Update state
    _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{location_id}.connected_locations", merged_connections, logger)
    
    # Ensure bidirectional connections
    ensure_bidirectional_connections(state, location_id, logger)
    
    return merged_connections

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
            results = get_nested(action_event, "results_for_narration_context", default={})
            outcome_desc = get_nested(action_event, "outcome_description", default="Something happened.")
            
            # Validate required fields
            if not actor_id:
                logger.warning(f"[NarrationTask] Received narration event without actor_id: {action_event}")
                narration_queue.task_done()
                continue

            if not intent:
                logger.warning(f"[NarrationTask] Received narration event without action for {actor_id}: {action_event}")
                narration_queue.task_done()
                continue
            
            # Discovery details are passed for narrative context only - state is already updated
            discovered_objects_for_narration_ctx = action_event.get("discovered_objects_for_narration", [])
            discovered_npcs_for_narration_ctx = action_event.get("discovered_npcs_for_narration", [])
            discovered_connections_for_narration_ctx = action_event.get("discovered_connections_for_narration", [])

            completion_time = get_nested(action_event, "completion_time", default=state.get("world_time", 0.0))
            actor_location_at_action_time = get_nested(action_event, "actor_current_location_id")

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

                    # --- NEW: Process discovered_npcs from Narrator ---
                    narrator_discovered_npcs = getattr(validated_narrator_output, 'discovered_npcs', [])
                    if narrator_discovered_npcs:
                        # Determine the location where these NPCs should be placed.
                        # actor_location_at_action_time was already retrieved earlier in narration_task.
                        if actor_location_at_action_time:
                            current_location_npcs = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_at_action_time, "ephemeral_npcs", default=[])
                            
                            # Convert Pydantic models to dicts if they aren't already
                            npcs_to_add_as_dicts = _list_to_dicts_if_needed(narrator_discovered_npcs)

                            for new_npc_dict in npcs_to_add_as_dicts:
                                if not any(npc.get("id") == new_npc_dict.get("id") for npc in current_location_npcs):
                                    current_location_npcs.append(new_npc_dict)
                                    logger.info(f"[NarrationTask] Added new NPC '{new_npc_dict.get('name')}' (ID: {new_npc_dict.get('id')}) to location {actor_location_at_action_time} from Narrator output.")
                            
                            _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.ephemeral_npcs", current_location_npcs, logger)
                        else:
                            logger.warning(f"[NarrationTask] Narrator discovered NPCs, but actor_location_at_action_time for {actor_id} is unknown. NPCs not placed.")
                    # --- END NEW ---

                    if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                        _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", cleaned_narrative_text, logger)
                        
                        # --- NEW: Set agent to idle and update action description after narration ---
                        _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.status", "idle", logger)
                        _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.current_action_description", "Idle, observing outcome.", logger)
                        logger.info(f"[NarrationTask] Updated last_observation for {actor_id} and set status to idle.")
                        # --- END NEW ---
                    # Add after this line:
                    final_narrative_entry = f"[T{completion_time:.1f}] {cleaned_narrative_text}"
                    state.setdefault("narrative_log", []).append(final_narrative_entry)

                    # Add this code to log the narrative to the events file:
                    _log_event(
                        sim_time=completion_time,
                        agent_id="Narrator",
                        event_type="narration",
                        data={"narrative": cleaned_narrative_text},
                        logger_instance=logger,
                        event_logger_global=event_logger_global
                    )
                except Exception as e_processing:
                    logger.error(f"[NarrationTask] Error processing narrator output: {e_processing}")
                    fallback_narrative = f"{actor_name} completed an action, but the narrative could not be generated."
                    if live_display_object:
                        live_display_object.console.print(Panel(fallback_narrative, title=f"Narrator (Fallback) @ {completion_time:.1f}s", border_style="yellow", expand=False))
                    if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                        _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", fallback_narrative, logger)
            else:
                # Handle case where LLM call failed
                logger.error(f"[NarrationTask] Failed to get narrator output for {actor_name}")
                fallback_narrative = f"{actor_name} completed an action: {outcome_desc}"
                if live_display_object:
                    live_display_object.console.print(Panel(fallback_narrative, title=f"Narrator (Fallback) @ {completion_time:.1f}s", border_style="red", expand=False))
                if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", fallback_narrative, logger)

            # Always call task_done() after successful processing
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
            
            location_state_data["objects_present"] = location_state_data.get("ephemeral_objects", [])
            time_for_world_engine_prompt = get_time_string_for_prompt(state, sim_elapsed_time_seconds=current_sim_time)
            target_id_we = get_nested(intent, "target_id")
            target_state_data = get_target_entity_state(state, target_id_we, actor_location_id) or {}

            # Convert trigger information to a dictionary
            trigger_data_we = {
                "actor_name_and_id": f"{actor_name} ({actor_id})",
                "current_location_id": actor_location_id,
                "intent": intent,
                "target_entity_state": target_state_data,
                "target_entity_id_hint": target_id_we or 'N/A',
                "location_state": location_state_data,
                "world_rules": world_rules,
                "world_time_context": time_for_world_engine_prompt,
                "weather_context": get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.'),
                "instruction": "Resolve this intent based on your agent instructions and the provided JSON context."
            }
            trigger_text_we = json.dumps(trigger_data_we)

            logger.debug(f"[WorldEngineLLM] Sending prompt to LLM for {actor_id}'s intent ({action_type}).")

            trigger_content_we = genai_types.UserContent(parts=[genai_types.Part(text=trigger_text_we)])

            validated_data = await _call_adk_agent_and_parse(
                world_engine_runner, world_engine_agent, world_engine_session_id, USER_ID,
                trigger_content_we, WorldEngineResponse, f"WorldEngineLLM_{actor_id}", logger,
            )

            # validated_data can be Pydantic model, SimpleNamespace, or None
            if validated_data:
                outcome_description = getattr(validated_data, 'outcome_description', "Outcome not described.")
                parsed_resolution = _to_dict(validated_data)
                if live_display_object and parsed_resolution:
                    live_display_object.console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                    try: 
                        live_display_object.console.print(json.dumps(parsed_resolution, indent=2))
                    except TypeError: 
                        live_display_object.console.print(str(parsed_resolution))
                _log_event(sim_time=current_sim_time, agent_id="WorldEngine", event_type="resolution", data=parsed_resolution, logger_instance=logger, event_logger_global=event_logger_global)

            # Handle case where validated_data is None
            if not validated_data:
                outcome_description = "Action failed: No response from World Engine LLM."
            elif getattr(validated_data, 'valid_action', False):
                completion_time = current_sim_time + getattr(validated_data, 'duration', 0.0)
                
                # ADD YOUR UNIVERSAL INTERRUPTION CODE HERE - before any action-specific handling
                # Check if target is another Simulacra and handle interruption if needed
                if target_id_we and target_id_we in get_nested(state, SIMULACRA_KEY, default={}):
                # NOTE: The following block applies to any action targeting another Simulacra, not just "talk"
                    # Check if target is busy (in the middle of an action)
                    target_status = get_nested(state, f"{SIMULACRA_KEY}.{target_id_we}.status", default="idle")
                    target_name = get_nested(state, f"{SIMULACRA_KEY}.{target_id_we}.name", default=target_id_we)
                    
                    if target_status == "busy":
                        logger.info(f"[WorldEngineLLM] {actor_name}'s {action_type} action is interrupting {target_name}")
                        handle_action_interruption(state, target_id_we, actor_id, actor_name, logger)
                        
                    # Set target to busy regardless (this happens for all targeted actions)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{target_id_we}.status", "busy", logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{target_id_we}.current_action_description", 
                                       f"Responding to {actor_name}'s {action_type}", logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{target_id_we}.current_action_end_time", 
                                       completion_time, logger)
                
                # --- ADD TALK ACTION HANDLING ---
                skip_immediate_narration_queue = False # Initialize default
                if action_type == "talk":
                    target_id_we = get_nested(intent, "target_id")
                    speech_content = get_nested(intent, "details", default="")
                    
                    if target_id_we:
                        # Check if target is a Simulacra
                        if target_id_we in get_nested(state, SIMULACRA_KEY, default={}):
                            # Get target name for better status descriptions
                            target_name = get_nested(state, SIMULACRA_KEY, target_id_we, "persona_details", "Name", default=target_id_we)
                            
                            # IMMEDIATE BLOCKING for BOTH simulacra in conversation
                            # 1. Set listener (target) to busy
                            _update_state_value(state, f"{SIMULACRA_KEY}.{target_id_we}.status", "busy", logger)
                            _update_state_value(state, f"{SIMULACRA_KEY}.{target_id_we}.current_action_description", f"Listening to {actor_name}", logger)
                            _update_state_value(state, f"{SIMULACRA_KEY}.{target_id_we}.current_action_end_time", completion_time, logger)
                            
                            # 2. Set speaker (actor) to busy explicitly for conversation
                            _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.status", "busy", logger)
                            _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.current_action_description", f"Speaking to {target_name}", logger)
                            _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.current_action_end_time", completion_time, logger)
                            
                            logger.info(f"[WorldEngine] IMMEDIATELY set both simulacra to conversation mode: {actor_id} speaking to {target_id_we} for {getattr(validated_data, 'duration', 0.0)}s")
                            
                            # IMMEDIATE speech delivery (not scheduled) - but don't queue narration yet
                            speech_event = {
                                "event_type": "simulacra_speech_received_as_interrupt",
                                "target_agent_id": target_id_we,
                                "details": {
                                    "message_content": speech_content,
                                    "speaker_name": actor_name,
                                    "speech_duration": getattr(validated_data, 'duration', 3.0)
                                },
                                "trigger_sim_time": state.get("world_time", 0.0),  # IMMEDIATE
                                "source_actor_id": actor_id
                            }
                            state.setdefault("pending_simulation_events", []).append(speech_event)
                            logger.info(f"[WorldEngine] IMMEDIATELY queued speech delivery for {target_id_we}")
                            
                            # DON'T queue narration immediately for talk actions - it will be queued when action completes
                            # The narration_event will be created below but not queued until completion_time
                            skip_immediate_narration_queue = True
                        
                        # ADD THIS: Check if target is an NPC
                        else:
                            # Check if target is an NPC in current location
                            ephemeral_npcs = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, "ephemeral_npcs", default=[])
                            npc_found = any(npc.get("id") == target_id_we for npc in ephemeral_npcs)
                            
                            if npc_found:
                                # Valid NPC talk - Narrator will generate the NPC's response
                                logger.info(f"[WorldEngine] Validated talk to NPC {target_id_we} - response will be generated by Narrator")
                            else:
                                # Could also check global NPCs or objects here
                                logger.debug(f"[WorldEngine] Talk target {target_id_we} not found as Simulacra or NPC in current location")
                # --- Handle scheduled_future_event --- 
                # REMOVE THIS BLOCK FOR TALK ACTIONS - they should be handled immediately above
                sfe_dict = _to_dict(getattr(validated_data, 'scheduled_future_event', None))
                if sfe_dict and action_type != "talk":  # ADD THIS CONDITION
                    event_trigger_time = completion_time + sfe_dict.get('estimated_delay_seconds', 0.0)
                    event_to_schedule = {
                        "event_type": sfe_dict.get('event_type', 'unknown_event'),
                        "target_agent_id": sfe_dict.get('target_agent_id'), 
                        "location_id": sfe_dict.get('location_id', 'unknown_location'),
                        "details": sfe_dict.get('details', {}), 
                        "trigger_sim_time": event_trigger_time,
                        "source_actor_id": actor_id
                    }
                    state.setdefault("pending_simulation_events", []).append(event_to_schedule)
                    state["pending_simulation_events"].sort(key=lambda x: x.get("trigger_sim_time", float('inf')))
                    logger.info(f"[WorldEngineLLM] Scheduled future event '{event_to_schedule.get('event_type')}' for agent {event_to_schedule.get('target_agent_id')} at sim_time {event_trigger_time:.2f} triggered by {actor_id}.")

                # --- Get results and extract discoveries from INSIDE results ---
                pending_results_dict = _to_dict(getattr(validated_data, 'results', {}))
                action_completion_results_dict = {}

                # Extract discoveries from INSIDE the results
                discovered_objects_list = pending_results_dict.get('discovered_objects', [])
                discovered_npcs_list = pending_results_dict.get('discovered_npcs', [])
                discovered_connections_list = pending_results_dict.get('discovered_connections', [])

                # Determine effective location for discoveries
                effective_location_id_for_discoveries = actor_location_id

                if action_type == "move":
                    # For move actions, discoveries apply to the DESTINATION
                    destination_location_id = pending_results_dict.get(f"{SIMULACRA_KEY}.{actor_id}.location")
                    if destination_location_id:
                        effective_location_id_for_discoveries = destination_location_id
                        logger.info(f"[WorldEngineLLM] Move action: discoveries will apply to destination {destination_location_id}")
                        
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
                # Handle discovered objects and add them to location
                if discovered_objects_list:
                    # Add objects to location's ephemeral objects
                    current_ephemeral_objects = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, effective_location_id_for_discoveries, "ephemeral_objects", default=[])
                    
                    # Merge new objects with existing ones, avoiding duplicates by ID
                    for new_obj in discovered_objects_list:
                        # Check if object already exists in this location
                        if not any(obj.get("id") == new_obj.get("id") for obj in current_ephemeral_objects):
                            current_ephemeral_objects.append(new_obj)
                            logger.debug(f"[WorldEngineLLM] Added new ephemeral object {new_obj.get('id')} to location {effective_location_id_for_discoveries}")
                    
                    # Update the state with the merged list
                    _update_state_value(state, f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{effective_location_id_for_discoveries}.ephemeral_objects", current_ephemeral_objects, logger)
                    logger.info(f"[WorldEngineLLM] Updated ephemeral objects for location {effective_location_id_for_discoveries}")

                # Always apply connection merging for any action that might discover connections
                if discovered_connections_list or action_type in ["look_around", "move", "explore", "examine"]:
                    update_location_connections(state, effective_location_id_for_discoveries, discovered_connections_list, logger)
                    logger.debug(f"[WorldEngineLLM] Applied connection merging for {action_type} action at location {effective_location_id_for_discoveries}")

                narration_event = {
                    "type": "action_complete", 
                    "actor_id": actor_id, 
                    "action": intent,
                    "results_for_narration_context": pending_results_dict,
                    "outcome_description": getattr(validated_data, 'outcome_description', ""),
                    "completion_time": completion_time,
                    "current_action_description": f"Action: {intent.get('action_type', 'unknown')} - Details: {intent.get('details', 'N/A')[:100]}",
                    "actor_current_location_id": actor_location_id, 
                    "world_mood": world_mood_global, 
                    "discovered_objects_for_narration": discovered_objects_list,
                    "discovered_npcs_for_narration": discovered_npcs_list,
                    "discovered_connections_for_narration": discovered_connections_list,
                    # ADD THIS: Include target information for talk actions
                    # "target_entity_info": target_state_data if action_type == "talk" else None,
                }

                if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.status", "busy", logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.pending_results", pending_results_dict, logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.action_completion_results", action_completion_results_dict, logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.current_action_end_time", completion_time, logger)

                    # Refine action description for listening
                    action_desc_for_state = narration_event["current_action_description"]
                    if action_type == "wait" and "listened attentively to" in outcome_description:
                        action_desc_for_state = outcome_description
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.current_action_description", action_desc_for_state, logger)

                    action_duration = getattr(validated_data, 'duration', 0.0)

                    # Delay narration for actions with duration.
                    # 'talk' actions with skip_immediate_narration_queue=True are handled first.
                    # Then, other actions with duration > 0.1s.
                    # Otherwise, queue narration immediately.
                    if action_type == "talk" and skip_immediate_narration_queue:
                        # Schedule narration to be queued at completion time
                        delayed_narration_event = {
                            "event_type": "delayed_narration",
                            "narration_data": narration_event,
                            "trigger_sim_time": completion_time,
                            "source_actor_id": actor_id
                        }
                        state.setdefault("pending_simulation_events", []).append(delayed_narration_event)
                        state["pending_simulation_events"].sort(key=lambda x: x.get("trigger_sim_time", float('inf')))
                        logger.info(f"[WorldEngineLLM] Scheduled narration for talk action (Simulacra target) to trigger at {completion_time:.1f}s")
                    elif action_duration > 0.1: # For other actions with duration (e.g., move, use, talk to NPC with duration)
                        delayed_narration_event = {
                            "event_type": "delayed_narration",
                            "narration_data": narration_event,
                            "trigger_sim_time": completion_time,
                            "source_actor_id": actor_id
                        }
                        state.setdefault("pending_simulation_events", []).append(delayed_narration_event)
                        state["pending_simulation_events"].sort(key=lambda x: x.get("trigger_sim_time", float('inf')))
                        logger.info(f"[WorldEngineLLM] Scheduled narration for action '{action_type}' (duration: {action_duration:.1f}s) to trigger at {completion_time:.1f}s")
                    else: # For actions with very short/zero duration
                        narration_success = await safe_queue_put(narration_queue, narration_event, timeout=5.0, task_name=f"WorldEngineLLM_{actor_id}")
                        if not narration_success:
                            logger.error(f"[WorldEngineLLM] Failed to queue narration for {actor_id}")
                            _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", outcome_description, logger)
                        else:
                            logger.info(f"[WorldEngineLLM] Action VALID for {actor_id}. Updated state, set end time {completion_time:.1f}s. Queued narration immediately for short/zero duration action.")
                else:
                    logger.error(f"[WorldEngineLLM] Actor {actor_id} not found in state after valid action resolution.")
            else: 
                final_outcome_desc = getattr(validated_data, 'outcome_description', outcome_description) if validated_data else outcome_description
                logger.info(f"[WorldEngineLLM] Action INVALID for {actor_id}. Reason: {final_outcome_desc}")
                
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
                    try: 
                        live_display_object.console.print(json.dumps(resolution_details, indent=2))
                    except TypeError: 
                        live_display_object.console.print(str(resolution_details))
                
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
    # PerceptionManager will get location details internally from the global state reference
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
    agent_current_location_id = agent_state_data.get(CURRENT_LOCATION_KEY, DEFAULT_HOME_LOCATION_NAME) # Still needed for "You are at:" and connections
    agent_personal_location_details = agent_state_data.get(LOCATION_DETAILS_KEY, "You are unsure of your exact surroundings.") # This is the agent's memory/understanding
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

    # Build monologue history instead of narrative history
    raw_recent_monologues = agent_state_data.get("monologue_history", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
    monologue_history_str = "\n".join(raw_recent_monologues) if raw_recent_monologues else "None."
    
    # Get recent narrative for context (reduced length)
    raw_recent_narrative = global_state_ref.get("narrative_log", [])[-3:]  # Only last 3 entries
    cleaned_recent_narrative = [re.sub(r'^\[T\d+\.\d+\]\s*', '', entry).strip() for entry in raw_recent_narrative]
    recent_narrative_context = "\n".join(cleaned_recent_narrative) if cleaned_recent_narrative else "None."
    
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
        f"- Your Recent Thoughts (Internal Monologue History):\n{monologue_history_str}",
        f"- Recent Activity Context (What happened recently):\n{recent_narrative_context}",
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

    # Circuit breaker variables
    consecutive_failures = 0
    max_consecutive_failures = 5
    failure_backoff_time = 5.0
    
    try:
        # Validate and initialize required state fields
        sim_state_init = get_nested(state, SIMULACRA_KEY, agent_id, default={})
        if not sim_state_init:
            logger.error(f"[{agent_name}] Agent state not found in global state. Cannot proceed.")
            return
            
        if "last_interjection_sim_time" not in sim_state_init:
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.last_interjection_sim_time", 0.0, logger)
        if "current_interrupt_probability" not in sim_state_init:
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None, logger)
        if "monologue_history" not in sim_state_init:
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.monologue_history", [], logger)
        
        # Handle initial action if agent is idle
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
                # Store the monologue and update goal if needed
                monologue_text = getattr(validated_intent, 'internal_monologue', "")
                if monologue_text:
                    monologue_entry = f"[T{current_world_time_init:.1f}] {monologue_text}"
                    current_monologue_history = get_nested(state, SIMULACRA_KEY, agent_id, "monologue_history", default=[])
                    current_monologue_history.append(monologue_entry)
                    # Keep only recent monologues
                    if len(current_monologue_history) > MEMORY_LOG_CONTEXT_LENGTH:
                        current_monologue_history = current_monologue_history[-MEMORY_LOG_CONTEXT_LENGTH:]
                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.monologue_history", current_monologue_history, logger)
                
                # Extract and update goal from monologue if it contains goal-setting language
                if monologue_text and any(keyword in monologue_text.lower() for keyword in ["goal", "want to", "need to", "plan to", "going to", "should"]):
                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.goal", monologue_text[:100] + "..." if len(monologue_text) > 100 else monologue_text, logger)
                
                if live_display_object:
                    live_display_object.console.print(Panel(monologue_text, title=f"{agent_name} Monologue @ {current_world_time_init:.1f}s", border_style="yellow", expand=False))
                    live_display_object.console.print(f"\n[{agent_name} Intent @ {current_world_time_init:.1f}s]")
                    live_display_object.console.print(json.dumps(_to_dict(validated_intent, exclude={'internal_monologue'}), indent=2))
                
                success = await safe_queue_put(
                    event_bus, 
                    {"type": "intent_declared", "actor_id": agent_id, "intent": _to_dict(validated_intent, exclude={'internal_monologue'})},
                    timeout=5.0,
                    task_name=f"Simulacra_{agent_name}_Initial"
                )
                
                if success:
                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "thinking", logger)
                    consecutive_failures = 0
                else:
                    logger.error(f"[{agent_name}] Failed to queue initial intent")
                    consecutive_failures += 1
            else:
                logger.error(f"[{agent_name}] Failed to get initial intent from LLM")
                consecutive_failures += 1

        next_interjection_check_sim_time = state.get("world_time", 0.0) + AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS
        iteration_count = 0
        max_iterations = 10000  # Circuit breaker for infinite loops
        
        while iteration_count < max_iterations:
            iteration_count += 1
            
            # Apply failure backoff if needed
            if consecutive_failures >= max_consecutive_failures:
                logger.warning(f"[{agent_name}] Too many consecutive failures ({consecutive_failures}). Backing off for {failure_backoff_time}s")
                await asyncio.sleep(failure_backoff_time)
                consecutive_failures = 0 # Reset after backoff
                
            await asyncio.sleep(AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS)
            current_sim_time_busy_loop = state.get("world_time", 0.0)
            agent_state_busy_loop = get_nested(state, SIMULACRA_KEY, agent_id, default={})
            
            if not agent_state_busy_loop:
                logger.error(f"[{agent_name}] Agent state disappeared from global state. Stopping task.")
                break
                
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
                    # Store the monologue and update goal if needed
                    monologue_text_loop = getattr(validated_intent, 'internal_monologue', "")
                    if monologue_text_loop:
                        monologue_entry_loop = f"[T{current_sim_time_busy_loop:.1f}] {monologue_text_loop}"
                        current_monologue_history_loop = get_nested(state, SIMULACRA_KEY, agent_id, "monologue_history", default=[])
                        current_monologue_history_loop.append(monologue_entry_loop)
                        # Keep only recent monologues
                        if len(current_monologue_history_loop) > MEMORY_LOG_CONTEXT_LENGTH:
                            current_monologue_history_loop = current_monologue_history_loop[-MEMORY_LOG_CONTEXT_LENGTH:]
                        _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.monologue_history", current_monologue_history_loop, logger)
                    
                    # Extract and update goal from monologue if it contains goal-setting language
                    if monologue_text_loop and any(keyword in monologue_text_loop.lower() for keyword in ["goal", "want to", "need to", "plan to", "going to", "should"]):
                        _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.goal", monologue_text_loop[:100] + "..." if len(monologue_text_loop) > 100 else monologue_text_loop, logger)
                    
                    if live_display_object:
                        live_display_object.console.print(Panel(monologue_text_loop, title=f"{agent_name} Monologue @ {current_sim_time_busy_loop:.1f}s", border_style="yellow", expand=False))
                        live_display_object.console.print(f"\n[{agent_name} Intent @ {current_sim_time_busy_loop:.1f}s]")
                        live_display_object.console.print(json.dumps(_to_dict(validated_intent, exclude={'internal_monologue'}), indent=2))
                    
                    success = await safe_queue_put(
                        event_bus, 
                        {"type": "intent_declared", "actor_id": agent_id, "intent": _to_dict(validated_intent, exclude={'internal_monologue'})},
                        timeout=5.0,
                        task_name=f"Simulacra_{agent_name}_Subsequent"
                    )
                    
                    if success:
                        _log_event(
                            sim_time=current_sim_time_busy_loop, agent_id=agent_id, event_type="intent",
                            data=_to_dict(validated_intent, exclude={'internal_monologue'}),
                            logger_instance=logger, event_logger_global=event_logger_global
                        )
                        _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "thinking", logger)
                        consecutive_failures = 0
                    else:
                        logger.error(f"[{agent_name}] Failed to queue subsequent intent")
                        consecutive_failures += 1
                        # Keep status as idle so agent can retry
                else:
                    logger.error(f"[{agent_name}] Failed to get subsequent intent from LLM")
                    consecutive_failures += 1
                    
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
                        
                        # Include recent thoughts in reflection prompt
                        recent_monologues = agent_state_busy_loop.get("monologue_history", [])[-3:]
                        recent_thoughts_context = "\n".join(recent_monologues) if recent_monologues else "None."
                        
                        reflection_prompt_text = f"""You are {agent_name}. You are currently busy with: "{current_action_desc_for_prompt}".
This task is scheduled to continue for a while (until simulation time {agent_state_busy_loop.get("current_action_end_time", 0.0):.1f}).

Your recent thoughts:
{recent_thoughts_context}

It's time for a brief self-reflection. Your status is 'reflecting'.
Do you:
1. Wish to `continue_current_task` without interruption?
2. Feel the need to `initiate_change`? (e.g., take a break, address hunger, switch focus, react to boredom/monotony).
Your `internal_monologue` should explain your reasoning and connect to your recent thoughts and current goal.
If you choose to continue, your `action_type` must be `continue_current_task`.
If you choose to `initiate_change`, provide the `action_type` and `details` for that change as usual.
Output ONLY the JSON: `{{"internal_monologue": "...", "action_type": "...", "target_id": "...", "details": "..."}}`"""
                        reflection_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=reflection_prompt_text)])
                        
                        validated_reflection_intent = await _call_adk_agent_and_parse(
                            sim_runner, sim_agent, sim_session_id, USER_ID,
                            reflection_trigger_content, SimulacraIntentResponse, f"Simulacra_{agent_name}_Reflection", logger
                        )

                        if validated_reflection_intent:
                            action_type_reflect = getattr(validated_reflection_intent, 'action_type', 'unknown')
                            monologue_reflect = getattr(validated_reflection_intent, 'internal_monologue', '')
                            
                            # Store reflection monologue
                            if monologue_reflect:
                                monologue_entry_reflect = f"[T{current_sim_time_busy_loop:.1f}] {monologue_reflect}"
                                current_monologue_history_reflect = get_nested(state, SIMULACRA_KEY, agent_id, "monologue_history", default=[])
                                current_monologue_history_reflect.append(monologue_entry_reflect)
                                if len(current_monologue_history_reflect) > MEMORY_LOG_CONTEXT_LENGTH:
                                    current_monologue_history_reflect = current_monologue_history_reflect[-MEMORY_LOG_CONTEXT_LENGTH:]
                                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.monologue_history", current_monologue_history_reflect, logger)
                            
                            if action_type_reflect == "continue_current_task":
                                logger.info(f"[{agent_name}] Reflection: Chose to continue. Monologue: {monologue_reflect[:50]}...")
                                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", original_status_before_reflection, logger)
                                consecutive_failures = 0
                            else:
                                logger.info(f"[{agent_name}] Reflection: Chose to '{action_type_reflect}'. Monologue: {monologue_reflect[:50]}...")
                                success = await safe_queue_put(
                                    event_bus, 
                                    {"type": "intent_declared", "actor_id": agent_id, "intent": _to_dict(validated_reflection_intent, exclude={'internal_monologue'})},
                                    timeout=5.0,
                                    task_name=f"Simulacra_{agent_name}_Reflection"
                                )
                                
                                if success:
                                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None, logger)
                                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "thinking", logger)
                                    consecutive_failures = 0
                                else:
                                    logger.error(f"[{agent_name}] Failed to queue reflection intent")
                                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", original_status_before_reflection, logger)
                                    consecutive_failures += 1
                        else:
                            logger.error(f"[{agent_name}] Error processing reflection response. Staying busy.")
                            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", original_status_before_reflection, logger)
                            consecutive_failures += 1

        if iteration_count >= max_iterations:
            logger.warning(f"[{agent_name}] Hit maximum iterations ({max_iterations}). Stopping to prevent infinite loop.")

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

    # REMOVE ALL WORLDGENERATOR INITIALIZATION:

    # --- Initialize WorldGenerator Agent, Runner, and Session ---
    # world_generator_agent = create_world_generator_llm_agent(
    #     world_mood=world_mood_global, world_type=current_world_type, sub_genre=current_sub_genre
    # )
    # world_generator_runner = Runner(agent=world_generator_agent, app_name=APP_NAME + "_WorldGenerator", session_service=adk_session_service)
    # world_generator_session_id = f"world_generator_session_{world_instance_uuid}"
    # await adk_session_service.create_session(app_name=world_generator_runner.app_name, user_id=USER_ID, session_id=world_generator_session_id, state={})
    # logger.info(f"WorldGenerator Runner and Session ({world_generator_session_id}) initialized.")

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
            tasks.append(asyncio.create_task(narrative_image_generation_task(
                current_state=state,
                world_mood=world_mood_global,
                logger_instance=logger,
                event_logger_instance=event_logger_global
            ), name="NarrativeImageGenerationTask"))
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
