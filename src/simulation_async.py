# src/simulation_async.py - Core Simulation Orchestrator
import asyncio
import glob  # Keep for run_simulation profile verification
import json
import logging  # Explicitly import logging for type hinting
import os
import random  # Keep for interjection logic in simulacra_agent_task_llm
import re
import sys
from datetime import datetime, timezone
# from io import BytesIO # No longer used directly here
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai  # For direct API config
# from atproto import Client, models  # Moved to core_tasks
# from google import genai as genai_image  # Moved to core_tasks
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import google_search  # <<< Import the google_search tool
from google.genai import types as genai_types
# from PIL import Image # No longer used directly here
from pydantic import ValidationError # BaseModel, Field, etc. are no longer needed here
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from .socket_server import socket_server_task

console = Console() # Keep a global console for direct prints if needed by run_simulation
from .core_tasks import dynamic_interruption_task, narrative_image_generation_task

from .agents import create_narration_llm_agent  # Agent creation functions
from .agents import (create_search_llm_agent, create_simulacra_llm_agent,
                     create_world_engine_llm_agent)
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
from .models import NarratorOutput  # Pydantic models for tasks in this file
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
adk_session_id: Optional[str] = None # Main session for World Engine, Narrator, Simulacra
adk_session: Optional[Session] = None
adk_runner: Optional[Runner] = None # Main runner
adk_memory_service: Optional[InMemoryMemoryService] = None

world_engine_agent: Optional[LlmAgent] = None
narration_agent_instance: Optional[LlmAgent] = None # Renamed
simulacra_agents_map: Dict[str, LlmAgent] = {} # Renamed
search_llm_agent_instance: Optional[LlmAgent] = None # Renamed
search_agent_runner_instance: Optional[Runner] = None # Renamed
search_agent_session_id_val: Optional[str] = None # Renamed

world_mood_global: str = "The familiar, everyday real world; starting the morning routine at home."
live_display_object: Optional[Live] = None

def get_current_sim_time():
    return state.get("world_time", 0.0)

# --- Helper for Dynamic Instruction Context ---
def _prepare_dynamic_instruction_context(
    current_state_dict: Dict[str, Any],
    sim_time_seconds: float,
    include_news_in_instruction: bool = True,
    news_item_count: int = 1 
) -> Dict[str, str]:
    """Prepares common dynamic replacement values for agent instructions."""
    time_str = get_time_string_for_prompt(current_state_dict, sim_elapsed_time_seconds=sim_time_seconds)
    weather_str = get_nested(current_state_dict, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
    
    news_str = "No significant news." # Default if not included or not found
    if include_news_in_instruction:
        headlines = [
            item.get('headline', '') 
            for item in get_nested(current_state_dict, 'world_feeds', 'news_updates', default=[])[:news_item_count]
            if item.get('headline', '') # Ensure headline is not empty
        ]
        if headlines:
            news_str = headlines[0] # Use the first headline if available
            
    return {
        "{DYNAMIC_CURRENT_TIME}": time_str,
        "{DYNAMIC_CURRENT_WEATHER}": weather_str,
        "{DYNAMIC_CURRENT_NEWS}": news_str
    }
# --- Helper Function for ADK Agent Calls ---
async def _call_adk_agent_and_parse(
    runner: Runner,
    agent_instance: LlmAgent,
    session_id: str,
    user_id: str,
    trigger_content: genai_types.Content,
    expected_pydantic_model: type,
    agent_name_for_logging: str,
    logger_instance: logging.Logger,
    original_instruction: Optional[str] = None,
    instruction_replacements: Optional[Dict[str, str]] = None
) -> Optional[Any]:
    """
    Calls an ADK agent, processes its response, and parses it into the expected Pydantic model.
    Handles dynamic instruction updates if provided.
    """
    modified_instruction = False
    if original_instruction and instruction_replacements:
        current_instruction = original_instruction
        for placeholder, value in instruction_replacements.items():
            current_instruction = current_instruction.replace(placeholder, value)
        if agent_instance.instruction != current_instruction:
            agent_instance.instruction = current_instruction
            modified_instruction = True
    
    runner.agent = agent_instance

        llm_response_data = None
        raw_text_from_llm = ""

        async for event_llm in runner.run_async(user_id=user_id, session_id=session_id, new_message=trigger_content):
            if event_llm.error_message:
                logger_instance.error(f"[{agent_name_for_logging}] LLM Error: {event_llm.error_message}")
                return None # Early exit on error
            
            if event_llm.is_final_response() and event_llm.content:
                if isinstance(event_llm.content, expected_pydantic_model):
                    llm_response_data = event_llm.content
                    logger_instance.debug(f"[{agent_name_for_logging}] ADK successfully parsed {expected_pydantic_model.__name__} schema.")
                elif event_llm.content.parts:
                    raw_text_from_llm = event_llm.content.parts[0].text.strip()
                    logger_instance.debug(f"[{agent_name_for_logging}] LLM Final Raw Content: {raw_text_from_llm[:200]}...")
                    
                    # Assuming parse_json_output_last is defined
                    parsed_dict_from_llm = parse_json_output_last(raw_text_from_llm)
                    if parsed_dict_from_llm:
                        try:
                            llm_response_data = expected_pydantic_model.model_validate(parsed_dict_from_llm)
                        except ValidationError as ve:
                            logger_instance.error(f"[{agent_name_for_logging}] Pydantic validation failed for manual parse: {ve}. Raw: {raw_text_from_llm}")
                            llm_response_data = None
                    else:
                        logger_instance.error(f"[{agent_name_for_logging}] Failed to parse JSON (manual fallback). Raw: {raw_text_from_llm}")
                        # Assuming NarratorOutput is defined
                        if expected_pydantic_model == NarratorOutput:
                            llm_response_data = NarratorOutput(narrative=raw_text_from_llm)
                        else:
                            llm_response_data = None
                break # Exit loop after processing final response
    finally:
        # Restore original instruction if it was modified
        if modified_instruction and original_instruction:
            agent_instance.instruction = original_instruction
        
        # Restore original include_contents setting
        agent_instance.include_contents = original_include_contents

    return llm_response_data

# --- ADK-Dependent Tasks (Remain in this file for global context access) ---

async def narration_task():
    """Listens for completed actions on the narration queue and generates stylized narrative."""
    logger.info("[NarrationTask] Task started.")

    if not adk_runner or not narration_agent_instance or not adk_session:
        logger.error("[NarrationTask] Global ADK components (runner, agent, session) not initialized. Task cannot proceed.")
        return
    session_id_to_use = adk_session.id
    # Store original instruction to handle dynamic parts
    original_narration_instruction = narration_agent_instance.instruction

    while True:
        action_event = None
        try:
            action_event = await narration_queue.get()
            actor_id = get_nested(action_event, "actor_id")
            intent = get_nested(action_event, "action")
            results = get_nested(action_event, "results", default={})
            outcome_desc = get_nested(action_event, "outcome_description", default="Something happened.")
            completion_time = get_nested(action_event, "completion_time", default=state.get("world_time", 0.0))
            actor_location_at_action_time = get_nested(action_event, "actor_current_location_id") # Get location from event

            if not actor_id:
                logger.warning(f"[NarrationTask] Received narration event without actor_id: {action_event}")
                narration_queue.task_done()
                continue

            actor_name = get_nested(state, SIMULACRA_KEY, actor_id, "persona_details", "Name", default=actor_id)
            logger.debug(f"[NarrationTask] Using global world mood: '{world_mood_global}' for actor {actor_name} at location {actor_location_at_action_time or 'Unknown'}")

            def clean_history_entry(entry: str) -> str:
                cleaned = re.sub(r'^\[T\d+\.\d+\]\s*', '', entry)
                cleaned = re.sub(r'\[\w+Agent(?:_sim_\w+)?\] said: ```json.*?```', '', cleaned, flags=re.DOTALL).strip()
                return cleaned
            raw_recent_narrative = state.get("narrative_log", [])[-5:]
            cleaned_recent_narrative = [clean_history_entry(entry) for entry in raw_recent_narrative if clean_history_entry(entry)]
            history_str = "\n".join(cleaned_recent_narrative)

            # Fetch world feeds for the Narrator
            logger.info(f"[NarrationTask] Generating narrative for {actor_name}'s action completion. Outcome: '{outcome_desc}'")
            intent_json = json.dumps(intent, indent=2)
            results_json = json.dumps(results, indent=2)
            instruction_replacements_narrator = _prepare_dynamic_instruction_context(state, completion_time, include_news_in_instruction=True, news_item_count=1)

            trigger_text_narrator = f"""
Actor ID: {actor_id}
Actor Name: {actor_name}
Original Intent: {intent_json}
Factual Outcome Description: {outcome_desc}
State Changes (Results): {results_json}
Recent Narrative History (Cleaned):
{history_str}

Generate the narrative paragraph based on these details and your instructions (remembering the established world style '{world_mood_global}').
"""
            trigger_content_narrator = genai_types.UserContent(parts=[genai_types.Part(text=trigger_text_narrator)])

            validated_narrator_output = await _call_adk_agent_and_parse(
                adk_runner, narration_agent_instance, session_id_to_use, USER_ID,
                trigger_content_narrator, NarratorOutput, f"NarrationTask_{actor_name}", logger,
                original_instruction=original_narration_instruction, instruction_replacements=instruction_replacements_narrator
            )

            if 'validated_narrator_output' in locals() and validated_narrator_output:
                try:
                    logger.debug(f"[NarrationTask] Narrator output validated successfully: {validated_narrator_output.model_dump_json(indent=2, exclude_none=True)}")

                    actual_narrative_paragraph = validated_narrator_output.narrative
                    discovered_objects = validated_narrator_output.discovered_objects
                    discovered_connections = validated_narrator_output.discovered_connections
                    discovered_npcs = validated_narrator_output.discovered_npcs

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

                    if actor_location_at_action_time:
                        original_intent_from_event = get_nested(action_event, "action", default={})
                        if original_intent_from_event.get("action_type") == "look_around":
                            if validated_narrator_output.discovered_objects: # Use validated data
                                location_path_for_ephemeral_obj = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.ephemeral_objects"
                                _update_state_value(state, location_path_for_ephemeral_obj, [obj.model_dump() for obj in validated_narrator_output.discovered_objects], logger)
                                logger.info(f"[NarrationTask] Updated/Set {len(validated_narrator_output.discovered_objects)} ephemeral objects for location {actor_location_at_action_time} from look_around.")
                            else: 
                                location_path_for_ephemeral_obj = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.ephemeral_objects"
                                _update_state_value(state, location_path_for_ephemeral_obj, [], logger)
                                logger.info(f"[NarrationTask] Cleared ephemeral objects for location {actor_location_at_action_time} as none were discovered by look_around.")
                            
                            if validated_narrator_output.discovered_npcs: # Use validated data
                                location_path_for_ephemeral_npc = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.ephemeral_npcs"
                                _update_state_value(state, location_path_for_ephemeral_npc, [npc.model_dump() for npc in validated_narrator_output.discovered_npcs], logger)
                                logger.info(f"[NarrationTask] Updated/Set {len(validated_narrator_output.discovered_npcs)} ephemeral NPCs for location {actor_location_at_action_time} from look_around.")
                            else: 
                                location_path_for_ephemeral_npc = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.ephemeral_npcs"
                                _update_state_value(state, location_path_for_ephemeral_npc, [], logger)
                                logger.info(f"[NarrationTask] Cleared ephemeral NPCs for location {actor_location_at_action_time} as none were discovered by look_around.")
                            
                            # Phase 4: Store discovered connections
                            location_path_for_connections = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.connected_locations"
                            _update_state_value(state, location_path_for_connections, [conn.model_dump() for conn in validated_narrator_output.discovered_connections], logger)
                            logger.info(f"[NarrationTask] Updated/Set {len(validated_narrator_output.discovered_connections)} connected_locations for {actor_location_at_action_time} from look_around.")

                    # --- Log Narration Event ---
                    _log_event(
                        sim_time=completion_time,
                        agent_id="Narrator",
                        event_type="narration",
                        data=validated_narrator_output.model_dump(), # Log the full parsed and validated output
                        logger_instance=logger, event_logger_global=event_logger_global
                    )

                except ValidationError as e_val:
                    # This case should be less common if ADK parsing works or parse_json_output_last is robust
                    logger.error(f"[NarrationTask] Narrator output failed Pydantic validation (after potential ADK parse/fallback): {e_val}. Raw text: {getattr(validated_narrator_output, 'narrative', 'Unknown')}. Using raw text as narrative.")
                    cleaned_narrative_text = getattr(validated_narrator_output, 'narrative', "Error in narration processing.")
                    if cleaned_narrative_text and live_display_object:
                        live_display_object.console.print(Panel(cleaned_narrative_text, title=f"Narrator (Fallback) @ {completion_time:.1f}s", border_style="yellow", expand=False))
                    if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                        _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", cleaned_narrative_text, logger)

            narration_queue.task_done()

        except asyncio.CancelledError:
            logger.info("[NarrationTask] Task cancelled.")
            if action_event and narration_queue._unfinished_tasks > 0:
                try: narration_queue.task_done()
                except ValueError: pass
            break
        except Exception as e:
            logger.exception(f"[NarrationTask] Error processing event: {e}")
            if action_event and narration_queue._unfinished_tasks > 0:
                try: narration_queue.task_done()
                except ValueError: pass

async def world_engine_task_llm():
    """Listens for action requests, calls LLM to resolve, stores results, and triggers narration."""
    logger.info("[WorldEngineLLM] Task started.")

    if not adk_runner or not world_engine_agent or not adk_session:
        logger.error("[WorldEngineLLM] Global ADK components (runner, agent, session) not initialized. Task cannot proceed.")
        return
    session_id_to_use = adk_session.id
    # Store original instruction to handle dynamic parts
    original_world_engine_instruction = world_engine_agent.instruction

    while True:
        request_event = None
        actor_id = None 
        outcome_description = "Action failed due to internal error (pre-processing)."

        try:
            request_event = await event_bus.get()
            if get_nested(request_event, "type") != "intent_declared": # MODIFIED: Listen for intent_declared
                logger.debug(f"[WorldEngineLLM] Ignoring event type: {get_nested(request_event, 'type')}")
                event_bus.task_done()
                continue

            actor_id = get_nested(request_event, "actor_id")
            
            # Check if agent is in interaction mode
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
            # action_type already defined above
            actor_state_we = get_nested(state, SIMULACRA_KEY, actor_id, default={})
            actor_name = get_nested(actor_state_we, "persona_details", "Name", default=actor_id)
            current_sim_time = state.get("world_time", 0.0)
            # sim_current_location_id = state.get('location_details').get('name', DEFAULT_HOME_LOCATION_NAME)
            # sim_current_location_id_desc = state.get('location_details').get('description', DEFAULT_HOME_LOCATION_NAME)
            actor_location_id = get_nested(actor_state_we, "location")
            location_state_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, default={})
            world_rules = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'rules', default={})
            location_state_data["objects_present"] = location_state_data.get("ephemeral_objects", [])
            time_for_world_engine_prompt = get_time_string_for_prompt(state, sim_elapsed_time_seconds=current_sim_time) # type: ignore
            # Use the new helper function to get target_state_data
            target_id_we = get_nested(intent, "target_id")
            target_state_data = get_target_entity_state(state, target_id_we, actor_location_id) or {}

            intent_json = json.dumps(intent, indent=2)
            target_state_json = json.dumps(target_state_data, indent=2) if target_state_data else "{}" # Ensure valid JSON string
            location_state_json = json.dumps(location_state_data, indent=2)
            world_rules_json = json.dumps(world_rules, indent=2)
            trigger_text_we = f"""
Actor Name and ID: {actor_name} ({actor_id})
Current Location: {actor_location_id}
Intent: {intent_json}
Target Entity State ({target_id_we or 'N/A'}): {target_state_json}
Location State: {location_state_json}
World Rules: {world_rules_json}

Resolve this intent based on your instructions and the provided context.
"""
            logger.debug(f"[WorldEngineLLM] Sending prompt to LLM for {actor_id}'s intent ({action_type}).")

            instruction_replacements_we = _prepare_dynamic_instruction_context(state, current_sim_time, include_news_in_instruction=False)
            trigger_content_we = genai_types.UserContent(parts=[genai_types.Part(text=trigger_text_we)])

            validated_data = await _call_adk_agent_and_parse(
                adk_runner, world_engine_agent, session_id_to_use, USER_ID,
                trigger_content_we, WorldEngineResponse, f"WorldEngineLLM_{actor_id}", logger,
                original_instruction=original_world_engine_instruction, instruction_replacements=instruction_replacements_we
            )

            parsed_resolution = validated_data.model_dump() if validated_data else None
            if validated_data:
                outcome_description = validated_data.outcome_description
                if live_display_object and parsed_resolution:
                    live_display_object.console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                    try: live_display_object.console.print(json.dumps(parsed_resolution, indent=2))
                    except TypeError: live_display_object.console.print(str(parsed_resolution))
                _log_event(sim_time=current_sim_time, agent_id="WorldEngine", event_type="resolution", data=parsed_resolution or {}, logger_instance=logger, event_logger_global=event_logger_global)

            # Handle case where validated_data is still None after loop (e.g. LLM error or no response text)
            if 'validated_data' not in locals() or not validated_data:
                if not outcome_description.startswith("Action failed due to LLM error"): 
                    outcome_description = "Action failed: No response from World Engine LLM."

            if validated_data and validated_data.valid_action:
                completion_time = current_sim_time + validated_data.duration
                # --- BEGIN ADDITION: Handle scheduled_future_event ---
                if validated_data.scheduled_future_event:
                    sfe = validated_data.scheduled_future_event
                    # The event should trigger relative to when the actor's action (speaking) completes.
                    event_trigger_time = completion_time + sfe.estimated_delay_seconds

                    event_to_schedule = {
                        "event_type": sfe.event_type,
                        "target_agent_id": sfe.target_agent_id,
                        "location_id": sfe.location_id, # From World Engine prompt
                        "details": sfe.details,
                        "trigger_sim_time": event_trigger_time, # Absolute simulation time for the event
                        "source_actor_id": actor_id # The one whose action generated this event
                    }

                    state.setdefault("pending_simulation_events", []).append(event_to_schedule)
                    # Sort pending events by trigger time to process them in order
                    state["pending_simulation_events"].sort(key=lambda x: x.get("trigger_sim_time", float('inf')))

                    logger.info(
                        f"[WorldEngineLLM] Scheduled future event '{event_to_schedule.get('event_type')}' "
                        f"for agent {event_to_schedule.get('target_agent_id')} at sim_time {event_trigger_time:.2f} "
                        f"triggered by {actor_id}."
                    )
                # --- END ADDITION ---
                narration_event = {
                    "type": "action_complete", "actor_id": actor_id, "action": intent,
                    "results": validated_data.results, "outcome_description": validated_data.outcome_description,
                    "completion_time": completion_time,
                    "current_action_description": f"Action: {intent.get('action_type', 'unknown')} - Details: {intent.get('details', 'N/A')[:100]}",
                    "actor_current_location_id": actor_location_id, 
                    "world_mood": world_mood_global, 
                }
                # --- Phase 3: Update location_details on successful move ---
                if action_type == "move" and validated_data.results.get(f"{SIMULACRA_KEY}.{actor_id}.location"):
                    new_location_id = validated_data.results[f"{SIMULACRA_KEY}.{actor_id}.location"]
                    new_location_name = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, new_location_id, "name", default=new_location_id)
                    new_location_details_text = f"You have arrived in {new_location_name}."
                    # Add this to the results that TimeManager will apply
                    validated_data.results[f"{SIMULACRA_KEY}.{actor_id}.location_details"] = new_location_details_text
                    logger.info(f"[WorldEngineLLM] Queuing update for {actor_id}'s location_details to: '{new_location_details_text}'")
                # --- End Phase 3 Change ---
                if actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.status", "busy", logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.pending_results", validated_data.results, logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.current_action_end_time", completion_time, logger)
                    _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.current_action_description", narration_event["current_action_description"], logger)
                    await narration_queue.put(narration_event)
                    logger.info(f"[WorldEngineLLM] Action VALID for {actor_id}. Stored results, set end time {completion_time:.1f}s. Outcome: {outcome_description}")
                else:
                    logger.error(f"[WorldEngineLLM] Actor {actor_id} not found in state after valid action resolution.")
            else: 
                final_outcome_desc = validated_data.outcome_description if validated_data else outcome_description
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
                try: event_bus.task_done()
                except ValueError: pass
            break
        except Exception as e:
            logger.exception(f"[WorldEngineLLM] Error processing event for actor {actor_id}: {e}")
            if actor_id and actor_id in get_nested(state, SIMULACRA_KEY, default={}):
                 _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.status", "idle", logger)
                 _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.pending_results", {}, logger)
                 _update_state_value(state, f"{SIMULACRA_KEY}.{actor_id}.last_observation", f"Action failed unexpectedly: {e}", logger)
            if request_event and event_bus._unfinished_tasks > 0: 
                try: event_bus.task_done()
                except ValueError: pass 
            await asyncio.sleep(1) 
        finally: 
            if request_event and event_bus._unfinished_tasks > 0:
                try: event_bus.task_done()
                except ValueError: logger.warning("[WorldEngineLLM] task_done() called too many times in finally.")
                except Exception as td_e: logger.error(f"[WorldEngineLLM] Error calling task_done() in finally: {td_e}")


def _build_simulacra_prompt(
    agent_id: str,
    agent_name: str,
    agent_state_data: Dict[str, Any], 
    current_sim_time: float,
    global_state_ref: Dict[str, Any], 
    perception_manager_instance: PerceptionManager, 
    status_message_for_prompt: str
) -> str:
    """Builds the detailed prompt for the Simulacra agent."""
    # --- Perception Gathering ---
    fresh_percepts = perception_manager_instance.get_percepts_for_simulacrum(agent_id)
    _log_event(
        sim_time=current_sim_time, agent_id=agent_id, event_type="agent_perception_generated",
        data=fresh_percepts, logger_instance=logger, event_logger_global=event_logger_global
    )
    logger.debug(f"[{agent_name}] Built percepts for prompt: {json.dumps(fresh_percepts, sort_keys=True)[:250]}...")

    # --- Format Percepts for LLM Prompt ---
    perceptual_summary_for_prompt = "Perception system error or offline."
    audible_env_str = "  The environment is quiet."
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

        perceptual_summary_for_prompt = (
            f"Official Location Description: \"{loc_desc_from_percepts}\"\n"
            f"Visible Entities:\n{visible_sim_str}\n{visible_static_obj_str}\n{visible_eph_obj_str}\n{visible_eph_npc_str}"
        )

    # --- Other Contextual Information ---
    agent_current_location_id = agent_state_data.get(CURRENT_LOCATION_KEY, DEFAULT_HOME_LOCATION_NAME)
    agent_personal_location_details = agent_state_data.get(LOCATION_DETAILS_KEY, "You are unsure of your exact surroundings.")
    connected_locations = get_nested(global_state_ref, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, agent_current_location_id, "connected_locations", default=[])
    raw_recent_narrative = global_state_ref.get("narrative_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
    cleaned_recent_narrative = [re.sub(r'^\[T\d+\.\d+\]\s*', '', entry).strip() for entry in raw_recent_narrative]
    history_str = "\n".join(cleaned_recent_narrative)
    weather_summary = get_nested(global_state_ref, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
    latest_news_headlines = [item.get('headline', '') for item in get_nested(global_state_ref, 'world_feeds', 'news_updates', default=[])[:2]]
    news_summary = " ".join(h for h in latest_news_headlines if h) or "No major news."

    prompt_text_parts = [
        f"**Current State Info for {agent_name} ({agent_id}):**",
        f"- Persona: {agent_state_data.get('persona_details', {})}",
        f"- Current Location ID: {agent_current_location_id or 'Unknown'}",
        f"- Your Personal Understanding of this Location: \"{agent_personal_location_details}\"",
        f"- Perceived Environment:\n{perceptual_summary_for_prompt}",
        f"- Status: {agent_state_data.get('status', 'idle')} ({status_message_for_prompt})",
        f"- Current Weather: {weather_summary}",
        f"- Recent News Snippet: {news_summary}",
        f"- Current Goal: {agent_state_data.get('goal', 'Determine goal.')}",
        f"- Current Time: {get_time_string_for_prompt(global_state_ref, sim_elapsed_time_seconds=current_sim_time)}",
        f"- Last Observation/Event: {agent_state_data.get('last_observation', 'None.')}",
        f"- Audible Environment:\n{audible_env_str}",
        f"- Recent Narrative History:\n{history_str if history_str else 'None.'}",
        f"- Exits/Connections from this location: {json.dumps(connected_locations) if connected_locations else 'None observed.'}",
        "\nFollow your thinking process and provide your response ONLY in the specified JSON format."
    ]
    return "\n".join(prompt_text_parts)


async def simulacra_agent_task_llm(agent_id: str):
    """Asynchronous task for managing a single Simulacra LLM agent."""
    agent_name = get_nested(state, SIMULACRA_KEY, agent_id, "persona_details", "Name", default=agent_id)
    logger.info(f"[{agent_name}] LLM Agent task started.")

    if not adk_runner or not adk_session:
        logger.error(f"[{agent_name}] Global ADK Runner or Session not initialized. Task cannot proceed.")
        return

    session_id_to_use = adk_session.id
    sim_agent = simulacra_agents_map.get(agent_id)
    if not sim_agent:
        logger.error(f"[{agent_name}] Could not find agent instance in simulacra_agents_map. Task cannot proceed.")
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
            
            instruction_replacements_sim = _prepare_dynamic_instruction_context(
                state, current_world_time_init, include_news_in_instruction=True, news_item_count=1
            )

            initial_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=initial_trigger_text)])
            
            validated_intent = await _call_adk_agent_and_parse(
                adk_runner, sim_agent, session_id_to_use, USER_ID,
                initial_trigger_content, SimulacraIntentResponse, f"Simulacra_{agent_name}_Initial", logger,
                original_instruction=original_simulacra_agent_instruction, instruction_replacements=instruction_replacements_sim
            )
            
            if validated_intent:
                if live_display_object:
                    live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_world_time_init:.1f}s", border_style="yellow", expand=False))
                    live_display_object.console.print(f"\n[{agent_name} Intent @ {current_world_time_init:.1f}s]")
                    live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": validated_intent.model_dump(exclude={'internal_monologue'})})
                _log_event(
                    sim_time=current_world_time_init, agent_id=agent_id, event_type="intent",
                    data=validated_intent.model_dump(exclude={'internal_monologue'}), 
                    logger_instance=logger, event_logger_global=event_logger_global
                )
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
                
                # Also update instruction for subsequent calls
                instruction_replacements_sim_loop = _prepare_dynamic_instruction_context(
                    state, current_sim_time_busy_loop, include_news_in_instruction=True, news_item_count=1
                )

                trigger_content_loop = genai_types.UserContent(parts=[genai_types.Part(text=prompt_text)])

                validated_intent = await _call_adk_agent_and_parse(
                    adk_runner, sim_agent, session_id_to_use, USER_ID,
                    trigger_content_loop, SimulacraIntentResponse, f"Simulacra_{agent_name}_Subsequent", logger,
                    original_instruction=original_simulacra_agent_instruction, instruction_replacements=instruction_replacements_sim_loop
                )

                if validated_intent:
                    if live_display_object:
                        live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_sim_time_busy_loop:.1f}s", border_style="yellow", expand=False))
                        live_display_object.console.print(f"\n[{agent_name} Intent @ {current_sim_time_busy_loop:.1f}s]")
                        live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                    await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": validated_intent.model_dump(exclude={'internal_monologue'})})
                    _log_event(
                        sim_time=current_sim_time_busy_loop, agent_id=agent_id, event_type="intent",
                        data=validated_intent.model_dump(exclude={'internal_monologue'}),
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
                            adk_runner, sim_agent, session_id_to_use, USER_ID,
                            reflection_trigger_content, SimulacraIntentResponse, f"Simulacra_{agent_name}_Reflection", logger
                            # No original_instruction/replacements needed here as reflection prompt is specific
                        )

                        if validated_reflection_intent:
                            if validated_reflection_intent.action_type == "continue_current_task":
                                logger.info(f"[{agent_name}] Reflection: Chose to continue. Monologue: {validated_reflection_intent.internal_monologue[:50]}...")
                                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", original_status_before_reflection, logger)
                            else:
                                logger.info(f"[{agent_name}] Reflection: Chose to '{validated_reflection_intent.action_type}'. Monologue: {validated_reflection_intent.internal_monologue[:50]}...")
                                await event_bus.put({ "type": "intent_declared", "actor_id": agent_id, "intent": validated_reflection_intent.model_dump(exclude={'internal_monologue'}) })
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
    global adk_session_service, adk_session_id, adk_session, adk_runner, adk_memory_service
    global world_engine_agent, simulacra_agents_map, state, live_display_object, narration_agent_instance
    global world_mood_global, search_llm_agent_instance, search_agent_runner_instance, search_agent_session_id_val, perception_manager_global
    global event_logger_global # Ensure we can assign to the global

    console.rule("[bold green]Starting Async Simulation[/]")

    event_logger_global = event_logger_instance # Assign the passed logger to our global variable

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        logger.info(f"Global random seed initialized to: {RANDOM_SEED}")

    adk_memory_service = InMemoryMemoryService()
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

    adk_session_service = InMemorySessionService()
    adk_session_id = f"sim_session_{world_instance_uuid}"
    adk_session = adk_session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=adk_session_id, state=state
    )
    # logger.info(f"ADK Session created: {adk_session.id if adk_session else 'None'}.") # Log actual session ID if available
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

    world_engine_agent = create_world_engine_llm_agent(
        sim_id=first_sim_id, persona_name=first_persona_name,
        world_type=current_world_type, sub_genre=current_sub_genre
    )
    narration_agent_instance = create_narration_llm_agent(
        sim_id=first_sim_id, persona_name=first_persona_name,
        world_mood=world_mood_global, world_type=current_world_type, sub_genre=current_sub_genre
    )
    search_llm_agent_instance = create_search_llm_agent()

    simulacra_agents_map.clear()
    for sim_id_val in final_active_sim_ids:
        sim_profile_data = get_nested(state, SIMULACRA_KEY, sim_id_val, default={})
        persona_name = get_nested(sim_profile_data, "persona_details", "Name", default=sim_id_val)
        sim_agent_instance = create_simulacra_llm_agent(sim_id_val, persona_name, world_mood=world_mood_global)
        simulacra_agents_map[sim_id_val] = sim_agent_instance
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

    adk_runner = Runner(
        agent=world_engine_agent, app_name=APP_NAME,
        session_service=adk_session_service, memory_service=adk_memory_service
    )
    logger.info(f"Main ADK Runner initialized.")

    search_agent_session_id_val = f"world_feed_search_session_{world_instance_uuid}"
    # adk_session_service.create_session(app_name=APP_NAME + "_Search", user_id=USER_ID, session_id=search_agent_session_id_val)
    # Ensure the search agent session creation is awaited
    search_adk_session = adk_session_service.create_session(
        app_name=APP_NAME + "_Search", 
        user_id=USER_ID, # This user_id is used by the world feed fetcher in simulation_utils.py
        session_id=search_agent_session_id_val,
        state={} # Initial empty state for the search session
    )
    if search_adk_session:
        logger.info(f"ADK Search Agent Session created: {search_adk_session.id}")
    else:
        logger.error(f"Failed to create ADK Search Agent Session: {search_agent_session_id_val}. World feeds may not function correctly.")
    search_agent_runner_instance = Runner(
        agent=search_llm_agent_instance, app_name=APP_NAME + "_Search",
        session_service=adk_session_service
    )
    logger.info(f"Dedicated Search Agent Runner initialized.")

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

            if not tasks:
                 logger.error("No tasks were created. Simulation cannot run.")
                 console.print("[bold red]Error: No simulation tasks started.[/bold red]")
            else:
                logger.info(f"Started {len(tasks)} tasks.")
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try: task.result()
                    except asyncio.CancelledError: logger.info(f"Task {task.get_name()} was cancelled.")
                    except Exception as task_exc: logger.error(f"Task {task.get_name()} raised: {task_exc}", exc_info=task_exc)
                logger.info("One main task completed/failed. Initiating shutdown.")

    except Exception as e:
        logger.exception(f"Error during simulation setup or execution: {e}")
        console.print(f"[bold red]Unexpected error during simulation run: {e}[/bold red]")
    finally:
        logger.info("Cancelling remaining tasks...")
        if 'tasks' in locals() and tasks: 
            for task in tasks:
                if not task.done(): task.cancel()
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