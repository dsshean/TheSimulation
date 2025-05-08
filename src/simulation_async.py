# src/simulation_async.py - Core Simulation Orchestrator

import asyncio
import datetime
import glob  # Keep for run_simulation profile verification
import json
import logging
import os
import random  # Keep for interjection logic in simulacra_agent_task_llm
import re
import string  # For default sim_id generation if needed
import sys
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai  # For direct API config
from google import genai as genai_image  # For direct API config
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import google_search  # <<< Import the google_search tool
from google.genai import types as genai_types
from PIL import Image
from pydantic import (BaseModel, Field,  # Keep for models defined here
                      ValidationError, ValidationInfo, field_validator)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console() # Keep a global console for direct prints if needed by run_simulation

from .agents import create_narration_llm_agent  # Agent creation functions
from .agents import (create_search_llm_agent, create_simulacra_llm_agent,
                     create_world_engine_llm_agent)
# Import from our new/refactored modules
from .config import (  # For run_simulation; For self-reflection; New constants for dynamic_interruption_task; PROB_INTERJECT_AS_NARRATIVE removed; from this import list
    ACTIVE_SIMULACRA_IDS_KEY, AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS,
    AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS, API_KEY, APP_NAME,
    CURRENT_LOCATION_KEY, DEFAULT_HOME_DESCRIPTION, DEFAULT_HOME_LOCATION_NAME,
    DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS, DYNAMIC_INTERRUPTION_MAX_PROB_CAP,
    DYNAMIC_INTERRUPTION_MIN_PROB,
    DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION,
    DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS,
    ENABLE_NARRATIVE_IMAGE_GENERATION, HOME_LOCATION_KEY,
    IMAGE_GENERATION_INTERVAL_REAL_SECONDS, IMAGE_GENERATION_MODEL_NAME,
    IMAGE_GENERATION_OUTPUT_DIR, INTERJECTION_COOLDOWN_SIM_SECONDS,
    LIFE_SUMMARY_DIR, LOCATION_DETAILS_KEY, LOCATION_KEY,
    LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS, MAX_MEMORY_LOG_ENTRIES,
    MAX_SIMULATION_TIME, MEMORY_LOG_CONTEXT_LENGTH,
    MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK, MODEL_NAME,
    PROB_INTERJECT_AS_SELF_REFLECTION, RANDOM_SEED, SEARCH_AGENT_MODEL_NAME,
    SIMULACRA_PROFILES_KEY, SIMULATION_SPEED_FACTOR, STATE_DIR,
    UPDATE_INTERVAL, USER_ID, WORLD_STATE_KEY, WORLD_TEMPLATE_DETAILS_KEY)
from .core_tasks import interaction_dispatcher_task  # ADK-independent tasks
from .core_tasks import time_manager_task, world_info_gatherer_task
from .loop_utils import (get_nested, load_json_file,
                         load_or_initialize_simulation, parse_json_output_last,
                         save_json_file)
from .models import (  # Pydantic models for tasks in this file
    SimulacraIntentResponse, WorldEngineResponse)
from .simulation_utils import (  # Utility functions; generate_llm_interjection_detail REMOVED
    _update_state_value, generate_table)
from .state_loader import parse_location_string  # Used in run_simulation

logger = logging.getLogger(__name__) # Use logger from main entry point setup

# --- Core Components (Module Scope) ---
# These are the "globals" that will be managed here and passed around or accessed directly by tasks in this file.
event_bus = asyncio.Queue()
narration_queue = asyncio.Queue()
state: Dict[str, Any] = {} # Global state dictionary

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


# --- ADK-Dependent Tasks (Remain in this file for global context access) ---

async def narration_task():
    """Listens for completed actions on the narration queue and generates stylized narrative."""
    # Accesses global: state, adk_runner, narration_agent_instance, adk_session, narration_queue, world_mood_global, live_display_object, logger
    logger.info("[NarrationTask] Task started.")

    if not adk_runner or not narration_agent_instance or not adk_session:
        logger.error("[NarrationTask] Global ADK components (runner, agent, session) not initialized. Task cannot proceed.")
        return
    session_id_to_use = adk_session.id

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

            actor_name = get_nested(state, "simulacra", actor_id, "name", default=actor_id)
            logger.debug(f"[NarrationTask] Using global world mood: '{world_mood_global}' for actor {actor_name} at location {actor_location_at_action_time or 'Unknown'}")

            def clean_history_entry(entry: str) -> str:
                cleaned = re.sub(r'^\[T\d+\.\d+\]\s*', '', entry)
                cleaned = re.sub(r'\[\w+Agent(?:_sim_\w+)?\] said: ```json.*?```', '', cleaned, flags=re.DOTALL).strip()
                return cleaned
            raw_recent_narrative = state.get("narrative_log", [])[-5:]
            cleaned_recent_narrative = [clean_history_entry(entry) for entry in raw_recent_narrative if clean_history_entry(entry)]
            history_str = "\n".join(cleaned_recent_narrative)

            logger.info(f"[NarrationTask] Generating narrative for {actor_name}'s action completion. Outcome: '{outcome_desc}'")
            intent_json = json.dumps(intent, indent=2)
            results_json = json.dumps(results, indent=2)

            prompt = f"""
Actor ID: {actor_id}
Actor Name: {actor_name}
Original Intent: {intent_json}
Factual Outcome Description: {outcome_desc}
State Changes (Results): {results_json}
Current World Time: {completion_time:.1f}
Recent Narrative History (Cleaned):
{history_str}

Generate the narrative paragraph based on these details and your instructions (remembering the established world style '{world_mood_global}').
"""
            adk_runner.agent = narration_agent_instance # Use the instance
            narrative_text = ""
            trigger_content = genai_types.Content(parts=[genai_types.Part(text=prompt)])

            async for event_llm in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=trigger_content):
                if event_llm.is_final_response() and event_llm.content:
                    narrative_text = event_llm.content.parts[0].text.strip()
                    logger.debug(f"NarrationLLM Final Content: {narrative_text[:100]}...")
                elif event_llm.error_message:
                    logger.error(f"NarrationLLM Error: {event_llm.error_message}")
                    narrative_text = f"[{actor_name}'s action resulted in: {outcome_desc}]"

            cleaned_narrative_text = narrative_text
            if narrative_text:
                try:
                    narrator_output_str = parse_json_output_last(narrative_text.strip())
                    if narrator_output_str is None:
                        raise json.JSONDecodeError("No JSON found by parse_json_output_last", narrative_text, 0)
                    narrator_output = json.loads(narrator_output_str)
                    
                    actual_narrative_paragraph = narrator_output.get("narrative", "An event occurred.")
                    discovered_objects = narrator_output.get("discovered_objects", [])
                    discovered_npcs = narrator_output.get("discovered_npcs", []) 

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

                    if actor_id in state.get("simulacra", {}):
                        _update_state_value(state, f"simulacra.{actor_id}.last_observation", cleaned_narrative_text, logger)
                    logger.info(f"[NarrationTask] Appended narrative for {actor_name}: {cleaned_narrative_text[:80]}...")

                    if actor_location_at_action_time:
                        original_intent_from_event = get_nested(action_event, "action", default={})
                        if original_intent_from_event.get("action_type") == "look_around":
                            if discovered_objects:
                                location_path_for_ephemeral_obj = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.ephemeral_objects"
                                _update_state_value(state, location_path_for_ephemeral_obj, discovered_objects, logger)
                                logger.info(f"[NarrationTask] Updated/Set {len(discovered_objects)} ephemeral objects for location {actor_location_at_action_time} from look_around.")
                            else: 
                                location_path_for_ephemeral_obj = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.ephemeral_objects"
                                _update_state_value(state, location_path_for_ephemeral_obj, [], logger)
                                logger.info(f"[NarrationTask] Cleared ephemeral objects for location {actor_location_at_action_time} as none were discovered by look_around.")
                            
                            if discovered_npcs:
                                location_path_for_ephemeral_npc = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.ephemeral_npcs"
                                _update_state_value(state, location_path_for_ephemeral_npc, discovered_npcs, logger)
                                logger.info(f"[NarrationTask] Updated/Set {len(discovered_npcs)} ephemeral NPCs for location {actor_location_at_action_time} from look_around.")
                            else: 
                                location_path_for_ephemeral_npc = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{actor_location_at_action_time}.ephemeral_npcs"
                                _update_state_value(state, location_path_for_ephemeral_npc, [], logger)
                                logger.info(f"[NarrationTask] Cleared ephemeral NPCs for location {actor_location_at_action_time} as none were discovered by look_around.")

                except json.JSONDecodeError:
                    logger.error(f"[NarrationTask] Failed to parse JSON from Narrator: {narrative_text}. Using raw text as narrative.")
                    cleaned_narrative_text = narrative_text 
                    if cleaned_narrative_text and live_display_object:
                        live_display_object.console.print(Panel(cleaned_narrative_text, title=f"Narrator (Fallback) @ {completion_time:.1f}s", border_style="yellow", expand=False))
                    if actor_id in state.get("simulacra", {}):
                        _update_state_value(state, f"simulacra.{actor_id}.last_observation", cleaned_narrative_text, logger)

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

    while True:
        request_event = None
        actor_id = None 
        outcome_description = "Action failed due to internal error (pre-processing)."

        try:
            request_event = await event_bus.get()
            if get_nested(request_event, "type") != "resolve_action_request":
                logger.debug(f"[WorldEngineLLM] Ignoring event type: {get_nested(request_event, 'type')}")
                event_bus.task_done()
                continue

            actor_id = get_nested(request_event, "actor_id")
            intent = get_nested(request_event, "intent")
            interaction_class = get_nested(request_event, "interaction_class", default="environment")
            if not actor_id or not intent:
                logger.warning(f"[WorldEngineLLM] Received invalid action request event: {request_event}")
                event_bus.task_done()
                continue

            logger.info(f"[WorldEngineLLM] Received '{interaction_class}' action request from {actor_id}: {intent}")
            action_type = intent.get("action_type")
            actor_state_we = get_nested(state, 'simulacra', actor_id, default={}) 
            actor_name = actor_state_we.get('name', actor_id)
            current_sim_time = state.get("world_time", 0.0)
            sim_current_location_id = state.get('location_details').get('name', DEFAULT_HOME_LOCATION_NAME)
            sim_current_location_id_desc = state.get('location_details').get('description', DEFAULT_HOME_LOCATION_NAME)
            actor_location_id = get_nested(actor_state_we, "location")
            location_state_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, default={})
            world_rules = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'rules', default={})
            location_state_data["objects_present"] = location_state_data.get("ephemeral_objects", [])
            
            target_id = get_nested(intent, "target_id")
            target_state_data = {}
            if target_id:
                target_state_data = get_nested(state, 'objects', target_id, default={}) or get_nested(state, 'simulacra', target_id, default={})

            intent_json = json.dumps(intent, indent=2)
            target_state_json = json.dumps(target_state_data, indent=2) if target_state_data else "N/A"
            location_state_json = json.dumps(location_state_data, indent=2)
            world_rules_json = json.dumps(world_rules, indent=2)

            prompt = f"""
Actor: {actor_name} ({actor_id})
Location: {sim_current_location_id}
Time: {current_sim_time:.1f}
Intent: {intent_json}
Target Entity State ({target_id or 'N/A'}): {target_state_json}
Location State: {location_state_json}
World Rules: {world_rules_json}

Resolve this intent based on your instructions and the provided context.
"""
            logger.debug(f"[WorldEngineLLM] Sending prompt to LLM for {actor_id}'s intent ({action_type}).")
            adk_runner.agent = world_engine_agent 
            response_text = ""
            trigger_content = genai_types.Content(parts=[genai_types.Part(text=prompt)])

            async for event_llm in adk_runner.run_async(
                user_id=USER_ID, session_id=session_id_to_use, new_message=trigger_content
            ):
                if event_llm.is_final_response() and event_llm.content:
                    response_text = event_llm.content.parts[0].text
                    logger.debug(f"WorldLLM Final Content: {response_text[:100]}...")
                elif event_llm.error_message:
                    logger.error(f"WorldLLM Error: {event_llm.error_message}")
                    outcome_description = f"Action failed due to LLM error: {event_llm.error_message}"
                    break 

            validated_data: Optional[WorldEngineResponse] = None
            parsed_resolution = None
            if response_text:
                try:
                    response_text_clean_str = parse_json_output_last(response_text.strip()) 
                    if response_text_clean_str is None:
                        raise json.JSONDecodeError("No JSON found by parse_json_output_last", response_text, 0)
                    
                    json_str_to_parse = response_text_clean_str
                    correct_actor_name = actor_state_we.get('name', actor_id)
                    agent_internal_name = f"SimulacraLLM_{actor_id}" 
                    json_str_to_parse = json_str_to_parse.replace(agent_internal_name, correct_actor_name)
                    json_str_to_parse = json_str_to_parse.replace("[Actor Name]", actor_name) 
                    json_str_to_parse = json_str_to_parse.replace("[ACTOR_ID]", actor_id)
                    if target_id:
                        json_str_to_parse = json_str_to_parse.replace("[target_id]", target_id)
                    if target_state_data: 
                         obj_name = target_state_data.get("name", target_id) 
                         json_str_to_parse = json_str_to_parse.replace("[Object Name]", obj_name)
                         if action_type == 'talk': 
                             target_name = target_state_data.get("name", target_id)
                             json_str_to_parse = json_str_to_parse.replace("[Target Name]", target_name)
                    target_object_state = get_nested(state, 'objects', target_id, default={}) if target_id else {}
                    if target_object_state and target_object_state.get("destination"):
                         dest_name = target_object_state.get("destination") 
                         json_str_to_parse = json_str_to_parse.replace("[Destination Name]", dest_name)
                         json_str_to_parse = json_str_to_parse.replace("[Destination]", dest_name)

                    raw_data = json.loads(json_str_to_parse)
                    parsed_resolution = raw_data
                    validated_data = WorldEngineResponse.model_validate(raw_data)
                    logger.debug(f"[WorldEngineLLM] LLM response validated successfully for {actor_id}.")
                    outcome_description = validated_data.outcome_description
                    if live_display_object:
                        live_display_object.console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                        try: live_display_object.console.print(json.dumps(parsed_resolution, indent=2))
                        except TypeError: live_display_object.console.print(str(parsed_resolution)) 
                except json.JSONDecodeError as e:
                    logger.error(f"[WorldEngineLLM] Failed to decode JSON response for {actor_id}: {e}\nResponse:\n{response_text}", exc_info=True)
                    outcome_description = "Action failed due to internal error (JSON decode)."
                except ValidationError as e:
                    logger.error(f"[WorldEngineLLM] Failed to validate LLM response structure for {actor_id}: {e}\nResponse:\n{response_text}", exc_info=True)
                    outcome_description = "Action failed due to internal error (invalid structure)."
                except Exception as e: 
                     logger.error(f"[WorldEngineLLM] Unexpected error parsing/validating response for {actor_id}: {e}\nResponse:\n{response_text}", exc_info=True)
                     outcome_description = "Action failed due to internal error (unexpected)."
            else: 
                if not outcome_description.startswith("Action failed due to LLM error"): 
                    outcome_description = "Action failed: No response from World Engine LLM."

            if validated_data and validated_data.valid_action:
                completion_time = current_sim_time + validated_data.duration
                narration_event = {
                    "type": "action_complete", "actor_id": actor_id, "action": intent,
                    "results": validated_data.results, "outcome_description": validated_data.outcome_description,
                    "completion_time": completion_time,
                    "current_action_description": f"Action: {intent.get('action_type', 'unknown')} - Details: {intent.get('details', 'N/A')[:100]}",
                    "actor_current_location_id": actor_location_id, 
                    "world_mood": world_mood_global, 
                }
                if actor_id in state.get("simulacra", {}):
                    _update_state_value(state, f"simulacra.{actor_id}.status", "busy", logger)
                    _update_state_value(state, f"simulacra.{actor_id}.pending_results", validated_data.results, logger)
                    _update_state_value(state, f"simulacra.{actor_id}.current_action_end_time", completion_time, logger)
                    _update_state_value(state, f"simulacra.{actor_id}.current_action_description", narration_event["current_action_description"], logger)
                    await narration_queue.put(narration_event)
                    logger.info(f"[WorldEngineLLM] Action VALID for {actor_id}. Stored results, set end time {completion_time:.1f}s. Outcome: {outcome_description}")
                else:
                    logger.error(f"[WorldEngineLLM] Actor {actor_id} not found in state after valid action resolution.")
            else: 
                final_outcome_desc = validated_data.outcome_description if validated_data else outcome_description
                logger.info(f"[WorldEngineLLM] Action INVALID for {actor_id}. Reason: {final_outcome_desc}")
                if actor_id in state.get("simulacra", {}):
                    _update_state_value(state, f"simulacra.{actor_id}.last_observation", final_outcome_desc, logger)
                    _update_state_value(state, f"simulacra.{actor_id}.status", "idle", logger)
                actor_name_for_log = get_nested(state, 'simulacra', actor_id, 'name', default=actor_id)
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
            if actor_id and actor_id in get_nested(state, "simulacra", default={}): 
                 _update_state_value(state, f"simulacra.{actor_id}.status", "idle", logger)
                 _update_state_value(state, f"simulacra.{actor_id}.pending_results", {}, logger)
                 _update_state_value(state, f"simulacra.{actor_id}.last_observation", f"Action failed unexpectedly: {e}", logger)
            if request_event and event_bus._unfinished_tasks > 0: 
                try: event_bus.task_done()
                except ValueError: pass 
            await asyncio.sleep(1) 
        finally: 
            if request_event and event_bus._unfinished_tasks > 0:
                try: event_bus.task_done()
                except ValueError: logger.warning("[WorldEngineLLM] task_done() called too many times in finally.")
                except Exception as td_e: logger.error(f"[WorldEngineLLM] Error calling task_done() in finally: {td_e}")

async def dynamic_interruption_task():
    """
    Periodically checks busy simulacra and probabilistically interrupts them
    with a direct narrative observation.
    """
    logger.info("[DynamicInterruptionTask] Task started.")

    # Initial delay can be shorter now that checks are more frequent    
    await asyncio.sleep(random.uniform(5.0, 15.0)) 

    while True:
        # await asyncio.sleep(AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS / 4.0 * UPDATE_INTERVAL / SIMULATION_SPEED_FACTOR) 
        await asyncio.sleep(DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS) 

        current_sim_time_dit = state.get("world_time", 0.0)
        active_sim_ids_dit = list(state.get(ACTIVE_SIMULACRA_IDS_KEY, []))

        for agent_id_to_check in active_sim_ids_dit:
            agent_state_to_check = get_nested(state, "simulacra", agent_id_to_check, default={})
            if not agent_state_to_check or agent_state_to_check.get("status") != "busy":
                continue
            
            # If agent is busy but not eligible for other reasons, clear its stored probability
            _update_state_value(state, f"simulacra.{agent_id_to_check}.current_interrupt_probability", None, logger)

            agent_name_to_check = agent_state_to_check.get("name", agent_id_to_check)
            last_interruption_time = agent_state_to_check.get("last_interjection_sim_time", 0.0) 
            cooldown_passed = (current_sim_time_dit - last_interruption_time) >= INTERJECTION_COOLDOWN_SIM_SECONDS 
            
            if not cooldown_passed:
                continue

            remaining_duration = agent_state_to_check.get("current_action_end_time", 0.0) - current_sim_time_dit
            if remaining_duration < MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK: 
                continue
            
            # Agent is eligible, calculate and store probability
            interrupt_probability = 0.0
            if DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS > 0:
                # Duration factor can now exceed 1.0 for actions longer than the target duration
                duration_factor = remaining_duration / DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS
                scaled_prob = duration_factor * DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION
                # Apply min probability and then cap at the absolute maximum
                interrupt_probability = min(DYNAMIC_INTERRUPTION_MAX_PROB_CAP, max(DYNAMIC_INTERRUPTION_MIN_PROB, scaled_prob))
            else: # Fallback if target duration is zero, use min_prob capped by max_prob
                interrupt_probability = min(DYNAMIC_INTERRUPTION_MAX_PROB_CAP, DYNAMIC_INTERRUPTION_MIN_PROB)
            _update_state_value(state, f"simulacra.{agent_id_to_check}.current_interrupt_probability", interrupt_probability, logger)

            if random.random() < interrupt_probability:
                logger.info(f"[DynamicInterruptionTask] Triggering dynamic interruption for {agent_name_to_check} (Prob: {interrupt_probability:.3f}, RemDur: {remaining_duration:.1f}s).")
                
                interruption_text = f"A minor unexpected event occurs, breaking {agent_name_to_check}'s concentration." 
                try:
                    interrupt_llm = genai.GenerativeModel(MODEL_NAME)
                    narrative_prompt = f"""Agent {agent_name_to_check} is currently busy with: "{agent_state_to_check.get("current_action_description", "their current activity")}".
The general world mood is: "{world_mood_global}".
An unexpected minor interruption occurs. Describe this interruption in one or two engaging narrative sentences from an observational perspective, suitable for {agent_name_to_check} to perceive.
Example: "Suddenly, a loud crash from the kitchen shatters the quiet, making {agent_name_to_check} jump."
Example: "The lights in the room flicker ominously for a moment, then stabilize."
Example: "A faint, unidentifiable melody seems to drift in from outside."
Output ONLY the narrative sentence(s).
"""
                    response = await interrupt_llm.generate_content_async(narrative_prompt)
                    if response.text:
                        interruption_text = response.text.strip()
                except Exception as e_interrupt_text:
                    logger.error(f"[DynamicInterruptionTask] Failed to generate LLM text for interruption, using default. Error: {e_interrupt_text}")
                
                logger.info(f"[DynamicInterruptionTask] Interrupting {agent_name_to_check} with: {interruption_text}")

                _update_state_value(state, f"simulacra.{agent_id_to_check}.status", "idle", logger) 
                _update_state_value(state, f"simulacra.{agent_id_to_check}.last_observation", interruption_text, logger)
                _update_state_value(state, f"simulacra.{agent_id_to_check}.pending_results", {}, logger)
                _update_state_value(state, f"simulacra.{agent_id_to_check}.current_action_end_time", current_sim_time_dit, logger) 
                _update_state_value(state, f"simulacra.{agent_id_to_check}.current_action_description", "Interrupted by a dynamic event.", logger)
                _update_state_value(state, f"simulacra.{agent_id_to_check}.last_interjection_sim_time", current_sim_time_dit, logger) 
                _update_state_value(state, f"simulacra.{agent_id_to_check}.current_interrupt_probability", None, logger) # Clear after interruption
                break 

        await asyncio.sleep(0.1) 


async def simulacra_agent_task_llm(agent_id: str):
    """Asynchronous task for managing a single Simulacra LLM agent."""
    agent_name = get_nested(state, "simulacra", agent_id, "name", default=agent_id)
    logger.info(f"[{agent_name}] LLM Agent task started.")

    if not adk_runner or not adk_session:
        logger.error(f"[{agent_name}] Global ADK Runner or Session not initialized. Task cannot proceed.")
        return

    session_id_to_use = adk_session.id
    sim_agent = simulacra_agents_map.get(agent_id)
    if not sim_agent:
        logger.error(f"[{agent_name}] Could not find agent instance in simulacra_agents_map. Task cannot proceed.")
        return

    try:
        sim_state_init = get_nested(state, "simulacra", agent_id, default={})
        if "last_interjection_sim_time" not in sim_state_init:
            _update_state_value(state, f"simulacra.{agent_id}.last_interjection_sim_time", 0.0, logger)
        if "current_interrupt_probability" not in sim_state_init: # Initialize if not present
            _update_state_value(state, f"simulacra.{agent_id}.current_interrupt_probability", None, logger)
        # next_simple_timer_interjection_sim_time is no longer managed by this task

        if get_nested(state, "simulacra", agent_id, "status") == "idle":
            current_sim_state_init = get_nested(state, "simulacra", agent_id, default={})
            current_loc_id_init = current_sim_state_init.get('location_details')
            current_loc_state_init = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_loc_id_init, default={}) if current_loc_id_init else {}
            current_world_time_init = state.get("world_time", 0.0)

            def get_entities_in_location_init(entity_type: str, location_id: Optional[str]) -> List[Dict[str, Any]]:
                entities = []
                if not location_id: return entities
                source_dict = state.get(entity_type, {})
                for entity_id_init_loop, entity_data_init in source_dict.items(): 
                    if entity_data_init.get('location') == location_id:
                        entities.append({"id": entity_id_init_loop, "name": entity_data_init.get("name", entity_id_init_loop)})
                return entities

            ephemeral_objects_init = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_loc_id_init, "ephemeral_objects", default=[])
            objects_in_room_for_prompt_init = []
            for obj_data in ephemeral_objects_init: 
                objects_in_room_for_prompt_init.append({"id": obj_data.get("id"), "name": obj_data.get("name")})
            agents_in_room_init = [a for a in get_entities_in_location_init("simulacra", current_loc_id_init) if a["id"] != agent_id]
            
            raw_recent_narrative_init = state.get("narrative_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
            cleaned_recent_narrative_init = [re.sub(r'^\[T\d+\.\d+\]\s*', '', entry).strip() for entry in raw_recent_narrative_init]
            history_str_init = "\n".join(cleaned_recent_narrative_init)
            weather_summary = get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
            latest_news_headlines = [item.get('headline', '') for item in get_nested(state, 'world_feeds', 'news_updates', default=[])[:2]]
            news_summary = " ".join(h for h in latest_news_headlines if h) or "No major news."

            sim_current_location_id = state.get('location_details').get('name', DEFAULT_HOME_LOCATION_NAME)
            sim_current_location_id_desc = state.get('location_details').get('description', DEFAULT_HOME_LOCATION_NAME)

            prompt_text_parts_init = [
                 f"**Current State Info for {agent_name} ({agent_id}):**",
                 f"- Persona Summary: {current_sim_state_init.get('persona', {}).get('summary', 'Not available.')}",
                 f"- Location ID: {current_loc_id_init or 'Unknown'}",
                 f"- Location Description: {current_loc_state_init.get('description', 'Not available.')}",
                 f"- Status: {current_sim_state_init.get('status', 'idle')} (You should act now)",
                 f"- Current Weather: {weather_summary}",
                 f"- Recent News Snippet: {news_summary}",
                 f"- Current Goal: {current_sim_state_init.get('goal', 'Determine goal.')}",
                 f"- Current Time: {current_world_time_init:.1f}s",
                 f"- Last Observation/Event: {current_sim_state_init.get('last_observation', 'None.')}",
                 f"- Recent History:\n{history_str_init if history_str_init else 'None.'}",
                 f"- Objects in Room: {json.dumps(objects_in_room_for_prompt_init) if objects_in_room_for_prompt_init else 'None.'}",
                 f"- Other Agents in Room: {json.dumps(agents_in_room_init) if agents_in_room_init else 'None.'}",
                 "\nFollow your thinking process and provide your response ONLY in the specified JSON format."
            ]
            initial_trigger_text = "\n".join(prompt_text_parts_init)
            logger.debug(f"[{agent_name}] Sending initial context prompt as agent is idle.")
            initial_trigger_content = genai_types.Content(parts=[genai_types.Part(text=initial_trigger_text)])
            adk_runner.agent = sim_agent 
            async for event in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=initial_trigger_content):
                if event.is_final_response() and event.content:
                    response_text = event.content.parts[0].text
                    try:
                        response_text_clean_str = parse_json_output_last(response_text.strip())
                        if response_text_clean_str is None:
                            raise json.JSONDecodeError("No JSON found by parse_json_output_last", response_text, 0)
                        parsed_data = json.loads(response_text_clean_str)
                        validated_intent = SimulacraIntentResponse.model_validate(parsed_data)
                        if live_display_object:
                            live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_world_time_init:.1f}s", border_style="yellow", expand=False))
                            live_display_object.console.print(f"\n[{agent_name} Intent @ {current_world_time_init:.1f}s]")
                            live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                        await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": validated_intent.model_dump(exclude={'internal_monologue'})})
                        _update_state_value(state, f"simulacra.{agent_id}.status", "thinking", logger)
                    except (json.JSONDecodeError, ValidationError) as e_init:
                        logger.error(f"[{agent_name}] Error processing initial response: {e_init}\nResponse:\n{response_text}", exc_info=True)
                    break 
                elif event.error_message:
                    logger.error(f"[{agent_name}] LLM Error during initial prompt: {event.error_message}")
                    break 

        next_interjection_check_sim_time = state.get("world_time", 0.0) + AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS
        while True:
            await asyncio.sleep(AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS)
            current_sim_time_busy_loop = state.get("world_time", 0.0)
            agent_state_busy_loop = get_nested(state, "simulacra", agent_id, default={})
            current_status_busy_loop = agent_state_busy_loop.get("status")

            if current_status_busy_loop == "idle":
                logger.debug(f"[{agent_name}] Status is idle. Proceeding to plan next action.")
                current_loc_id = agent_state_busy_loop.get('location')
                current_loc_state = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_loc_id, default={}) if current_loc_id else {}
                def get_entities_in_location(entity_type: str, location_id: Optional[str]) -> List[Dict[str, Any]]:
                    entities = []
                    if not location_id: return entities
                    source_dict = state.get(entity_type, {})
                    for entity_id_loop, entity_data_loop in source_dict.items():
                        if entity_data_loop.get('location') == location_id:
                            entities.append({"id": entity_id_loop, "name": entity_data_loop.get("name", entity_id_loop)})
                    return entities
                
                ephemeral_objects_loop = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_loc_id, "ephemeral_objects", default=[])
                objects_in_room_for_prompt_loop = []
                for obj_data in ephemeral_objects_loop:
                    objects_in_room_for_prompt_loop.append({"id": obj_data.get("id"), "name": obj_data.get("name")})
                agents_in_room = [a for a in get_entities_in_location("simulacra", current_loc_id) if a["id"] != agent_id]
                raw_recent_narrative = state.get("narrative_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
                cleaned_recent_narrative = [re.sub(r'^\[T\d+\.\d+\]\s*', '', entry).strip() for entry in raw_recent_narrative]
                history_str = "\n".join(cleaned_recent_narrative)
                weather_summary_loop = get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
                latest_news_headlines_loop = [item.get('headline', '') for item in get_nested(state, 'world_feeds', 'news_updates', default=[])[:2]]
                news_summary_loop = " ".join(h for h in latest_news_headlines_loop if h) or "No major news."
                prompt_text_parts = [
                     f"**Current State Info for {agent_name} ({agent_id}):**",
                     f"- Persona Summary: {agent_state_busy_loop.get('persona', {}).get('summary', 'Not available.')}",
                     f"- Location ID: {current_loc_id or 'Unknown'}",
                     f"- Location Description: {current_loc_state.get('description', 'Not available.')}",
                     f"- Status: {agent_state_busy_loop.get('status', 'idle')} (You should act now)",
                     f"- Current Weather: {weather_summary_loop}",
                     f"- Recent News Snippet: {news_summary_loop}",
                     f"- Current Goal: {agent_state_busy_loop.get('goal', 'Determine goal.')}",
                     f"- Current Time: {current_sim_time_busy_loop:.1f}s",
                     f"- Last Observation/Event: {agent_state_busy_loop.get('last_observation', 'None.')}",
                     f"- Recent History:\n{history_str if history_str else 'None.'}",
                     f"- Objects in Room: {json.dumps(objects_in_room_for_prompt_loop) if objects_in_room_for_prompt_loop else 'None.'}",
                     f"- Other Agents in Room: {json.dumps(agents_in_room) if agents_in_room else 'None.'}",
                     "\nFollow your thinking process and provide your response ONLY in the specified JSON format."
                ]
                prompt_text = "\n".join(prompt_text_parts)
                logger.debug(f"[{agent_name}] Sending subsequent prompt.")
                trigger_content = genai_types.Content(parts=[genai_types.Part(text=prompt_text)])
                adk_runner.agent = sim_agent 
                async for event in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=trigger_content):
                    if event.is_final_response() and event.content:
                        response_text = event.content.parts[0].text
                        try:
                            response_text_clean_str = parse_json_output_last(response_text.strip())
                            if response_text_clean_str is None:
                                raise json.JSONDecodeError("No JSON found by parse_json_output_last", response_text, 0)
                            parsed_data = json.loads(response_text_clean_str)
                            validated_intent = SimulacraIntentResponse.model_validate(parsed_data)
                            if live_display_object:
                                live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_sim_time_busy_loop:.1f}s", border_style="yellow", expand=False))
                                live_display_object.console.print(f"\n[{agent_name} Intent @ {current_sim_time_busy_loop:.1f}s]")
                                live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                            await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": validated_intent.model_dump(exclude={'internal_monologue'})})
                            _update_state_value(state, f"simulacra.{agent_id}.status", "thinking", logger)
                        except (json.JSONDecodeError, ValidationError) as e_idle:
                            logger.error(f"[{agent_name}] Error processing subsequent response: {e_idle}\nResponse:\n{response_text}", exc_info=True)
                        break 
                    elif event.error_message:
                        logger.error(f"[{agent_name}] LLM Error during subsequent prompt: {event.error_message}")
                        break 
                continue 

            # Busy Action Interjection Logic - RETAINED FOR SELF-REFLECTION ONLY
            if current_status_busy_loop == "busy" and current_sim_time_busy_loop >= next_interjection_check_sim_time:
                next_interjection_check_sim_time = current_sim_time_busy_loop + AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS
                remaining_duration = agent_state_busy_loop.get("current_action_end_time", 0.0) - current_sim_time_busy_loop
                last_interjection_time_busy = agent_state_busy_loop.get("last_interjection_sim_time", 0.0) 
                cooldown_passed_busy = (current_sim_time_busy_loop - last_interjection_time_busy) >= INTERJECTION_COOLDOWN_SIM_SECONDS

                if remaining_duration > LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS and cooldown_passed_busy:
                    logger.info(f"[{agent_name}] Busy with long task (rem: {remaining_duration:.1f}s). Checking for self-reflection.")
                    _update_state_value(state, f"simulacra.{agent_id}.last_interjection_sim_time", current_sim_time_busy_loop, logger)

                    if random.random() < PROB_INTERJECT_AS_SELF_REFLECTION: 
                        original_status_before_reflection = agent_state_busy_loop.get("status")
                        _update_state_value(state, f"simulacra.{agent_id}.status", "reflecting", logger)
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
                        reflection_trigger_content = genai_types.Content(parts=[genai_types.Part(text=reflection_prompt_text)])
                        adk_runner.agent = sim_agent 
                        async for event in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=reflection_trigger_content):
                            if event.is_final_response() and event.content:
                                response_text = event.content.parts[0].text
                                try:
                                    response_text_clean_str = parse_json_output_last(response_text.strip())
                                    if response_text_clean_str is None:
                                        raise json.JSONDecodeError("No JSON found by parse_json_output_last", response_text, 0)
                                    parsed_data = json.loads(response_text_clean_str)
                                    validated_reflection_intent = SimulacraIntentResponse.model_validate(parsed_data)
                                    if validated_reflection_intent.action_type == "continue_current_task":
                                        logger.info(f"[{agent_name}] Reflection: Chose to continue. Monologue: {validated_reflection_intent.internal_monologue[:50]}...")
                                        _update_state_value(state, f"simulacra.{agent_id}.status", original_status_before_reflection, logger)
                                        # Probability remains as it was, task continues
                                    else:
                                        logger.info(f"[{agent_name}] Reflection: Chose to '{validated_reflection_intent.action_type}'. Monologue: {validated_reflection_intent.internal_monologue[:50]}...")
                                        await event_bus.put({ "type": "intent_declared", "actor_id": agent_id, "intent": validated_reflection_intent.model_dump(exclude={'internal_monologue'}) })
                                        _update_state_value(state, f"simulacra.{agent_id}.current_interrupt_probability", None, logger) # Clear as action is changing
                                        _update_state_value(state, f"simulacra.{agent_id}.status", "thinking", logger)
                                except Exception as e_reflect:
                                    logger.error(f"[{agent_name}] Error processing reflection response: {e_reflect}. Staying busy. Response: {response_text}")
                                    _update_state_value(state, f"simulacra.{agent_id}.status", original_status_before_reflection, logger)
                                break 

    except asyncio.CancelledError:
        logger.info(f"[{agent_name}] Task cancelled.")
    except Exception as e:
        logger.error(f"[{agent_name}] Error in agent task: {e}", exc_info=True)
        if agent_id in get_nested(state, "simulacra", default={}):
            _update_state_value(state, f"simulacra.{agent_id}.status", "idle", logger)
    finally:
        logger.info(f"[{agent_name}] Task finished.")


async def run_simulation(
    instance_uuid_arg: Optional[str] = None,
    location_override_arg: Optional[str] = None,
    mood_override_arg: Optional[str] = None
    ):
    global adk_session_service, adk_session_id, adk_session, adk_runner, adk_memory_service
    global world_engine_agent, simulacra_agents_map, state, live_display_object, narration_agent_instance
    global world_mood_global, search_llm_agent_instance, search_agent_runner_instance, search_agent_session_id_val

    console.rule("[bold green]Starting Async Simulation[/]")

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


    active_sim_ids_from_state = list(state.get(ACTIVE_SIMULACRA_IDS_KEY, []))
    sim_profiles_from_state = state.get(SIMULACRA_PROFILES_KEY, {})
    
    for sim_id_val_init in active_sim_ids_from_state:
        if sim_id_val_init not in state.get("simulacra", {}): 
            profile = sim_profiles_from_state.get(sim_id_val_init, {})
            persona = profile.get("persona_details")
            if persona:
                default_loc = list(state.get(WORLD_STATE_KEY, {}).get(LOCATION_DETAILS_KEY, {}).keys())[0] if state.get(WORLD_STATE_KEY, {}).get(LOCATION_DETAILS_KEY) else DEFAULT_HOME_LOCATION_NAME
                start_loc = profile.get(CURRENT_LOCATION_KEY, default_loc)
                home_loc = profile.get(HOME_LOCATION_KEY, default_loc)

                state.setdefault("simulacra", {})[sim_id_val_init] = {
                    "id": sim_id_val_init, "name": persona.get("Name", sim_id_val_init), "persona": persona,
                    "location": start_loc, "home_location": home_loc, "status": "idle",
                    "current_action_end_time": state.get('world_time', 0.0),
                    "goal": profile.get("goal", persona.get("Initial_Goal", "Determine goals.")),
                    "last_observation": profile.get("last_observation", "Waking up."),
                    "memory_log": profile.get("memory_log", []),
                    "pending_results": {}, "last_interjection_sim_time": 0.0,
                    "next_simple_timer_interjection_sim_time": 0.0,
                    "current_action_description": "N/A",
                    "current_interrupt_probability": None # Initialize the new field
                }
                logger.info(f"Ensured runtime state for simulacrum: {sim_id_val_init}")
            else:
                logger.warning(f"Missing persona for {sim_id_val_init} in profiles. Cannot fully populate runtime state.")

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
    logger.info(f"ADK Session created: {adk_session_id}.")

    simulacra_agents_map.clear()
    for sim_id_val in final_active_sim_ids:
        sim_state_data = state.get("simulacra", {}).get(sim_id_val, {})
        persona_name = sim_state_data.get("name", sim_id_val)
        sim_agent_instance = create_simulacra_llm_agent(sim_id_val, persona_name, world_mood=world_mood_global)
        world_engine_agent = create_world_engine_llm_agent(sim_id_val, persona_name)
        narration_agent_instance = create_narration_llm_agent(sim_id_val, persona_name,world_mood=world_mood_global)
        search_llm_agent_instance = create_search_llm_agent()
        simulacra_agents_map[sim_id_val] = sim_agent_instance
    logger.info(f"Created {len(simulacra_agents_map)} simulacra agents.")

    adk_runner = Runner(
        agent=world_engine_agent, app_name=APP_NAME,
        session_service=adk_session_service, memory_service=adk_memory_service
    )
    logger.info(f"Main ADK Runner initialized.")

    search_agent_session_id_val = f"world_feed_search_session_{world_instance_uuid}"
    adk_session_service.create_session(app_name=APP_NAME + "_Search", user_id=USER_ID, session_id=search_agent_session_id_val)
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

            tasks.append(asyncio.create_task(time_manager_task(
                current_state=state, 
                event_bus_qsize_func=lambda: event_bus.qsize(), 
                narration_qsize_func=lambda: narration_queue.qsize(), 
                live_display=live, 
                logger_instance=logger
            ), name="TimeManager"))
            tasks.append(asyncio.create_task(interaction_dispatcher_task(state, event_bus, logger), name="InteractionDispatcher"))
            tasks.append(asyncio.create_task(world_info_gatherer_task(state, world_mood_global, search_agent_runner_instance, search_agent_session_id_val, logger), name="WorldInfoGatherer"))
            tasks.append(asyncio.create_task(narrative_image_generation_task(), name="NarrativeImageGenerator"))
            tasks.append(asyncio.create_task(dynamic_interruption_task(), name="DynamicInterruptionTask")) 
            
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
        if final_uuid_to_save:
            final_state_path_to_save = os.path.join(STATE_DIR, f"simulation_state_{final_uuid_to_save}.json") 
            logger.info("Saving final simulation state.")
            try:
                if not isinstance(state.get("world_time"), (int, float)):
                     logger.warning(f"Final world_time is not a number ({type(state.get('world_time'))}). Saving as 0.0.")
                     state["world_time"] = 0.0
                save_json_file(final_state_path_to_save, state) 
                logger.info(f"Final simulation state saved to {final_state_path_to_save}")
                console.print(f"Final state saved to {final_state_path_to_save}")
            except Exception as save_e:
                 logger.error(f"Failed to save final state to {final_state_path_to_save}: {save_e}", exc_info=True)
                 console.print(f"[red]Error saving final state: {save_e}[/red]")
        else:
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

async def narrative_image_generation_task():
    """
    Periodically generates an image based on the latest narrative log entry.
    """
    if not ENABLE_NARRATIVE_IMAGE_GENERATION:
        logger.info("[NarrativeImageGenerator] Task is disabled by configuration.")
        try:
            while True: # Keep the task alive but idle
                await asyncio.sleep(3600) # Sleep for a long time
        except asyncio.CancelledError:
            logger.info("[NarrativeImageGenerator] Idling task cancelled.")
            raise # Re-raise CancelledError to allow proper cleanup

    logger.info(f"[NarrativeImageGenerator] Task started. Will generate images every {IMAGE_GENERATION_INTERVAL_REAL_SECONDS} real seconds.")
    logger.info(f"[NarrativeImageGenerator] Using model: {IMAGE_GENERATION_MODEL_NAME}")
    logger.info(f"[NarrativeImageGenerator] Saving images to: {IMAGE_GENERATION_OUTPUT_DIR}")

    # Initialize the image generation model (outside the loop for efficiency)
   
    client = genai_image.Client()
    
    await asyncio.sleep(random.uniform(5.0, 10.0)) # Initial delay

    while True:
        await asyncio.sleep(IMAGE_GENERATION_INTERVAL_REAL_SECONDS)
        
        if not state or not state.get("narrative_log"):
            logger.debug("[NarrativeImageGenerator] No narrative log found or empty. Skipping image generation.")
            continue

        narrative_log_entries = state.get("narrative_log", [])
        if not narrative_log_entries:
            logger.debug("[NarrativeImageGenerator] Narrative log is empty. Skipping image generation.")
            continue

        latest_narrative_full = narrative_log_entries[-1]
        # Strip the timestamp like "[T123.4] " from the narrative for a cleaner image prompt
        narrative_prompt_text = re.sub(r'^\[T\d+\.\d+\]\s*', '', latest_narrative_full).strip()

        if not narrative_prompt_text:
            logger.debug("[NarrativeImageGenerator] Latest narrative entry is empty after stripping timestamp. Skipping.")
            continue

        current_sim_time_for_filename = state.get("world_time", 0.0)
        prompt_for_image = f"Visually represent this scene based on the following narrative, imagine it as an instagram post in style of a cartoon: \"{narrative_prompt_text}\""
        
        logger.info(f"[NarrativeImageGenerator] Requesting image for narrative (T{current_sim_time_for_filename:.1f}): \"{narrative_prompt_text[:100]}...\"")

        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=IMAGE_GENERATION_MODEL_NAME,
                contents=prompt_for_image,  # Pass as a list for contents
                config=genai_types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )

            image_generated = False
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        try:
                            image_bytes = part.inline_data.data
                            image = Image.open(BytesIO(image_bytes))
                            
                            # Create a unique filename
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            sim_time_str = f"T{current_sim_time_for_filename:.0f}"
                            image_filename = f"narrative_{sim_time_str}_{timestamp_str}.png"
                            image_path = os.path.join(IMAGE_GENERATION_OUTPUT_DIR, image_filename)
                            
                            image.save(image_path)
                            logger.info(f"[NarrativeImageGenerator] Successfully generated and saved image: {image_path}")
                            image_generated = True
                            # image.show() # Optional: for interactive testing, but usually not for a background task
                            break # Assuming one image per request
                        except Exception as e_img_proc:
                            logger.error(f"[NarrativeImageGenerator] Error processing image data: {e_img_proc}")
            
            if not image_generated:
                logger.warning(f"[NarrativeImageGenerator] No image data found in response for prompt: {prompt_for_image[:100]}...")
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.text:
                             logger.warning(f"[NarrativeImageGenerator] Model text response (if any): {part.text}")


        except Exception as e:
            logger.error(f"[NarrativeImageGenerator] Error during image generation API call: {e}")
            if hasattr(e, 'response') and e.response: # type: ignore
                 logger.error(f"[NarrativeImageGenerator] API Response (if available): {e.response}") # type: ignore