# src/simulation_async.py - Core Simulation Orchestrator
import asyncio
import datetime
import glob  # Keep for run_simulation profile verification
import json
import logging  # Explicitly import logging for type hinting
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
from atproto import Client, models  # Import atproto client and models
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

from .socket_server import socket_server_task

console = Console() # Keep a global console for direct prints if needed by run_simulation

from .agents import create_narration_llm_agent  # Agent creation functions
from .agents import (create_search_llm_agent, create_simulacra_llm_agent,
                     create_world_engine_llm_agent)
# Import from our new/refactored modules
from .config import (  # For run_simulation; For self-reflection; New constants for dynamic_interruption_task; PROB_INTERJECT_AS_NARRATIVE removed; from this import list; Import Bluesky and social post config; Import SIMULACRA_KEY
    ACTIVE_SIMULACRA_IDS_KEY, AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS,
    AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS, API_KEY, APP_NAME,
    BLUESKY_APP_PASSWORD, BLUESKY_HANDLE, CURRENT_LOCATION_KEY,
    DEFAULT_HOME_DESCRIPTION, DEFAULT_HOME_LOCATION_NAME,
    DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS, DYNAMIC_INTERRUPTION_MAX_PROB_CAP,
    DYNAMIC_INTERRUPTION_MIN_PROB,
    DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION,
    DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS, ENABLE_BLUESKY_POSTING,
    ENABLE_NARRATIVE_IMAGE_GENERATION, HOME_LOCATION_KEY,
    IMAGE_GENERATION_INTERVAL_REAL_SECONDS, IMAGE_GENERATION_MODEL_NAME,
    IMAGE_GENERATION_OUTPUT_DIR, INTERJECTION_COOLDOWN_SIM_SECONDS,
    LIFE_SUMMARY_DIR, LOCATION_DETAILS_KEY, LOCATION_KEY,
    LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS, MAX_MEMORY_LOG_ENTRIES,
    MAX_SIMULATION_TIME, MEMORY_LOG_CONTEXT_LENGTH,
    MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK, MODEL_NAME,
    PROB_INTERJECT_AS_SELF_REFLECTION, RANDOM_SEED, SEARCH_AGENT_MODEL_NAME,
    SIMULACRA_KEY, SIMULACRA_PROFILES_KEY, SIMULATION_SPEED_FACTOR,
    SOCIAL_POST_HASHTAGS, SOCIAL_POST_TEXT_LIMIT, STATE_DIR, UPDATE_INTERVAL,
    USER_ID, WORLD_STATE_KEY, WORLD_TEMPLATE_DETAILS_KEY)
from .core_tasks import time_manager_task, world_info_gatherer_task
from .loop_utils import (get_nested, load_json_file,
                         load_or_initialize_simulation, parse_json_output_last,
                         save_json_file)
from .models import NarratorOutput  # Pydantic models for tasks in this file
from .models import SimulacraIntentResponse, WorldEngineResponse
from .simulation_utils import (  # Utility functions; generate_llm_interjection_detail REMOVED
    _update_state_value, generate_table, get_time_string_for_prompt)
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

# --- Helper for Event Logging ---
def _log_event(sim_time: float, agent_id: str, event_type: str, data: Dict[str, Any]):
    """Logs a structured event to the dedicated event logger."""
    if event_logger_global:
        log_entry = {
            "sim_time_s": round(sim_time, 2), # Round time for cleaner logs
            "agent_id": agent_id,
            "event_type": event_type,
            "data": data
        }
        try:
            event_logger_global.info(json.dumps(log_entry))
        except Exception as e:
            logger.error(f"Failed to log event (type: {event_type}, agent: {agent_id}) to event log: {e}", exc_info=True)

# --- Perception Manager ---
class PerceptionManager:
    """
    Manages what each simulacrum perceives in its environment based on the global state.
    This is primarily logic-based.
    """
    def __init__(self, global_state_ref: Dict[str, Any]):
        self.state = global_state_ref  # Reference to the main state dictionary

    def get_percepts_for_simulacrum(self, perceiving_sim_id: str) -> Dict[str, Any]:
        """
        Generates a structured perceptual package for a given simulacrum.
        """
        perceiving_sim_data = get_nested(self.state, SIMULACRA_KEY, perceiving_sim_id)
        if not perceiving_sim_data:
            logger.warning(f"[PerceptionManager] Perceiving simulacrum '{perceiving_sim_id}' not found in state.")
            return {"error": f"Perceiving simulacrum '{perceiving_sim_id}' not found."}

        current_location_id = perceiving_sim_data.get(CURRENT_LOCATION_KEY)
        if not current_location_id:
            logger.warning(f"[PerceptionManager] Perceiving simulacrum '{perceiving_sim_id}' has no current location.")
            return {
                "current_location_id": None,
                "location_description": "You are in an undefined space.",
                "visible_simulacra": [],
                "visible_static_objects": [],
                "visible_ephemeral_objects": [], # Renamed for clarity
                "visible_npcs": [],
                "audible_events": [], # Placeholder for future sound perception
                "error": "Perceiving simulacrum has no current location."
            }

        percepts: Dict[str, Any] = {
            "current_location_id": current_location_id,
            "location_description": get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "description", default="An undescribed location."),
            "visible_simulacra": [],
            "visible_static_objects": [],
            "visible_ephemeral_objects": [], # Renamed for clarity
            "visible_npcs": [],
            "audible_events": [],
        }

        # Perceive other Simulacra in the same location
        all_simulacra = get_nested(self.state, SIMULACRA_KEY, default={})
        for sim_id, sim_data in all_simulacra.items():
            if sim_id == perceiving_sim_id:
                continue
            if sim_data.get(CURRENT_LOCATION_KEY) == current_location_id:
                percepts["visible_simulacra"].append({
                    "id": sim_id,
                    "name": get_nested(sim_data, "persona_details", "Name", default=sim_id),
                    "status": sim_data.get("status", "unknown") # Observable status
                })

        # Perceive Static Objects in the same location
        all_static_objects = self.state.get("objects", []) # Assuming state["objects"] is a list of dicts
        for obj_data in all_static_objects:
            if isinstance(obj_data, dict) and obj_data.get("location") == current_location_id:
                percepts["visible_static_objects"].append({
                    "id": obj_data.get("id", "unknown_static_object"),
                    "name": obj_data.get("name", "Unnamed Static Object"),
                    "description": obj_data.get("description", "A static object is here.")
                    # Add other relevant observable properties if needed
                })

        # Perceive Ephemeral Objects and NPCs (already handled by Narrator's look_around discoveries)
        # These are stored under state[WORLD_STATE_KEY][LOCATION_DETAILS_KEY][current_location_id]["ephemeral_objects" / "ephemeral_npcs"]
        percepts["visible_ephemeral_objects"] = get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "ephemeral_objects", default=[])
        percepts["visible_npcs"] = get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "ephemeral_npcs", default=[])

        # Perceive Ambient Sound of the location
        ambient_sound_desc = get_nested(self.state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_location_id, "ambient_sound_description")
        if ambient_sound_desc:
            percepts["audible_events"].append({"source_id": "environment", "description": ambient_sound_desc, "type": "ambient"})

        return percepts
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
            weather_summary_narrator = get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
            latest_news_headlines_narrator = [item.get('headline', '') for item in get_nested(state, 'world_feeds', 'news_updates', default=[])[:1]] # Just one for brevity
            news_summary_narrator = latest_news_headlines_narrator[0] if latest_news_headlines_narrator else "No major news."
            logger.info(f"[NarrationTask] Generating narrative for {actor_name}'s action completion. Outcome: '{outcome_desc}'")
            intent_json = json.dumps(intent, indent=2)
            results_json = json.dumps(results, indent=2)
            time_for_narrator_prompt = get_time_string_for_prompt(state, sim_elapsed_time_seconds=completion_time)
            # original_narration_agent_instruction = narration_agent_instance.instruction
            # narration_agent_instance.instruction = original_narration_agent_instruction.replace("{DYNAMIC_CURRENT_TIME}", time_for_narrator_prompt).replace("{DYNAMIC_CURRENT_WEATHER}", weather_summary_narrator).replace("{DYNAMIC_CURRENT_NEWS}", news_summary_narrator)
            adk_runner.agent = narration_agent_instance

            prompt = f"""
Actor ID: {actor_id}
Actor Name: {actor_name}
Original Intent: {intent_json}
Factual Outcome Description: {outcome_desc}
State Changes (Results): {results_json}
Current World Time: {time_for_narrator_prompt}
Current Weather: {weather_summary_narrator}
Latest News Headline: {news_summary_narrator}
Recent Narrative History (Cleaned):
{history_str}

Generate the narrative paragraph based on these details and your instructions (remembering the established world style '{world_mood_global}').
"""
# Use the instance
            narrative_text = ""
            trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=prompt)])

            async for event_llm in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=trigger_content):
                if event_llm.error_message:
                    logger.error(f"NarrationLLM Error: {event_llm.error_message}")
                    narrative_text = f"[{actor_name}'s action resulted in: {outcome_desc}]"
                    # Attempt to process this fallback as if it were the narrative paragraph
                    validated_narrator_output = NarratorOutput(narrative=narrative_text)
                    # Skip further parsing attempts if there was an LLM error
                    break 
                
                if event_llm.is_final_response() and event_llm.content:
                    if isinstance(event_llm.content, NarratorOutput): # ADK parsed it!
                        validated_narrator_output = event_llm.content
                        logger.debug(f"[NarrationTask] ADK successfully parsed NarratorOutput schema.")
                        narrative_text = validated_narrator_output.narrative # For logging snippet
                    elif event_llm.content.parts: # Fallback to manual parsing
                        narrative_text = event_llm.content.parts[0].text.strip()
                        logger.debug(f"NarrationLLM Final Raw Content: {narrative_text[:100]}...")
                        parsed_dict_from_llm = parse_json_output_last(narrative_text)
                        if parsed_dict_from_llm:
                            validated_narrator_output = NarratorOutput.model_validate(parsed_dict_from_llm)
                        else: # If manual parsing also fails
                            logger.error(f"[NarrationTask] Failed to parse JSON from Narrator (manual fallback). Raw text: {narrative_text}. Using raw text as narrative.")
                            validated_narrator_output = NarratorOutput(narrative=narrative_text) # Use raw text as narrative
                    break # Process the final response

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
                        data=validated_narrator_output.model_dump() # Log the full parsed and validated output
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
            
            target_id = get_nested(intent, "target_id")
            target_state_data = {}
            if target_id:
                # Try to find the target in various places
                # 1. Simulacra
                target_state_data = get_nested(state, SIMULACRA_KEY, target_id, default=None)

                # 2. Static Objects (list of dicts)
                if not target_state_data:
                    static_objects_list = state.get("objects", [])
                    for obj in static_objects_list:
                        if isinstance(obj, dict) and obj.get("id") == target_id:
                            target_state_data = obj
                            break
                
                # 3. Ephemeral Objects in current location
                if not target_state_data and actor_location_id:
                    ephemeral_objects_list = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, "ephemeral_objects", default=[])
                    for eph_obj in ephemeral_objects_list:
                        if isinstance(eph_obj, dict) and eph_obj.get("id") == target_id:
                            target_state_data = eph_obj
                            break

                # 4. Ephemeral NPCs in current location
                if not target_state_data and actor_location_id:
                    ephemeral_npcs_list = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, "ephemeral_npcs", default=[])
                    for eph_npc in ephemeral_npcs_list:
                        if isinstance(eph_npc, dict) and eph_npc.get("id") == target_id:
                            target_state_data = eph_npc # NPCs might have a different structure, but treat as dict for now
                            break
                if not target_state_data: target_state_data = {} # Ensure it's a dict if not found
            
            # Fetch world feeds for the World Engine
            weather_summary_we = get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')

            intent_json = json.dumps(intent, indent=2)
            target_state_json = json.dumps(target_state_data, indent=2) if target_state_data else "N/A"
            location_state_json = json.dumps(location_state_data, indent=2)
            world_rules_json = json.dumps(world_rules, indent=2)
            # News is intentionally omitted from World Engine's direct prompt now
            prompt = f"""
Actor Name and ID: {actor_name} ({actor_id})
Current Location: {actor_location_id}
Current World Time: {time_for_world_engine_prompt}
Intent: {intent_json}
Current Weather: {weather_summary_we}
Target Entity State ({target_id or 'N/A'}): {target_state_json}
Location State: {location_state_json}
World Rules: {world_rules_json}

Resolve this intent based on your instructions and the provided context.
"""
            logger.debug(f"[WorldEngineLLM] Sending prompt to LLM for {actor_id}'s intent ({action_type}).")

            original_narration_agent_instruction = world_engine_agent.instruction
            # Remove DYNAMIC_CURRENT_NEWS from instruction replacement for World Engine
            world_engine_agent.instruction = original_narration_agent_instruction.replace("{DYNAMIC_CURRENT_TIME}", time_for_world_engine_prompt).replace("{DYNAMIC_CURRENT_WEATHER}", weather_summary_we)
            adk_runner.agent = world_engine_agent

            response_text = ""
            trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=prompt)])

            async for event_llm in adk_runner.run_async(
                user_id=USER_ID, session_id=session_id_to_use, new_message=trigger_content
            ):
                if event_llm.error_message:
                    logger.error(f"WorldLLM Error: {event_llm.error_message}")
                    outcome_description = f"Action failed due to LLM error: {event_llm.error_message}"
                    break 
                
                if event_llm.is_final_response() and event_llm.content:
                    if isinstance(event_llm.content, WorldEngineResponse): # ADK parsed it!
                        validated_data = event_llm.content
                        logger.debug(f"[WorldEngineLLM] ADK successfully parsed WorldEngineResponse schema for {actor_id}.")
                        parsed_resolution = validated_data.model_dump() # For logging and display
                    elif event_llm.content.parts: # Fallback to manual parsing
                        response_text = event_llm.content.parts[0].text
                        logger.debug(f"WorldLLM Final Raw Content: {response_text[:100]}...")
                        parsed_resolution_dict = parse_json_output_last(response_text.strip())
                        if parsed_resolution_dict:
                            validated_data = WorldEngineResponse.model_validate(parsed_resolution_dict)
                            parsed_resolution = validated_data.model_dump() # For logging and display
                        else:
                            logger.error(f"[WorldEngineLLM] Failed to parse JSON from WorldEngine (manual fallback) for {actor_id}. Response: {response_text}")
                            outcome_description = "Action failed due to internal error (JSON decode)."
                    # Common processing after successful parse (ADK or manual)
                    if validated_data:
                        logger.debug(f"[WorldEngineLLM] LLM response processed successfully for {actor_id}.")
                        outcome_description = validated_data.outcome_description
                        if live_display_object and parsed_resolution:
                            live_display_object.console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                            try: live_display_object.console.print(json.dumps(parsed_resolution, indent=2))
                            except TypeError: live_display_object.console.print(str(parsed_resolution)) 
                        _log_event(sim_time=current_sim_time, agent_id="WorldEngine", event_type="resolution", data=parsed_resolution or {})
                    break # Process the final response

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
                    data={"valid_action": False, "duration": 0.0, "results": {}, "outcome_description": final_outcome_desc}
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
            agent_state_to_check = get_nested(state, SIMULACRA_KEY, agent_id_to_check, default={})
            if not agent_state_to_check or agent_state_to_check.get("status") != "busy":
                continue
            
            # If agent is busy but not eligible for other reasons, clear its stored probability
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_interrupt_probability", None, logger)

            agent_name_to_check = get_nested(agent_state_to_check, "persona_details", "Name", default=agent_id_to_check)
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
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_interrupt_probability", interrupt_probability, logger)

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

                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id_to_check}.status", "idle", logger)
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id_to_check}.last_observation", interruption_text, logger)
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id_to_check}.pending_results", {}, logger)
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_action_end_time", current_sim_time_dit, logger)
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_action_description", "Interrupted by a dynamic event.", logger)
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id_to_check}.last_interjection_sim_time", current_sim_time_dit, logger)
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_interrupt_probability", None, logger) # Clear after interruption
                break 

        await asyncio.sleep(0.1) 


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
            # --- Phase 3: Location Awareness Enhancement ---
            agent_current_location_id_init = current_sim_state_init.get(CURRENT_LOCATION_KEY, DEFAULT_HOME_LOCATION_NAME)
            agent_personal_location_details_init = current_sim_state_init.get(LOCATION_DETAILS_KEY, "You are unsure of your exact surroundings.")
            world_location_data_init = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, agent_current_location_id_init, default={})
            world_location_description_init = world_location_data_init.get("description", "An unknown place.")
            current_world_time_init = state.get("world_time", 0.0)

            def get_entities_in_location_init(entity_type: str, location_id: Optional[str]) -> List[Dict[str, Any]]:
                entities = []
                if not location_id: return entities
                source_dict = state.get(entity_type, {})
                for entity_id_init_loop, entity_data_init in source_dict.items():
                    if entity_data_init.get('location') == location_id: # For objects
                        entities.append({"id": entity_id_init_loop, "name": entity_data_init.get("name", entity_id_init_loop)})
                    elif entity_type == SIMULACRA_KEY and get_nested(entity_data_init, "location") == location_id: # For simulacra
                        entities.append({"id": entity_id_init_loop, "name": get_nested(entity_data_init, "persona_details", "Name", default=entity_id_init_loop)})
                return entities

            ephemeral_objects_init = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, agent_current_location_id_init, "ephemeral_objects", default=[])
            objects_in_room_for_prompt_init = []
            for obj_data in ephemeral_objects_init: 
                objects_in_room_for_prompt_init.append({"id": obj_data.get("id"), "name": obj_data.get("name")})
            # Phase 4: Get connected locations for prompt
            connected_locations_init = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, agent_current_location_id_init, "connected_locations", default=[])
            agents_in_room_init = [a for a in get_entities_in_location_init(SIMULACRA_KEY, agent_current_location_id_init) if a["id"] != agent_id]
            
            raw_recent_narrative_init = state.get("narrative_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
            cleaned_recent_narrative_init = [re.sub(r'^\[T\d+\.\d+\]\s*', '', entry).strip() for entry in raw_recent_narrative_init]
            history_str_init = "\n".join(cleaned_recent_narrative_init)
            weather_summary = get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
            latest_news_headlines = [item.get('headline', '') for item in get_nested(state, 'world_feeds', 'news_updates', default=[])[:2]]
            news_summary = " ".join(h for h in latest_news_headlines if h) or "No major news."
            
            prompt_text_parts_init = [
                f"**Current State Info for {agent_name} ({agent_id}):**",
                f"- Persona: {current_sim_state_init.get('persona_details', {})}",
                f"- Current Location ID: {agent_current_location_id_init}",
                f"- Your understanding of this place: \"{agent_personal_location_details_init}\"",
                f"- Official Location Description: \"{world_location_description_init}\"",
                f"- Status: {current_sim_state_init.get('status', 'idle')} (You wake up and are ready to act.)",
                f"- Current Weather: {weather_summary}",
                f"- Recent News Snippet: {news_summary}",
                f"- Current Goal: {current_sim_state_init.get('goal', 'Determine goal.')}",
                f"- Current Time: {get_time_string_for_prompt(state, sim_elapsed_time_seconds=current_world_time_init)}",
                f"- Last Observation/Event: {current_sim_state_init.get('last_observation', 'None.')}",
                f"- Recent History:\n{history_str_init if history_str_init else 'None.'}",
                f"- Objects in area: {json.dumps(objects_in_room_for_prompt_init) if objects_in_room_for_prompt_init else 'None.'}",
                f"- Other Agents in area: {json.dumps(agents_in_room_init) if agents_in_room_init else 'None.'}",
                f"- Exits/Connections from this location: {json.dumps(connected_locations_init) if connected_locations_init else 'None observed.'}", # Phase 4
                f"- Current State: You wake up at home and are ready to act.",
                 "\nFollow your thinking process and provide your response ONLY in the specified JSON format."
            # --- End Phase 3 Change ---
            ]
            initial_trigger_text = "\n".join(prompt_text_parts_init)
            logger.debug(f"[{agent_name}] Sending initial context prompt as agent is idle.")
            current_sim_time = state.get("world_time", 0.0)
            time_for_simulacra = get_time_string_for_prompt(state, sim_elapsed_time_seconds=current_sim_time)

            # Fetch weather and news for instruction replacement
            weather_for_instruction = get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
            news_headlines_for_instruction = [item.get('headline', '') for item in get_nested(state, 'world_feeds', 'news_updates', default=[])[:1]] # Keep it brief for instruction
            news_for_instruction = news_headlines_for_instruction[0] if news_headlines_for_instruction else "No significant news."

            initial_trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=initial_trigger_text)])
            original_simulacra_agent_instruction = sim_agent.instruction
            sim_agent.instruction = original_simulacra_agent_instruction.replace("{DYNAMIC_CURRENT_TIME}", time_for_simulacra).replace("{DYNAMIC_CURRENT_WEATHER}", weather_for_instruction).replace("{DYNAMIC_CURRENT_NEWS}", news_for_instruction)
            adk_runner.agent = sim_agent
            async for event in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=initial_trigger_content):
                if event.error_message:
                    logger.error(f"[{agent_name}] LLM Error during initial prompt: {event.error_message}")
                    break
                
                if event.is_final_response() and event.content:
                    validated_intent: Optional[SimulacraIntentResponse] = None
                    if isinstance(event.content, SimulacraIntentResponse): # ADK parsed it!
                        validated_intent = event.content
                        logger.debug(f"[{agent_name}] ADK successfully parsed SimulacraIntentResponse schema (initial).")
                    elif event.content.parts: # Fallback to manual parsing
                        response_text = event.content.parts[0].text
                        response_dict = parse_json_output_last(response_text.strip())
                        if response_dict:
                            validated_intent = SimulacraIntentResponse.model_validate(response_dict)
                        else:
                            logger.error(f"[{agent_name}] Failed to parse JSON from Simulacra (initial, manual fallback). Response: {response_text}")
                    
                    if validated_intent:
                        if live_display_object:
                            live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_world_time_init:.1f}s", border_style="yellow", expand=False))
                            live_display_object.console.print(f"\n[{agent_name} Intent @ {current_world_time_init:.1f}s]")
                            live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                        await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": validated_intent.model_dump(exclude={'internal_monologue'})})
                        
                        # --- Log Simulacra Intent Event ---
                        _log_event(
                            sim_time=current_world_time_init,
                            agent_id=agent_id,
                            event_type="intent",
                            data=validated_intent.model_dump(exclude={'internal_monologue'}) # Log the intent data
                        )
                        _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "thinking", logger)
                    # else: Error already logged by parsing attempts

                    break 

        next_interjection_check_sim_time = state.get("world_time", 0.0) + AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS
        while True:
            await asyncio.sleep(AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS)
            current_sim_time_busy_loop = state.get("world_time", 0.0)
            agent_state_busy_loop = get_nested(state, SIMULACRA_KEY, agent_id, default={})
            current_status_busy_loop = agent_state_busy_loop.get("status")

            if current_status_busy_loop == "idle":
                # --- Generate and Log Percepts ---
                if perception_manager_global:
                    fresh_percepts = perception_manager_global.get_percepts_for_simulacrum(agent_id)
                    _log_event(
                        sim_time=current_sim_time_busy_loop,
                        agent_id=agent_id,
                        event_type="agent_perception_generated", # New event type for logging
                        data=fresh_percepts
                    )
                    logger.debug(f"[{agent_name}] Generated percepts: {json.dumps(fresh_percepts, sort_keys=True)[:250]}...") # Slightly more log
                else:
                    logger.warning(f"[{agent_name}] Perception Manager not available. Skipping fresh percept generation.")
                    fresh_percepts = {"error": "Perception system offline."} # Provide a fallback
                # --- End Perception Generation ---

                logger.debug(f"[{agent_name}] Status is idle. Proceeding to plan next action.")

                # --- Format Percepts for LLM Prompt ---
                perceptual_summary_for_prompt = "Perception system error or offline."
                if fresh_percepts and not fresh_percepts.get("error"):
                    loc_desc_from_percepts = fresh_percepts.get("location_description", "An unknown place.")
                    
                    visible_sim_text_parts = []
                    for sim_info in fresh_percepts.get("visible_simulacra", []):
                        visible_sim_text_parts.append(f"  - Simulacra: {sim_info.get('name', sim_info.get('id'))} (ID: {sim_info.get('id')}, Status: {sim_info.get('status', 'unknown')})")
                    visible_sim_str = "\n".join(visible_sim_text_parts) if visible_sim_text_parts else "  No other simulacra perceived."

                    visible_static_obj_text_parts = []
                    for static_obj_info in fresh_percepts.get("visible_static_objects", []):
                        visible_static_obj_text_parts.append(f"  - Static Object: {static_obj_info.get('name', static_obj_info.get('id'))} (ID: {static_obj_info.get('id')}, Desc: {static_obj_info.get('description', '')[:30]}...)")
                    visible_static_obj_str = "\n".join(visible_static_obj_text_parts) if visible_static_obj_text_parts else "  No static objects perceived."

                    visible_obj_text_parts = []
                    for obj_info in fresh_percepts.get("visible_ephemeral_objects", []): # Use renamed key
                        visible_obj_text_parts.append(f"  - Ephemeral Object: {obj_info.get('name', obj_info.get('id'))} (ID: {obj_info.get('id')})")
                    visible_eph_obj_str = "\n".join(visible_obj_text_parts) if visible_obj_text_parts else "  No ephemeral objects perceived."
                    
                    visible_npc_text_parts = []
                    for npc_info in fresh_percepts.get("visible_npcs", []): # Ephemeral NPCs
                        visible_npc_text_parts.append(f"  - Ephemeral NPC: {npc_info.get('name', npc_info.get('id'))} (ID: {npc_info.get('id')})")
                    visible_eph_npc_str = "\n".join(visible_npc_text_parts) if visible_npc_text_parts else "  No ephemeral NPCs perceived."

                    audible_events_text_parts = []
                    for sound_info in fresh_percepts.get("audible_events", []):
                        audible_events_text_parts.append(f"  - Sound ({sound_info.get('type', 'general')} from {sound_info.get('source_id', 'unknown')}): {sound_info.get('description', 'An indistinct sound.')}")
                    audible_env_str = "\n".join(audible_events_text_parts) if audible_events_text_parts else "  The environment is quiet."

                    perceptual_summary_for_prompt = (
                        f"Official Location Description: \"{loc_desc_from_percepts}\"\n"
                        f"Visible Entities:\n{visible_sim_str}\n{visible_static_obj_str}\n{visible_eph_obj_str}\n{visible_eph_npc_str}"
                    )
                # --- End Format Percepts ---

                # --- Phase 3: Location Awareness Enhancement ---
                agent_current_location_id_loop = agent_state_busy_loop.get(CURRENT_LOCATION_KEY, DEFAULT_HOME_LOCATION_NAME)
                agent_personal_location_details_loop = agent_state_busy_loop.get(LOCATION_DETAILS_KEY, "You are unsure of your exact surroundings.")
                world_location_data_loop = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, agent_current_location_id_loop, default={})
                world_location_description_loop = world_location_data_loop.get("description", "An unknown place.")
                def get_entities_in_location(entity_type: str, location_id: Optional[str]) -> List[Dict[str, Any]]:
                    entities = []
                    # This function is now largely superseded by PerceptionManager for simulacra/objects/NPCs.
                    # It might still be useful for other entity types if you add them.
                    # For now, we'll rely on the PerceptionManager's output.
                    return entities
                
                # Phase 4: Get connected locations for prompt
                connected_locations_loop = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, agent_current_location_id_loop, "connected_locations", default=[])
                raw_recent_narrative = state.get("narrative_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
                cleaned_recent_narrative = [re.sub(r'^\[T\d+\.\d+\]\s*', '', entry).strip() for entry in raw_recent_narrative]
                history_str = "\n".join(cleaned_recent_narrative)
                weather_summary_loop = get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
                latest_news_headlines_loop = [item.get('headline', '') for item in get_nested(state, 'world_feeds', 'news_updates', default=[])[:2]]
                news_summary_loop = " ".join(h for h in latest_news_headlines_loop if h) or "No major news."
                prompt_text_parts = [
                     f"**Current State Info for {agent_name} ({agent_id}):**",
                     f"- Persona: {agent_state_busy_loop.get('persona_details', {})}", # Use agent_state_busy_loop
                     f"- Current Location ID: {agent_current_location_id_loop or 'Unknown'}",
                     f"- Your Personal Understanding of this Location: \"{agent_personal_location_details_loop}\"",
                     f"- Perceived Environment:\n{perceptual_summary_for_prompt}", # Use the formatted percepts
                     f"- Status: {agent_state_busy_loop.get('status', 'idle')} (You should act now)",
                     f"- Current Weather: {weather_summary_loop}",
                     f"- Recent News Snippet: {news_summary_loop}",
                     f"- Current Goal: {agent_state_busy_loop.get('goal', 'Determine goal.')}",
                     f"- Current Time: {get_time_string_for_prompt(state, sim_elapsed_time_seconds=current_sim_time_busy_loop)}",
                     f"- Last Observation/Event: {agent_state_busy_loop.get('last_observation', 'None.')}",
                     f"- Audible Environment:\n{audible_env_str}", # Add audible environment
                     f"- Recent Narrative History:\n{history_str if history_str else 'None.'}",
                     f"- Exits/Connections from this location: {json.dumps(connected_locations_loop) if connected_locations_loop else 'None observed.'}", # Phase 4
                     "\nFollow your thinking process and provide your response ONLY in the specified JSON format."
                # --- End Phase 3 Change ---
                ]
                prompt_text = "\n".join(prompt_text_parts)
                logger.debug(f"[{agent_name}] Sending subsequent prompt.")
                
                # Also update instruction for subsequent calls
                time_for_simulacra_loop = get_time_string_for_prompt(state, sim_elapsed_time_seconds=current_sim_time_busy_loop)
                weather_for_instruction_loop = weather_summary_loop # Already fetched for prompt
                news_for_instruction_loop = news_summary_loop.split('.')[0] if news_summary_loop else "No significant news." # First sentence or default
                sim_agent.instruction = original_simulacra_agent_instruction.replace("{DYNAMIC_CURRENT_TIME}", time_for_simulacra_loop).replace("{DYNAMIC_CURRENT_WEATHER}", weather_for_instruction_loop).replace("{DYNAMIC_CURRENT_NEWS}", news_for_instruction_loop)

                trigger_content = genai_types.UserContent(parts=[genai_types.Part(text=prompt_text)])
                adk_runner.agent = sim_agent 
                async for event in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=trigger_content):
                    if event.error_message:
                        logger.error(f"[{agent_name}] LLM Error during subsequent prompt: {event.error_message}")
                        break
                    
                    if event.is_final_response() and event.content:
                        validated_intent: Optional[SimulacraIntentResponse] = None
                        if isinstance(event.content, SimulacraIntentResponse): # ADK parsed it!
                            validated_intent = event.content
                            logger.debug(f"[{agent_name}] ADK successfully parsed SimulacraIntentResponse schema (subsequent).")
                        elif event.content.parts: # Fallback to manual parsing
                            response_text = event.content.parts[0].text
                            response_dict = parse_json_output_last(response_text.strip())
                            if response_dict:
                                validated_intent = SimulacraIntentResponse.model_validate(response_dict)
                            else:
                                logger.error(f"[{agent_name}] Failed to parse JSON from Simulacra (subsequent, manual fallback). Response: {response_text}")
                        
                        if validated_intent:
                            if live_display_object:
                                live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_sim_time_busy_loop:.1f}s", border_style="yellow", expand=False))
                                live_display_object.console.print(f"\n[{agent_name} Intent @ {current_sim_time_busy_loop:.1f}s]")
                                live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                            await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": validated_intent.model_dump(exclude={'internal_monologue'})})
                            
                            # --- Log Simulacra Intent Event ---
                            _log_event(
                                sim_time=current_sim_time_busy_loop,
                                agent_id=agent_id,
                                event_type="intent",
                                data=validated_intent.model_dump(exclude={'internal_monologue'}) # Log the intent data
                            )
                            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "thinking", logger)
                        # else: Error already logged by parsing attempts

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
                        adk_runner.agent = sim_agent 
                        async for event in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=reflection_trigger_content):
                            if event.error_message:
                                logger.error(f"[{agent_name}] LLM Error during reflection prompt: {event.error_message}")
                                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", original_status_before_reflection, logger)
                                break
                            
                            if event.is_final_response() and event.content:
                                validated_reflection_intent: Optional[SimulacraIntentResponse] = None
                                if isinstance(event.content, SimulacraIntentResponse): # ADK parsed it!
                                    validated_reflection_intent = event.content
                                    logger.debug(f"[{agent_name}] ADK successfully parsed SimulacraIntentResponse schema (reflection).")
                                elif event.content.parts: # Fallback to manual parsing
                                    response_text = event.content.parts[0].text
                                    response_dict = parse_json_output_last(response_text.strip())
                                    if response_dict:
                                        validated_reflection_intent = SimulacraIntentResponse.model_validate(response_dict)
                                
                                if validated_reflection_intent:
                                    if validated_reflection_intent.action_type == "continue_current_task":
                                        logger.info(f"[{agent_name}] Reflection: Chose to continue. Monologue: {validated_reflection_intent.internal_monologue[:50]}...")
                                        _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", original_status_before_reflection, logger)
                                        # Probability remains as it was, task continues
                                    else:
                                        logger.info(f"[{agent_name}] Reflection: Chose to '{validated_reflection_intent.action_type}'. Monologue: {validated_reflection_intent.internal_monologue[:50]}...")
                                        await event_bus.put({ "type": "intent_declared", "actor_id": agent_id, "intent": validated_reflection_intent.model_dump(exclude={'internal_monologue'}) })
                                        _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None, logger) # Clear as action is changing
                                        _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "thinking", logger)
                                else: # Parsing failed (ADK and manual)
                                    logger.error(f"[{agent_name}] Error processing reflection response (parsing failed). Staying busy. Raw response was logged if available.")
                                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", original_status_before_reflection, logger)
                                break 

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
    logger.info(f"ADK Session created: {adk_session_id}.")
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
        sim_id=first_sim_id, persona_name=first_persona_name, # Generic actor context for agent creation
        world_type=current_world_type, sub_genre=current_sub_genre
    )
    narration_agent_instance = create_narration_llm_agent(
        sim_id=first_sim_id, persona_name=first_persona_name, # Generic actor context
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
            tasks.append(asyncio.create_task(
                socket_server_task(
                    state=state,
                    narration_queue=narration_queue,
                    world_mood=world_mood_global,
                    simulation_time_getter=get_current_sim_time
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

# Get a combination with one general, one lighting, and one color style
# random_style = get_random_style_combination(num_general=1, num_lighting=1, num_color=1, include_technique=False)
# print(f"Random Style Combination 1: {random_style}")

# Get a combination with two styles from any category
# random_style_2 = get_random_style_combination(num_general=1, num_lighting=1, num_color=1, num_technique=1)
# print(f"Random Style Combination 2: {random_style_2}")

# Get a combination focusing only on lighting and technique
# random_style_3 = get_random_style_combination(include_general=False, include_color=False, num_lighting=1, num_technique=1)
# print(f"Random Style Combination 3: {random_style_3}")

# Get just one random style from any category (approx)
# random_style_4 = get_random_style_combination(num_general=1, num_lighting=1, num_color=1, num_technique=1)
# print(f"Random Style Combination 4: {random_style_4}")


async def narrative_image_generation_task():
    """
    Periodically generates an image based on the latest narrative log entry.
    Logs the filename to the event logger.
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

    # Initialize Bluesky client if enabled
    bluesky_client = None
    if ENABLE_BLUESKY_POSTING:
        bluesky_client = Client()

    # Updated example call to use new style categories for striking real-world photos
    random_style = get_random_style_combination(
        include_general=True, num_general=0,  # Often better to be specific than too general
        include_lighting=True, num_lighting=1,
        include_color=True, num_color=1,
        include_technique=True, num_technique=1,
        include_composition=True, num_composition=1,
        include_atmosphere=True, num_atmosphere=1
    )
    logger.info(f"[NarrativeImageGenerator] Random style for image generation: {random_style}")
    # Add a random delay before starting the image generation loop
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

        latest_narrative_full = narrative_log_entries[-1] # Get the actual latest narrative entry
        # Strip the timestamp like "[T123.4] " from the narrative for a cleaner image prompt
        original_narrative_prompt_text = re.sub(r'^\[T\d+\.\d+\]\s*', '', latest_narrative_full).strip()

        if not original_narrative_prompt_text:
            logger.debug("[NarrativeImageGenerator] Latest narrative entry is empty after stripping timestamp. Skipping.")
            continue

        current_sim_time_for_filename = state.get("world_time", 0.0)

        # Attempt to extract an actor name from the narrative for a more personal prompt.
        # This is a heuristic and assumes the narrative often features the actor's name.
        actor_name_in_narrative = "the observer" # Default, more generic
        match = re.match(r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)", original_narrative_prompt_text)
        if match:
            # Avoid common sentence-starting words that aren't names
            common_words_to_avoid = ["As", "The", "A", "An", "It", "He", "She", "They", "Then", "Suddenly", "During", "While"]
            if match.group(1) not in common_words_to_avoid:
                actor_name_in_narrative = match.group(1)
        
        # Get current time string, weather, and mood for grounding
        time_string_for_image_prompt = get_time_string_for_prompt(state, sim_elapsed_time_seconds=current_sim_time_for_filename)
        weather_condition_for_image_prompt = get_nested(state, 'world_feeds', 'weather', 'condition', default='The weather is unknown.')
        current_world_mood_ig = state.get(WORLD_TEMPLATE_DETAILS_KEY, {}).get('mood', world_mood_global)
        
        # --- LLM Call 1: Refine narrative for image generation ---
        # Use the standard text model (MODEL_NAME from config) for this refinement task
        refined_narrative_for_image = original_narrative_prompt_text # Fallback
        try:
            refinement_llm = genai.GenerativeModel(MODEL_NAME)
            prompt_for_refinement = f"""You are an expert at transforming narrative text into concise, visually descriptive prompts ideal for an image generation model. Your goal is to focus on a single, clear subject, potentially with a naturally blurred background.

Original Narrative Context: "{original_narrative_prompt_text}"
Current Time: "{time_string_for_image_prompt}"
Current Weather: "{weather_condition_for_image_prompt}"
World Mood: "{current_world_mood_ig}"

Instructions for Refinement:
1.  Identify a single, compelling visual element or a very brief, static moment from the 'Original Narrative Context'.
2.  Describe this single subject clearly and vividly. Use descriptive language for the subject and its relationship to any implied background.
3.  If appropriate, suggest a composition that would naturally lead to a blurred background (e.g., "A close-up of...", "A detailed shot of...", "A lone figure with the background softly blurred...").
4.  Keep the refined description concise (preferably 1-2 sentences).
5.  The refined description should be purely visual and directly usable as an image prompt.
6.  Do NOT include any instructions for the image generation model itself (like "Generate an image of..."). Just provide the refined descriptive text.

Examples of good refined descriptions:
- "A close-up photograph of a single red rose in a glass vase, with a soft, blurred background."
- "A lone figure walking along a cobblestone street on a foggy morning, the background intentionally blurred to emphasize the person."
- "A detailed shot of a vintage leather-bound book lying open on a wooden table, with a shallow depth of field creating a blurred background."

Refined Visual Description:
"""
            logger.info(f"[NarrativeImageGenerator] Requesting LLM to refine narrative for image prompt. Original: '{original_narrative_prompt_text[:100]}...'")
            response_refinement = await refinement_llm.generate_content_async(prompt_for_refinement)
            if response_refinement.text:
                refined_narrative_for_image = response_refinement.text.strip()
                logger.info(f"[NarrativeImageGenerator] LLM refined narrative to: '{refined_narrative_for_image}'")
            else:
                logger.warning("[NarrativeImageGenerator] LLM refinement call returned no text. Using original narrative.")
        except Exception as e_refine:
            logger.error(f"[NarrativeImageGenerator] Error during LLM narrative refinement: {e_refine}. Using original narrative.", exc_info=True)
        # --- End of LLM Call 1 ---

        # --- LLM Call 2: Image Generation ---
# World Mood: "{current_world_mood_ig}"
        prompt_for_image = f"""
Generate a high-quality, visually appealing, **photo-realistic** photograph of a scene or subject directly related to the following narrative context, as if captured by {actor_name_in_narrative}.
Narrative Context: "{refined_narrative_for_image}" # Use the refined narrative
Style: "{random_style}"

Instructions for the Image:
The image should feature:
-   **Time of Day:** Reflect the lighting and atmosphere typical of "{time_string_for_image_prompt}".
-   **Weather:** Depict the conditions described by "{weather_condition_for_image_prompt}".
-   **Season:** Infer the season from the date in the time string and depict it (e.g., foliage, clothing).

-   A clear subject directly related to the Narrative Context.
-   Lighting, composition, and focus that give it the aesthetic of a professional, high-engagement social media photograph (like those popular on Instagram or Twitter).
-   Details that align with the World Mood.
-   A composition that is balanced and aesthetically pleasing, **with a strong emphasis on a clear, well-defined subject within a natural or slightly blurred background.**

Style:
-   The overall aesthetic should be that of a **modern, editorial-quality, photo-realistic photograph** suitable for a popular social media feed, emphasizing photographic quality, clarity, **authentic textures, and natural colors**.
-   The image MUST represent what {actor_name_in_narrative} is seeing or a photograph they would take of their environment or an object of interest. This means it should be from a first-person perspective or a shot of the scene/subject in front of them.
-   Consider an aspect ratio common on social media, such as **4:5 (portrait) or 1:1 (square) as a primary preference to ensure optimal display on mobile feeds.**

Crucial Exclusions:
-   **The image itself must NOT contain any digital overlays, app interfaces, Instagram/Twitter frames, borders, like buttons, comment icons, usernames, text captions, or any other UI elements.**
-   **No watermarks or logos should be embedded in the image.**
-   The output should be the pure photographic image of the subject as described.
-   **The actor ({actor_name_in_narrative}) themselves MUST NOT be visible in the image. No selfies or third-person shots of the actor.**

Generate this image.
"""
        logger.info(f"[NarrativeImageGenerator] Requesting image generation with prompt (T{current_sim_time_for_filename:.1f}): \"{refined_narrative_for_image}\"")

        try:
            response = await asyncio.to_thread(
                client.models.generate_images,
                model=IMAGE_GENERATION_MODEL_NAME,
                prompt=prompt_for_image,  # Pass as a list for contents
                config=genai_types.GenerateImagesConfig(
                    number_of_images=1,
                )
            )

            image_generated = False # Mark success
            saved_image_path_for_social_post: Optional[str] = None # Store path of successfully saved image

            for generated_image in response.generated_images:
                try:
                    image = Image.open(BytesIO(generated_image.image.image_bytes))
                    
                    # Create a unique filename
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    sim_time_str = f"T{current_sim_time_for_filename:.0f}"
                    image_filename = f"narrative_{sim_time_str}_{timestamp_str}.png"
                    image_path = os.path.join(IMAGE_GENERATION_OUTPUT_DIR, image_filename)
                    
                    image.save(image_path)
                    logger.info(f"[NarrativeImageGenerator] Successfully generated and saved image: {image_path}")
                    image_generated = True # Mark success
                    saved_image_path_for_social_post = image_path # Store for Bluesky

                    # --- Log Image Generation Event ---
                    _log_event(
                        sim_time=current_sim_time_for_filename,
                        agent_id="ImageGenerator", # Or a more specific ID if you have one
                        event_type="image_generation",
                        data={"image_filename": image_filename, "prompt_snippet": refined_narrative_for_image} # Log refined prompt
                    )
                    break # Assuming one image per request
                except Exception as e_img_proc:
                    logger.error(f"[NarrativeImageGenerator] Error processing image data (PIL/save): {e_img_proc}", exc_info=True)
            
            # --- Bluesky Posting (occurs if image_generated is True and we have a path) ---
            if ENABLE_BLUESKY_POSTING and bluesky_client and image_generated and saved_image_path_for_social_post:
                logger.info(f"[NarrativeImageGenerator] Attempting to post image to Bluesky: {saved_image_path_for_social_post}")
                try:
                    # Login to Bluesky (idempotent check)
                    if not bluesky_client.me:
                        logger.info("[NarrativeImageGenerator] Logging into Bluesky...")
                        bluesky_client.login(BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)
                        logger.info("[NarrativeImageGenerator] Bluesky login successful.")

                    # --- Image Resizing/Compression for Bluesky ---
                    BLUESKY_MAX_IMAGE_SIZE_BYTES = 976 * 1024 # 976KB as a safe upper limit
                    image_bytes_for_upload = None
                    original_file_size = os.path.getsize(saved_image_path_for_social_post)

                    if original_file_size > BLUESKY_MAX_IMAGE_SIZE_BYTES:
                        logger.warning(f"[NarrativeImageGenerator] Image {saved_image_path_for_social_post} ({original_file_size / (1024*1024):.2f}MB) exceeds Bluesky limit ({BLUESKY_MAX_IMAGE_SIZE_BYTES / (1024*1024):.2f}MB). Attempting to compress/resize.")
                        try:
                            img_pil = Image.open(saved_image_path_for_social_post)
                            # Convert to RGB if it's RGBA (PNGs can have alpha) to save as JPEG
                            if img_pil.mode == 'RGBA':
                                img_pil = img_pil.convert('RGB')

                            temp_image_buffer = BytesIO()
                            quality = 85 # Start with a decent quality for JPEG
                            
                            # Attempt to save with decreasing quality
                            while quality >= 50:
                                temp_image_buffer.seek(0) # Reset buffer
                                temp_image_buffer.truncate() # Clear buffer
                                img_pil.save(temp_image_buffer, format="JPEG", quality=quality, optimize=True)
                                if temp_image_buffer.tell() <= BLUESKY_MAX_IMAGE_SIZE_BYTES:
                                    logger.info(f"[NarrativeImageGenerator] Compressed image to {temp_image_buffer.tell() / 1024:.2f}KB with JPEG quality {quality}.")
                                    image_bytes_for_upload = temp_image_buffer.getvalue()
                                    break
                                quality -= 10
                            
                            # If still too large after quality reduction, try resizing (optional, can be added if needed)
                            # For now, if quality reduction isn't enough, we might fail or skip posting this image.
                            if not image_bytes_for_upload:
                                logger.error(f"[NarrativeImageGenerator] Could not compress {saved_image_path_for_social_post} sufficiently for Bluesky. Final attempt size: {temp_image_buffer.tell() / 1024:.2f}KB.")
                                # Optionally, you could try resizing here as a further step.
                                # For simplicity, we'll skip posting if compression alone isn't enough.

                        except Exception as e_compress:
                            logger.error(f"[NarrativeImageGenerator] Error during image compression for Bluesky: {e_compress}", exc_info=True)
                    else:
                        # Image is already small enough, read its bytes
                        with open(saved_image_path_for_social_post, 'rb') as f:
                            image_bytes_for_upload = f.read()
                    
                    if not image_bytes_for_upload:
                        logger.error(f"[NarrativeImageGenerator] Failed to prepare image for Bluesky (too large or compression error). Skipping post for this image.")
                        continue # Skip to the next image generation cycle

                    # Prepare post text
                    alt_text_for_image = f"Image from simulation at T{current_sim_time_for_filename:.0f}s: {original_narrative_prompt_text}"
                    post_text_raw = alt_text_for_image # Use refined narrative for social post text as well
                    # alt_text_for_image = f"Image from simulation at T{current_sim_time_for_filename:.0f}s: {refined_narrative_for_image[:250]}..."
                    
                    max_text_length = SOCIAL_POST_TEXT_LIMIT
                    hashtags_str = SOCIAL_POST_HASHTAGS
                    estimated_hashtag_space = len(hashtags_str) + (len(hashtags_str.split()) * 1) 
                    effective_text_limit = max(0, max_text_length - estimated_hashtag_space)
                    post_text = post_text_raw[:effective_text_limit]
                    if len(post_text_raw) > effective_text_limit:
                        post_text = post_text.rsplit(' ', 1)[0] + '...' if ' ' in post_text.rsplit(' ', 1)[0] else post_text + '...' # Ensure rsplit doesn't fail on no space
                    final_post_content = f"{post_text}\n\n{hashtags_str}".strip()

                    logger.info(f"[NarrativeImageGenerator] Sending image and text to Bluesky: '{final_post_content[:50]}...'")
                    post_response = bluesky_client.send_image(text=final_post_content, image=image_bytes_for_upload, image_alt=alt_text_for_image)
                    logger.info(f"[NarrativeImageGenerator] Successfully posted to Bluesky. Post URI: {post_response.uri}")

                except Exception as e_bsky:
                    logger.error(f"[NarrativeImageGenerator] Error posting image to Bluesky: {e_bsky}", exc_info=True)
            elif ENABLE_BLUESKY_POSTING and not bluesky_client:
                 logger.warning("[NarrativeImageGenerator] Bluesky posting is enabled but client failed to initialize.")
            
            if not image_generated:
                logger.warning(f"[NarrativeImageGenerator] No image data found in response for prompt: {prompt_for_image}...")
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.text:
                             logger.warning(f"[NarrativeImageGenerator] Model text response (if any): {part.text}")


        except Exception as e:
            logger.error(f"[NarrativeImageGenerator] Error during image generation API call: {e}")
            if hasattr(e, 'response') and e.response: # type: ignore
                 logger.error(f"[NarrativeImageGenerator] API Response (if available): {e.response}") # type: ignore
