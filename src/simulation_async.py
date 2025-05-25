# src/simulation_async.py - Core Simulation Orchestrator
import asyncio
import datetime
import glob  # Keep for run_simulation profile verification
import json
import logging  # Explicitly import logging for type hinting
import os
import random  # Keep for interjection logic in dynamic_interruption_task
import re
import string  # For default sim_id generation if needed
import sys
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import google.generativeai as genai  # For direct API config
from atproto import Client, models  # Import atproto client and models
from google import genai as genai_image  # For direct API config
from google.adk.agents import (BaseAgent, LlmAgent,  # ADK Workflow Agents
                               LoopAgent, ParallelAgent, SequentialAgent)
from google.adk.agents.invocation_context import \
    InvocationContext  # For ADK agent development
from google.adk.events import Event, EventActions  # For ADK agent development
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import agent_tool  # For wrapping agents as tools
from google.adk.tools import google_search
from google.genai import types as genai_types
from PIL import Image
from pydantic import (  # Keep for models defined here if any, or by auxiliary tasks
    BaseModel, Field, ValidationError, ValidationInfo, field_validator)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from .socket_server import socket_server_task

console = Console() # Keep a global console for direct prints if needed by run_simulation

from .agents import (  # Import the main architecture builder and individual agent creators
    create_narration_llm_agent, create_search_llm_agent,
    create_simulacra_llm_agent, create_simulation_architecture,
    create_world_engine_llm_agent)
# Import from our new/refactored modules
from .config import (  # AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS used by old simulacra_task
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
    MAX_SIMULATION_TICKS, MAX_SIMULATION_TIME, MEMORY_LOG_CONTEXT_LENGTH,
    MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK, MODEL_NAME,
    PROB_INTERJECT_AS_SELF_REFLECTION, RANDOM_SEED, SEARCH_AGENT_MODEL_NAME,
    SIMULACRA_KEY, SIMULACRA_PROFILES_KEY, SIMULATION_SPEED_FACTOR,
    SOCIAL_POST_HASHTAGS, SOCIAL_POST_TEXT_LIMIT, STATE_DIR, UPDATE_INTERVAL,
    USER_ID, WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS, WORLD_STATE_KEY,
    WORLD_TEMPLATE_DETAILS_KEY)
from .core_tasks import time_manager_task
from .loop_utils import (get_nested, load_json_file,
                         load_or_initialize_simulation, parse_json_output_last,
                         save_json_file)
# Pydantic models might still be used by auxiliary tasks or for type hinting
from .models import (SimulacraIntentResponse,
                     WorldEngineResponse)
from .simulation_utils import (  # _update_state_value removed as it's in simulation_utils
    generate_table, get_random_style_combination, get_time_string_for_prompt)
from .state_loader import parse_location_string

logger = logging.getLogger(__name__)

# --- Core Components (Module Scope) ---
state: Dict[str, Any] = {} # Global state dictionary, will be synced with ADK session
event_logger_global: Optional[logging.Logger] = None

adk_session_service: Optional[InMemorySessionService] = None
adk_session_id: Optional[str] = None
adk_session: Optional[Session] = None
adk_runner: Optional[Runner] = None
adk_memory_service: Optional[InMemoryMemoryService] = None

world_mood_global: str = "The familiar, everyday real world; starting the morning routine at home."
live_display_object: Optional[Live] = None

def get_current_sim_time():
    return state.get("world_time", 0.0)

# --- Helper for Event Logging ---
def _log_event(sim_time: float, agent_id: str, event_type: str, data: Dict[str, Any]):
    """Logs a structured event to the dedicated event logger."""
    if event_logger_global:
        log_entry = {
            "sim_time_s": round(sim_time, 2),
            "agent_id": agent_id,
            "event_type": event_type,
            "data": data
        }
        try:
            event_logger_global.info(json.dumps(log_entry))
        except Exception as e:
            logger.error(f"Failed to log event (type: {event_type}, agent: {agent_id}) to event log: {e}", exc_info=True)


async def run_simulation(
    instance_uuid_arg: Optional[str] = None,
    location_override_arg: Optional[str] = None,
    mood_override_arg: Optional[str] = None,
    event_logger_instance: Optional[logging.Logger] = None
    ):
    global adk_session_service, adk_session_id, adk_session, adk_runner, adk_memory_service
    global state, live_display_object, world_mood_global
    global event_logger_global

    console.rule("[bold green]Starting Async Simulation with ADK Workflow[/]")
    event_logger_global = event_logger_instance

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

    final_active_sim_ids = state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
    if not final_active_sim_ids:
         logger.critical("No active simulacra available after state load. Cannot proceed.")
         console.print("[bold red]Error:[/bold red] No verified Simulacra available.")
         sys.exit(1)
    logger.info(f"Initialization complete. Instance {world_instance_uuid} ready with {len(final_active_sim_ids)} simulacra.")
    console.print(f"Running simulation with: {', '.join(final_active_sim_ids)}")

    # --- Initialize ADK Session and Runner ---
    adk_session_service = InMemorySessionService()
    adk_session_id = f"sim_master_loop_session_{world_instance_uuid}"
    
    # Populate Python state with necessary config for ADK agents
    state["config"] = {
        "UPDATE_INTERVAL": UPDATE_INTERVAL,
        "SIMULATION_SPEED_FACTOR": SIMULATION_SPEED_FACTOR,
        "USER_ID": USER_ID,
        "MODEL_NAME": MODEL_NAME,
        "SEARCH_AGENT_MODEL_NAME": SEARCH_AGENT_MODEL_NAME,
        "MEMORY_LOG_CONTEXT_LENGTH": MEMORY_LOG_CONTEXT_LENGTH,
        "WORLD_TEMPLATE_DETAILS_KEY": WORLD_TEMPLATE_DETAILS_KEY,
        "SIMULACRA_KEY": SIMULACRA_KEY,
        "LOCATION_DETAILS_KEY": LOCATION_DETAILS_KEY,
        "WORLD_STATE_KEY": WORLD_STATE_KEY,
        "ACTIVE_SIMULACRA_IDS_KEY": ACTIVE_SIMULACRA_IDS_KEY,
        "MAX_SIMULATION_TICKS": MAX_SIMULATION_TICKS,
        "WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS": WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS,
        "IMAGE_GENERATION_INTERVAL_REAL_SECONDS": IMAGE_GENERATION_INTERVAL_REAL_SECONDS,
        "DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS": DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS,
        "INTERJECTION_COOLDOWN_SIM_SECONDS": INTERJECTION_COOLDOWN_SIM_SECONDS,
        "MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK": MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK,
        "DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS": DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS,
        "DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION": DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION,
        "DYNAMIC_INTERRUPTION_MIN_PROB": DYNAMIC_INTERRUPTION_MIN_PROB,
        "DYNAMIC_INTERRUPTION_MAX_PROB_CAP": DYNAMIC_INTERRUPTION_MAX_PROB_CAP,
    }
    state["world_mood_global"] = world_mood_global
    state["simulacra_profiles"] = state.get(SIMULACRA_KEY, {})

    # --- Create Individual LLM Agent Instances ---
    simulacra_llm_agents_list = []
    # This map might still be useful for auxiliary tasks or debugging
    simulacra_agents_map: Dict[str, LlmAgent] = {}
    for sim_id_val in final_active_sim_ids:
        sim_profile_data = state.get(SIMULACRA_KEY, {}).get(sim_id_val, {})
        persona_name = get_nested(sim_profile_data, "persona_details", "Name", default=sim_id_val)
        sim_agent_instance = create_simulacra_llm_agent(sim_id_val, persona_name, world_mood=world_mood_global)
        simulacra_llm_agents_list.append(sim_agent_instance)
        simulacra_agents_map[sim_id_val] = sim_agent_instance
    
    world_type = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'world_type', default="real")
    sub_genre = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'sub_genre', default="realtime")
    world_engine_llm = create_world_engine_llm_agent("system", "WorldEngine", world_type, sub_genre)
    narration_llm = create_narration_llm_agent("system", "Narrator", world_mood_global, world_type, sub_genre)
    search_llm = create_search_llm_agent()

    # --- Create the Multi-Agent Architecture ---
    master_loop_agent = create_simulation_architecture(
        simulacra_llm_agents=simulacra_llm_agents_list,
        world_engine_llm=world_engine_llm,
        narration_llm=narration_llm,
        search_llm_agent=search_llm,
        max_ticks=state.get("config", {}).get("MAX_SIMULATION_TICKS", MAX_SIMULATION_TICKS)
    )
    logger.info(f"ADK Multi-agent simulation architecture created with {len(simulacra_llm_agents_list)} simulacra.")

    adk_session = await adk_session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=adk_session_id, state=state
    )
    logger.info(f"ADK Session for MasterLoopAgent created: {adk_session_id} with initial state from Python state.")

    adk_runner = Runner(
        agent=master_loop_agent, 
        app_name=APP_NAME,
        session_service=adk_session_service,
        memory_service=adk_memory_service
    )
    logger.info(f"Main ADK Runner initialized with MasterLoopAgent.")

    # --- Synchronize Simulation State with ADK Session ---
    def sync_state_to_adk_session(current_session_obj: Session):
        """Sync current Python state to ADK session state."""
        global state # state is the global Python state dictionary
        try:
            current_session_obj.state.update({ # Use current_session_obj passed as argument
                "world_time": state.get("world_time", 0.0),
                SIMULACRA_KEY: state.get(SIMULACRA_KEY, {}), # Use constant
                WORLD_STATE_KEY: state.get(WORLD_STATE_KEY, {}), # Use constant
                "world_feeds": state.get("world_feeds", {}),
                "narrative_log": state.get("narrative_log", []),
                ACTIVE_SIMULACRA_IDS_KEY: state.get(ACTIVE_SIMULACRA_IDS_KEY, []), # Use constant
                WORLD_TEMPLATE_DETAILS_KEY: state.get(WORLD_TEMPLATE_DETAILS_KEY, {}), # Use constant
                "config": state.get("config", {}),
                "world_mood_global": state.get("world_mood_global", "neutral"), # Get from global state
                "pending_simulation_events": state.get("pending_simulation_events", []),
                "event_log": state.get("event_log", []),
                "world_instance_uuid": state.get("world_instance_uuid", "unknown")
            })
            logger.debug("Python state synced to ADK session")
        except Exception as e:
            logger.error(f"Error syncing Python state to ADK session: {e}")
    
    def sync_state_from_adk_session(current_session_obj: Session):
        """Sync ADK session state back to Python state."""
        global state # state is the global Python state dictionary
        try:
            if current_session_obj and current_session_obj.state: # Use current_session_obj
                state.update({ # Update the global Python state using current_session_obj
                    "world_time": current_session_obj.state.get("world_time", state.get("world_time", 0.0)),
                    SIMULACRA_KEY: current_session_obj.state.get(SIMULACRA_KEY, state.get(SIMULACRA_KEY, {})),
                    WORLD_STATE_KEY: current_session_obj.state.get(WORLD_STATE_KEY, state.get(WORLD_STATE_KEY, {})),
                    "world_feeds": current_session_obj.state.get("world_feeds", state.get("world_feeds", {})),
                    "narrative_log": current_session_obj.state.get("narrative_log", state.get("narrative_log", [])),
                    "pending_simulation_events": current_session_obj.state.get("pending_simulation_events", state.get("pending_simulation_events", [])),
                    "event_log": current_session_obj.state.get("event_log", state.get("event_log", []))
                    # Do not sync back config, world_mood_global, active_sim_ids, world_template_details from ADK to global state
                    # as these are primarily set up from Python state initially.
                })
                logger.debug("ADK session state synced back to Python state")
        except Exception as e:
            logger.error(f"Error syncing ADK session state to Python state: {e}")
    
    # Initial sync is handled by session creation with `state=state`
    logger.info("Initial ADK session state populated from Python state.")

    auxiliary_tasks = [] # Renamed from tasks to avoid conflict
    final_state_path = os.path.join(STATE_DIR, f"simulation_state_{world_instance_uuid}.json")

    try:
        def get_current_table_for_live():
            # Queues are removed, pass 0 or modify generate_table if it expects qsizes
            return generate_table(state, 0, 0) 

        with Live(get_current_table_for_live(), console=console, refresh_per_second=1.0/UPDATE_INTERVAL, vertical_overflow="visible") as live:
            live_display_object = live
            
            # --- Start Auxiliary Tasks (Non-ADK managed for Phase 1) ---
            auxiliary_tasks.append(asyncio.create_task(
                socket_server_task(
                    state=state,
                    world_mood=world_mood_global,
                    simulation_time_getter=get_current_sim_time
                ), 
                name="SocketServer"
            ))
            
            # time_manager_task's role is reduced; primarily for display updates in Phase 1
            auxiliary_tasks.append(asyncio.create_task(
                time_manager_task(
                    current_state=state,
                    event_bus_qsize_func=lambda: 0, 
                    narration_qsize_func=lambda: 0,
                    live_display=live, 
                    logger_instance=logger
                ), name="TimeManager"))

            # --- Main ADK Multi-Agent Loop ---
            master_loop_task = asyncio.create_task(
                run_adk_master_loop(adk_runner, adk_session, sync_state_to_adk_session, sync_state_from_adk_session, live),
                name="ADKMasterLoop"
            )
            
            all_tasks_to_monitor = auxiliary_tasks + [master_loop_task]
            
            if not all_tasks_to_monitor:
                 logger.error("No tasks were created. Simulation cannot run.")
                 console.print("[bold red]Error: No simulation tasks started.[/bold red]")
            else:
                logger.info(f"Started {len(all_tasks_to_monitor)} tasks.")
                done, pending = await asyncio.wait(all_tasks_to_monitor, return_when=asyncio.FIRST_COMPLETED)
                for task_item_done in done:
                    task_name = task_item_done.get_name() if hasattr(task_item_done, 'get_name') else str(task_item_done)
                    try: 
                        _ = task_item_done.result() # Check for exceptions
                        logger.info(f"Task {task_name} completed successfully.")
                    except asyncio.CancelledError: 
                        logger.info(f"Task {task_name} was cancelled.")
                    except Exception as task_exc: 
                        logger.error(f"Task {task_name} raised: {task_exc}", exc_info=True)
                        if task_name == "ADKMasterLoop":
                            logger.critical("ADK Master Loop failed - this is critical!")
                logger.info("One main task completed/failed. Initiating shutdown.")

                for task_item_pending in pending:
                    task_name_pending = task_item_pending.get_name() if hasattr(task_item_pending, 'get_name') else str(task_item_pending)
                    logger.info(f"Cancelling pending task: {task_name_pending}")
                    task_item_pending.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)


    except Exception as e:
        logger.exception(f"Error during simulation setup or execution: {e}")
        console.print(f"[bold red]Unexpected error during simulation run: {e}[/bold red]")
        raise 
    finally:
        logger.info("Cancelling remaining tasks if any (e.g., if error before all_tasks_to_monitor was populated)...")
        # This outer finally block ensures cleanup even if the try block for `with Live` fails early.
        # `all_tasks_to_monitor` might not be defined if an error occurs before its assignment.
        if 'all_tasks_to_monitor' in locals():
            for task_to_cancel in all_tasks_to_monitor:
                if not task_to_cancel.done():
                    task_name_cancel = task_to_cancel.get_name() if hasattr(task_to_cancel, 'get_name') else str(task_to_cancel)
                    logger.info(f"Ensuring task {task_name_cancel} is cancelled.")
                    task_to_cancel.cancel()
            if any(not t.done() for t in all_tasks_to_monitor): # Check if any are still not done
                try:
                    await asyncio.gather(*(t for t in all_tasks_to_monitor if not t.done()), return_exceptions=True)
                except Exception as gather_e:
                    logger.error(f"Error during final gather of tasks: {gather_e}")
        else:
            logger.warning("No 'all_tasks_to_monitor' list found during final cleanup.")
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
            console.print(generate_table(state, 0, 0)) # Queues removed
        else:
            console.print("[yellow]State dictionary is empty.[/yellow]")
        console.rule("[bold green]Simulation Shutdown Complete[/]")

async def run_adk_master_loop(
    runner: Runner,
    session_obj: Session, # Renamed to avoid potential clashes and be clear
    sync_to_adk_callback: callable, # Expects a function that takes a Session
    sync_from_adk_callback: callable, # Expects a function that takes a Session
    live_display: Live
):
    """Run the ADK master loop with proper state synchronization"""
    try:
        logger.info("Starting ADK Master Loop...")
        iteration_count = 0
        last_save_iteration = 0
        # Use the passed session_obj
        save_interval = session_obj.state.get("config", {}).get("ADK_STATE_SAVE_INTERVAL", 50) 
        
        async for event in runner.run_async(
            user_id=USER_ID, 
            session_id=session_obj.id, 
            new_message=None # Add the required new_message argument
        ):
            # Event is yielded after MasterLoopAgent (and its sub-agents) complete one tick.
            # session_obj.state now contains the results of this tick.

            # 1. Sync from ADK session (fresh data) to global Python `state`
            sync_from_adk_callback(session_obj)

            iteration_count += 1
            
            # 2. Update live display using the now-current global `state`
            live_display.update(generate_table(state, 0, 0))
            
            # 3. Periodic save (using session_obj.state is fine as it's the direct ADK output)
            #    Alternatively, could save global `state` after sync_from_adk.
            #    The current logic saves session_obj.state, which is the most direct ADK state.
            if iteration_count - last_save_iteration >= save_interval: # Check if save_interval is > 0
                try:
                    world_uuid_for_save = session_obj.state.get('world_instance_uuid', 'unknown_adk_save')
                    periodic_state_path = os.path.join(STATE_DIR, f"simulation_state_{world_uuid_for_save}.json")
                    save_json_file(periodic_state_path, dict(session_obj.state)) # Corrected order: path, then data
                    logger.info(f"Periodic ADK state save at iteration {iteration_count} to {periodic_state_path}")
                    last_save_iteration = iteration_count
                except Exception as save_e:
                    logger.error(f"Failed periodic ADK state save: {save_e}")
            
            # 4. Log current simulation time from the updated global `state`
            if iteration_count % 10 == 0:
                current_sim_time_loop = state.get("world_time", 0.0) # Use global state for logging
                logger.info(f"ADK Master Loop iteration {iteration_count}, sim_time: {current_sim_time_loop:.1f}s")
            
            # 5. Sync global Python `state` back to ADK session_obj.state for the *next* tick's input.
            #    This is important if any non-ADK logic (e.g., socket server) modified global `state`.
            sync_to_adk_callback(session_obj)

            # 6. Termination conditions (use the updated global `state`)
            if event.actions and event.actions.escalate:
                logger.info(f"ADK MasterLoopAgent escalated, ending loop. Event: {event.author} - {str(event.content)[:100]}")
                break

            current_sim_time_for_check = state.get("world_time", 0.0) # Use global state
            if current_sim_time_for_check >= MAX_SIMULATION_TIME:
                logger.info(f"Simulation time limit reached: {current_sim_time_for_check:.1f}s >= {MAX_SIMULATION_TIME}s")
                break
            
            # MAX_SIMULATION_TICKS is loaded into state["config"] initially.
            max_ticks_from_config = state.get("config", {}).get("MAX_SIMULATION_TICKS", MAX_SIMULATION_TICKS)
            if iteration_count >= max_ticks_from_config:
                logger.info(f"Max simulation ticks reached: {iteration_count} >= {max_ticks_from_config}")
                break
            
    except Exception as e:
        logger.error(f"Error in ADK Master Loop: {e}", exc_info=True)
        raise
    finally:
        logger.info(f"ADK Master Loop completed after {iteration_count} iterations.")

# Note: The get_random_style_combination and narrative_image_generation_task (old version)
# were removed from this file as their logic will be part of ADK agents or utils.
# For Phase 1, narrative_image_generation_task is imported from core_tasks.
# If get_random_style_combination was used by it, it should be in simulation_utils.py.
