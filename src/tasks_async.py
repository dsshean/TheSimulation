# tasks_async.py - Implementations for the asynchronous simulation tasks

import asyncio
import random
import heapq
import json
import logging
import re
import time
import uuid # Keep uuid for temporary session test if needed later
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from google.genai import types
# from google.adk.runners import Runner # No longer needed directly
# Import Pydantic for validation
from pydantic import (BaseModel, Field, ValidationError, field_validator,
                      ValidationInfo)
# Import ADK types for runner call
from google.genai import types as genai_types
# from google.adk.sessions import InMemorySessionService, Session # No longer needed directly
# Import LlmAgent for type hints
from google.adk.agents import LlmAgent

# --- Remove Import of Global ADK Variables ---
# from src.simulation_async import (adk_session_service, adk_session, world_engine_runner,
#                                   world_engine_agent, simulacra_agents)
# --- # Tasks will use context object


# --- Rich Imports ---
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
except ImportError:
    # Define DummyConsole and other Rich fallbacks if needed
    class DummyConsole:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print(f"\n--- {args[0] if args else ''} ---")
        def table(self, *args, **kwargs): return self
        def add_column(self, *args, **kwargs): pass
        def add_row(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, *args, **kwargs): pass
    console = DummyConsole()
    print("Rich console not found, using basic print.")
    # Define dummy Live if needed for type hints, though it's passed in
    class Live:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass


# Import SimulationContext for type hinting only to avoid circular dependency
if TYPE_CHECKING:
    from src.simulation_async import SimulationContext

logger = logging.getLogger(__name__)

# --- Pydantic Models for LLM Response Validation ---

class WorldEngineResponse(BaseModel):
    """Pydantic model for validating the World Engine LLM response."""
    valid_action: bool
    duration: float = Field(ge=0.0)
    results: Dict[str, Any] = Field(default_factory=dict)
    narrative: str

    @field_validator('duration')
    @classmethod
    def duration_must_be_zero_if_invalid(cls, v: float, info: ValidationInfo):
        """Ensure duration is 0 if action is invalid."""
        if info.data and 'valid_action' in info.data and not info.data['valid_action'] and v != 0.0:
            logger.warning(f"Invalid action returned non-zero duration ({v}). Forcing to 0.0.")
            return 0.0
        return v

    @field_validator('results')
    @classmethod
    def results_must_be_empty_if_invalid(cls, v: Dict, info: ValidationInfo):
        """Ensure results dict is empty if action is invalid."""
        if info.data and 'valid_action' in info.data and not info.data['valid_action'] and v:
            logger.warning(f"Invalid action returned non-empty results ({v}). Forcing to empty dict.")
            return {}
        return v

class SimulacraIntentResponse(BaseModel):
    """Pydantic model for validating the Simulacra LLM response."""
    internal_monologue: str
    action_type: str
    target_id: Optional[str] = None
    details: str = ""


# --- Helper Functions ---

def get_nested(data: Dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve nested dictionary values."""
    current = data
    for key in keys:
        # Check if current level is a dictionary before accessing key
        if isinstance(current, dict):
            current = current.get(key)
        # Allow integer indexing for lists
        elif isinstance(current, list) and isinstance(key, int):
            try:
                current = current[key]
            except IndexError:
                return default # Index out of bounds
        else:
            return default # Not a dict or list, cannot traverse further
    return current if current is not None else default

def _update_state_value(state: Dict[str, Any], key_path: str, value: Any):
    """
    Safely updates a nested value in the state dictionary using a dot-notation path.
    Logs warnings or errors if the path is invalid.
    """
    try:
        keys = key_path.split('.')
        target = state
        # Traverse path up to the second-to-last key
        for i, key in enumerate(keys[:-1]):
            # Check if the current target is a dictionary
            if not isinstance(target, dict):
                logger.error(f"Invalid path '{key_path}': Segment '{keys[i-1]}' is not a dictionary.")
                return False # Indicate failure

            if key not in target:
                 logger.warning(f"Path segment '{key}' not found in '{key_path}'. Cannot update.")
                 return False

            target = target[key] # Move to the next level

        # Set the value at the final key
        final_key = keys[-1]
        if not isinstance(target, dict):
             logger.error(f"Invalid path '{key_path}': Segment before final key '{final_key}' is not a dictionary.")
             return False

        target[final_key] = value
        logger.info(f"[StateUpdate] Applied: {key_path} = {value}")
        return True # Indicate success

    except Exception as e:
        logger.error(f"Error updating state for path '{key_path}' with value '{value}': {e}", exc_info=True)
        return False # Indicate failure


def generate_table(context: 'SimulationContext') -> Table:
    """Generates the Rich table for live display based on the central state."""
    state = context.state
    table = Table(title=f"Simulation State @ {state.get('world_time', 0.0):.2f}s",
                  show_header=True,
                  header_style="bold magenta",
                  box=None,
                  padding=(0, 1),
                  expand=True)

    table.add_column("Parameter", style="dim", no_wrap=True)
    table.add_column("Value", overflow="fold", no_wrap=False)

    # World Info
    table.add_row("World Time", f"{state.get('world_time', 0.0):.2f}s")
    table.add_row("World UUID", str(get_nested(state, 'world_instance_uuid', default='N/A')))
    table.add_row("World Desc", get_nested(state, 'world_details', 'description', default='N/A')[:80] + "...")

    # Simulacra Info
    active_sim_ids = state.get("active_simulacra_ids", [])
    sim_limit = 3
    for i, sim_id in enumerate(active_sim_ids):
        if i >= sim_limit:
            table.add_row(f"... ({len(active_sim_ids) - sim_limit} more)", "...")
            break
        sim_state = state.get("simulacra", {}).get(sim_id, {})
        table.add_row(f"--- Sim: {get_nested(sim_state, 'name', default=sim_id)} ---", "---")
        table.add_row(f"  Status", get_nested(sim_state, 'status', default="Unknown"))
        table.add_row(f"  Location", get_nested(sim_state, 'location', default="Unknown"))
        table.add_row(f"  Goal", get_nested(sim_state, 'goal', default="Unknown")[:60] + "...")
        table.add_row(f"  Action End", f"{get_nested(sim_state, 'current_action_end_time', default=0.0):.2f}s" if get_nested(sim_state, 'status')=='busy' else "N/A")
        table.add_row(f"  Last Obs.", get_nested(sim_state, 'last_observation', default="None")[:80] + "...")

    # Object Info
    object_limit = 3
    objects_dict = get_nested(state, 'objects', default={})
    table.add_row("--- Objects ---", f"({len(objects_dict)} total)")
    for i, (obj_id, obj_state) in enumerate(objects_dict.items()):
         if i >= object_limit:
             table.add_row(f"... ({len(objects_dict) - object_limit} more)", "...")
             break
         obj_name = get_nested(obj_state, 'name', default=obj_id)
         obj_loc = get_nested(obj_state, 'location', default='Unknown')
         obj_power = get_nested(obj_state, 'power')
         obj_locked = get_nested(obj_state, 'locked')
         obj_status = get_nested(obj_state, 'status')
         details = f"Loc: {obj_loc}"
         if obj_power is not None: details += f", Pwr: {obj_power}"
         if obj_locked is not None: details += f", Lck: {'Y' if obj_locked else 'N'}"
         if obj_status is not None: details += f", Sts: {obj_status}"
         table.add_row(f"  {obj_name}", details)

    # System Info
    table.add_row("--- System ---", "---")
    table.add_row("Schedule Size", str(len(context.schedule)))
    table.add_row("Event Bus Size", str(context.event_bus.qsize()))
    log_display = "\n".join(get_nested(state, 'narrative_log', default=[])[-6:])
    table.add_row("Narrative Log", log_display)

    return table

# --- Core Async Tasks ---

async def time_manager_task(
    context: 'SimulationContext',
    live_display: Live
):
    """Advances time, processes scheduled events, and updates state."""
    logger.info("[TimeManager] Task started.")
    last_real_time = time.monotonic()
    state = context.state
    schedule = context.schedule

    try:
        while state.get("world_time", 0.0) < context.config.max_simulation_time:
            current_real_time = time.monotonic()
            real_delta_time = current_real_time - last_real_time
            last_real_time = current_real_time
            sim_delta_time = real_delta_time * context.config.simulation_speed_factor
            current_sim_time = state.setdefault("world_time", 0.0)
            new_sim_time = current_sim_time + sim_delta_time
            state["world_time"] = new_sim_time

            while schedule and schedule[0][0] <= new_sim_time:
                completion_time, event_id, event_data = heapq.heappop(schedule)
                logger.info(f"[TimeManager] Processing scheduled event {event_id} ({event_data.get('type')}) due at {completion_time:.1f}")

                results = event_data.get("results", {})
                actor_id_for_memory = event_data.get("actor_id")
                memory_log_updated = False

                # Apply state changes using the helper function
                for key_path, value in results.items():
                    success = _update_state_value(state, key_path, value)
                    if success and actor_id_for_memory and key_path == f"simulacra.{actor_id_for_memory}.memory_log":
                        memory_log_updated = True

                # Log the narrative
                narrative = event_data.get("narrative")
                if narrative:
                    state.setdefault("narrative_log", []).append(f"[T{completion_time:.1f}] {narrative}")

                # --- Prune Memory Log if it was updated ---
                if memory_log_updated and actor_id_for_memory:
                    actor_state_for_mem = get_nested(state, "simulacra", actor_id_for_memory)
                    if actor_state_for_mem and "memory_log" in actor_state_for_mem:
                        max_len = context.config.max_memory_log_entries
                        current_mem_log = actor_state_for_mem["memory_log"]
                        if isinstance(current_mem_log, list) and len(current_mem_log) > max_len:
                            actor_state_for_mem["memory_log"] = current_mem_log[-max_len:]
                            logger.debug(f"[TimeManager] Pruned memory log for {actor_id_for_memory} to {max_len} entries.")

                # Update actor status and last observation
                actor_id = event_data.get("actor_id")
                if actor_id and "simulacra" in state:
                    actor_state = state["simulacra"].get(actor_id)
                    if actor_state:
                        if actor_state.get("status") == "busy" and abs(get_nested(actor_state, "current_action_end_time", default=-1) - completion_time) < 0.01:
                            actor_state["status"] = "idle"
                            actor_state["current_action_end_time"] = completion_time
                            logger.info(f"[TimeManager] Set {actor_id} status to idle.")
                            if narrative:
                                actor_state["last_observation"] = narrative
                                logger.info(f"[TimeManager] Stored action narrative in {actor_id}'s last_observation.")
                        elif actor_state.get("status") == "busy":
                            logger.warning(f"[TimeManager] Action completion for {actor_id} at {completion_time:.1f} did not match current busy state end time {get_nested(actor_state, 'current_action_end_time', default=-1)}. Status left as busy.")
                        else:
                            logger.debug(f"[TimeManager] Action completion for {actor_id} at {completion_time:.1f} occurred while actor was {actor_state.get('status')}.")
                            if narrative:
                                 actor_state["last_observation"] = narrative
                    else:
                        logger.warning(f"[TimeManager] Actor state for '{actor_id}' not found for completed event.")

            live_display.update(generate_table(context))
            await asyncio.sleep(context.config.update_interval)

    except asyncio.CancelledError:
        logger.info("[TimeManager] Task cancelled.")
    except Exception as e:
        logger.exception(f"[TimeManager] Error in main loop: {e}")
    finally:
        logger.info(f"[TimeManager] Loop finished at sim time {state.get('world_time', 0.0):.1f}")


async def world_engine_task_llm(
    context: 'SimulationContext' # Task still receives context
):
    """Listens for intents, calls LLM to resolve, and schedules completions."""
    logger.info("[WorldEngineLLM] Task started.")
    state = context.state # Access state via context
    event_bus = context.event_bus # Access event_bus via context
    schedule = context.schedule # Access schedule via context
    # --- Get runner and agent from context ---
    world_engine_runner = context.get_runner("world_engine") # Get via context method
    world_engine_agent = context.get_agent("world_engine") # Get via context method
    # ---

    # Access globals directly for checks
    if not world_engine_runner:
        logger.error("[WorldEngineLLM] Runner not found in context. Task cannot proceed.")
        return
    if not world_engine_agent:
        logger.error("[WorldEngineLLM] Agent instance not found in context. Task cannot proceed.")
        return

    while True:
        intent_event = None
        actor_id = None
        actor_state = {}
        narrative = "Action failed due to internal error (pre-processing)."

        try:
            intent_event = await event_bus.get()

            if get_nested(intent_event, "type") != "intent_declared":
                logger.debug(f"[WorldEngineLLM] Ignoring event type: {get_nested(intent_event, 'type')}")
                continue

            actor_id = get_nested(intent_event, "actor_id")
            intent = get_nested(intent_event, "intent")
            if not actor_id or not intent:
                logger.warning(f"[WorldEngineLLM] Received invalid intent event: {intent_event}")
                continue

            logger.info(f"[WorldEngineLLM] Received intent from {actor_id}: {intent}")
            action_type = intent.get("action_type")
            actor_state = get_nested(state, 'simulacra', actor_id, default={})
            actor_name = actor_state.get('name', actor_id)
            current_sim_time = state.get("world_time", 0.0)

            # --- Handle 'talk' action specifically ---
            if action_type == 'talk':
                target_id = intent.get("target_id")
                message = intent.get("details", "")
                target_state = get_nested(state, 'simulacra', target_id) if target_id else None
                target_name = target_state.get('name', target_id) if target_state else target_id

                # Validity checks
                if not target_id:
                    is_valid = False
                    narrative = f"{actor_name} tries to talk, but doesn't specify who."
                elif not target_state:
                    is_valid = False
                    narrative = f"{actor_name} tries to talk to {target_id}, but they don't exist."
                elif get_nested(actor_state, 'location') != get_nested(target_state, 'location'):
                    is_valid = False
                    narrative = f"{actor_name} tries to talk to {target_name}, but they are in different locations."
                else:
                    is_valid = True
                    duration = 5.0 + len(message) * 0.1 # Simple duration based on message length
                    results = {
                        f"simulacra.{target_id}.last_observation": f"{actor_name} said to you: \"{message}\""
                    }
                    narrative = f"{actor_name} says \"{message}\" to {target_name}."

                # Process outcome (schedule or handle failure)
                if is_valid:
                    completion_time = current_sim_time + duration
                    context.schedule_event_counter += 1
                    event_id = context.schedule_event_counter
                    completion_event = {
                        "type": "action_complete", "actor_id": actor_id,
                        "action": intent, "results": results, "narrative": narrative
                    }
                    heapq.heappush(schedule, (completion_time, event_id, completion_event))

                    if actor_id in state.get("simulacra", {}):
                        state["simulacra"][actor_id]["status"] = "busy"
                        state["simulacra"][actor_id]["current_action_end_time"] = completion_time
                    logger.info(f"[WorldEngineLLM] 'talk' Action VALID for {actor_id}. Scheduled completion at {completion_time:.1f}s. Narrative: {narrative}")
                else:
                    logger.info(f"[WorldEngineLLM] 'talk' Action INVALID for {actor_id}. Reason: {narrative}")
                    if actor_id in state.get("simulacra", {}):
                        state["simulacra"][actor_id]["last_observation"] = narrative
                        state["simulacra"][actor_id]["status"] = "idle"
                    state.setdefault("narrative_log", []).append(f"[T{current_sim_time:.1f}] {narrative}")

                continue # Go to finally block

            # --- Handle other actions via LLM ---
            else:
                # --- Prepare Context for LLM ---
                actor_location_id = get_nested(actor_state, "location")
                location_state = get_nested(state, 'location_details', actor_location_id, default={})
                world_rules = get_nested(state, 'world_details', 'rules', default={})
                target_id = get_nested(intent, "target_id")
                target_object_state = get_nested(state, 'objects', target_id, default={}) if target_id else {}

                # --- Construct Prompt ---
                world_rules_json = json.dumps(world_rules, indent=2)
                location_state_json = json.dumps(location_state, indent=2)
                intent_json = json.dumps(intent, indent=2)
                target_object_json = json.dumps(target_object_state, indent=2) if target_object_state else "No specific target object."

                prompt = f"""
You are the World Engine... (rest of prompt identical) ...
Example Invalid Output:
{{
  "valid_action": false,
  "duration": 0.0,
  "results": {{}},
  "narrative": "{actor_name} tries to use the computer, but it has no power."
}}
"""
                # --- Call LLM ---
                logger.debug(f"[WorldEngineLLM] Sending prompt to LLM for {actor_id}'s intent ({action_type}).")
                # --- Set agent before call ---
                world_engine_runner.agent = world_engine_agent # Use global runner/agent
                response_text = ""
                trigger_content = genai_types.Content(parts=[types.Part(text=prompt)])
                # --- Use session_id from context ---
                session_id_to_use = context.adk_session_id
                if not session_id_to_use:
                     logger.error("[WorldEngineLLM] Session ID not found in context.")
                     raise RuntimeError("ADK Session ID not initialized in context.")
                # ---
                async for event in world_engine_runner.run_async( # Use the single runner
                    user_id=context.config.user_id, # Use user_id from context config
                    session_id=session_id_to_use, # Use ID from global session
                    new_message=trigger_content
                ):
                    if event.is_final_response() and event.content:
                        response_text = event.content.parts[0].text
                        logger.debug(f"WorldLLM Final Content: {response_text[:100]}...")
                    elif event.error_message:
                        logger.error(f"WorldLLM Error: {event.error_message}")
                        narrative = f"Action failed due to LLM error: {event.error_message}"
                        break
                    # else: logger.debug(f"WorldLLM Event: Type={type(event.content)}")
                # --- END LLM Call ---

                # --- Parse and Validate LLM Response ---
                validated_data: Optional[WorldEngineResponse] = None
                if response_text:
                    try:
                        response_text = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                        raw_data = json.loads(response_text)
                        validated_data = WorldEngineResponse.model_validate(raw_data)
                        logger.debug(f"[WorldEngineLLM] LLM response validated successfully for {actor_id}.")
                        narrative = validated_data.narrative

                    except json.JSONDecodeError as e:
                        logger.error(f"[WorldEngineLLM] Failed to decode JSON response for {actor_id}: {e}\nResponse:\n{response_text}", exc_info=True)
                        narrative = "Action failed due to internal error (JSON decode)."
                    except ValidationError as e:
                        logger.error(f"[WorldEngineLLM] Failed to validate LLM response structure for {actor_id}: {e}\nResponse:\n{response_text}", exc_info=True)
                        narrative = "Action failed due to internal error (invalid structure)."
                    except Exception as e:
                         logger.error(f"[WorldEngineLLM] Unexpected error parsing/validating response for {actor_id}: {e}\nResponse:\n{response_text}", exc_info=True)
                         narrative = "Action failed due to internal error (unexpected)."

                # --- Process Validated Data (or handle validation/LLM failure) ---
                if validated_data and validated_data.valid_action:
                    completion_time = current_sim_time + validated_data.duration
                    context.schedule_event_counter += 1
                    event_id = context.schedule_event_counter
                    completion_event = {
                        "type": "action_complete", "actor_id": actor_id,
                        "action": intent,
                        "results": validated_data.results,
                        "narrative": validated_data.narrative
                    }
                    heapq.heappush(schedule, (completion_time, event_id, completion_event))

                    if actor_id in state.get("simulacra", {}):
                        state["simulacra"][actor_id]["status"] = "busy"
                        state["simulacra"][actor_id]["current_action_end_time"] = completion_time
                    logger.info(f"[WorldEngineLLM] Action VALID for {actor_id}. Scheduled completion at {completion_time:.1f}s (Event ID: {event_id}). Narrative: {validated_data.narrative}")

                else:
                    # Action is invalid or validation failed or LLM error
                    logger.info(f"[WorldEngineLLM] Action INVALID for {actor_id}. Reason: {narrative}")
                    if actor_id in state.get("simulacra", {}):
                        state["simulacra"][actor_id]["last_observation"] = narrative
                        state["simulacra"][actor_id]["status"] = "idle"
                    actor_name_for_log = get_nested(actor_state, 'name', default=actor_id) if actor_id else "Unknown Actor"
                    state.setdefault("narrative_log", []).append(f"[T{current_sim_time:.1f}] {actor_name_for_log}'s action failed: {narrative}")

        except asyncio.CancelledError:
            logger.info("[WorldEngineLLM] Task cancelled.")
            if intent_event and event_bus._unfinished_tasks > 0:
                try: event_bus.task_done()
                except ValueError: pass
            break
        except Exception as e:
            logger.exception(f"[WorldEngineLLM] Error processing event for actor {actor_id}: {e}")
            if actor_id:
                 actor_name_for_log = get_nested(state, 'simulacra', actor_id, 'name', default=actor_id)
                 state.setdefault("narrative_log", []).append(f"[T{state.get('world_time', 0.0):.1f}] {actor_name_for_log}'s action failed unexpectedly: {e}")
                 if actor_id in get_nested(state, "simulacra", default={}):
                     state["simulacra"][actor_id]["status"] = "idle"
                     state["simulacra"][actor_id]["last_observation"] = f"Action failed unexpectedly: {e}"
            await asyncio.sleep(5)
        finally:
            if intent_event and event_bus._unfinished_tasks > 0:
                try:
                    event_bus.task_done()
                except ValueError:
                    logger.warning("[WorldEngineLLM] task_done() called too many times.")
                except Exception as td_e:
                    logger.error(f"[WorldEngineLLM] Error calling task_done(): {td_e}")


async def simulacra_agent_task_llm(
    agent_id: str,
    context: 'SimulationContext' # Task still receives context
):
    """Represents the thinking and acting loop for a single simulacrum using an LLM."""
    state = context.state # Access state via context
    event_bus = context.event_bus # Access event_bus via context
    # --- Get runner and agent from context ---
    simulacra_runner = context.get_runner(agent_id) # Get via context method
    simulacra_agent = context.get_agent(agent_id) # Get via context method
    # ---
    agent_name = get_nested(state, "simulacra", agent_id, "name", default=agent_id)

    logger.info(f"[{agent_name}] LLM Agent task started.")
    if not simulacra_runner:
        logger.error(f"[{agent_name}] Runner not found in context. Task cannot proceed.")
        return # Should not happen if world_engine_runner is global
    if not simulacra_agent:
        logger.error(f"[{agent_name}] Global agent instance not found for {agent_id}. Task cannot proceed.")
        return

    while True:
        try:
            agent_state = get_nested(state, "simulacra", agent_id)
            if not agent_state:
                 logger.error(f"[{agent_name}] State not found for agent {agent_id}. Stopping task.")
                 break

            while agent_state.get("status") == "busy":
                await asyncio.sleep(0.5 + random.uniform(0, 0.5))
                agent_state = get_nested(state, "simulacra", agent_id)
                if not agent_state: break
            if not agent_state: break

            # --- Prepare Context for LLM ---
            persona = agent_state.get("persona", {})
            goal = agent_state.get("goal", "Survive and observe.")
            location_id = agent_state.get("location")
            location_details = get_nested(state, "location_details", location_id, default={})

            objects_in_loc_data = {}
            agents_in_loc_data = {}
            for obj_id, obj_data in state.get("objects", {}).items():
                if obj_data.get("location") == location_id:
                    objects_in_loc_data[obj_id] = {
                        "name": obj_data.get("name", obj_id),
                        "description": obj_data.get("description", ""),
                        "properties": obj_data.get("properties", [])
                    }
            for other_agent_id, other_agent_data in state.get("simulacra", {}).items():
                if other_agent_id != agent_id and other_agent_data.get("location") == location_id:
                     agents_in_loc_data[other_agent_id] = {
                         "name": other_agent_data.get("name", other_agent_id),
                         "status": other_agent_data.get("status", "unknown")
                     }

            last_observation = agent_state.get("last_observation", "Just arrived.")
            mem_len = context.config.memory_log_context_length
            memory_log = agent_state.get("memory_log", [])[-mem_len:]
            current_time_for_prompt = state.get("world_time", 0.0)

            # --- Construct Prompt ---
            persona_json = json.dumps(persona, indent=2)
            objects_json = json.dumps(objects_in_loc_data, indent=2) if objects_in_loc_data else "None"
            agents_json = json.dumps(agents_in_loc_data, indent=2) if agents_in_loc_data else "None"
            memory_json = json.dumps(memory_log, indent=2)

            prompt = f"""
You are {agent_name}.
Your Persona:
{persona_json}
... (rest of prompt identical) ...
Example Output (Talking):
{{
  "internal_monologue": "Bob is here, maybe I should ask him about the key.",
  "action_type": "talk",
  "target_id": "sim_bob_123",
  "details": "Hey Bob, have you seen a small brass key around?"
}}
"""
            # --- Call LLM ---
            logger.debug(f"[{agent_name}] Sending prompt to LLM.")
            # --- DIAGNOSTIC LOGGING (Check context service) ---
            if simulacra_runner.session_service:
                logger.debug(f"[{agent_name}] Runner Service ID: {id(simulacra_runner.session_service)}")
            if context.adk_session_service: # Check context service
                logger.debug(f"[{agent_name}] Context Service ID: {id(context.adk_session_service)}")
            if context.adk_session_service and hasattr(context.adk_session_service, '_sessions'):
                 logger.debug(f"[{agent_name}] Sessions in Context Service: {list(context.adk_session_service._sessions.keys())}")

            # --- Set agent before call ---
            simulacra_runner.agent = simulacra_agent
            response_text = ""
            trigger_content = genai_types.Content(parts=[types.Part(text=prompt)])
            # --- Use session_id from context ---
            session_id_to_use = context.adk_session_id # Use context object
            if not session_id_to_use: # Check if it exists on context
                 logger.error(f"[{agent_name}] Global adk_session not available for run_async.")
                 raise RuntimeError("ADK Session not initialized globally.")
            # ---
            async for event in simulacra_runner.run_async( # Use the single runner
                user_id=context.config.user_id,
                session_id=session_id_to_use, # Use ID from context
                new_message=trigger_content
            ):
                if event.is_final_response() and event.content:
                    response_text = event.content.parts[0].text
                    logger.debug(f"SimLLM ({agent_name}) Final Content: {response_text[:100]}...")
                elif event.error_message:
                    logger.error(f"SimLLM ({agent_name}) Error: {event.error_message}")
                    response_text = ""
                    break
                # else: logger.debug(f"SimLLM ({agent_name}) Event: Type={type(event.content)}")
            # --- END LLM Call ---

            # --- Parse and Validate LLM Response ---
            validated_intent: Optional[SimulacraIntentResponse] = None
            if response_text:
                try:
                   response_text = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                   raw_data = json.loads(response_text)
                   validated_intent = SimulacraIntentResponse.model_validate(raw_data)
                   logger.debug(f"[{agent_name}] LLM response validated successfully.")

                   context.console.print(f"[cyan][{agent_name} Thought][/] {validated_intent.internal_monologue}")
                   intent = {
                       "action_type": validated_intent.action_type,
                       "target_id": validated_intent.target_id,
                       "details": validated_intent.details
                   }
                   context.console.print(f"[yellow][{agent_name} Intent][/] {intent}")

                   await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": intent})

                except json.JSONDecodeError as e:
                   logger.error(f"[{agent_name}] Failed to decode JSON response: {e}\nResponse:\n{response_text}", exc_info=True)
                   context.console.print(f"[red][{agent_name} Error][/] Failed to decode LLM response. Agent will wait.")
                   await asyncio.sleep(10 + random.uniform(0, 10))
                except ValidationError as e:
                   logger.error(f"[{agent_name}] Failed to validate LLM response structure: {e}\nResponse:\n{response_text}", exc_info=True)
                   context.console.print(f"[red][{agent_name} Error][/] LLM response has invalid structure. Agent will wait.")
                   await asyncio.sleep(10 + random.uniform(0, 10))
                except Exception as e:
                   logger.error(f"[{agent_name}] Unexpected error parsing/validating response: {e}\nResponse:\n{response_text}", exc_info=True)
                   context.console.print(f"[red][{agent_name} Error][/] Unexpected error processing LLM response. Agent will wait.")
                   await asyncio.sleep(10 + random.uniform(0, 10))
            else:
                # Handle case where LLM call failed (error already logged)
                logger.warning(f"[{agent_name}] No valid response received from LLM. Agent will wait.")
                context.console.print(f"[yellow][{agent_name} Warning][/] No valid response from LLM. Agent will wait.")
                await asyncio.sleep(10 + random.uniform(0, 10))


            # Wait a bit before next decision cycle
            await asyncio.sleep(1 + random.uniform(0, 2))

        except asyncio.CancelledError:
            logger.info(f"[{agent_name}] Task cancelled.")
            break
        except Exception as e:
            logger.exception(f"[{agent_name}] Error in main loop: {e}")
            if agent_id in get_nested(state, "simulacra", default={}):
                 # --- Store the actual error message ---
                 state["simulacra"][agent_id]["last_observation"] = f"Encountered unexpected error: {e}"
                 # ---
            await asyncio.sleep(10)
