# src/simulation_async.py - Core Simulation Logic

import asyncio
import glob
import heapq
import json
import logging
import os
import re
import sys
import time
import uuid
import random
import string # For default sim_id generation if needed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# --- ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types as genai_types # Renamed to avoid conflict
import google.generativeai as genai # For direct API config

# --- Pydantic Validation ---
from pydantic import (BaseModel, Field, ValidationError, field_validator,
                      ValidationInfo)

# --- Rich Imports ---
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    console = Console()
except ImportError:
    # Define DummyConsole and Live for environments without Rich
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
    class Live:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

# --- Logging Setup (from main_async.py, adjusted) ---
logger = logging.getLogger(__name__) # Use logger from main entry point setup

# --- Configuration (Directly in script or from .env) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info(".env file loaded.")
except ImportError:
    logger.info("dotenv not installed, ensure GOOGLE_API_KEY is set in environment.")

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_GEMINI_PRO", "gemini-1.5-flash-latest") # Use PRO or FLASH
APP_NAME = "TheSimulationAsync" # Consistent App Name
USER_ID = "player1"

# --- Simulation Parameters ---
SIMULATION_SPEED_FACTOR = float(os.getenv("SIMULATION_SPEED_FACTOR", 1.0))
UPDATE_INTERVAL = float(os.getenv("UPDATE_INTERVAL", 0.1))
MAX_SIMULATION_TIME = float(os.getenv("MAX_SIMULATION_TIME", 1800.0))
MEMORY_LOG_CONTEXT_LENGTH = 10 # Max number of recent memories in prompt
MAX_MEMORY_LOG_ENTRIES = 500 # Max total memories stored per agent

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project Root
STATE_DIR = os.path.join(BASE_DIR, "data", "states")
LIFE_SUMMARY_DIR = os.path.join(BASE_DIR, "data", "life_summaries")
WORLD_CONFIG_DIR = os.path.join(BASE_DIR, "data")

# Ensure directories exist (redundant with main_async.py but safe)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)
os.makedirs(WORLD_CONFIG_DIR, exist_ok=True)

# --- Core Components (Module Scope) ---
event_bus = asyncio.Queue()
schedule: List[Tuple[float, int, Dict[str, Any]]] = []
schedule_event_counter = 0
state: Dict[str, Any] = {} # Global state dictionary, initialized empty

# --- ADK Components (Module Scope - Initialized in run_simulation) ---
adk_session_service: Optional[InMemorySessionService] = None
adk_session_id: Optional[str] = None
adk_session: Optional[Session] = None
adk_runner: Optional[Runner] = None # Renamed from world_engine_runner for clarity
world_engine_agent: Optional[LlmAgent] = None
simulacra_agents: Dict[str, LlmAgent] = {}

# --- State Keys ---
WORLD_STATE_KEY = "current_world_state"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
LOCATION_DETAILS_KEY = "location_details"
SIMULACRA_PROFILES_KEY = "simulacra_profiles"
CURRENT_LOCATION_KEY = "current_location"
HOME_LOCATION_KEY = "home_location"
WORLD_TEMPLATE_DETAILS_KEY = "world_template_details"
LOCATION_KEY = "location"

# --- Initialization Functions (Integrated from main_async2.py) ---

def load_json_file(path: str, default: Optional[Any] = None) -> Optional[Any]:
    """Loads JSON from a file, returning default if file not found or invalid."""
    if not os.path.exists(path):
        logger.debug(f"File not found: {path}. Returning default.")
        return default
    try:
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {path}. Returning default.")
        return default
    except Exception as e:
        logger.error(f"Error loading file {path}: {e}. Returning default.")
        return default

def save_json_file(path: str, data: Any):
    """Saves data to a JSON file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved data to {path}")
    except Exception as e:
        logger.error(f"Error saving file {path}: {e}")
        raise

def find_latest_simulation_state_file(state_dir: str = STATE_DIR) -> Optional[str]:
    """Finds the most recently modified simulation state file."""
    try:
        os.makedirs(state_dir, exist_ok=True)
        state_file_pattern = os.path.join(state_dir, "simulation_state_*.json")
        list_of_files = glob.glob(state_file_pattern)
        if not list_of_files:
            logger.info(f"No existing simulation state files found in {state_dir}.")
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        logger.info(f"Found latest simulation state file: {latest_file}")
        return latest_file
    except Exception as e:
        logger.error(f"Error finding latest state file in {state_dir}: {e}")
        return None

def ensure_state_structure(state_dict: Dict[str, Any]) -> bool:
    """Checks and adds missing essential keys/structures to a state dictionary."""
    modified = False
    if not isinstance(state_dict, dict): return False

    if ACTIVE_SIMULACRA_IDS_KEY not in state_dict:
        state_dict[ACTIVE_SIMULACRA_IDS_KEY] = []
        logger.warning(f"Added missing '{ACTIVE_SIMULACRA_IDS_KEY}' key.")
        modified = True

    if WORLD_STATE_KEY not in state_dict:
        state_dict[WORLD_STATE_KEY] = {}
        logger.warning(f"Added missing '{WORLD_STATE_KEY}' key structure.")
        modified = True

    world_state_dict = state_dict.get(WORLD_STATE_KEY, {})
    if not isinstance(world_state_dict, dict):
        state_dict[WORLD_STATE_KEY] = {}
        world_state_dict = state_dict[WORLD_STATE_KEY]
        modified = True

    if LOCATION_DETAILS_KEY not in world_state_dict:
        world_state_dict[LOCATION_DETAILS_KEY] = {}
        logger.warning(f"Added missing '{LOCATION_DETAILS_KEY}' key to '{WORLD_STATE_KEY}'.")
        modified = True

    if SIMULACRA_PROFILES_KEY not in state_dict:
        state_dict[SIMULACRA_PROFILES_KEY] = {}
        logger.warning(f"Added missing '{SIMULACRA_PROFILES_KEY}' key.")
        modified = True

    if "narrative_log" not in state_dict:
        state_dict["narrative_log"] = []
        logger.warning("Added missing 'narrative_log' key.")
        modified = True
    elif not isinstance(state_dict["narrative_log"], list):
        state_dict["narrative_log"] = []
        logger.warning("Corrected 'narrative_log' key to be a list.")
        modified = True

    if "objects" not in state_dict:
        state_dict["objects"] = {}
        logger.warning("Added missing 'objects' key.")
        modified = True
    elif not isinstance(state_dict["objects"], dict):
        state_dict["objects"] = {}
        logger.warning("Corrected 'objects' key to be a dict.")
        modified = True

    if WORLD_TEMPLATE_DETAILS_KEY not in state_dict:
        state_dict[WORLD_TEMPLATE_DETAILS_KEY] = {"description": "Default", "rules": {}}
        logger.warning(f"Added missing '{WORLD_TEMPLATE_DETAILS_KEY}' key.")
        modified = True
    elif not isinstance(state_dict[WORLD_TEMPLATE_DETAILS_KEY], dict):
        state_dict[WORLD_TEMPLATE_DETAILS_KEY] = {"description": "Default", "rules": {}}
        logger.warning(f"Corrected '{WORLD_TEMPLATE_DETAILS_KEY}' key to be a dict.")
        modified = True

    # Ensure 'simulacra' state section exists
    if "simulacra" not in state_dict:
        state_dict["simulacra"] = {}
        logger.warning("Added missing 'simulacra' key.")
        modified = True
    elif not isinstance(state_dict["simulacra"], dict):
        state_dict["simulacra"] = {}
        logger.warning("Corrected 'simulacra' key to be a dict.")
        modified = True

    return modified

# --- Helper Functions (Integrated from main_async2.py) ---

def get_nested(data: Dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve nested dictionary values."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and isinstance(key, int):
            try:
                current = current[key]
            except IndexError: return default
        else: return default
    return current if current is not None else default

def _update_state_value(target_state: Dict[str, Any], key_path: str, value: Any):
    """Safely updates a nested value in the state dictionary."""
    try:
        keys = key_path.split('.')
        target = target_state
        for i, key in enumerate(keys[:-1]):
            if not isinstance(target, dict):
                logger.error(f"Invalid path '{key_path}': Segment '{keys[i-1]}' is not a dictionary.")
                return False
            # Create missing dictionaries along the path if necessary
            if key not in target or not isinstance(target[key], dict):
                logger.warning(f"Path segment '{key}' not found or not dict in '{key_path}'. Creating.")
                target[key] = {}
            target = target[key]

        final_key = keys[-1]
        if not isinstance(target, dict):
             logger.error(f"Invalid path '{key_path}': Segment before final key '{final_key}' is not a dictionary.")
             return False

        target[final_key] = value
        logger.info(f"[StateUpdate] Applied: {key_path} = {value}")
        return True
    except Exception as e:
        logger.error(f"Error updating state for path '{key_path}' with value '{value}': {e}", exc_info=True)
        return False

# --- Pydantic Models (Integrated from main_async2.py) ---

class WorldEngineResponse(BaseModel):
    valid_action: bool
    duration: float = Field(ge=0.0)
    results: Dict[str, Any] = Field(default_factory=dict)
    narrative: str

    @field_validator('duration')
    @classmethod
    def duration_must_be_zero_if_invalid(cls, v: float, info: ValidationInfo):
        if info.data and 'valid_action' in info.data and not info.data['valid_action'] and v != 0.0:
            logger.warning(f"Invalid action returned non-zero duration ({v}). Forcing to 0.0.")
            return 0.0
        return v

    @field_validator('results')
    @classmethod
    def results_must_be_empty_if_invalid(cls, v: Dict, info: ValidationInfo):
        if info.data and 'valid_action' in info.data and not info.data['valid_action'] and v:
            logger.warning(f"Invalid action returned non-empty results ({v}). Forcing to empty dict.")
            return {}
        return v

class SimulacraIntentResponse(BaseModel):
    internal_monologue: str
    action_type: str
    target_id: Optional[str] = None
    details: str = ""

# --- Agent Definitions (Integrated from main_async2.py) ---

def create_simulacra_llm_agent(sim_id: str, persona_name: str) -> LlmAgent:
    """Creates the LLM agent representing the character."""
    agent_name = f"SimulacraLLM_{sim_id}"
    # Note: Cannot directly reference global state['sim_{sim_id}'] here as it might not exist yet
    # The prompt needs to rely on context passed during the run_async call.
    instruction = f"""
You are {persona_name} ({sim_id}). Your current state (goal, location, observations, memory) will be provided in the trigger message.
You think step-by-step to decide your next action based on your observations, goal, and internal state.

**Current State Info (Provided via trigger message):**
- Your Goal: Provided in trigger.
- Your Location ID: Provided in trigger.
- Your Status: Provided in trigger (Should be 'idle' when you plan).
- Current Time: Provided in trigger.
- Last Observation/Event: Provided in trigger.
- Recent History (Last ~{MEMORY_LOG_CONTEXT_LENGTH} events): Provided in trigger.
- Objects in Room (IDs and Names): Provided in trigger.
- Other Agents in Room: Provided in trigger.
- Location Description: Provided in trigger.

**Thinking Process (Internal Monologue - Follow this process and INCLUDE it in your output):**
1.  **Recall & React:** What was the last thing I observed or did (`last_observation`)? What happened just before that (`Recent History`)? Did my recent actions work? How does it relate to my goal? How am I feeling?
2.  **Analyze Goal:** What is my goal? What do I want to achieve? How does it relate to my current state? What are the obstacles in my way?
3.  **Identify Options:** Based on the current state and my recent history/last observation, what actions could I take *right now* using the available object IDs or interacting with other agents?
    *   `look_around`: Get a detailed description of the room.
    *   `use [object_id]`: Interact with an object (e.g., `use door_office`, `use computer_office`). Specify `details`.
    *   `talk [agent_id]`: Speak to another agent. Specify `details` (the message).
    *   `wait`: If stuck or waiting.
    *   `think`: If needing to pause and reflect.
4.  **Prioritize:** If the door is open, leave! If locked, find a key (check desk, bookshelf?) or alternative (computer?). If computer login failed, try a different password or check elsewhere. If a search failed, try searching somewhere else. If someone spoke to me, should I respond? Don't repeat the exact same failed action immediately based on recent history.
5.  **Formulate Intent:** Choose the single best action. Use the correct `target_id` from the objects/agents list. Be specific in 'details'.

**Output:**
- Output ONLY a JSON object representing your chosen intent AND your internal monologue.
- Format: `{{"internal_monologue": "...", "action_type": "...", "target_id": "...", "details": "..."}}`
- Valid `action_type`: "use", "wait", "look_around", "think", "talk"
- Use `target_id` from the provided object/agent list (e.g., "door_office", "sim_bob_123"). Omit if not applicable (e.g., look_around, wait).
- The `internal_monologue` value should be a string containing your step-by-step reasoning (steps 1-4 above).
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        instruction=instruction,
        description=f"LLM Simulacra agent for {persona_name}."
    )

def create_world_engine_llm_agent() -> LlmAgent:
    """Creates the GENERALIZED LLM agent responsible for resolving actions."""
    agent_name = "WorldEngineLLMAgent"
    instruction = f"""
You are the World Engine, simulating the physics and state changes of the environment. You process a single declared intent from a Simulacra and determine its outcome, duration, and narrative description based on the current world state. **Crucially, your narrative must describe the RESULT or CONSEQUENCE of the action attempt, not just the attempt itself.**

**Input (Provided via trigger message):**
- Actor ID: e.g., "alex"
- Actor Name: e.g., "Alex"
- Actor Location ID: e.g., "MysteriousOffice"
- Intent: `{{"action_type": "...", "target_id": "...", "details": "..."}}`
- Current World Time: e.g., 15.3
- Target Object State: The current state dictionary of the object specified in `intent['target_id']` (if applicable).
- Location State: The state dictionary of the actor's current location.
- World Rules: General rules of the simulation.

**Your Task:**
1.  Examine the actor's intent (`action_type`, `target_id`, `details`).
2.  **Determine Validity & Outcome based on Intent, Target Object State, Location State, and World Rules:**
    *   **Location Check:** Is the actor in the same location as the target object? If not, `valid_action: false`, narrative: "[Actor Name] tries to use [Object Name] but it's not here." Duration 0s. Results empty.
    *   **Action Type Check:** Is `action_type` valid ("use", "wait", "look_around", "think")? (Note: "talk" is handled separately). If not, `valid_action: false`, narrative: "[Actor Name] attempts to [invalid action], which seems impossible." Duration 0s. Results empty.
    *   **If `action_type` is "use":**
        *   Look at the `Target Object State` (description, properties, status like power, locked).
        *   Is the intended `details` plausible for this object and its current state?
        *   **Determine Outcome & Narrative (Focus on Result):**
            *   **Turning On/Off:** If `details` is "turn on" and object is `powerable` and `power: "off"`. Results: `{{"objects.[target_id].power": "on"}}`. Narrative: "The [Object Name] powers on, [brief description of effect, e.g., screen flickers to life]." Duration 2s. `valid_action: true`. If already on, Narrative: "The [Object Name] is already on." Duration 1s. `valid_action: true`. Results empty. (Similar logic for "turn off").
            *   **Opening/Closing:** If `details` is "open" and object is `openable`, `locked: false`, `status: "closed"`. Results: `{{"objects.[target_id].status": "open"}}`. Narrative: "The [Object Name] swings open, revealing [brief description of what's revealed, e.g., a dark interior, the corridor]." Duration 2s. `valid_action: true`. If locked, Narrative: "The [Object Name] remains firmly locked." Duration 2s. `valid_action: true`. Results empty. If already open, Narrative: "The [Object Name] is already open." Duration 1s. `valid_action: true`. Results empty. (Similar logic for "close").
            *   **Locking/Unlocking:** If `details` is "unlock" and object is `lockable`. Check if actor has `key_required` (assume NO for now). Narrative: "[Actor Name] tries the lock on the [Object Name], but doesn't have the right key." Duration 3s. `valid_action: true`. Results empty. (If they had key: Results: `{{"objects.[target_id].locked": false}}`. Narrative: "The key turns smoothly, and the [Object Name] unlocks!").
            *   **Going Through (Doors):** If `target_id` is a door and `details` is "go through" (or similar). Check if `status` is "open". If yes, Results: `{{"simulacra.[ACTOR_ID].location": "[Destination]"}}`. Narrative: "[Actor Name] steps through the open [Object Name] into the [Destination Name]." Duration 3s. `valid_action: true`. If closed/locked, `valid_action: false`, narrative: "The [Object Name] blocks the way; it's closed/locked." Duration 0s. Results empty.
            *   **Computer Login:** If `target_id` is a computer, `power: "on"`, `logged_in: false`. If `details` contain "login" or "password". Extract password attempt. Compare to `Target Object State['password']`.
                *   Success: Results: `{{"objects.[target_id].logged_in": true, "objects.[target_id].current_user": "[ACTOR_ID]"}}`. Narrative: "Login successful! The computer displays a simple desktop interface." Duration 5s. `valid_action: true`.
                *   Failure: Results: `{{"objects.[target_id].last_login_attempt_failed": true}}`. Narrative: "Login failed. The screen displays 'Incorrect password'." Duration 5s. `valid_action: true`.
            *   **Computer Use (Logged In):** If computer `power: "on"`, `logged_in: true`. Handle "search files", "log off". Narrative describes outcome (e.g., "After searching the files, Alex finds an email mentioning...", "Alex logs off the computer, returning it to the login screen."). Duration 10-20s. Update `current_user` on log off. `valid_action: true`.
            *   **Searching:** If `details` involve "search", "look for", "check" and object is `searchable`. **Decide outcome (e.g., 30% chance find).**
                *   **Found:** Narrative: "[Actor Name] searches the [Object Name] and finds [Specific Item, e.g., a small brass key, a crumpled note]!" Duration 10s. `results: {{}}` (or update inventory later). `valid_action: true`.
                *   **Not Found:** Narrative: "[Actor Name] searches the [Object Name] thoroughly but finds nothing useful or out of the ordinary." Duration 10s. `results: {{}}`. `valid_action: true`.
            *   **Other Plausible 'use':** Narrative describes the *result* of the interaction (e.g., "Alex examines the window closely; the bars are solid and rusted.", "Alex reads the book title: 'Advanced Circuitry'."). Short/medium duration. `valid_action: true`. Results empty.
            *   **Implausible 'use':** `valid_action: false`, narrative: "[Actor Name] tries to [details] the [Object Name], but nothing happens / that doesn't seem possible." Duration 0s. Results empty.
    *   **If `action_type` is "look_around":** Narrative MUST BE the exact `Location State['description']` provided in the input. Duration 3s. No results. `valid_action: true`.
    *   **If `action_type` is "wait" or "think":** Narrative: "[Actor Name] waits, observing the quiet room." or "[Actor Name] pauses, considering the situation." Duration 5s or 1s. No results. `valid_action: true`.
3.  Calculate `duration` (float, simulation seconds). If invalid, duration is 0.0.
4.  Determine `results` (dict, state changes on completion, using dot notation). Replace '[ACTOR_ID]' and '[target_id]' appropriately. If invalid, results is {{}}.
5.  Generate `narrative` (string, present tense, descriptive, **focusing on the outcome/result**). Replace placeholders like '[Object Name]', '[Actor Name]', '[Destination Name]', etc. appropriately based on the input context.
6.  Determine `valid_action` (boolean).

**Output:**
- Output ONLY a JSON object with the keys: "valid_action", "duration", "results", "narrative". Ensure results dictionary keys use dot notation.
- Example Valid Output: `{{"valid_action": true, "duration": 2.0, "results": {{"objects.computer_office.power": "on"}}, "narrative": "The Old Computer powers on, its screen flickering to life."}}`
- Example Invalid Output: `{{"valid_action": false, "duration": 0.0, "results": {{}}, "narrative": "Alex tries to use the door but it's not here."}}`
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        instruction=instruction,
        description="LLM World Engine: Resolves actions based on target state, calculates duration/results, generates outcome-focused narrative."
    )

# --- Task Functions (Integrated from main_async2.py) ---
# These functions now rely on module-level variables like `state`, `schedule`, `event_bus`,
# `adk_runner`, `adk_session`, `world_engine_agent`, `simulacra_agents`.

def generate_table() -> Table:
    """Generates the Rich table for live display based on the module-level state."""
    # Access module-level state directly
    table = Table(title=f"Simulation State @ {state.get('world_time', 0.0):.2f}s",
                  show_header=True, header_style="bold magenta", box=None,
                  padding=(0, 1), expand=True)
    table.add_column("Parameter", style="dim", no_wrap=True)
    table.add_column("Value", overflow="fold", no_wrap=False)

    table.add_row("World Time", f"{state.get('world_time', 0.0):.2f}s")
    table.add_row("World UUID", str(get_nested(state, 'world_instance_uuid', default='N/A')))
    world_desc = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'description', default='N/A')
    table.add_row("World Desc", world_desc[:80] + ("..." if len(world_desc) > 80 else ""))

    active_sim_ids = state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
    sim_limit = 3
    for i, sim_id in enumerate(active_sim_ids):
        if i >= sim_limit:
            table.add_row(f"... ({len(active_sim_ids) - sim_limit} more)", "...")
            break
        sim_state_data = state.get("simulacra", {}).get(sim_id, {}) # Renamed sim_state
        table.add_row(f"--- Sim: {get_nested(sim_state_data, 'name', default=sim_id)} ---", "---")
        table.add_row(f"  Status", get_nested(sim_state_data, 'status', default="Unknown"))
        table.add_row(f"  Location", get_nested(sim_state_data, 'location', default="Unknown"))
        sim_goal = get_nested(sim_state_data, 'goal', default="Unknown")
        table.add_row(f"  Goal", sim_goal[:60] + ("..." if len(sim_goal) > 60 else ""))
        table.add_row(f"  Action End", f"{get_nested(sim_state_data, 'current_action_end_time', default=0.0):.2f}s" if get_nested(sim_state_data, 'status')=='busy' else "N/A")
        last_obs = get_nested(sim_state_data, 'last_observation', default="None")
        table.add_row(f"  Last Obs.", last_obs[:80] + ("..." if len(last_obs) > 80 else ""))

    object_limit = 3
    objects_dict = get_nested(state, 'objects', default={})
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
         details = f"Loc: {obj_loc}"
         if obj_power is not None: details += f", Pwr: {obj_power}"
         if obj_locked is not None: details += f", Lck: {'Y' if obj_locked else 'N'}"
         if obj_status is not None: details += f", Sts: {obj_status}"
         table.add_row(f"  {obj_name}", details)

    table.add_row("--- System ---", "---")
    table.add_row("Schedule Size", str(len(schedule)))
    table.add_row("Event Bus Size", str(event_bus.qsize()))
    log_display = "\n".join(get_nested(state, 'narrative_log', default=[])[-6:])
    table.add_row("Narrative Log", log_display)
    return table

async def time_manager_task(live_display: Live):
    """Advances time, processes scheduled events, and updates state."""
    global state, schedule # Explicitly mention modification of module-level vars
    logger.info("[TimeManager] Task started.")
    last_real_time = time.monotonic()

    try:
        while state.get("world_time", 0.0) < MAX_SIMULATION_TIME:
            current_real_time = time.monotonic()
            real_delta_time = current_real_time - last_real_time
            last_real_time = current_real_time
            sim_delta_time = real_delta_time * SIMULATION_SPEED_FACTOR
            current_sim_time = state.setdefault("world_time", 0.0)
            new_sim_time = current_sim_time + sim_delta_time
            state["world_time"] = new_sim_time

            while schedule and schedule[0][0] <= new_sim_time:
                completion_time, event_id, event_data = heapq.heappop(schedule)
                logger.info(f"[TimeManager] Processing scheduled event {event_id} ({event_data.get('type')}) due at {completion_time:.1f}")

                results = event_data.get("results", {})
                actor_id_for_memory = event_data.get("actor_id")
                memory_log_updated = False

                for key_path, value in results.items():
                    success = _update_state_value(state, key_path, value) # Pass module-level state
                    if success and actor_id_for_memory and key_path == f"simulacra.{actor_id_for_memory}.memory_log":
                        memory_log_updated = True

                narrative = event_data.get("narrative")
                if narrative:
                    state.setdefault("narrative_log", []).append(f"[T{completion_time:.1f}] {narrative}")

                # Prune memory log if it was updated
                if memory_log_updated and actor_id_for_memory:
                    actor_state_for_mem = get_nested(state, "simulacra", actor_id_for_memory)
                    if actor_state_for_mem and "memory_log" in actor_state_for_mem:
                        current_mem_log = actor_state_for_mem["memory_log"]
                        if isinstance(current_mem_log, list) and len(current_mem_log) > MAX_MEMORY_LOG_ENTRIES:
                            actor_state_for_mem["memory_log"] = current_mem_log[-MAX_MEMORY_LOG_ENTRIES:]
                            logger.debug(f"[TimeManager] Pruned memory log for {actor_id_for_memory} to {MAX_MEMORY_LOG_ENTRIES} entries.")

                # Update actor status to idle upon action completion
                actor_id = event_data.get("actor_id")
                if actor_id and "simulacra" in state:
                    actor_sim_state = state["simulacra"].get(actor_id)
                    if actor_sim_state:
                        # Check if the completion time matches the expected end time
                        expected_end_time = get_nested(actor_sim_state, "current_action_end_time", default=-1)
                        if actor_sim_state.get("status") == "busy" and abs(expected_end_time - completion_time) < 0.01:
                            actor_sim_state["status"] = "idle"
                            actor_sim_state["current_action_end_time"] = completion_time # Update to actual completion
                            logger.info(f"[TimeManager] Set {actor_id} status to idle.")
                            if narrative:
                                actor_sim_state["last_observation"] = narrative
                                logger.info(f"[TimeManager] Stored action narrative in {actor_id}'s last_observation.")
                        elif actor_sim_state.get("status") == "busy":
                            logger.warning(f"[TimeManager] Action completion for {actor_id} at {completion_time:.1f} did not match current busy state end time {expected_end_time}. Status left as busy.")
                        else: # Actor wasn't busy, but action completed (e.g., immediate effect)
                            logger.debug(f"[TimeManager] Action completion for {actor_id} at {completion_time:.1f} occurred while actor was {actor_sim_state.get('status')}.")
                            if narrative:
                                 actor_sim_state["last_observation"] = narrative # Still update observation
                    else:
                        logger.warning(f"[TimeManager] Actor state for '{actor_id}' not found for completed event.")

            live_display.update(generate_table()) # Pass no args, uses module-level state
            await asyncio.sleep(UPDATE_INTERVAL)

    except asyncio.CancelledError:
        logger.info("[TimeManager] Task cancelled.")
    except Exception as e:
        logger.exception(f"[TimeManager] Error in main loop: {e}")
    finally:
        logger.info(f"[TimeManager] Loop finished at sim time {state.get('world_time', 0.0):.1f}")

async def world_engine_task_llm():
    """Listens for intents, calls LLM to resolve, and schedules completions."""
    global schedule_event_counter, state, schedule # Explicitly mention modification
    # Access module-level components
    logger.info("[WorldEngineLLM] Task started.")

    if not adk_runner:
        logger.error("[WorldEngineLLM] Module-level runner not initialized. Task cannot proceed.")
        return
    if not world_engine_agent:
        logger.error("[WorldEngineLLM] Module-level world_engine_agent not initialized. Task cannot proceed.")
        return
    if not adk_session:
        logger.error("[WorldEngineLLM] Module-level ADK session not initialized. Task cannot proceed.")
        return

    session_id_to_use = adk_session.id # Get ID from module-level session

    while True:
        intent_event = None
        actor_id = None
        actor_state_we = {}
        narrative = "Action failed due to internal error (pre-processing)."

        try:
            intent_event = await event_bus.get()

            if get_nested(intent_event, "type") != "intent_declared":
                logger.debug(f"[WorldEngineLLM] Ignoring event type: {get_nested(intent_event, 'type')}")
                event_bus.task_done()
                continue

            actor_id = get_nested(intent_event, "actor_id")
            intent = get_nested(intent_event, "intent")
            if not actor_id or not intent:
                logger.warning(f"[WorldEngineLLM] Received invalid intent event: {intent_event}")
                event_bus.task_done()
                continue

            logger.info(f"[WorldEngineLLM] Received intent from {actor_id}: {intent}")
            action_type = intent.get("action_type")
            actor_state_we = get_nested(state, 'simulacra', actor_id, default={})
            actor_name = actor_state_we.get('name', actor_id)
            current_sim_time = state.get("world_time", 0.0)

            # --- Handle 'talk' action specifically (Bypass LLM) ---
            if action_type == 'talk':
                target_id = intent.get("target_id")
                message = intent.get("details", "")
                target_state = get_nested(state, 'simulacra', target_id) if target_id else None
                target_name = target_state.get('name', target_id) if target_state else target_id

                is_valid = False
                if not target_id: narrative = f"{actor_name} tries to talk, but doesn't specify who."
                elif not target_state: narrative = f"{actor_name} tries to talk to {target_id}, but they don't exist."
                elif get_nested(actor_state_we, 'location') != get_nested(target_state, 'location'): narrative = f"{actor_name} tries to talk to {target_name}, but they are in different locations."
                else:
                    is_valid = True
                    duration = 5.0 + len(message) * 0.1
                    results = {f"simulacra.{target_id}.last_observation": f"{actor_name} said to you: \"{message}\""}
                    narrative = f"{actor_name} says \"{message}\" to {target_name}."

                if is_valid:
                    completion_time = current_sim_time + duration
                    schedule_event_counter += 1
                    completion_event = {"type": "action_complete", "actor_id": actor_id, "action": intent, "results": results, "narrative": narrative}
                    heapq.heappush(schedule, (completion_time, schedule_event_counter, completion_event))
                    if actor_id in state.get("simulacra", {}):
                        state["simulacra"][actor_id]["status"] = "busy"
                        state["simulacra"][actor_id]["current_action_end_time"] = completion_time
                    logger.info(f"[WorldEngineLLM] 'talk' Action VALID for {actor_id}. Scheduled completion at {completion_time:.1f}s. Narrative: {narrative}")
                else:
                    logger.info(f"[WorldEngineLLM] 'talk' Action INVALID for {actor_id}. Reason: {narrative}")
                    if actor_id in state.get("simulacra", {}):
                        state["simulacra"][actor_id]["last_observation"] = narrative
                        state["simulacra"][actor_id]["status"] = "idle" # Set back to idle immediately
                    state.setdefault("narrative_log", []).append(f"[T{current_sim_time:.1f}] {narrative}")

                event_bus.task_done()
                continue # Go to next event

            # --- Handle other actions via LLM ---
            else:
                actor_location_id = get_nested(actor_state_we, "location")
                location_state_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, default={}) # Use constants
                world_rules = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'rules', default={}) # Use constants
                target_id = get_nested(intent, "target_id")
                target_object_state = get_nested(state, 'objects', target_id, default={}) if target_id else {}

                world_rules_json = json.dumps(world_rules, indent=2)
                location_state_json = json.dumps(location_state_data, indent=2)
                intent_json = json.dumps(intent, indent=2)
                target_object_json = json.dumps(target_object_state, indent=2) if target_object_state else "No specific target object."

                # --- TRIGGER TEXT (Context Only) ---
                prompt = f"""
Resolve action for {actor_name} ({actor_id}) at time {current_sim_time:.1f}.
Location: {actor_location_id}
Intent: {intent_json}

Current Relevant State:
Target Object State ({target_id}): {target_object_json}
Location State ({actor_location_id}): {location_state_json}
World Rules: {world_rules_json}

Based on the above state and your instructions, determine the outcome and output the JSON result.
"""
                # --- END TRIGGER TEXT ---

                logger.debug(f"[WorldEngineLLM] Sending prompt to LLM for {actor_id}'s intent ({action_type}).")
                adk_runner.agent = world_engine_agent # Set agent before call
                response_text = ""
                trigger_content = genai_types.Content(parts=[genai_types.Part(text=prompt)])

                async for event_llm in adk_runner.run_async( # Renamed event to event_llm
                    user_id=USER_ID,
                    session_id=session_id_to_use,
                    new_message=trigger_content
                ):
                    if event_llm.is_final_response() and event_llm.content:
                        response_text = event_llm.content.parts[0].text
                        logger.debug(f"WorldLLM Final Content: {response_text[:100]}...")
                    elif event_llm.error_message:
                        logger.error(f"WorldLLM Error: {event_llm.error_message}")
                        narrative = f"Action failed due to LLM error: {event_llm.error_message}"
                        break

                validated_data: Optional[WorldEngineResponse] = None
                if response_text:
                    try:
                        response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                        # --- Placeholder Replacement ---
                        json_str_to_parse = response_text_clean
                        json_str_to_parse = json_str_to_parse.replace("[ACTOR_ID]", actor_id)
                        if target_id:
                             json_str_to_parse = json_str_to_parse.replace("[target_id]", target_id)
                        if target_object_state:
                             obj_name = target_object_state.get("name", target_id)
                             json_str_to_parse = json_str_to_parse.replace("[Object Name]", obj_name)
                        if target_object_state and target_object_state.get("destination"):
                             dest_name = target_object_state.get("destination")
                             json_str_to_parse = json_str_to_parse.replace("[Destination Name]", dest_name)
                             json_str_to_parse = json_str_to_parse.replace("[Destination]", dest_name)
                        json_str_to_parse = json_str_to_parse.replace("[Actor Name]", actor_name)
                        # --- End Placeholder Replacement ---

                        raw_data = json.loads(json_str_to_parse)
                        validated_data = WorldEngineResponse.model_validate(raw_data)
                        logger.debug(f"[WorldEngineLLM] LLM response validated successfully for {actor_id}.")
                        narrative = validated_data.narrative # Use narrative from validated data
                    except json.JSONDecodeError as e:
                        logger.error(f"[WorldEngineLLM] Failed to decode JSON response for {actor_id}: {e}\nResponse:\n{response_text}", exc_info=True)
                        narrative = "Action failed due to internal error (JSON decode)."
                    except ValidationError as e:
                        logger.error(f"[WorldEngineLLM] Failed to validate LLM response structure for {actor_id}: {e}\nResponse:\n{response_text}", exc_info=True)
                        narrative = "Action failed due to internal error (invalid structure)."
                    except Exception as e:
                         logger.error(f"[WorldEngineLLM] Unexpected error parsing/validating response for {actor_id}: {e}\nResponse:\n{response_text}", exc_info=True)
                         narrative = "Action failed due to internal error (unexpected)."
                else:
                    if not narrative.startswith("Action failed due to LLM error"):
                        narrative = "Action failed: No response from World Engine LLM."

                if validated_data and validated_data.valid_action:
                    completion_time = current_sim_time + validated_data.duration
                    schedule_event_counter += 1
                    completion_event = {
                        "type": "action_complete", "actor_id": actor_id,
                        "action": intent, "results": validated_data.results,
                        "narrative": validated_data.narrative # Use validated narrative
                    }
                    heapq.heappush(schedule, (completion_time, schedule_event_counter, completion_event))
                    if actor_id in state.get("simulacra", {}):
                        state["simulacra"][actor_id]["status"] = "busy"
                        state["simulacra"][actor_id]["current_action_end_time"] = completion_time
                    logger.info(f"[WorldEngineLLM] Action VALID for {actor_id}. Scheduled completion at {completion_time:.1f}s. Narrative: {narrative}")
                else:
                    # Use narrative from validated_data if available, otherwise use the error narrative
                    final_narrative = validated_data.narrative if validated_data else narrative
                    logger.info(f"[WorldEngineLLM] Action INVALID for {actor_id}. Reason: {final_narrative}")
                    if actor_id in state.get("simulacra", {}):
                        state["simulacra"][actor_id]["last_observation"] = final_narrative
                        state["simulacra"][actor_id]["status"] = "idle" # Set back to idle immediately
                    actor_name_for_log = get_nested(actor_state_we, 'name', default=actor_id)
                    state.setdefault("narrative_log", []).append(f"[T{current_sim_time:.1f}] {actor_name_for_log}'s action failed: {final_narrative}")

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
            await asyncio.sleep(5) # Wait before processing next event after error
        finally:
            if intent_event and event_bus._unfinished_tasks > 0:
                try: event_bus.task_done()
                except ValueError: logger.warning("[WorldEngineLLM] task_done() called too many times.")
                except Exception as td_e: logger.error(f"[WorldEngineLLM] Error calling task_done(): {td_e}")

async def simulacra_agent_task_llm(agent_id: str):
    """Represents the thinking and acting loop for a single simulacrum using LLM."""
    global state # Explicitly mention modification
    # Access module-level components
    logger.info(f"[{agent_id}] LLM Agent task started.")

    simulacra_agent = simulacra_agents.get(agent_id)
    if not adk_runner:
        logger.error(f"[{agent_id}] Module-level runner not initialized. Task cannot proceed.")
        return
    if not simulacra_agent:
        logger.error(f"[{agent_id}] Module-level agent instance not found for {agent_id}. Task cannot proceed.")
        return
    if not adk_session:
        logger.error(f"[{agent_id}] Module-level ADK session not initialized. Task cannot proceed.")
        return

    session_id_to_use = adk_session.id # Get ID from module-level session

    while True:
        agent_state_sim = None
        try:
            agent_state_sim = get_nested(state, "simulacra", agent_id)
            if not agent_state_sim:
                 logger.error(f"[{agent_id}] State not found for agent {agent_id}. Stopping task.")
                 break

            agent_name = agent_state_sim.get("name", agent_id)

            # Wait while busy
            while agent_state_sim.get("status") == "busy":
                await asyncio.sleep(0.5 + random.uniform(0, 0.5))
                agent_state_sim = get_nested(state, "simulacra", agent_id) # Re-fetch state
                if not agent_state_sim: break # Exit if state disappeared during wait
            if not agent_state_sim: break # Exit outer loop if state disappeared

            # --- Prepare Context for LLM ---
            persona = agent_state_sim.get("persona", {})
            goal = agent_state_sim.get("goal", "Survive and observe.")
            location_id = agent_state_sim.get("location")
            location_details = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, location_id, default={}) # Use constants
            location_desc = location_details.get("description", "An unknown place.")

            # Find objects and other agents in the same location
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

            last_observation = agent_state_sim.get("last_observation", "Just arrived.")
            memory_log = agent_state_sim.get("memory_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
            current_time_for_prompt = state.get("world_time", 0.0)
            recent_history_log = state.get("narrative_log", [])[-6:] # Get recent global narrative
            recent_history_str = "\n".join(recent_history_log)

            # --- TRIGGER TEXT (Context Only) ---
            persona_json = json.dumps(persona, indent=2)
            objects_json = json.dumps(objects_in_loc_data, indent=2) if objects_in_loc_data else "None"
            agents_json = json.dumps(agents_in_loc_data, indent=2) if agents_in_loc_data else "None"
            memory_json = json.dumps(memory_log, indent=2) # Agent's own memory

            prompt = f"""
Current State for {agent_name} ({agent_id}):
Goal: {goal}
Location: {location_id} ({location_desc})
Status: {agent_state_sim.get('status', 'Unknown')}
Time: {current_time_for_prompt:.1f}
Last Observation/Event: {last_observation}
Your Recent Memory Log: {memory_json}
Recent Global History (Last ~6 Events):
{recent_history_str}
Objects in Room (ID: Details): {objects_json}
Other Agents in Room (ID: Details): {agents_json}

Based on this state and your goal, follow your instructions (thinking process, output format) to decide your next action intent.
"""
            # --- END TRIGGER TEXT ---

            logger.debug(f"[{agent_name}] Sending prompt to LLM.")

            # --- Set agent before call ---
            adk_runner.agent = simulacra_agent # Use the specific agent for this simulacra
            response_text = ""
            trigger_content = genai_types.Content(parts=[genai_types.Part(text=prompt)])

            async for event_llm in adk_runner.run_async( # Renamed event to event_llm
                user_id=USER_ID,
                session_id=session_id_to_use,
                new_message=trigger_content
            ):
                if event_llm.is_final_response() and event_llm.content:
                    response_text = event_llm.content.parts[0].text
                    logger.debug(f"SimLLM ({agent_name}) Final Content: {response_text[:100]}...")
                elif event_llm.error_message:
                    logger.error(f"SimLLM ({agent_name}) Error: {event_llm.error_message}")
                    response_text = "" # Clear response on error
                    break

            validated_intent: Optional[SimulacraIntentResponse] = None
            if response_text:
                try:
                   response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                   raw_data = json.loads(response_text_clean)
                   validated_intent = SimulacraIntentResponse.model_validate(raw_data)
                   logger.debug(f"[{agent_name}] LLM response validated successfully.")

                   # Print Monologue and Intent
                   console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_time_for_prompt:.1f}s", border_style="dim yellow", expand=False))
                   intent_dict = {
                       "action_type": validated_intent.action_type,
                       "target_id": validated_intent.target_id,
                       "details": validated_intent.details
                   }
                   console.print(f"\n[bold yellow][{agent_name} Intent @ {current_time_for_prompt:.1f}s][/bold yellow]")
                   console.print(json.dumps(intent_dict, indent=2))

                   # Put intent on the event bus
                   await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": intent_dict})
                   # Update status immediately after putting intent on bus
                   agent_state_sim["status"] = "busy"
                   agent_state_sim["current_action_end_time"] = float('inf') # Mark as busy until resolved

                except json.JSONDecodeError as e:
                   logger.error(f"[{agent_name}] Failed to decode JSON response: {e}\nResponse:\n{response_text}", exc_info=True)
                   console.print(f"[red][{agent_name} Error][/] Failed to decode LLM response. Agent will wait.")
                   await asyncio.sleep(10 + random.uniform(0, 10)) # Wait longer after error
                except ValidationError as e:
                   logger.error(f"[{agent_name}] Failed to validate LLM response structure: {e}\nResponse:\n{response_text}", exc_info=True)
                   console.print(f"[red][{agent_name} Error][/] LLM response has invalid structure. Agent will wait.")
                   await asyncio.sleep(10 + random.uniform(0, 10))
                except Exception as e:
                   logger.error(f"[{agent_name}] Unexpected error parsing/validating response: {e}\nResponse:\n{response_text}", exc_info=True)
                   console.print(f"[red][{agent_name} Error][/] Unexpected error processing LLM response. Agent will wait.")
                   await asyncio.sleep(10 + random.uniform(0, 10))
            else:
                logger.warning(f"[{agent_name}] No valid response received from LLM. Agent will wait.")
                console.print(f"[yellow][{agent_name} Warning][/] No valid response from LLM. Agent will wait.")
                await asyncio.sleep(10 + random.uniform(0, 10))

            # Wait a bit before next decision cycle (only if not busy)
            # The busy check at the start handles waiting if an action is in progress
            await asyncio.sleep(1 + random.uniform(0, 2))

        except asyncio.CancelledError:
            logger.info(f"[{agent_name}] Task cancelled.")
            break
        except Exception as e:
            logger.exception(f"[{agent_name}] Error in main loop: {e}")
            # Attempt to set state back to idle on error
            try:
                if agent_id in get_nested(state, "simulacra", default={}):
                    state["simulacra"][agent_id]["status"] = "idle"
                    state["simulacra"][agent_id]["last_observation"] = f"Encountered unexpected error: {e}"
            except Exception as state_err:
                 logger.error(f"[{agent_name}] Failed to update state after error: {state_err}")
            await asyncio.sleep(10) # Wait longer after an error

# --- Main Execution Logic ---
async def run_simulation(instance_uuid_arg: Optional[str] = None):
    """Sets up ADK and runs all concurrent tasks."""
    # Declare modification of module-level variables
    global adk_session_service, adk_session_id, adk_session, adk_runner
    global world_engine_agent, simulacra_agents, state

    console.rule("[bold green]Starting Async Simulation[/]")

    # --- API Key Check ---
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

    # --- Load State ---
    console.print(Panel(f"[[bold yellow]{APP_NAME}[/]] - Initializing Simulation State...", title="Startup", border_style="blue"))
    logger.info("Starting simulation initialization.")

    world_instance_uuid: Optional[str] = None
    state_file_path: Optional[str] = None
    loaded_state_data: Optional[Dict[str, Any]] = None

    # --- Determine Instance UUID and State File Path ---
    if instance_uuid_arg:
        logger.info(f"Attempting to load specified instance UUID: {instance_uuid_arg}")
        potential_state_path = os.path.join(STATE_DIR, f"simulation_state_{instance_uuid_arg}.json")
        if os.path.exists(potential_state_path):
            world_instance_uuid = instance_uuid_arg
            state_file_path = potential_state_path
            console.print(f"Targeting specified instance state file: {state_file_path}")
        else:
            logger.error(f"State file not found for specified UUID: {instance_uuid_arg} at {potential_state_path}")
            console.print(f"[bold red]Error:[/bold red] State file for specified UUID '{instance_uuid_arg}' not found.")
            sys.exit(1)
    else:
        logger.info("No instance UUID specified, attempting to load the latest state file.")
        latest_state_file = find_latest_simulation_state_file(STATE_DIR)
        if latest_state_file:
            state_file_path = latest_state_file
            match = re.search(r"simulation_state_([a-f0-9\-]+)\.json", os.path.basename(latest_state_file))
            if match:
                world_instance_uuid = match.group(1)
                logger.info(f"Found latest state file: {latest_state_file} (UUID from filename: {world_instance_uuid})")
                console.print(f"Loading latest instance state file: {state_file_path}")
            else:
                logger.error(f"Could not extract UUID from latest state file name: {latest_state_file}")
                console.print(f"[bold red]Error:[/bold red] Could not determine UUID from latest state file '{os.path.basename(latest_state_file)}'. Exiting.")
                sys.exit(1)
        else:
            logger.error("No instance UUID specified and no existing state files found.")
            console.print(f"[bold red]Error:[/bold red] No simulation state files found in '{STATE_DIR}'.")
            console.print("Please run 'setup_simulation.py' first or specify an instance UUID.")
            sys.exit(1)

    # --- Load Simulation State ---
    try:
        logger.info(f"Attempting to load state file: {state_file_path}")
        loaded_state_data = load_json_file(state_file_path)
        if loaded_state_data is None:
             raise FileNotFoundError(f"State file found but failed to load content: {state_file_path}")

        uuid_from_state = loaded_state_data.get("world_instance_uuid")
        if not uuid_from_state:
            logger.critical(f"State file {state_file_path} is missing 'world_instance_uuid'. Cannot proceed.")
            console.print(f"[bold red]Error:[/bold red] State file is missing the 'world_instance_uuid' key.")
            sys.exit(1)
        if uuid_from_state != world_instance_uuid:
            logger.critical(f"UUID mismatch! Filename/Arg suggested '{world_instance_uuid}', but state file contains '{uuid_from_state}'.")
            console.print(f"[bold red]Error:[/bold red] UUID mismatch between state file content ('{uuid_from_state}') and expected UUID ('{world_instance_uuid}').")
            sys.exit(1)

        logger.info(f"Successfully loaded and verified simulation state for UUID: {world_instance_uuid}")
        console.print(f"State File Loaded: {state_file_path}")
        state = loaded_state_data # Assign loaded data to module-level state

    except Exception as e:
        logger.critical(f"Failed to load simulation instance state from {state_file_path}: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to load simulation state file '{state_file_path}'. Check logs. Error: {e}")
        sys.exit(1)

    # --- Ensure State Structure ---
    logger.info("Ensuring essential state structure...")
    state_modified = ensure_state_structure(state) # Pass module-level state
    if state_modified:
        logger.info("State structure updated. Saving state file.")
        try:
            save_json_file(state_file_path, state) # Save module-level state
            logger.info(f"State saved to {state_file_path} after ensuring structure.")
        except Exception as save_e:
             logger.error(f"Failed to save state update after ensuring structure: {save_e}")
             console.print(f"[bold red]Error:[/bold red] Failed to save state update. Check logs.")

    # --- Verify Simulacra & Populate Runtime State ---
    console.rule("[cyan]Verifying Simulacra & Populating Runtime State[/cyan]")
    state_sim_ids = list(state.get(ACTIVE_SIMULACRA_IDS_KEY, []))
    verified_active_sim_ids: List[str] = []

    life_summary_pattern_instance = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_*_{world_instance_uuid}.json")
    available_summary_files = glob.glob(life_summary_pattern_instance)
    available_sim_ids_from_files = set()
    valid_summary_files_map = {}
    for filepath in available_summary_files:
        summary = load_json_file(filepath)
        if summary and summary.get("world_instance_uuid") == world_instance_uuid:
            sim_id_from_file = summary.get("simulacra_id")
            if sim_id_from_file:
                available_sim_ids_from_files.add(sim_id_from_file)
                valid_summary_files_map[sim_id_from_file] = filepath

    if state_sim_ids:
        verified_active_sim_ids = [sid for sid in state_sim_ids if sid in available_sim_ids_from_files]
        missing_ids = set(state_sim_ids) - set(verified_active_sim_ids)
        if missing_ids: logger.warning(f"Simulacra from state ({missing_ids}) missing valid summary files. Ignoring.")
    else:
        logger.warning(f"No Simulacra IDs found in state. Activating based on available files.")
        verified_active_sim_ids = list(available_sim_ids_from_files)

    sim_profiles_from_state = state.get(SIMULACRA_PROFILES_KEY, {})
    # state["simulacra"] is ensured by ensure_state_structure

    final_active_sim_ids = []
    for sim_id in verified_active_sim_ids:
        profile = sim_profiles_from_state.get(sim_id, {})
        persona = profile.get("persona_details")
        if not persona:
            fallback_file = valid_summary_files_map.get(sim_id)
            if fallback_file:
                life_data = load_json_file(fallback_file)
                if life_data and "persona_details" in life_data:
                    persona = life_data["persona_details"]
                    logger.info(f"Loaded persona for {sim_id} from fallback.")
                    # Ensure profile exists before updating
                    state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})["persona_details"] = persona

        if persona:
            current_location = profile.get(CURRENT_LOCATION_KEY)
            home_location = profile.get(HOME_LOCATION_KEY)
            # Ensure location exists, fallback to home or first available
            if not current_location or current_location not in state.get(WORLD_STATE_KEY, {}).get(LOCATION_DETAILS_KEY, {}):
                valid_locations = list(state.get(WORLD_STATE_KEY, {}).get(LOCATION_DETAILS_KEY, {}).keys())
                fallback_loc = home_location if home_location in valid_locations else (valid_locations[0] if valid_locations else "UnknownLocation")
                current_location = fallback_loc
                logger.warning(f"Simulacrum '{sim_id}' missing or invalid current location. Setting to '{current_location}'.")
                state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[CURRENT_LOCATION_KEY] = current_location
                if not home_location or home_location not in valid_locations:
                     state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[HOME_LOCATION_KEY] = current_location

            # Populate the main 'simulacra' runtime state section
            state["simulacra"][sim_id] = {
                "id": sim_id,
                "name": persona.get("Name", sim_id),
                "persona": persona, # Store full persona details here for agent use
                "location": current_location,
                "home_location": home_location or current_location,
                "status": "idle", # Start as idle
                "current_action_end_time": state.get('world_time', 0.0), # Initialize based on current time
                "goal": profile.get("goal", persona.get("Initial_Goal", "Survive.")),
                "last_observation": profile.get("last_observation", "Just arrived."),
                "memory_log": profile.get("memory_log", []) # Load existing memory log
            }
            final_active_sim_ids.append(sim_id)
            logger.info(f"Populated runtime state for simulacrum: {sim_id}")
        else:
            logger.error(f"Could not load persona for active sim {sim_id}. Skipping.")

    state[ACTIVE_SIMULACRA_IDS_KEY] = final_active_sim_ids # Update state with final list

    if not final_active_sim_ids:
         logger.critical("No active simulacra available. Cannot proceed.")
         console.print("[bold red]Error:[/bold red] No verified Simulacra available.")
         sys.exit(1)

    logger.info(f"Initialization complete. Instance {world_instance_uuid} ready with {len(final_active_sim_ids)} simulacra.")
    console.print(f"Running simulation with: {', '.join(final_active_sim_ids)}")
    console.rule()

    # --- Initialize ADK Components (Module Scope) ---
    adk_session_service = InMemorySessionService()
    adk_session_id = f"sim_session_{world_instance_uuid}" # Use consistent ID format
    adk_session = adk_session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=adk_session_id,
        state=state # Pass the fully loaded module-level state here
    )
    logger.info(f"ADK Session created: {adk_session_id} with initial state.")

    # --- Instantiate Agents (Module Scope) ---
    world_engine_agent = create_world_engine_llm_agent()
    logger.info(f"World Engine Agent '{world_engine_agent.name}' created.")

    simulacra_agents = {} # Clear just in case
    for sim_id in final_active_sim_ids:
        sim_state_data = state.get("simulacra", {}).get(sim_id, {})
        persona_name = sim_state_data.get("name", sim_id)
        sim_agent = create_simulacra_llm_agent(sim_id, persona_name)
        simulacra_agents[sim_id] = sim_agent # Store in module-level dict
        logger.info(f"Simulacra Agent '{sim_agent.name}' created for {sim_id}.")

    # --- Create Runner (Module Scope) ---
    # Initialize with a default agent (e.g., world engine)
    adk_runner = Runner(
        agent=world_engine_agent, # Start with world engine as default
        app_name=APP_NAME,
        session_service=adk_session_service
    )
    logger.info(f"ADK Runner initialized with default agent '{world_engine_agent.name}'.")

    # --- Create and Start Tasks ---
    tasks = []
    final_state_path = os.path.join(STATE_DIR, f"simulation_state_{world_instance_uuid}.json")

    try:
        # Use the imported Live class (either real or dummy)
        with Live(generate_table(), console=console, refresh_per_second=1.0/UPDATE_INTERVAL, vertical_overflow="visible") as live:
            # Pass the live display object to the time manager
            tasks.append(asyncio.create_task(time_manager_task(live_display=live), name="TimeManager"))
            tasks.append(asyncio.create_task(world_engine_task_llm(), name="WorldEngine"))

            for sim_id in final_active_sim_ids:
                tasks.append(asyncio.create_task(simulacra_agent_task_llm(agent_id=sim_id), name=f"Simulacra_{sim_id}"))

            if not tasks:
                 logger.error("No tasks were created. Simulation cannot run.")
                 console.print("[bold red]Error: No simulation tasks started.[/bold red]")
            else:
                logger.info(f"Started {len(tasks)} tasks.")
                # Wait for the first task to complete (or fail)
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                # Log results/exceptions from completed tasks
                for task in done:
                    try:
                        task.result() # Raise exception if task failed
                        logger.info(f"Task {task.get_name()} completed normally.")
                    except asyncio.CancelledError:
                        logger.info(f"Task {task.get_name()} was cancelled.")
                    except Exception as task_exc:
                        logger.error(f"Task {task.get_name()} raised an exception: {task_exc}", exc_info=task_exc)
                        console.print(f"[bold red]Error in task {task.get_name()}: {task_exc}[/bold red]")

                logger.info("One of the main tasks completed or failed. Initiating shutdown.")

    except Exception as e:
        logger.exception(f"Error during simulation setup or execution: {e}")
        console.print(f"[bold red]Unexpected error during simulation run: {e}[/bold red]")
    finally:
        logger.info("Cancelling remaining tasks...")
        # Use the 'tasks' list defined in the try block
        if 'tasks' in locals() and tasks:
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to finish cancellation
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Log any exceptions that occurred during cancellation/cleanup
            for i, result in enumerate(results):
                 if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                     task_name = tasks[i].get_name() if i < len(tasks) else f"Task_{i}"
                     logger.error(f"Error during task cleanup for {task_name}: {result}", exc_info=result)
        else:
            logger.warning("No tasks list found or empty during cleanup.")

        logger.info("All tasks cancelled or finished.")

        # --- Save Final State ---
        final_uuid = state.get("world_instance_uuid") # Use module-level state
        if final_uuid:
            final_state_path = os.path.join(STATE_DIR, f"simulation_state_{final_uuid}.json")
            logger.info("Saving final simulation state.")
            try:
                # Ensure world_time is serializable
                if not isinstance(state.get("world_time"), (int, float)):
                     logger.warning(f"Final world_time is not a number ({type(state.get('world_time'))}). Saving as 0.0.")
                     state["world_time"] = 0.0

                save_json_file(final_state_path, state) # Save module-level state
                logger.info(f"Final simulation state saved to {final_state_path}")
                console.print(f"Final state saved to {final_state_path}")
            except Exception as save_e:
                 logger.error(f"Failed to save final state to {final_state_path}: {save_e}", exc_info=True)
                 console.print(f"[red]Error saving final state: {save_e}[/red]")
        else:
             logger.error("Cannot save final state: world_instance_uuid is not defined in module state.")
             console.print("[bold red]Error: Cannot save final state (UUID unknown).[/bold red]")

        console.print("\nFinal State Table:")
        if state:
            console.print(generate_table()) # Use module-level state
        else:
            console.print("[yellow]State dictionary is empty.[/yellow]")

        console.rule("[bold green]Simulation Shutdown Complete[/]")

# Note: The entry point (__main__) is now in main_async.py