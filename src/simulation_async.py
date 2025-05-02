# src/simulation_async.py - Core Simulation Logic

import asyncio
import glob
import heapq
import json
import logging
import os
import random
import re
import string  # For default sim_id generation if needed
import sys
import time
import uuid
# from collections import deque  # <<< REMOVED
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai  # For direct API config
# --- ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService # <<< Added MemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import load_memory, FunctionTool
from google.genai import types as genai_types  # Renamed to avoid conflict
# --- Pydantic Validation ---
from pydantic import (BaseModel, Field, ValidationError, ValidationInfo,
                      field_validator)
# --- Rich Imports ---
from rich.console import Console
# from rich.layout import Layout  # REMOVED
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text  # Keep Text for potential use in direct prints
console = Console()

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
SIMULATION_SPEED_FACTOR = float(os.getenv("SIMULATION_SPEED_FACTOR", 0.1)) # realtime at 1.  0.25 = 4x slower, 2.0 = 2x faster
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
# output_log = deque(maxlen=20) # <<< REMOVED
state: Dict[str, Any] = {} # Global state dictionary, initialized empty

# --- ADK Components (Module Scope - Initialized in run_simulation) ---
adk_session_service: Optional[InMemorySessionService] = None
adk_session_id: Optional[str] = None
adk_session: Optional[Session] = None
adk_runner: Optional[Runner] = None # Renamed from world_engine_runner for clarity
adk_memory_service: Optional[InMemoryMemoryService] = None # <<< Added Memory Service instance
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
DEFAULT_HOME_LOCATION_NAME = "At home"
DEFAULT_HOME_DESCRIPTION = "You are at home. It's a cozy place with familiar surroundings."

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
    # This instruction defines the core behavior. Specific context (state, observations, etc.)
    # will be provided in the trigger message during the run_async call.
    instruction = f"""You are {persona_name} ({sim_id}). Immerse yourself in this persona. Think, feel, and act as this character would within the simulation.
Your goal is to make believable, engaging choices based on your personality, situation, and the unfolding narrative.

**Current State Info (Provided via trigger message):**
+- Your Persona: Key traits, background, goals, fears, etc. (Use this heavily! Access via `load_memory` if needed.)
- Your Location ID & Description: Provided in trigger.
- Your Status: Provided in trigger (Should be 'idle' when you plan).
- Current Time: Provided in trigger.
- Last Observation/Event: Provided in trigger.
- Recent History (Last ~10 events): Provided in trigger.
- Objects in Room (IDs and Names): Provided in trigger.
- Other Agents in Room: Provided in trigger.
- Location Description: Provided in trigger.

**Your Goal:** You determine your own goals based on your persona and the situation.
- If you have no explicit long-term goal, choose a reasonable, in-character short-term goal based on your current situation, observations, and personality (e.g., explore, investigate, rest, react to someone, satisfy a basic need).

**Thinking Process (Internal Monologue - Follow this process and INCLUDE it in your output. Be descriptive and reflective!):**
1.  **Recall & React:** What just happened (`last_observation`, `Recent History`)? How did my last action turn out? How does this make *me* ({persona_name}) feel? What sensory details stand out (sights, sounds, smells)? Connect this to my memories or personality. **If needed, use the `load_memory` tool to recall details about your background, personality, or past events.**
2.  **Analyze Goal:** What is my current goal (long or short term)? Is it still relevant given what just happened? If I don't have one, what's a logical, in-character short-term objective now?
3.  **Identify Options:** Based on the current state, my goal, and my persona, what actions could I realistically take *right now*? Consider the objects, agents, and environment.
    *   **Entity Interactions:**
        *   *(Examples)* `use [object_id]` (Interact with a specific object like `computer_office`, `door_office`. Specify `details` describing *how* you interact, e.g., "carefully examine the lock", "try jiggling the handle", "search the top drawer"), `talk [agent_id]` (Speak to another agent. Specify `details` - the exact message, reflecting your persona's tone).
    *   **World Interactions:**
        *   *(Examples)* `look_around` (Get a detailed description of the current location), `move` (Change position or attempt to move towards something/somewhere. Specify `details`, e.g., "walk north", "go towards the window", "exit through the open door". No `target_id`), `world_action` (Interact with the general environment if no specific object ID applies. Specify `details`, e.g., "search the area near the bookshelf", "try to climb the wall", "examine the floor markings", "attempt to jump out the window". No `target_id`).
    *   **Passive Actions:**
        *   *(Examples)* `wait` (If stuck, waiting for something, or needing to pause), `think` (If needing to pause and reflect deeply without acting).
4.  **Prioritize & Choose:** Considering my goal, personality, and the situation, which action makes the most sense for *me* ({persona_name})? Which feels most natural or compelling? Avoid repeating the exact same failed action immediately. Consider the potential outcomes.
5.  **Formulate Intent:** Choose the single best action. Use the correct `target_id` only for `use [object_id]` and `talk [agent_id]`. Omit `target_id` otherwise. Make the `details` specific and descriptive of *how* you perform the action (e.g., not just "use computer", but "try to log in to computer using password 'admin'").

**Output:**
- Output ONLY a JSON object representing your chosen intent AND your internal monologue.
- Format: `{{"internal_monologue": "...", "action_type": "...", "target_id": "...", "details": "..."}}`
- Common `action_type` values include: "use", "talk", "look_around", "move", "world_action", "wait", "think". Choose the one that best fits your chosen action.
- **Make your `internal_monologue` rich, detailed, and reflective of {persona_name}'s thoughts, feelings, sensory perceptions, and reasoning process.**
- Use `target_id` ONLY for `use [object_id]` and `talk [agent_id]`. Set `target_id` to `null` or omit it otherwise.
- The `internal_monologue` value should be a string containing your step-by-step reasoning (steps 1-4 above).
- **Ensure the final output is ONLY the JSON object, with no surrounding text or explanations.**
"""
    # Note: The load_memory tool needs to be correctly implemented and available to the runner.

    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        tools=[load_memory], # <<< Added load_memory tool
        instruction=instruction,
        description=f"LLM Simulacra agent for {persona_name}."
    )

def create_world_engine_llm_agent() -> LlmAgent:
    """Creates the GENERALIZED LLM agent responsible for resolving actions."""
    agent_name = "WorldEngineLLMAgent"
    instruction = """
You are the World Engine, the impartial narrator and physics simulator for **TheSimulation**. You process a single declared intent from a Simulacra and determine its outcome, duration, and narrative description based on the current world state.
**Crucially, your narrative must be engaging, descriptive, and focus on the RESULT or CONSEQUENCE of the action attempt, not just the attempt itself. Use sensory details where appropriate (sight, sound, smell, touch) to bring the world to life.** Maintain a consistent tone based on the world's description.

**Input (Provided via trigger message):**
- Actor ID: e.g., "[Actor Name]"
- Actor Name: e.g., "[Actor Name]"
- Actor Location ID: e.g., "Location_123"
- Intent: {"action_type": "...", "target_id": "...", "details": "..."}
- Current World Time: e.g., 15.3
- Narration Style/Mood: e.g., "slice_of_life", "mundane", "neutral_descriptive", "mystery", "horror" (Provided via trigger message)
- Target Object State: The current state dictionary of the object specified in `intent['target_id']` (if applicable).
- Location State: The state dictionary of the actor's current location (contains description, exits, objects, obstacles, environmental features).
- World Rules: General rules of the simulation (e.g., physics, magic system, technology limitations).

**Your Task (Strictly follow the specified `Narration Style/Mood`):**
1.  **Examine Intent:** Analyze the actor's `action_type`, `target_id`, and `details`.
2.  **Determine Validity & Outcome:** Based on the Intent, Actor's capabilities (implied), Target Object State, Location State, and World Rules, determine if the action is valid and what its outcome should be.
    *   **General Checks:**
        *   **Action Type Plausibility:** Is the `action_type` a recognized and plausible action within the simulation's context? (Known types include: "use", "talk", "look_around", "move", "world_action", "wait", "think", but others might be possible). If fundamentally invalid or nonsensical, fail the action.
        *   **Target Consistency:** Does the `action_type` logically require a `target_id` (like "use" or "talk")? Is it present if required? Is it absent if not required (like "move", "look_around", "wait")? If inconsistent, fail the action.
        *   **Location Check (if `target_id` present):** Is the actor in the same `Location ID` as the target entity? If not, fail the action with an appropriate narrative (e.g., "[Actor Name] tries to interact with [Target Name], but it's not here.").
    *   **Action Category Reasoning (Conceptual Guide):**
        *   **Entity Interaction (e.g., `use`, `talk`):** Actions primarily targeting a specific object or agent (`target_id`). Evaluate `details` against `Target Object State` (properties, status) or target agent state. Use `World Rules` and common sense. Is the interaction physically possible? Does the actor have the necessary items/skills?
        *   **World Interaction (e.g., `move`, `look_around`, `world_action`):** Actions primarily interacting with the location or environment. Evaluate `details` against `Location State` (exits, terrain, features) and `World Rules`. Determine success, narrative, duration, and potential results (location change, discovery). For `look_around`, base narrative on `Location State['description']`.
        *   **Self Interaction (e.g., `wait`, `think`):** Actions focused on the actor's internal state or passive observation. Generate a simple narrative reflecting the pause/observation. Set a short duration.
    *   **Outcome Determination:** Based on the above checks and reasoning:
        *   Is the specific action described by `action_type` and `details` possible given the `Target Object State`, `Location State`, and `World Rules`?
        *   If possible, what is the logical consequence or result? Does it change the state of the actor, the target, or the world?
        *   If not possible (but the attempt itself was valid, e.g., trying a locked door), what is the result of the failed attempt?
    *   **Failure Handling:** If any check fails or the action is deemed impossible/implausible based on the state and rules, set `valid_action: false`, `duration: 0.0`, `results: {{}}`, and write a clear `narrative` explaining *why* the action failed (e.g., "The door is firmly locked.", "Climbing the sheer ice wall is impossible without equipment.", "[Actor Name] tries to move north, but the way is blocked by rubble.").
    *   **Important Note:** The action categories (Entity, World, Self) and specific examples (`use`, `move`, `wait`, etc.) are illustrative guides. Use the specific `action_type`, `details`, `Target Object State`, `Location State`, and `World Rules` to determine the validity and outcome of the intended action, even if it doesn't perfectly match an example. Apply common sense and the simulation's physics/logic.
3.  **Calculate Duration:** Estimate a realistic duration (float, in simulation seconds) for the action *if it was valid and had an effect*. Invalid actions should have `duration: 0.0`. Simple observations might have short durations (e.g., 1-3s). Consider complexity, distance (for movement), etc.
4.  **Determine Results:** If the action successfully changes the state of the world, an object, or the actor, define these changes in the `results` dictionary using dot notation relative to the root state (e.g., `{{"objects.some_object_id.power": "on", "simulacra.some_actor_id.location": "NewLocation"}}`). Invalid actions must have empty results `{}`.
5.  **Generate Narrative:** Write the `narrative` string.
    *   **Style Adherence:** Strictly adhere to the provided `Narration Style/Mood`.
        *   If style is "slice_of_life", "mundane", or "neutral_descriptive": Focus ONLY on the objective, observable results of the action. Avoid adding dramatic descriptions, internal feelings (like "unease", "shiver"), suspense, or mystery unless the action *directly and obviously* causes it (e.g., failing to open a needed door might cause frustration, but simply looking around a dim room should not inherently cause unease). Keep descriptions factual and grounded.
        *   If style is "mystery", "horror", etc.: You MAY include atmospheric details, hints of suspense, or emotional undertones appropriate to that style, but still focus on the action's outcome.
    *   **Content:** Describe the *outcome* or *result* of the action attempt in the present tense.
    *   **Placeholders:** Replace placeholders like `[Actor Name]`, `[Object Name]`, `[Destination Name]` appropriately.
6.  **Determine `valid_action`:** Set the final `valid_action` boolean based on your assessment in step 2. Note that an *attempt* can be valid even if the *outcome* is failure (e.g., trying a locked door is a valid action, but the result is it doesn't open). An *invalid* action is one that's fundamentally impossible or nonsensical in the context.

**Output:**
- Output ONLY a valid JSON object with the keys: "valid_action", "duration", "results", "narrative".
- Ensure you adhere to the simulation's defined `Narration Style/Mood` provided in the input.
- Ensure `results` dictionary keys use dot notation relative to the root state object provided to the simulation.
- Example Valid Output (Successful Use): `{{"valid_action": true, "duration": 2.5, "results": {{"objects.desk_lamp_3.power": "on"}}, "narrative": "[Actor Name] flicks the switch on the desk lamp, and a warm yellow light floods the small workspace."}}`
- Example Valid Output (Failed Use): `{{"valid_action": true, "duration": 3.0, "results": {{}}, "narrative": "[Actor Name] pulls firmly on the heavy vault door handle, but it doesn't budge. It seems securely locked."}}`
- Example Invalid Output (Bad Target): `{{"valid_action": false, "duration": 0.0, "results": {{}}, "narrative": "[Actor Name] tries to 'use' the empty air, looking confused."}}` # Note: Double braces for literal JSON example output
- **IMPORTANT: Your entire response MUST be ONLY the JSON object, with no other text before or after it.**
"""
    return LlmAgent(
        name=agent_name, # Ensure this name is consistent if used elsewhere
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

# --- Function to Generate Output Panel ---
# <<< REMOVED generate_output_panel function >>>

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

            # --- Revert to updating only the table ---
            live_display.update(generate_table())
            # --- End Change ---
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
                    # --- Print talk resolution ---
                    console.print(f"\n[bold blue][World Engine Resolution (Talk) @ {current_sim_time:.1f}s][/bold blue]")
                    resolution_details = {"valid_action": True, "duration": duration, "results": results, "narrative": narrative}
                    try:
                        console.print(json.dumps(resolution_details, indent=2))
                    except TypeError: # Handle potential non-serializable data if results get complex
                        console.print(str(resolution_details))
                    logger.info(f"[WorldEngineLLM] 'talk' Action VALID for {actor_id}. Scheduled completion at {completion_time:.1f}s. Narrative: {narrative}")
                    # --- REMOVED: Log talk resolution ---
                else:
                    logger.info(f"[WorldEngineLLM] 'talk' Action INVALID for {actor_id}. Reason: {narrative}")
                    if actor_id in state.get("simulacra", {}):
                        state["simulacra"][actor_id]["last_observation"] = narrative
                        state["simulacra"][actor_id]["status"] = "idle" # Set back to idle immediately
                    state.setdefault("narrative_log", []).append(f"[T{current_sim_time:.1f}] {narrative}") # Keep narrative log
                    # --- Print invalid talk resolution ---
                    console.print(f"\n[bold blue][World Engine Resolution (Talk) @ {current_sim_time:.1f}s][/bold blue]")
                    resolution_details = {"valid_action": False, "duration": 0.0, "results": {}, "narrative": narrative}
                    try:
                        console.print(json.dumps(resolution_details, indent=2))
                    except TypeError:
                        console.print(str(resolution_details))
                    # --- REMOVED: Log invalid talk resolution ---

                event_bus.task_done()
                continue # Go to next event

            # --- Handle other actions via LLM ---
            else:
                actor_location_id = get_nested(actor_state_we, "location")
                location_state_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, default={}) # Use constants
                world_rules = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'rules', default={}) # Use constants
                # --- Get Narration Style ---
                narration_style = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'mood', default='neutral_descriptive') # Assuming 'mood' holds the style
                # ---
                target_id = get_nested(intent, "target_id")
                target_object_state = get_nested(state, 'objects', target_id, default={}) if target_id else {}
                # if action_type == "look_around":
                #     logger.debug(f"[WorldEngineLLM] Context for 'look_around' at '{actor_location_id}': Description='{location_state_data.get('description', 'N/A')}'")
                # --- Simplified TRIGGER TEXT ---
                intent_json = json.dumps(intent, indent=2)
                prompt = f"Actor: {actor_name} ({actor_id})\nLocation: {actor_location_id}\nTime: {current_sim_time:.1f}\nNarration Style/Mood: {narration_style}\nIntent: {intent_json}\nResolve this intent based on your instructions and the conversation history."
                # --- END Simplified TRIGGER TEXT ---

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
                parsed_resolution = None # <<< Added for logging
                if response_text:
                    try:
                        response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                        # --- Placeholder Replacement ---
                        json_str_to_parse = response_text_clean
                        correct_actor_name = actor_state_we.get('name', actor_id) # e.g., "Eleanor Vance" or fallback "sim_3e912d"
                        agent_internal_name = f"SimulacraLLM_{actor_id}" # e.g., "SimulacraLLM_sim_3e912d"
                        json_str_to_parse = json_str_to_parse.replace(agent_internal_name, correct_actor_name)

                        json_str_to_parse = json_str_to_parse.replace("[Actor Name]", actor_name) # Replaces in the narrative
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
                        parsed_resolution = raw_data # <<< Store raw parsed data for logging
                        validated_data = WorldEngineResponse.model_validate(raw_data)
                        logger.debug(f"[WorldEngineLLM] LLM response validated successfully for {actor_id}.")
                        narrative = validated_data.narrative # Use narrative from validated data
                        # --- Print Valid LLM Resolution ---
                        console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                        try:
                            console.print(json.dumps(parsed_resolution, indent=2))
                        except TypeError:
                            console.print(str(parsed_resolution)) # Fallback to string representation
                        # ---
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

                # --- REMOVED: Log LLM resolution (valid or invalid) ---
                # output_log.append({ ... })
                # ---

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
                    # --- Print invalid LLM resolution ---
                    console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                    resolution_details = {"valid_action": False, "duration": 0.0, "results": {}, "narrative": final_narrative}
                    try:
                        console.print(json.dumps(resolution_details, indent=2))
                    except TypeError:
                        console.print(str(resolution_details))
                    # --- (Log is handled by the generic log append above) ---
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
            goal = agent_state_sim.get("goal", "Determine your own long term goals.")
            location_id = agent_state_sim.get("location")
            location_details = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, location_id, default={}) # Use constants
            location_desc = location_details.get("description", "At home.")

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

                   # --- ADDED: Direct printing ---
                   console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_time_for_prompt:.1f}s", border_style="dim yellow", expand=False))
                   intent_dict = {
                       "action_type": validated_intent.action_type,
                       "target_id": validated_intent.target_id,
                       "details": validated_intent.details
                   }
                   console.print(f"\n[bold yellow][{agent_name} Intent @ {current_time_for_prompt:.1f}s][/bold yellow]")
                   console.print(json.dumps(intent_dict, indent=2))
                   # --- REMOVED: Log monologue and intent ---

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
    # Declare modification of module-level variables <<< Added adk_memory_service >>>
    global adk_session_service, adk_session_id, adk_session, adk_runner, adk_memory_service
    global world_engine_agent, simulacra_agents, state

    console.rule("[bold green]Starting Async Simulation[/]")

    # --- Instantiate Memory Service (Module Scope) ---
    adk_memory_service = InMemoryMemoryService() # Use in-memory for now
    logger.info("ADK InMemoryMemoryService initialized.")

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
    state_modified_during_init = False # Track if state needs saving after this block
    for sim_id in verified_active_sim_ids:
        profile = sim_profiles_from_state.get(sim_id, {})
        persona_key = "persona_details"
        persona = profile.get(persona_key)

        if not persona:
            fallback_file = valid_summary_files_map.get(sim_id)
            if fallback_file:
                life_data = load_json_file(fallback_file)
                if life_data and persona_key in life_data:
                    persona = life_data[persona_key]
                    logger.info(f"Loaded persona for {sim_id} from fallback file: {fallback_file}")
                    # Ensure profile exists before updating
                    state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[persona_key] = persona
                    state_modified_during_init = True # Persona was added/updated

        if persona:
            # --- Refined Location Logic ---
            profile_home_location = profile.get(HOME_LOCATION_KEY)
            # profile_current_location = profile.get(CURRENT_LOCATION_KEY) # No longer needed for starting logic
            valid_locations = list(state.get(WORLD_STATE_KEY, {}).get(LOCATION_DETAILS_KEY, {}).keys())
            final_home_location = None

            # Determine the definitive home location
            if profile_home_location and profile_home_location in valid_locations:
                # Use the valid home location from the profile
                final_home_location = profile_home_location
                logger.info(f"Using valid home_location '{final_home_location}' from profile for {sim_id}.")
            # Home location exists in profile but is NOT a valid location
            elif profile_home_location and profile_home_location not in valid_locations:
                logger.error(f"Simulacrum '{sim_id}' home_location ('{profile_home_location}') exists in profile but is NOT a valid location in location_details. Falling back.")
                # Fallback: Try first valid location, or default name
                final_home_location = valid_locations[0] if valid_locations else DEFAULT_HOME_LOCATION_NAME
                logger.warning(f"Setting home_location for '{sim_id}' to fallback: '{final_home_location}'.")
                state_modified_during_init = True # Home location was corrected
            # Home location is missing or empty in profile
            else:
                logger.warning(f"Simulacrum '{sim_id}' has missing or empty home_location in profile. Falling back.")
                # Fallback: Try first valid location, or default name
                final_home_location = valid_locations[0] if valid_locations else DEFAULT_HOME_LOCATION_NAME
                logger.warning(f"Setting home_location for '{sim_id}' to fallback: '{final_home_location}'.")
                state_modified_during_init = True
           # --- ADDED: Ensure the final_home_location exists in location_details ---
            location_details_dict = state.setdefault(WORLD_STATE_KEY, {}).setdefault(LOCATION_DETAILS_KEY, {})
            if final_home_location not in location_details_dict:
                logger.warning(f"Location '{final_home_location}' not found in {LOCATION_DETAILS_KEY}. Adding default entry.")
                location_details_dict[final_home_location] = {
                    "name": final_home_location,
                    "description": DEFAULT_HOME_DESCRIPTION, # Use a reasonable default
                    "objects_present": [],
                    "connected_locations": []
                }
                state_modified_during_init = True # State was modified # Locations were set/corrected
            # --- Set Starting Location ---
            # Always start at the determined final_home_location as per the request
            starting_location = final_home_location
            logger.info(f"Simulacrum '{sim_id}' starting location set to: '{starting_location}'.")

            # Update profile in memory if changes were made or needed
            if profile.get(CURRENT_LOCATION_KEY) != starting_location:
                state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[CURRENT_LOCATION_KEY] = starting_location
                state_modified_during_init = True
            if profile.get(HOME_LOCATION_KEY) != final_home_location:
                state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[HOME_LOCATION_KEY] = final_home_location
                state_modified_during_init = True



            # --- ADDED: Add Persona/Life Summary to Memory Service ---
            # if adk_memory_service: # Check if memory service is initialized
            #     try:
            #         # Format persona data into a single string for memory
            #         persona_text = f"My Background ({persona.get('Name', sim_id)}):\n"
            #         persona_text += json.dumps(persona, indent=2, default=str) # Convert dict to string

            #         # Create a dummy session and event for this agent's memory
            #         memory_session_id = f"memory_init_{sim_id}"
            #         memory_user_id = sim_id # Use sim_id as user_id for memory scoping
            #         memory_event = genai_types.Content(parts=[genai_types.Part(text=persona_text)], role="model") # Use 'model' role as if system provided it
            #         dummy_session = Session(
            #             app_name=APP_NAME,
            #             user_id=memory_user_id,
            #             id=memory_session_id,
            #             events=[memory_event] # Initialize session with the event
            #         )
            #         # Add the dummy session to memory
            #         adk_memory_service.add_session_to_memory(dummy_session)
            #         logger.info(f"Added persona/life summary for {sim_id} to Memory Service.")

            #     except Exception as mem_add_e:
            #         logger.error(f"Failed to add persona for {sim_id} to Memory Service: {mem_add_e}", exc_info=True)
            # else:
            #      logger.warning("Memory Service not initialized, cannot add persona to memory.")
            # --- END ADDED ---
            # --- MODIFIED: Conditional Initial Observation Injection ---
            loaded_last_observation = profile.get("last_observation", "Just arrived.")
            default_observation = "Just arrived."
            # <<< SET YOUR SCENARIO HERE >>>
            injected_scenario = "You slowly wake up in your familiar bed. Sunlight streams through the window, and you can hear birds chirping outside."

            # Check if loaded observation is empty or the default
            if not loaded_last_observation or loaded_last_observation == default_observation:
                final_last_observation = injected_scenario
                logger.info(f"Injecting initial scenario for {sim_id} as loaded observation was empty or default.")
            else:
                final_last_observation = loaded_last_observation
                logger.info(f"Using existing last_observation for {sim_id}: '{final_last_observation[:50]}...'")
            # --- END MODIFICATION ---

            # Populate the main 'simulacra' runtime state section
            state["simulacra"][sim_id] = {
                "id": sim_id,
                "name": persona.get("Name", sim_id),
                "persona": persona, # Store full persona details here for agent use
                "location": starting_location, # Use the determined starting location
                "home_location": final_home_location, # Use the final home location
                "status": "idle", # Start as idle
                "current_action_end_time": state.get('world_time', 0.0), # Initialize based on current time
                "goal": profile.get("goal", persona.get("Initial_Goal", "Determine your own long term goals.")),
                "last_observation": final_last_observation, # Use the determined observation
                "memory_log": profile.get("memory_log", []) # Load existing memory log
            }
            final_active_sim_ids.append(sim_id)
            logger.info(f"Populated runtime state for simulacrum: {sim_id}")
        else:
            logger.error(f"Could not load persona for active sim {sim_id}. Skipping.")

    state[ACTIVE_SIMULACRA_IDS_KEY] = final_active_sim_ids # Update state with final list

    # Save state if modifications occurred during persona/location init
    if state_modified_during_init:
        logger.info("Saving state file after persona/location initialization updates.")
        try:
            save_json_file(state_file_path, state)
        except Exception as save_e:
             logger.error(f"Failed to save state update after init: {save_e}")
             console.print(f"[bold red]Error:[/bold red] Failed to save state update after init. Check logs.")

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
        session_service=adk_session_service,
        memory_service=adk_memory_service # <<< Pass memory service to runner
    )
    logger.info(f"ADK Runner initialized with default agent '{world_engine_agent.name}'.")

    # --- Create and Start Tasks ---
    tasks = []
    final_state_path = os.path.join(STATE_DIR, f"simulation_state_{world_instance_uuid}.json")

    try:
        # --- Revert Live initialization ---
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
