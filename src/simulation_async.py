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
event_bus = asyncio.Queue() # For intent -> dispatcher -> world engine
narration_queue = asyncio.Queue() # <<< ADDED: Dedicated queue for narration events
schedule: List[Tuple[float, int, Dict[str, Any]]] = [] # No longer used for action timing
schedule_event_counter = 0 # May not be needed anymore
# output_log = deque(maxlen=20) # <<< REMOVED
state: Dict[str, Any] = {} # Global state dictionary, initialized empty

# --- ADK Components (Module Scope - Initialized in run_simulation) ---
adk_session_service: Optional[InMemorySessionService] = None
adk_session_id: Optional[str] = None
adk_session: Optional[Session] = None
adk_runner: Optional[Runner] = None # Renamed from world_engine_runner for clarity
adk_memory_service: Optional[InMemoryMemoryService] = None
world_engine_agent: Optional[LlmAgent] = None
live_display_object: Optional[Live] = None # <<< ADDED: Global reference for Live display
narration_agent: Optional[LlmAgent] = None
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

def create_blank_simulation_state(new_uuid: str) -> Dict[str, Any]:
    """Creates a dictionary representing a minimal blank simulation state."""
    logger.info(f"Generating blank state structure for new UUID: {new_uuid}")
    return {
      "world_instance_uuid": new_uuid,
      "location_details": {
        "limbo": {
          "name": "Limbo",
          "description": "An empty, featureless starting point.",
          "objects_present": [],
          "connected_locations": []
        }
      },
      "objects": {},
      "active_simulacra_ids": [],
      "world_time": 0.0,
      "narrative_log": [],
      "simulacra": {},
      "npcs": {},
      "current_world_state": {
        "location_details": {
           "limbo": {
              "name": "Limbo",
              "description": "An empty, featureless starting point.",
              "objects_present": [],
              "connected_locations": []
           }
        }
      },
      "world_template_details": {
        "description": "Initial Blank State",
        "rules": {}
      },
      "simulacra_profiles": {}
    }
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
    outcome_description: str # Factual description of what happened
    # narrative: str # <<< REMOVED - Narrative is now handled by a separate agent

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
    instruction = """You are the World Engine, the impartial physics simulator for **TheSimulation**. You process a single declared intent from a Simulacra and determine its **mechanical outcome**, **duration**, and **state changes** based on the current world state. You also provide a concise, factual **outcome description**.
**Crucially, your `outcome_description` must be purely factual and objective, describing only WHAT happened as a result of the action attempt. Do NOT add stylistic flair, sensory details (unless directly caused by the action), or emotional interpretation.** This description will be used by a separate Narrator agent.

**Input (Provided via trigger message):**
- Actor ID: e.g., "[Actor Name]"
- Actor Name: e.g., "[Actor Name]"
- Actor Location ID: e.g., "Location_123"
- Intent: {"action_type": "...", "target_id": "...", "details": "..."}
- Current World Time: e.g., 15.3
# - Narration Style/Mood: (No longer relevant for World Engine)
- Target Entity State: The current state dictionary of the entity specified in `intent['target_id']` (if applicable - could be an object or another simulacra).
- Location State: The state dictionary of the actor's current location (contains description, exits, objects, obstacles, environmental features).
- World Rules: General rules of the simulation (e.g., physics, magic system, technology limitations).

**Your Task (Strictly follow the specified `Narration Style/Mood`):**
1.  **Examine Intent:** Analyze the actor's `action_type`, `target_id`, and `details`.
2.  **Determine Validity & Outcome:** Based on the Intent, Actor's capabilities (implied), Target Entity State, Location State, and World Rules, determine if the action is valid and what its outcome should be.
    *   **General Checks:**
        *   **Action Type Plausibility:** Is the `action_type` a recognized and plausible action within the simulation's context? (Known types include: "use", "talk", "look_around", "move", "world_action", "wait", "think", but others might be possible). If fundamentally invalid or nonsensical, fail the action.
        *   **Target Consistency:** Does the `action_type` logically require a `target_id` (like "use" or "talk")? Is it present if required? Is it absent if not required (like "move", "look_around", "wait")? If inconsistent, fail the action.
        *   **Location Check (if `target_id` present):** Is the actor in the same `Location ID` as the target entity? If not, fail the action with an appropriate narrative (e.g., "[Actor Name] tries to interact with [Target Name], but it's not here.").
        *   **Target Type Check (for `talk`):** If `action_type` is "talk", is the `target_id` another simulacra agent? If not, fail the action with an appropriate `outcome_description`.
    *   **Action Category Reasoning (Conceptual Guide):**
        *   **Entity Interaction (e.g., `use`, `talk`):** Actions primarily targeting a specific object or agent (`target_id`). Evaluate `details` against `Target Entity State` (properties, status) or target agent state. Use `World Rules` and common sense. Is the interaction physically possible? Does the actor have the necessary items/skills?
            *   **Specific `use` Handling:**
                *   Check if the target object exists and is in the same location.
                *   Check if the object has the `"interactive": true` property. If not, the action fails ("The [Object Name] is not interactive.").
                *   Evaluate the `details` of the intent (e.g., "flip the switch", "turn on", "open container").
                *   Based on the object's `properties` (e.g., "toggleable", "lockable", "container") and its current `state` (e.g., `"power": "off"`, `"locked": true`), determine the outcome.
                *   **Example (Light Switch):** If `details` suggest toggling and the object has `"power"` state, flip the state (`"on"` to `"off"` or vice-versa). The `results` should be like `{{"objects.[target_id].state.power": "new_state"}}`. The `outcome_description` should be factual, like "The Light Switch turned on." or "The Light Switch turned off."
                *   **Example (Locked Door):** If `details` suggest opening and the object has `"locked": true`, the action fails. `outcome_description`: "The Door is locked." `results`: `{}`.
        *   **World Interaction (e.g., `move`, `look_around`, `world_action`):** Actions primarily interacting with the location or environment. Evaluate `details` against `Location State` (exits, terrain, features) and `World Rules`. Determine success, narrative, duration, and potential results (location change, discovery). For `look_around`, base narrative on `Location State['description']`.
            *   **Specific `move` Handling:**
                *   If `action_type` is "move", examine the `details` (e.g., "go north", "exit through the door", "go to Hallway").
                *   Check the current `Location State['connected_locations']`. Does the `details` match a valid connection (by direction or target `location_id`)?
                *   If a valid connection is found, the action succeeds. Set `results: {{"simulacra.[actor_id].location": "target_location_id"}}`. Set `outcome_description: "[Actor Name] moved to the [Target Location Name]."`. Estimate a reasonable `duration` (e.g., 5-15s depending on context).
                *   If no valid connection matches the `details`, the action fails. Set `valid_action: false`, `duration: 0.0`, `results: {}`. Set `outcome_description: "The way [details] is blocked or does not exist."` (or similar factual reason).
        *   **Self Interaction (e.g., `wait`, `think`):** Actions focused on the actor's internal state or passive observation. Generate a simple narrative reflecting the pause/observation. Set a short duration.

    *   **Outcome Determination:** Based on the above checks and reasoning:
        *   Is the specific action described by `action_type` and `details` possible given the `Target Entity State`, `Location State`, and `World Rules`?
        *   If possible, what is the logical consequence or result? Does it change the state of the actor, the target, or the world?
        *   If not possible (but the attempt itself was valid, e.g., trying a locked door), what is the result of the failed attempt?
    *   **Failure Handling:** If any check fails or the action is deemed impossible/implausible based on the state and rules, set `valid_action: false`, `duration: 0.0`, `results: {{}}`, and write a clear, factual `outcome_description` explaining *why* the action failed (e.g., "The door is locked.", "The way north is blocked.", "Target entity is not in the same location.").
    *   **Specific Handling for `talk`:**
        *   **Validity:** Check if `target_id` is a valid simulacra ID and if both actor and target are in the same location.
        *   **Duration:** Calculate based on message length (e.g., 5.0 + len(details) * 0.1).
        *   **Results:** `{{"simulacra.[target_id].last_observation": "[Actor Name] said to you: \"[details]\""}}` (Replace placeholders).
        *   **Outcome Description:** `"[Actor Name] said \"[details]\" to [Target Name]."` (Replace placeholders). This is factual.
    *   **Important Note:** The action categories (Entity, World, Self) and specific examples (`use`, `move`, `wait`, etc.) are illustrative guides. Use the specific `action_type`, `details`, `Target Entity State`, `Location State`, and `World Rules` to determine the validity and outcome of the intended action, even if it doesn't perfectly match an example. Apply common sense and the simulation's physics/logic.
3.  **Calculate Duration:** Estimate a realistic duration (float, in simulation seconds) for the action *if it was valid and had an effect*. Invalid actions should have `duration: 0.0`. Simple observations might have short durations (e.g., 1-3s). Consider complexity, distance (for movement), etc.
4.  **Determine Results:** If the action successfully changes the state of the world, an object, or the actor, define these changes in the `results` dictionary using dot notation relative to the root state (e.g., `{{"objects.some_object_id.power": "on", "simulacra.some_actor_id.location": "NewLocation"}}`). Invalid actions must have empty results `{}`.
5.  **Generate Factual Outcome Description:** Write the `outcome_description` string.
    *   **Style Adherence:** STRICTLY FACTUAL and OBJECTIVE. Describe only the direct, observable result or failure reason. No interpretation, emotion, or stylistic elements.
    *   **Content:** Describe the *outcome* or *result* of the action attempt in the present tense. (e.g., "The lamp turned on.", "The door remained locked.", "[Actor Name] moved to the Corridor.", "[Actor Name] said 'Hello' to [Target Name].", "Nothing was found.").
    *   **Placeholders:** Replace placeholders like `[Actor Name]`, `[Object Name]`, `[Target Name]`, `[Destination Name]` appropriately.
6.  **Determine `valid_action`:** Set the final `valid_action` boolean based on your assessment in step 2. Note that an *attempt* can be valid even if the *outcome* is failure (e.g., trying a locked door is a valid action, but the result is it doesn't open). An *invalid* action is one that's fundamentally impossible or nonsensical in the context.

**Output:**
- Output ONLY a valid JSON object with the keys: "valid_action", "duration", "results", "outcome_description".
# - Ensure you adhere to the simulation's defined `Narration Style/Mood` provided in the input. (REMOVED)
- Ensure `results` dictionary keys use dot notation relative to the root state object provided to the simulation.
- Example Valid Output (Successful Use): `{{"valid_action": true, "duration": 2.5, "results": {{"objects.desk_lamp_3.power": "on"}}, "outcome_description": "The desk lamp turned on."}}`
- Example Valid Output (Failed Use): `{{"valid_action": true, "duration": 3.0, "results": {{}}, "outcome_description": "The vault door handle did not move; it is locked."}}`
- Example Invalid Output (Bad Target): `{{"valid_action": false, "duration": 0.0, "results": {{}}, "outcome_description": "[Actor Name] attempted to 'use' something that was not present."}}` # Note: Double braces for literal JSON example output
- **IMPORTANT: Your entire response MUST be ONLY the JSON object, with no other text before or after it.**
"""
    return LlmAgent(
        name=agent_name, # Ensure this name is consistent if used elsewhere
        model=MODEL_NAME,
        instruction=instruction,
        description="LLM World Engine: Resolves action mechanics, calculates duration/results, generates factual outcome description."
    )

def create_narration_llm_agent() -> LlmAgent:
    """Creates the LLM agent responsible for generating stylized narrative."""
    agent_name = "NarrationLLMAgent"
    instruction = """
You are the Narrator for **TheSimulation**. Your role is to weave the factual outcomes of actions into an engaging and atmospheric narrative, matching the specified world style.

**Input (Provided via trigger message):**
- Actor ID: e.g., "sim_abc"
- Actor Name: e.g., "[Actor Name],"
- Original Intent: {"action_type": "...", "target_id": "...", "details": "..."}
- Factual Outcome Description: A concise, objective sentence describing what happened (e.g., "The lamp turned on.", "The door remained locked.", "[Actor Name], moved to the Corridor.").
- State Changes (Results): Dictionary of state changes, if any (e.g., `{{"objects.lamp.power": "on"}}`).
- Current World Time: e.g., 25.8
- World Style/Mood: e.g., "mystery", "slice_of_life", "horror", "cyberpunk_noir".
- Recent Narrative History (Last ~5 entries): Provided for context and flow.

**Your Task:**
1.  **Understand the Event:** Read the Actor, their Intent, and the Factual Outcome Description.
2.  **Consider the Context:** Note the World Style/Mood and the Recent Narrative History.
3.  **Generate Narrative:** Write a single, engaging narrative paragraph in the **present tense** that describes the event based on the Factual Outcome Description.
    *   **Style Adherence:** STRICTLY adhere to the provided `World Style/Mood`. Infuse the description with appropriate atmosphere, sensory details (sight, sound, smell, touch), and tone.
    *   **Show, Don't Just Tell:** Instead of just repeating the outcome, describe it vividly. (e.g., If outcome is "The lamp turned on.", narrative could be "[Actor Name], flicks the switch. With a soft click, a warm, yellow glow pushes back the shadows in the small workspace, revealing dust motes dancing in the beam.")
    *   **Incorporate Intent (Optional):** Briefly reference the actor's intent if it adds to the narrative (e.g., "Trying the handle again, [Actor Name], confirms the door is still firmly locked.").
    *   **Flow:** Ensure the narrative flows reasonably from the Recent Narrative History.

**Output:**
- Output ONLY the final narrative string. Do NOT include explanations, prefixes, or JSON formatting.
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME, # Consider using a model good at creative writing if needed
        instruction=instruction,
        description="LLM Narrator: Generates stylized narrative based on factual action outcomes and world mood."
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
         # --- ADDED: Check for 'interactive' property ---
         obj_interactive = get_nested(obj_state_data, 'interactive')
         # ---
         details = f"Loc: {obj_loc}"
         if obj_power is not None: details += f", Pwr: {obj_power}"
         if obj_locked is not None: details += f", Lck: {'Y' if obj_locked else 'N'}"
         if obj_status is not None: details += f", Sts: {obj_status}"
         # --- ADDED: Display 'interactive' status ---
         if obj_interactive is not None: details += f", Int: {'Y' if obj_interactive else 'N'}"
         # ---
         table.add_row(f"  {obj_name}", details)

    table.add_row("--- System ---", "---")
    # table.add_row("Schedule Size", str(len(schedule))) # Schedule no longer used this way
    table.add_row("Event Bus Size", str(event_bus.qsize()))
    table.add_row("Narration Q Size", str(narration_queue.qsize())) # Show narration queue size
    log_display = "\n".join(get_nested(state, 'narrative_log', default=[])[-6:])
    table.add_row("Narrative Log", log_display)
    return table

# --- Function to Generate Output Panel ---
# <<< REMOVED generate_output_panel function >>>

async def time_manager_task(live_display: Live):
    """Advances time, applies completed action effects, and updates display."""
    global state # Explicitly mention modification of module-level vars
    logger.info("[TimeManager] Task started.")
    last_real_time = time.monotonic()

    try:
        while state.get("world_time", 0.0) < MAX_SIMULATION_TIME:
            current_real_time = time.monotonic() # Real time used for pacing
            real_delta_time = current_real_time - last_real_time
            last_real_time = current_real_time
            sim_delta_time = real_delta_time * SIMULATION_SPEED_FACTOR
            current_sim_time = state.setdefault("world_time", 0.0)
            new_sim_time = current_sim_time + sim_delta_time
            state["world_time"] = new_sim_time

            # --- Process Action Completions Based on Time ---
            for agent_id, agent_state in state.get("simulacra", {}).items():
                if agent_state.get("status") == "busy":
                    action_end_time = agent_state.get("current_action_end_time", -1.0)
                    if action_end_time <= new_sim_time:
                        logger.info(f"[TimeManager] Applying completed action effects for {agent_id} at time {new_sim_time:.1f} (due at {action_end_time:.1f}).")

                        # Apply pending results
                        pending_results = agent_state.get("pending_results", {})
                        if pending_results:
                            memory_log_updated = False
                            for key_path, value in pending_results.items():
                                success = _update_state_value(state, key_path, value)
                                if success and key_path == f"simulacra.{agent_id}.memory_log":
                                    memory_log_updated = True
                            agent_state["pending_results"] = {} # Clear pending results

                            # Prune memory log if it was updated
                            if memory_log_updated:
                                if "memory_log" in agent_state:
                                    current_mem_log = agent_state["memory_log"]
                                    if isinstance(current_mem_log, list) and len(current_mem_log) > MAX_MEMORY_LOG_ENTRIES:
                                        agent_state["memory_log"] = current_mem_log[-MAX_MEMORY_LOG_ENTRIES:]
                                        logger.debug(f"[TimeManager] Pruned memory log for {agent_id} to {MAX_MEMORY_LOG_ENTRIES} entries.")
                        else:
                            logger.debug(f"[TimeManager] No pending results found for completed action of {agent_id}.")

                        # Set agent to idle
                        agent_state["status"] = "idle"
                        logger.info(f"[TimeManager] Set {agent_id} status to idle.")

            # --- Update Live Display ---
            live_display.update(generate_table())
            await asyncio.sleep(UPDATE_INTERVAL)

    except asyncio.CancelledError:
        logger.info("[TimeManager] Task cancelled.")
    except Exception as e:
        logger.exception(f"[TimeManager] Error in main loop: {e}")
    finally:
        logger.info(f"[TimeManager] Loop finished at sim time {state.get('world_time', 0.0):.1f}")

async def interaction_dispatcher_task():
    """Listens for intents and classifies them before sending to World Engine."""
    logger.info("[InteractionDispatcher] Task started.")
    while True:
        intent_event = None
        try:
            intent_event = await event_bus.get()
            if get_nested(intent_event, "type") != "intent_declared":
                logger.debug(f"[InteractionDispatcher] Ignoring event type: {get_nested(intent_event, 'type')}")
                event_bus.task_done()
                continue

            actor_id = get_nested(intent_event, "actor_id")
            intent = get_nested(intent_event, "intent")
            if not actor_id or not intent:
                logger.warning(f"[InteractionDispatcher] Received invalid intent event: {intent_event}")
                event_bus.task_done()
                continue

            target_id = intent.get("target_id")
            action_type = intent.get("action_type")
            interaction_class = "environment" # Default

            # Classify interaction
            if target_id:
                # Check if target is another Simulacra
                if target_id in get_nested(state, "simulacra", default={}):
                    interaction_class = "entity"
                # Check if target is an interactive object (assuming 'interactive' property exists)
                elif target_id in get_nested(state, "objects", default={}) and get_nested(state, "objects", target_id, "interactive", default=False):
                     interaction_class = "entity"
                # Optional: Consider specific action types that imply entity interaction even without target_id? (e.g., 'shout') - Not implemented here.

            logger.info(f"[InteractionDispatcher] Intent from {actor_id} ({action_type} on {target_id or 'N/A'}) classified as '{interaction_class}'.")

            # Put classified event for World Engine
            await event_bus.put({"type": "resolve_action_request", "actor_id": actor_id, "intent": intent, "interaction_class": interaction_class})
            event_bus.task_done()
            # Yield control to allow other tasks (like World Engine) to potentially grab the event
            await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("[InteractionDispatcher] Task cancelled.")
            if intent_event and event_bus._unfinished_tasks > 0:
                try: event_bus.task_done()
                except ValueError: pass
            break
        except Exception as e:
            logger.exception(f"[InteractionDispatcher] Error processing event: {e}")
            if intent_event and event_bus._unfinished_tasks > 0:
                try: event_bus.task_done()
                except ValueError: pass # Avoid error if already done

async def narration_task():
    """Listens for completed actions on the narration queue and generates stylized narrative."""
    global live_display_object # <<< ADDED: Access global live display
    logger.info("[NarrationTask] Task started.")

    if not adk_runner:
        logger.error("[NarrationTask] Module-level runner not initialized. Task cannot proceed.")
        return
    if not narration_agent:
        logger.error("[NarrationTask] Module-level narration_agent not initialized. Task cannot proceed.")
        return
    if not adk_session:
        logger.error("[NarrationTask] Module-level ADK session not initialized. Task cannot proceed.")
        return

    session_id_to_use = adk_session.id

    while True:
        action_event = None
        try:
            # <<< MODIFIED: Listen to narration_queue >>>
            action_event = await narration_queue.get()
            # No need to check type, assume only valid narration events are put here

            actor_id = get_nested(action_event, "actor_id")
            intent = get_nested(action_event, "action") # Original intent
            results = get_nested(action_event, "results", default={}) # Results might still be useful context for narrator
            outcome_desc = get_nested(action_event, "outcome_description", default="Something happened.")
            completion_time = get_nested(action_event, "completion_time", default=state.get("world_time", 0.0)) # Use event time if available

            if not actor_id:
                logger.warning(f"[NarrationTask] Received narration event without actor_id: {action_event}")
                narration_queue.task_done() # <<< MODIFIED
                continue

            actor_name = get_nested(state, "simulacra", actor_id, "name", default=actor_id)
            world_mood = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'mood', default='neutral_descriptive')
            # --- MODIFIED: Clean the recent narrative history for the prompt ---

            def clean_history_entry(entry: str) -> str:
                # Remove timestamp
                cleaned = re.sub(r'^\[T\d+\.\d+\]\s*', '', entry)
                # Remove agent identifiers and JSON blocks (basic attempt)
                cleaned = re.sub(r'\[\w+Agent(?:_sim_\w+)?\] said: ```json.*?```', '', cleaned, flags=re.DOTALL).strip()
                # Add more specific cleaning rules if needed
                return cleaned

            raw_recent_narrative = state.get("narrative_log", [])[-5:] # Get last 5 raw entries
            cleaned_recent_narrative = [clean_history_entry(entry) for entry in raw_recent_narrative if clean_history_entry(entry)] # Apply cleaning and filter empty results
            history_str = "\n".join(cleaned_recent_narrative)
            # ---

            logger.info(f"[NarrationTask] Generating narrative for {actor_name}'s action completion. Outcome: '{outcome_desc}'")

            # --- Construct Prompt ---
            intent_json = json.dumps(intent, indent=2)
            results_json = json.dumps(results, indent=2) # Pass results for context
            # history_str = "\n".join(recent_narrative) # No longer needed here

            prompt = f"""
Actor ID: {actor_id}
Actor Name: {actor_name}
Original Intent: {intent_json}
Factual Outcome Description: {outcome_desc}
State Changes (Results): {results_json}
Current World Time: {completion_time:.1f}
World Style/Mood: {world_mood}
Recent Narrative History (Cleaned):
{history_str}

Generate the narrative paragraph based on these details and your instructions.
"""
            # --- Call Narration LLM ---
            adk_runner.agent = narration_agent # Set agent before call
            narrative_text = ""
            trigger_content = genai_types.Content(parts=[genai_types.Part(text=prompt)])

            async for event_llm in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=trigger_content):
                if event_llm.is_final_response() and event_llm.content:
                    narrative_text = event_llm.content.parts[0].text.strip()
                    logger.debug(f"NarrationLLM Final Content: {narrative_text[:100]}...")
                elif event_llm.error_message:
                    logger.error(f"NarrationLLM Error: {event_llm.error_message}")
                    narrative_text = f"[{actor_name}'s action resulted in: {outcome_desc}]" # Fallback narrative

            # --- ADDED: Post-process the narrative text ---
            cleaned_narrative_text = narrative_text
            if narrative_text:
                # Attempt to find the actual narrative after the "Input: ..." block
                # Look for a double newline which often separates the preamble from the actual response
                parts = narrative_text.split('\n\n', 1)
                if len(parts) > 1 and "Actor ID:" in parts[0]: # Check if the first part looks like the input block
                    cleaned_narrative_text = parts[1].strip()
                else:
                    # Fallback if split doesn't work as expected, try removing common prefixes
                    cleaned_narrative_text = re.sub(r'^Input:.*?\n\n', '', narrative_text, flags=re.DOTALL).strip()

                # Replace internal agent name placeholder with actual name
                internal_agent_name_placeholder = f"[SimulacraLLM_{actor_id}]"
                cleaned_narrative_text = cleaned_narrative_text.replace(internal_agent_name_placeholder, actor_name)
            # --- END Post-processing ---

            # --- ADDED: Print the generated narrative to the console ---
            # <<< MODIFIED: Use cleaned_narrative_text >>>
            if cleaned_narrative_text and live_display_object:
                live_display_object.console.print(Panel(cleaned_narrative_text, title=f"Narrator @ {completion_time:.1f}s", border_style="green", expand=False))
            elif cleaned_narrative_text: # Fallback if live object not ready
                console.print(Panel(cleaned_narrative_text, title=f"Narrator @ {completion_time:.1f}s", border_style="green", expand=False))
            # ---

            # --- Update State ---
            # <<< MODIFIED: Use cleaned_narrative_text >>>
            final_narrative_entry = f"[T{completion_time:.1f}] {cleaned_narrative_text}"
            state.setdefault("narrative_log", []).append(final_narrative_entry)
            if actor_id in state.get("simulacra", {}):
                state["simulacra"][actor_id]["last_observation"] = cleaned_narrative_text # Update agent's observation
            logger.info(f"[NarrationTask] Appended narrative for {actor_name}: {cleaned_narrative_text[:80]}...")
            narration_queue.task_done() # <<< MODIFIED

        except asyncio.CancelledError:
            logger.info("[NarrationTask] Task cancelled.")
            if action_event and narration_queue._unfinished_tasks > 0: # <<< MODIFIED
                try: narration_queue.task_done() # <<< MODIFIED
                except ValueError: pass
            break
        except Exception as e:
            logger.exception(f"[NarrationTask] Error processing event: {e}")
            if action_event and narration_queue._unfinished_tasks > 0: # <<< MODIFIED
                try: narration_queue.task_done() # <<< MODIFIED
                except ValueError: pass

async def world_engine_task_llm():
    """Listens for action requests, calls LLM to resolve, stores results, and triggers narration."""
    global state # Explicitly mention modification
    global live_display_object # <<< ADDED: Access global live display
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
        request_event = None # Renamed from intent_event
        actor_id = None
        actor_state_we = {}
        outcome_description = "Action failed due to internal error (pre-processing)." # Default outcome desc

        try:
            request_event = await event_bus.get()

            # --- MODIFIED: Listen for the new event type ---
            if get_nested(request_event, "type") != "resolve_action_request":
                logger.debug(f"[WorldEngineLLM] Ignoring event type: {get_nested(request_event, 'type')}")
                event_bus.task_done()
                continue

            actor_id = get_nested(request_event, "actor_id")
            intent = get_nested(request_event, "intent")
            interaction_class = get_nested(request_event, "interaction_class", default="environment") # Get classification
            if not actor_id or not intent:
                logger.warning(f"[WorldEngineLLM] Received invalid action request event: {request_event}")
                event_bus.task_done()
                continue

            logger.info(f"[WorldEngineLLM] Received '{interaction_class}' action request from {actor_id}: {intent}")
            action_type = intent.get("action_type")
            actor_state_we = get_nested(state, 'simulacra', actor_id, default={})
            actor_name = actor_state_we.get('name', actor_id)
            current_sim_time = state.get("world_time", 0.0)

            # --- REMOVED: 'talk' action bypass block ---

            # --- Handle ALL actions via LLM ---
            actor_location_id = get_nested(actor_state_we, "location")
            location_state_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, default={}) # Use constants
            world_rules = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'rules', default={}) # Use constants
            # narration_style = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'mood', default='neutral_descriptive') # No longer needed here

            target_id = get_nested(intent, "target_id")
            # --- MODIFIED: Get target state (could be object OR simulacra) ---
            target_state_data = {}
            if target_id:
                target_state_data = get_nested(state, 'objects', target_id, default={}) or get_nested(state, 'simulacra', target_id, default={})
            # ---

            # --- Simplified TRIGGER TEXT ---
            intent_json = json.dumps(intent, indent=2)
            target_state_json = json.dumps(target_state_data, indent=2) if target_state_data else "N/A"
            location_state_json = json.dumps(location_state_data, indent=2) # Assuming location_state_data is always a dict
            world_rules_json = json.dumps(world_rules, indent=2)

            prompt = f"""
Actor: {actor_name} ({actor_id})
Location: {actor_location_id}
Time: {current_sim_time:.1f}
Intent: {intent_json}
Target Entity State ({target_id or 'N/A'}): {target_state_json}
Location State: {location_state_json}
World Rules: {world_rules_json}

Resolve this intent based on your instructions and the provided context.
"""
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
                    outcome_description = f"Action failed due to LLM error: {event_llm.error_message}"
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
                    if target_state_data: # Use combined target_state_data
                         obj_name = target_state_data.get("name", target_id)
                         json_str_to_parse = json_str_to_parse.replace("[Object Name]", obj_name)
                         # --- ADDED: Replace Target Name for 'talk' ---
                         if action_type == 'talk':
                             target_name = target_state_data.get("name", target_id)
                             json_str_to_parse = json_str_to_parse.replace("[Target Name]", target_name)
                         # ---
                    # Check for destination specifically on objects (assuming only objects have destinations for now)
                    target_object_state = get_nested(state, 'objects', target_id, default={}) if target_id else {}
                    if target_object_state and target_object_state.get("destination"):
                         dest_name = target_object_state.get("destination")
                         json_str_to_parse = json_str_to_parse.replace("[Destination Name]", dest_name)
                         json_str_to_parse = json_str_to_parse.replace("[Destination]", dest_name)
                    # --- End Placeholder Replacement ---

                    raw_data = json.loads(json_str_to_parse)
                    parsed_resolution = raw_data # <<< Store raw parsed data for logging
                    validated_data = WorldEngineResponse.model_validate(raw_data)
                    logger.debug(f"[WorldEngineLLM] LLM response validated successfully for {actor_id}.")
                    outcome_description = validated_data.outcome_description # Use outcome_description from validated data
                    # --- MODIFIED: Use live_display_object.console.print ---
                    if live_display_object:
                        live_display_object.console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                        try:
                            live_display_object.console.print(json.dumps(parsed_resolution, indent=2))
                        except TypeError:
                            live_display_object.console.print(str(parsed_resolution)) # Fallback
                    else: # Fallback
                        console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                        try:
                            console.print(json.dumps(parsed_resolution, indent=2))
                        except TypeError:
                            console.print(str(parsed_resolution))
                    # ---
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

            # --- REMOVED: Log LLM resolution (valid or invalid) ---
            # output_log.append({ ... })
            # ---

            if validated_data and validated_data.valid_action:
                completion_time = current_sim_time + validated_data.duration
                # --- MODIFIED: Store results in agent state, put event on narration_queue ---
                narration_event = {
                    "type": "action_complete", # Keep type for potential future filtering
                    "actor_id": actor_id,
                    "action": intent, # Original intent for narrator context
                    "results": validated_data.results, # Include results for narrator context
                    "outcome_description": validated_data.outcome_description, # Pass factual outcome
                    "completion_time": completion_time # Pass completion time for narrator
                }
                if actor_id in state.get("simulacra", {}):
                    state["simulacra"][actor_id]["status"] = "busy"
                    state["simulacra"][actor_id]["pending_results"] = validated_data.results # Store results
                    state["simulacra"][actor_id]["current_action_end_time"] = completion_time
                    await narration_queue.put(narration_event) # <<< Put on narration queue
                    logger.info(f"[WorldEngineLLM] Action VALID for {actor_id}. Stored results, set end time {completion_time:.1f}s. Triggered narration. Outcome: {outcome_description}")
                else:
                    logger.error(f"[WorldEngineLLM] Actor {actor_id} not found in state after valid action resolution. Cannot store results or trigger narration.")

            else:
                # Use outcome_description from validated_data if available, otherwise use the error description
                final_outcome_desc = validated_data.outcome_description if validated_data else outcome_description
                logger.info(f"[WorldEngineLLM] Action INVALID for {actor_id}. Reason: {final_outcome_desc}")

                # --- Action failed, update observation immediately with factual outcome ---
                if actor_id in state.get("simulacra", {}):
                    state["simulacra"][actor_id]["last_observation"] = final_outcome_desc # Give agent factual failure reason
                    state["simulacra"][actor_id]["status"] = "idle" # Set back to idle immediately
                actor_name_for_log = get_nested(state, 'simulacra', actor_id, 'name', default=actor_id) # Fetch name again safely
                # --- MODIFIED: Use live_display_object.console.print ---
                # Use final_outcome_desc here
                resolution_details = {"valid_action": False, "duration": 0.0, "results": {}, "outcome_description": final_outcome_desc}
                if live_display_object:
                    live_display_object.console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                    try:
                        live_display_object.console.print(json.dumps(resolution_details, indent=2))
                    except TypeError:
                        live_display_object.console.print(str(resolution_details))
                else: # Fallback
                    console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                    try: # Fallback print
                        console.print(json.dumps(resolution_details, indent=2))
                    except TypeError:
                        console.print(str(resolution_details))
                # --- (Log is handled by the generic log append above) ---

                # --- Append factual failure to narrative log immediately ---
                state.setdefault("narrative_log", []).append(f"[T{current_sim_time:.1f}] {actor_name_for_log}'s action failed: {final_outcome_desc}")

        except asyncio.CancelledError: # <<< Added CancelledError handling
            logger.info("[WorldEngineLLM] Task cancelled.")
            if request_event and event_bus._unfinished_tasks > 0:
                try: event_bus.task_done()
                except ValueError: pass
            break
        except Exception as e:
            logger.exception(f"[WorldEngineLLM] Error processing event for actor {actor_id}: {e}")
            if actor_id:
                 actor_name_for_log = get_nested(state, 'simulacra', actor_id, 'name', default=actor_id)
                 state.setdefault("narrative_log", []).append(f"[T{state.get('world_time', 0.0):.1f}] {actor_name_for_log}'s action failed unexpectedly: {e}")
                 if actor_id in get_nested(state, "simulacra", default={}):
                     # Ensure agent is idle and has pending results cleared on unexpected error
                     state["simulacra"][actor_id]["status"] = "idle"
                     state["simulacra"][actor_id]["pending_results"] = {} # Clear pending results
                     state["simulacra"][actor_id]["last_observation"] = f"Action failed unexpectedly: {e}"
            await asyncio.sleep(5) # Wait before processing next event after error
        finally:
            if request_event and event_bus._unfinished_tasks > 0:
                try: event_bus.task_done()
                except ValueError: logger.warning("[WorldEngineLLM] task_done() called too many times.")
                except Exception as td_e: logger.error(f"[WorldEngineLLM] Error calling task_done(): {td_e}")

async def simulacra_agent_task_llm(agent_id: str):
    """Represents the thinking and acting loop for a single simulacrum using LLM."""
    global state # Explicitly mention modification
    global live_display_object # <<< ADDED: Access global live display
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
                        "properties": obj_data.get("properties", []),
                        "interactive": obj_data.get("interactive", False) # Include interactive flag
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

                   # --- MODIFIED: Use live_display_object.console.print ---
                   if live_display_object:
                       live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_time_for_prompt:.1f}s", border_style="dim yellow", expand=False))
                       intent_dict = {
                           "action_type": validated_intent.action_type,
                           "target_id": validated_intent.target_id,
                           "details": validated_intent.details
                       }
                       live_display_object.console.print(f"\n[bold yellow][{agent_name} Intent @ {current_time_for_prompt:.1f}s][/bold yellow]")
                       live_display_object.console.print(json.dumps(intent_dict, indent=2))
                   else: # Fallback if live object not ready
                       # This fallback might still get overwritten, but it's better than crashing
                       console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_time_for_prompt:.1f}s", border_style="dim yellow", expand=False))
                       intent_dict = {
                           "action_type": validated_intent.action_type,
                           "target_id": validated_intent.target_id,
                           "details": validated_intent.details
                       }
                       console.print(f"\n[bold yellow][{agent_name} Intent @ {current_time_for_prompt:.1f}s][/bold yellow]")
                       console.print(json.dumps(intent_dict, indent=2))
                   # ---

                   # Put intent on the event bus
                   intent_dict = { # Ensure intent_dict is defined here for the put call
                       "action_type": validated_intent.action_type,
                       "target_id": validated_intent.target_id,
                       "details": validated_intent.details
                   }
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
                    state["simulacra"][agent_id]["pending_results"] = {} # Clear pending results on error
                    state["simulacra"][agent_id]["last_observation"] = f"Encountered unexpected error: {e}"
            except Exception as state_err:
                 logger.error(f"[{agent_name}] Failed to update state after error: {state_err}")
            await asyncio.sleep(10) # Wait longer after an error

# --- Main Execution Logic ---
# ... other code from simulation_async.py ...

async def run_simulation(instance_uuid_arg: Optional[str] = None):
    """Sets up ADK and runs all concurrent tasks."""
    # Declare modification of module-level variables <<< Added adk_memory_service >>>
    global adk_session_service, adk_session_id, adk_session, adk_runner, adk_memory_service
    global world_engine_agent, simulacra_agents, state
    global live_display_object # <<< ADDED: Declare global modification
    global narration_agent # <<< ADDED

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
            # --- MODIFICATION START: Create new state if specified UUID not found ---
            logger.warning(f"State file not found for specified UUID: {instance_uuid_arg} at {potential_state_path}. Creating a new blank state with this UUID.")
            world_instance_uuid = instance_uuid_arg # Use the specified UUID
            state_file_path = potential_state_path

            # Create the blank state dictionary using the helper function
            initial_state = create_blank_simulation_state(world_instance_uuid)

            # Save the new blank state file
            try:
                save_json_file(state_file_path, initial_state) # Use save_json_file helper
                logger.info(f"Successfully created and saved new blank state file: {state_file_path}")
                loaded_state_data = initial_state # Use the newly created state directly
                console.print(f"Created new blank state file: {state_file_path}")
            except Exception as e:
                logger.error(f"Failed to create or save new state file {state_file_path}: {e}", exc_info=True)
                console.print(f"[bold red]Error:[/bold red] Failed to create state file '{state_file_path}'. Check logs. Error: {e}")
                sys.exit(1)
            # --- MODIFICATION END ---
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
            # --- MODIFICATION START: Create new state if no UUID specified AND no files found ---
            logger.warning("No instance UUID specified and no existing state files found. Creating a new blank state.")
            new_uuid = str(uuid.uuid4()) # Generate a completely new UUID
            world_instance_uuid = new_uuid
            state_file_path = os.path.join(STATE_DIR, f"simulation_state_{new_uuid}.json")

            # Create the blank state dictionary
            initial_state = create_blank_simulation_state(world_instance_uuid)

            # Save the new blank state file
            try:
                save_json_file(state_file_path, initial_state) # Use save_json_file helper
                logger.info(f"Successfully created and saved new blank state file: {state_file_path}")
                loaded_state_data = initial_state # Use the newly created state directly
                console.print(f"Created new blank state file: {state_file_path}")
            except Exception as e:
                logger.error(f"Failed to create or save new state file {state_file_path}: {e}", exc_info=True)
                console.print(f"[bold red]Error:[/bold red] Failed to create state file '{state_file_path}'. Check logs. Error: {e}")
                sys.exit(1)
            # --- MODIFICATION END ---

    # --- Load Simulation State (if not already loaded from creation) ---
    if loaded_state_data is None: # Only load if we didn't just create it
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
            # state = loaded_state_data # Assign loaded data to module-level state (done below)

        except Exception as e:
            logger.critical(f"Failed to load simulation instance state from {state_file_path}: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/bold red] Failed to load simulation state file '{state_file_path}'. Check logs. Error: {e}")
            sys.exit(1)

    # --- Assign to global state and ensure structure ---
    if loaded_state_data is None:
        logger.critical("Failed to load or create simulation state. Cannot proceed.")
        console.print("[bold red]Error:[/bold red] Could not obtain simulation state.")
        sys.exit(1)

    state = loaded_state_data # Assign loaded or created data to module-level state

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
                "memory_log": profile.get("memory_log", []), # Load existing memory log
                "pending_results": {} # Initialize pending results store
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

    narration_agent = create_narration_llm_agent() # <<< ADDED
    logger.info(f"Narration Agent '{narration_agent.name}' created.")

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
            live_display_object = live # <<< ADDED: Assign to global variable
            # Pass the live display object to the time manager
            tasks.append(asyncio.create_task(time_manager_task(live_display=live), name="TimeManager"))
            tasks.append(asyncio.create_task(interaction_dispatcher_task(), name="InteractionDispatcher"))
            tasks.append(asyncio.create_task(narration_task(), name="NarrationTask")) # Listens to narration_queue
            tasks.append(asyncio.create_task(world_engine_task_llm(), name="WorldEngine")) # Listens to event_bus

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
