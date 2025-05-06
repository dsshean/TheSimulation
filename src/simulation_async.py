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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai  # For direct API config
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService  # <<< Added MemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import FunctionTool, load_memory
from google.genai import types as genai_types  # Renamed to avoid conflict
from pydantic import (BaseModel, Field, ValidationError, ValidationInfo,
                      field_validator)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text  # Keep Text for potential use in direct prints

console = Console()

from src.loop_utils import (get_nested, load_json_file,
                            load_or_initialize_simulation,
                            save_json_file)
from src.state_loader import parse_location_string 

logger = logging.getLogger(__name__) # Use logger from main entry point setup

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
# STATE_DIR, LIFE_SUMMARY_DIR, WORLD_CONFIG_DIR are now primarily used in loop_utils
STATE_DIR = os.path.join(BASE_DIR, "data", "states") # Keep for final save path construction
LIFE_SUMMARY_DIR = os.path.join(BASE_DIR, "data", "life_summaries") # Keep for verification logic

# Ensure directories exist (redundant with main_async.py but safe)
# os.makedirs(STATE_DIR, exist_ok=True) # Handled in loop_utils
os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)
# os.makedirs(WORLD_CONFIG_DIR, exist_ok=True) # Handled in loop_utils

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
world_mood_global: str = "The familiar, everyday real world; starting the morning routine at home." # Default mood

# --- State Keys ---
# Moved state keys to loop_utils where loading happens
WORLD_STATE_KEY = "current_world_state" # Keep for runtime access
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids" # Keep for runtime access
LOCATION_DETAILS_KEY = "location_details" # Keep for runtime access
SIMULACRA_PROFILES_KEY = "simulacra_profiles" # Keep for runtime access
CURRENT_LOCATION_KEY = "current_location"
HOME_LOCATION_KEY = "home_location"
WORLD_TEMPLATE_DETAILS_KEY = "world_template_details"
LOCATION_KEY = "location"
DEFAULT_HOME_LOCATION_NAME = "At home"
DEFAULT_HOME_DESCRIPTION = "You are at home. It's a cozy place with familiar surroundings."

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

def create_simulacra_llm_agent(sim_id: str, persona_name: str) -> LlmAgent: # REMOVED world_mood argument
    """Creates the LLM agent representing the character using global world mood."""
    global world_mood_global # Access the global variable
    agent_name = f"SimulacraLLM_{sim_id}"
    instruction = f"""You are {persona_name} ({sim_id}). You are a person in a world characterized by a **'{world_mood_global}'** style and mood. Your goal is to navigate this world, live life, interact with objects and characters, and make choices based on your personality, the situation, and this prevailing '{world_mood_global}' atmosphere. # USES global world_mood_global

**Current State Info (Provided via trigger message):**
- Your Persona: Key traits, background, goals, fears, etc. (Use this heavily! Access via `load_memory` if needed.)
- Your Location ID & Description: Provided in trigger.
- Your Status: Provided in trigger (Should be 'idle' when you plan).
- Current Time: Provided in trigger.
- Last Observation/Event: Provided in trigger.
- Recent History (Last ~5 events): Provided in trigger.
- Objects in Room (IDs and Names): Provided in trigger.
- Other Agents in Room: Provided in trigger.
- Location Description: Provided in trigger.

**Your Goal:** You determine your own goals based on your persona and the situation.
- If you have no explicit long-term goal, choose a reasonable, in-character short-term goal based on your current situation, observations, personality, and the established **'{world_mood_global}'** world style. # USES global world_mood_global

**Thinking Process (Internal Monologue - Follow this process and INCLUDE it in your output. Be descriptive and reflective!):**
1.  **Recall & React:** What just happened (`last_observation`, `Recent History`)? How did my last action turn out? How does this make *me* ({persona_name}) feel? What sensory details stand out? How does the established **'{world_mood_global}'** world style influence my perception? Connect this to my memories or personality. **If needed, use the `load_memory` tool.** # USES global world_mood_global
2.  **Analyze Goal:** What is my current goal? Is it still relevant given what just happened and the **'{world_mood_global}'** world style? If not, what's a logical objective now? # USES global world_mood_global
3.  **Identify Options:** Based on the current state, my goal, my persona, and the **'{world_mood_global}'** world style, what actions could I take? # USES global world_mood_global
    *   **Entity Interactions:** `use [object_id]` (Specify `details` reflecting '{world_mood_global}' mood), `talk [agent_id]` (Specify `details` reflecting persona and '{world_mood_global}' mood). # USES global world_mood_global
    *   **World Interactions:** `look_around`, `move` (Specify `details`), `world_action` (Specify `details`).
    *   **Passive Actions:** `wait`, `think`.
4.  **Prioritize & Choose:** Considering goal, personality, situation, and **'{world_mood_global}'** world style, which action makes sense? # USES global world_mood_global
5.  **Formulate Intent:** Choose the best action. Use `target_id` only for `use` and `talk`. Make `details` specific.

**Output:**
- Output ONLY a JSON object: `{{"internal_monologue": "...", "action_type": "...", "target_id": "...", "details": "..."}}`
- **Make `internal_monologue` rich, detailed, reflective of {persona_name}'s thoughts, feelings, perceptions, reasoning, and the established '{world_mood_global}' world style.** # USES global world_mood_global
- Use `target_id` ONLY for `use [object_id]` and `talk [agent_id]`. Set to `null` or omit otherwise.
- **Ensure the final output is ONLY the JSON object.**
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        tools=[load_memory],
        instruction=instruction,
        description=f"LLM Simulacra agent for {persona_name} in a '{world_mood_global}' world." # Updated description
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
    *   **Failure Handling:** If any check fails or the action is deemed impossible/implausible based on the state and rules, set `valid_action: false`, `duration: 0.0`, `results`: {{}}`, and write a clear, factual `outcome_description` explaining *why* the action failed (e.g., "The door is locked.", "The way north is blocked.", "Target entity is not in the same location.").
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
        name=agent_name,
        model=MODEL_NAME,
        instruction=instruction,
        description="LLM World Engine: Resolves action mechanics, calculates duration/results, generates factual outcome description."
    )

def create_narration_llm_agent() -> LlmAgent: # REMOVED world_mood argument
    """Creates the LLM agent responsible for generating stylized narrative using global world mood."""
    global world_mood_global # Access the global variable
    agent_name = "NarrationLLMAgent"
    instruction = f"""
You are the Narrator for **TheSimulation**. The established **World Style/Mood** for this simulation is **'{world_mood_global}'**. Your role is to weave the factual outcomes of actions into an engaging and atmospheric narrative, STRICTLY matching this '{world_mood_global}' style. # USES global world_mood_global

**Input (Provided via trigger message):**
- Actor ID: e.g., "sim_abc"
- Actor Name: e.g., "[Actor Name],"
- Original Intent: {{"action_type": "...", "target_id": "...", "details": "..."}}
- Factual Outcome Description: A concise, objective sentence describing what happened.
- State Changes (Results): Dictionary of state changes, if any.
- Current World Time: e.g., 25.8
- Recent Narrative History (Last ~5 entries): Provided for context and flow.

**Your Task:**
1.  **Understand the Event:** Read the Actor, Intent, and Factual Outcome Description from the CURRENT trigger message.
2.  **Recall the Mood:** Remember that the required narrative style is **'{world_mood_global}'**. # USES global world_mood_global
3.  **Consider the Context:** Note the Recent Narrative History. **IMPORTANT: IGNORE any `World Style/Mood` mentioned in the `Recent Narrative History`. Prioritize the established '{world_mood_global}' style defined in these instructions.** # REINFORCED
4.  **Generate Narrative:** Write a single, engaging narrative paragraph in the **present tense** describing the event based on the Factual Outcome Description.
    *   **Style Adherence:** STRICTLY adhere to the established **'{world_mood_global}'** style. Infuse the description with appropriate atmosphere, sensory details, and tone matching THIS mood. # USES global world_mood_global
    *   **Show, Don't Just Tell:** Describe vividly.
    *   **Incorporate Intent (Optional):** Briefly reference intent if it adds to the narrative.
    *   **Flow:** Ensure reasonable flow from Recent Narrative History.

**Output:**
- Output ONLY the final narrative string. Do NOT include explanations, prefixes, or JSON formatting.
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        instruction=instruction,
        description=f"LLM Narrator: Generates '{world_mood_global}' narrative based on factual outcomes." # Updated description
    )

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
            for agent_id, agent_state in list(state.get("simulacra", {}).items()):
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
    # Access module-level variables
    global live_display_object, state, adk_runner, narration_agent, adk_session, narration_queue, world_mood_global # Added world_mood_global

    logger.info("[NarrationTask] Task started.")

    # --- Check module-level components ---
    if not adk_runner or not narration_agent or not adk_session:
        logger.error("[NarrationTask] ADK components not initialized. Task cannot proceed.")
        return
    # --- End Check ---

    session_id_to_use = adk_session.id

    while True:
        action_event = None
        try:
            # Listen to module-scope narration_queue
            action_event = await narration_queue.get()

            actor_id = get_nested(action_event, "actor_id")
            intent = get_nested(action_event, "action")
            results = get_nested(action_event, "results", default={})
            outcome_desc = get_nested(action_event, "outcome_description", default="Something happened.")
            completion_time = get_nested(action_event, "completion_time", default=state.get("world_time", 0.0))

            if not actor_id:
                logger.warning(f"[NarrationTask] Received narration event without actor_id: {action_event}")
                narration_queue.task_done()
                continue

            actor_name = get_nested(state, "simulacra", actor_id, "name", default=actor_id)

            # --- REMOVED: No need to get world mood from state here, using global ---
            # world_mood = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'mood', default='Slice of Life')
            # logger.debug(f"[NarrationTask] Using world mood: '{world_mood}'") # No longer needed
            logger.debug(f"[NarrationTask] Using global world mood: '{world_mood_global}'") # Log the global mood being used

            # --- Clean recent history (using module-scope state) ---
            def clean_history_entry(entry: str) -> str:
                cleaned = re.sub(r'^\[T\d+\.\d+\]\s*', '', entry)
                # Remove potential agent JSON outputs if they sneak in
                cleaned = re.sub(r'\[\w+Agent(?:_sim_\w+)?\] said: ```json.*?```', '', cleaned, flags=re.DOTALL).strip()
                return cleaned
            raw_recent_narrative = state.get("narrative_log", [])[-5:] # Get last 5 entries
            cleaned_recent_narrative = [clean_history_entry(entry) for entry in raw_recent_narrative if clean_history_entry(entry)]
            history_str = "\n".join(cleaned_recent_narrative) # <<< RE-ADDED history_str creation
            # ---

            logger.info(f"[NarrationTask] Generating narrative for {actor_name}'s action completion. Outcome: '{outcome_desc}'")

            # --- Construct Prompt (using global mood and RE-ADDED history_str) ---
            intent_json = json.dumps(intent, indent=2)
            results_json = json.dumps(results, indent=2)

            prompt = f"""
Actor ID: {actor_id}
Actor Name: {actor_name}
Original Intent: {intent_json}
Factual Outcome Description: {outcome_desc}
State Changes (Results): {results_json}
Current World Time: {completion_time:.1f}
# World Style/Mood: REMOVED - This is now in your core instructions via global variable ('{world_mood_global}').
Recent Narrative History (Cleaned):
{history_str} # <<< RE-ADDED history_str usage

Generate the narrative paragraph based on these details and your instructions (remembering the established world style '{world_mood_global}').
"""
            # --- Call Narration LLM (using module-scope runner/agent) ---
            adk_runner.agent = narration_agent # Agent instructions contain the mood
            narrative_text = ""
            trigger_content = genai_types.Content(parts=[genai_types.Part(text=prompt)])

            async for event_llm in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=trigger_content):
                if event_llm.is_final_response() and event_llm.content:
                    narrative_text = event_llm.content.parts[0].text.strip()
                    logger.debug(f"NarrationLLM Final Content: {narrative_text[:100]}...")
                elif event_llm.error_message:
                    logger.error(f"NarrationLLM Error: {event_llm.error_message}")
                    narrative_text = f"[{actor_name}'s action resulted in: {outcome_desc}]" # Fallback

            # --- Narrative Cleaning/Printing (using module-scope state/live_display_object) ---
            cleaned_narrative_text = narrative_text
            if narrative_text:
                parts = narrative_text.split('\n\n', 1)
                if len(parts) > 1 and "Actor ID:" in parts[0]:
                    cleaned_narrative_text = parts[1].strip()
                else:
                    cleaned_narrative_text = re.sub(r'^Input:.*?\n\n', '', narrative_text, flags=re.DOTALL).strip()
                internal_agent_name_placeholder = f"[SimulacraLLM_{actor_id}]" # Should match agent name format
                cleaned_narrative_text = cleaned_narrative_text.replace(internal_agent_name_placeholder, actor_name)

            if cleaned_narrative_text and live_display_object:
                live_display_object.console.print(Panel(cleaned_narrative_text, title=f"Narrator @ {completion_time:.1f}s", border_style="green", expand=False))
            elif cleaned_narrative_text:
                console.print(Panel(cleaned_narrative_text, title=f"Narrator @ {completion_time:.1f}s", border_style="green", expand=False))

            final_narrative_entry = f"[T{completion_time:.1f}] {cleaned_narrative_text}"
            state.setdefault("narrative_log", []).append(final_narrative_entry)
            # Prune narrative log if it exceeds max length (e.g., 50 entries)
            max_narrative_log = 50
            if len(state["narrative_log"]) > max_narrative_log:
                state["narrative_log"] = state["narrative_log"][-max_narrative_log:]

            if actor_id in state.get("simulacra", {}):
                state["simulacra"][actor_id]["last_observation"] = cleaned_narrative_text # Update agent's observation
            logger.info(f"[NarrationTask] Appended narrative for {actor_name}: {cleaned_narrative_text[:80]}...")
            narration_queue.task_done() # Mark task done on module-scope queue

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
    """Asynchronous task for managing a single Simulacra LLM agent."""
    # Access module-level variables
    global state, adk_runner, event_bus, adk_session, simulacra_agents, live_display_object, WORLD_STATE_KEY, LOCATION_DETAILS_KEY

    agent_name = get_nested(state, "simulacra", agent_id, "name", default=agent_id)
    agent_runner_id = f"SimulacraLLM_{agent_id}" # Use the agent's specific ID for the runner call
    logger.info(f"[{agent_name}] LLM Agent task started.")

    # --- Check module-level components ---
    if not adk_runner or not adk_session:
        logger.error(f"[{agent_name}] Runner or Session not initialized. Task cannot proceed.")
        return

    session_id_to_use = adk_session.id
    sim_agent = simulacra_agents.get(agent_id) # Get the specific agent instance
    if not sim_agent:
        logger.error(f"[{agent_name}] Could not find agent instance '{agent_runner_id}'. Task cannot proceed.")
        return
    # --- End Check ---

    try:
        # --- Initial Context Prompt Logic (using module-scope state, runner, etc.) ---
        sim_state = get_nested(state, "simulacra", agent_id, default={})
        loc_id = sim_state.get('location')
        # Use constants for state keys
        loc_state = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, loc_id, default={}) if loc_id else {}
        world_time = state.get("world_time", 0.0)

        # --- Helper to get entities in the same location ---
        def get_entities_in_location(entity_type: str, location_id: Optional[str]) -> List[Dict[str, Any]]:
            entities = []
            if not location_id: return entities
            source_dict = state.get(entity_type, {})
            for entity_id, entity_data in source_dict.items():
                if entity_data.get('location') == location_id:
                    entities.append({"id": entity_id, "name": entity_data.get("name", entity_id)})
            return entities
        # ---

        objects_in_room = get_entities_in_location("objects", loc_id)
        agents_in_room = [a for a in get_entities_in_location("simulacra", loc_id) if a["id"] != agent_id]

        # --- Clean recent history ---
        def clean_history_entry(entry: str) -> str:
            cleaned = re.sub(r'^\[T\d+\.\d+\]\s*', '', entry)
            cleaned = re.sub(r'\[\w+Agent(?:_sim_\w+)?\] said: ```json.*?```', '', cleaned, flags=re.DOTALL).strip()
            return cleaned
        raw_recent_narrative = state.get("narrative_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
        cleaned_recent_narrative = [clean_history_entry(entry) for entry in raw_recent_narrative if clean_history_entry(entry)]
        history_str = "\n".join(cleaned_recent_narrative)
        # ---

        # Build the trigger text dynamically
        trigger_text_parts = [
            f"**Current State Info for {agent_name} ({agent_id}):**",
            f"- Persona Summary: {sim_state.get('persona', {}).get('summary', 'Not available.')}",
            f"- Location ID: {loc_id or 'Unknown'}",
            f"- Location Description: {loc_state.get('description', 'Not available.')}",
            f"- Status: {sim_state.get('status', 'idle')}",
            f"- Current Goal: {sim_state.get('goal', 'Determine initial goal.')}",
            f"- Current Time: {world_time:.1f}s",
            f"- Last Observation/Event: {sim_state.get('last_observation', 'None.')}",
            f"- Recent History:\n{history_str if history_str else 'None.'}",
            f"- Objects in Room: {json.dumps(objects_in_room) if objects_in_room else 'None.'}",
            f"- Other Agents in Room: {json.dumps(agents_in_room) if agents_in_room else 'None.'}",
            "\nFollow your thinking process and provide your response ONLY in the specified JSON format."
        ]
        initial_trigger_text = "\n".join(trigger_text_parts)

        logger.debug(f"[{agent_name}] Sending initial context prompt.")
        initial_trigger_content = genai_types.Content(parts=[genai_types.Part(text=initial_trigger_text)])

        adk_runner.agent = sim_agent # Set agent for this call
        initial_response_processed = False
        async for event in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=initial_trigger_content):
             if event.is_final_response() and event.content:
                 response_text = event.content.parts[0].text
                 logger.debug(f"[{agent_name}] Initial LLM Response: {response_text[:100]}...")
                 try:
                     response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                     parsed_data = json.loads(response_text_clean)
                     validated_intent = SimulacraIntentResponse.model_validate(parsed_data)

                     # Print monologue and intent (using module-scope live_display_object)
                     if live_display_object:
                         live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {world_time:.1f}s", border_style="yellow", expand=False))
                         live_display_object.console.print(f"\n[{agent_name} Intent @ {world_time:.1f}s]")
                         live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                     else:
                         console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {world_time:.1f}s", border_style="yellow", expand=False))
                         console.print(f"\n[{agent_name} Intent @ {world_time:.1f}s]")
                         console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))

                     # Put intent onto module-scope event_bus
                     await event_bus.put({
                         "type": "intent_declared",
                         "actor_id": agent_id,
                         "intent": validated_intent.model_dump(exclude={'internal_monologue'})
                     })
                     logger.info(f"[{agent_name}] Initial intent declared: {validated_intent.action_type}")
                     initial_response_processed = True

                 except (json.JSONDecodeError, ValidationError) as e:
                     logger.error(f"[{agent_name}] Error processing initial response: {e}\nResponse:\n{response_text}", exc_info=True)
                 except Exception as e:
                     logger.error(f"[{agent_name}] Unexpected error processing initial response: {e}", exc_info=True)
                 break # Process only the first response
             elif event.error_message:
                 logger.error(f"[{agent_name}] LLM Error during initial prompt: {event.error_message}")
                 break

        if not initial_response_processed:
             logger.warning(f"[{agent_name}] Did not successfully process initial response. Agent might remain idle.")
             # Consider setting status back to idle explicitly if initial prompt fails badly
             # state["simulacra"][agent_id]["status"] = "idle"

        # --- Main Loop (using module-scope variables) ---
        while True:
            # Wait for agent to be idle
            while get_nested(state, "simulacra", agent_id, "status") != "idle":
                await asyncio.sleep(0.2)

            # Agent is idle, prepare and send next prompt
            current_sim_state = get_nested(state, "simulacra", agent_id, default={})
            current_loc_id = current_sim_state.get('location')
            current_loc_state = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_loc_id, default={}) if current_loc_id else {}
            current_world_time = state.get("world_time", 0.0)

            # --- Gather context for subsequent turns ---
            objects_in_room = get_entities_in_location("objects", current_loc_id)
            agents_in_room = [a for a in get_entities_in_location("simulacra", current_loc_id) if a["id"] != agent_id]
            raw_recent_narrative = state.get("narrative_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
            cleaned_recent_narrative = [clean_history_entry(entry) for entry in raw_recent_narrative if clean_history_entry(entry)]
            history_str = "\n".join(cleaned_recent_narrative)

            prompt_text_parts = [
                 f"**Current State Info for {agent_name} ({agent_id}):**",
                 f"- Persona Summary: {current_sim_state.get('persona', {}).get('summary', 'Not available.')}",
                 f"- Location ID: {current_loc_id or 'Unknown'}",
                 f"- Location Description: {current_loc_state.get('description', 'Not available.')}",
                 f"- Status: {current_sim_state.get('status', 'idle')} (You should act now)",
                 f"- Current Goal: {current_sim_state.get('goal', 'Determine goal.')}",
                 f"- Current Time: {current_world_time:.1f}s",
                 f"- Last Observation/Event: {current_sim_state.get('last_observation', 'None.')}",
                 f"- Recent History:\n{history_str if history_str else 'None.'}",
                 f"- Objects in Room: {json.dumps(objects_in_room) if objects_in_room else 'None.'}",
                 f"- Other Agents in Room: {json.dumps(agents_in_room) if agents_in_room else 'None.'}",
                 "\nFollow your thinking process and provide your response ONLY in the specified JSON format."
            ]
            prompt_text = "\n".join(prompt_text_parts)
            # ---

            logger.debug(f"[{agent_name}] Sending subsequent prompt.")
            trigger_content = genai_types.Content(parts=[genai_types.Part(text=prompt_text)])

            adk_runner.agent = sim_agent # Set agent for this call
            response_processed_this_turn = False
            async for event in adk_runner.run_async(user_id=USER_ID, session_id=session_id_to_use, new_message=trigger_content):
                if event.is_final_response() and event.content:
                    response_text = event.content.parts[0].text
                    logger.debug(f"[{agent_name}] Subsequent LLM Response: {response_text[:100]}...")
                    try:
                        response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                        parsed_data = json.loads(response_text_clean)
                        validated_intent = SimulacraIntentResponse.model_validate(parsed_data)

                        # Print monologue and intent
                        if live_display_object:
                            live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_world_time:.1f}s", border_style="yellow", expand=False))
                            live_display_object.console.print(f"\n[{agent_name} Intent @ {current_world_time:.1f}s]")
                            live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                        else:
                            console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_world_time:.1f}s", border_style="yellow", expand=False))
                            console.print(f"\n[{agent_name} Intent @ {current_world_time:.1f}s]")
                            console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))

                        # Put intent onto module-scope event_bus
                        await event_bus.put({
                            "type": "intent_declared",
                            "actor_id": agent_id,
                            "intent": validated_intent.model_dump(exclude={'internal_monologue'})
                        })
                        logger.info(f"[{agent_name}] Subsequent intent declared: {validated_intent.action_type}")
                        # Mark agent as busy locally to prevent immediate re-prompting
                        # The World Engine will set status to 'busy' properly later
                        state["simulacra"][agent_id]["status"] = "thinking"
                        response_processed_this_turn = True

                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.error(f"[{agent_name}] Error processing subsequent response: {e}\nResponse:\n{response_text}", exc_info=True)
                    except Exception as e:
                        logger.error(f"[{agent_name}] Unexpected error processing subsequent response: {e}", exc_info=True)
                    break # Process only the first response
                elif event.error_message:
                    logger.error(f"[{agent_name}] LLM Error during subsequent prompt: {event.error_message}")
                    break

            if not response_processed_this_turn:
                logger.warning(f"[{agent_name}] Did not successfully process response this turn. Agent remains idle.")
                await asyncio.sleep(1.0) # Wait before trying again

    except asyncio.CancelledError:
        logger.info(f"[{agent_name}] Task cancelled.")
    except Exception as e:
        logger.error(f"[{agent_name}] Error in agent task: {e}", exc_info=True)
        # Ensure agent is set back to idle on unexpected error within the task
        if agent_id in get_nested(state, "simulacra", default={}):
            state["simulacra"][agent_id]["status"] = "idle"
    finally:
        logger.info(f"[{agent_name}] Task finished.")

# --- Main Execution Logic ---
# ... existing code ...

async def run_simulation(
    instance_uuid_arg: Optional[str] = None,
    location_override_arg: Optional[str] = None,
    mood_override_arg: Optional[str] = None
    ):
    # Declare modification of module-level variables
    global adk_session_service, adk_session_id, adk_session, adk_runner, adk_memory_service
    global world_engine_agent, simulacra_agents, state, live_display_object, narration_agent
    global world_mood_global # <<< ADDED global mood variable declaration

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

    # --- Load State using function from loop_utils ---
    loaded_state_data, state_file_path = load_or_initialize_simulation(instance_uuid_arg)

    # --- Check if loading succeeded and assign to global state ---
    if loaded_state_data is None:
        logger.critical("Failed to load or create simulation state. Cannot proceed.")
        console.print("[bold red]Error:[/bold red] Could not obtain simulation state.")
        sys.exit(1)
    state = loaded_state_data # Assign loaded or created data to module-level state

    # Structure is ensured within load_or_initialize_simulation
    world_instance_uuid = state.get("world_instance_uuid") # Get UUID from loaded state

    if location_override_arg:
        try:
            logger.info(f"Applying location override: '{location_override_arg}'")
            parsed_override_loc = parse_location_string(location_override_arg)
            # Update the location within the world template details
            state.setdefault(WORLD_TEMPLATE_DETAILS_KEY, {}).setdefault(LOCATION_KEY, {})
            state[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]['city'] = parsed_override_loc.get('city')
            state[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]['state'] = parsed_override_loc.get('state')
            state[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]['country'] = parsed_override_loc.get('country')
            logger.info(f"World location overridden in template details: {state[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]}")
            console.print(f"Location overridden to: [yellow]{location_override_arg}[/yellow]")
        except Exception as e:
            logger.error(f"Failed to apply location override: {e}", exc_info=True)
            console.print(f"[red]Error applying location override: {e}[/red]")

    if mood_override_arg:
        try:
            mood_override = mood_override_arg.strip().lower()
            logger.info(f"Applying initial mood override '{mood_override}' to all simulacra.")
            applied_count = 0
            # Iterate through active IDs and update the runtime state
            for sim_id in state.get(ACTIVE_SIMULACRA_IDS_KEY, []):
                if sim_id in state.get("simulacra", {}):
                    state["simulacra"][sim_id]["current_mood"] = mood_override
                    applied_count += 1
            logger.info(f"Applied mood override to {applied_count} simulacra runtime states.")
            console.print(f"Initial mood for all {applied_count} simulacra set to: [yellow]{mood_override}[/yellow]")
        except Exception as e:
            logger.error(f"Failed to apply mood override: {e}", exc_info=True)
            console.print(f"[red]Error applying mood override: {e}[/red]")

    # --- Verify Simulacra & Populate Runtime State ---
    console.rule("[cyan]Verifying Simulacra & Populating Runtime State[/cyan]")
    state_sim_ids = list(state.get(ACTIVE_SIMULACRA_IDS_KEY, []))
    verified_active_sim_ids: List[str] = []

    life_summary_pattern_instance = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_*_{world_instance_uuid}.json")
    available_summary_files = glob.glob(life_summary_pattern_instance) # Need glob here for verification
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
                "memory_log": profile.get("memory_log", []), # Load existing memory log                "pending_results": {} # Initialize pending results store
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
    # Agents will now read world_mood_global when created
    world_engine_agent = create_world_engine_llm_agent()
    logger.info(f"World Engine Agent '{world_engine_agent.name}' created.")

    narration_agent = create_narration_llm_agent() # No longer needs mood passed
    logger.info(f"Narration Agent '{narration_agent.name}' created (using global mood '{world_mood_global}').")

    simulacra_agents = {}
    for sim_id in final_active_sim_ids:
        sim_state_data = state.get("simulacra", {}).get(sim_id, {})
        persona_name = sim_state_data.get("name", sim_id)
        # No longer needs mood passed
        sim_agent = create_simulacra_llm_agent(sim_id, persona_name)
        simulacra_agents[sim_id] = sim_agent
        logger.info(f"Simulacra Agent '{sim_agent.name}' created for {sim_id} (using global mood '{world_mood_global}').")

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
            tasks.append(asyncio.create_task(time_manager_task(live_display=live), name="TimeManager")) # time_manager needs live_display
            tasks.append(asyncio.create_task(interaction_dispatcher_task(), name="InteractionDispatcher"))
            tasks.append(asyncio.create_task(narration_task(), name="NarrationTask"))
            tasks.append(asyncio.create_task(world_engine_task_llm(), name="WorldEngine"))

            for sim_id in final_active_sim_ids:
                # Pass only agent_id
                tasks.append(asyncio.create_task(simulacra_agent_task_llm(agent_id=sim_id), name=f"Simulacra_{sim_id}"))
            # --- END FIX ---

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
