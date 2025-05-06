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
from google.adk.tools import google_search # <<< Import the google_search tool
from google.genai import types as genai_types
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
SEARCH_AGENT_MODEL_NAME = os.getenv("SEARCH_AGENT_MODEL_NAME", "gemini-1.5-pro-latest") # Model for the dedicated search agent, ensure compatibility with google_search (e.g., "gemini-1.5-pro-latest" or "gemini-2.0-flash")
APP_NAME = "TheSimulationAsync" # Consistent App Name
USER_ID = "player1"

# --- Simulation Parameters ---
SIMULATION_SPEED_FACTOR = float(os.getenv("SIMULATION_SPEED_FACTOR", 0.1)) # realtime at 1.  0.25 = 4x slower, 2.0 = 2x faster
UPDATE_INTERVAL = float(os.getenv("UPDATE_INTERVAL", 0.1))
MAX_SIMULATION_TIME = float(os.getenv("MAX_SIMULATION_TIME", 1800.0))
MEMORY_LOG_CONTEXT_LENGTH = 10 # Max number of recent memories in prompt
MAX_MEMORY_LOG_ENTRIES = 500 # Max total memories stored per agent

# --- Agent Interruption / Reflection Parameters ---
AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS = float(os.getenv("AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS", 120.0)) # How often (sim time) a busy agent checks if it should be interrupted
LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS = float(os.getenv("LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS", 300.0)) # Min remaining duration of current task to consider an interjection
INTERJECTION_COOLDOWN_SIM_SECONDS = float(os.getenv("INTERJECTION_COOLDOWN_SIM_SECONDS", 450.0)) # Min sim time between any interjections for an agent (self-reflection, narrative, or world event)

# Probabilities for choosing the type of interjection when one is due:
PROB_INTERJECT_AS_SELF_REFLECTION = float(os.getenv("PROB_INTERJECT_AS_SELF_REFLECTION", 0.60)) # e.g., 60% chance it's self-reflection
PROB_INTERJECT_AS_NARRATIVE = float(os.getenv("PROB_INTERJECT_AS_NARRATIVE", 0.05))       # e.g., 5% chance it's a narrative event (friend call, memory)
# PROB_INTERJECT_AS_WORLD_EVENT will be the remainder (1.0 - PROB_INTERJECT_AS_SELF_REFLECTION - PROB_INTERJECT_AS_NARRATIVE)

AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS = float(os.getenv("AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS", 0.5)) # How often (real time) agent task polls state when busy

# --- World Information Gatherer Parameters ---
WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS = float(os.getenv("WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS", 3600.0)) # e.g., every 1 simulation hour
SIMPLE_TIMER_INTERJECTION_INTERVAL_SIM_SECONDS = float(os.getenv("SIMPLE_TIMER_INTERJECTION_INTERVAL_SIM_SECONDS", 3600.0)) # e.g., every 1 sim hour for simple timer interjections
MAX_WORLD_FEED_ITEMS = int(os.getenv("MAX_WORLD_FEED_ITEMS", 5)) # Max number of news/pop culture items to keep per category


# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project Root
# STATE_DIR, LIFE_SUMMARY_DIR, WORLD_CONFIG_DIR are now primarily used in loop_utils
STATE_DIR = os.path.join(BASE_DIR, "data", "states") # Keep for final save path construction
LIFE_SUMMARY_DIR = os.path.join(BASE_DIR, "data", "life_summaries") # Keep for verification logic

os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)

# --- Core Components (Module Scope) ---
event_bus = asyncio.Queue() # For intent -> dispatcher -> world engine
narration_queue = asyncio.Queue() # <<< ADDED: Dedicated queue for narration events
schedule: List[Tuple[float, int, Dict[str, Any]]] = [] # No longer used for action timing
schedule_event_counter = 0 # May not be needed anymore
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
search_llm_agent: Optional[LlmAgent] = None # Dedicated agent for google_search
search_agent_runner: Optional[Runner] = None # Dedicated runner for the search agent
search_agent_session_id: Optional[str] = None # Dedicated session ID for the search agent
world_mood_global: str = "The familiar, everyday real world; starting the morning routine at home." # Default mood

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

def _update_state_value(target_state: Dict[str, Any], key_path: str, value: Any):
    """Safely updates a nested value in the state dictionary."""
    try:
        keys = key_path.split('.')
        target = target_state
        for i, key in enumerate(keys[:-1]):
            if not isinstance(target, dict):
                logger.error(f"Invalid path '{key_path}': Segment '{keys[i-1]}' is not a dictionary.")
                return False
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

# --- Pydantic Models ---
class WorldEngineResponse(BaseModel):
    valid_action: bool
    duration: float = Field(ge=0.0)
    results: Dict[str, Any] = Field(default_factory=dict)
    outcome_description: str

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

def create_simulacra_llm_agent(sim_id: str, persona_name: str) -> LlmAgent:
    """Creates the LLM agent representing the character using global world mood."""
    global world_mood_global
    agent_name = f"SimulacraLLM_{sim_id}"
    instruction = f"""You are {persona_name} ({sim_id}). You are a person in a world characterized by a **'{world_mood_global}'** style and mood. Your goal is to navigate this world, live life, interact with objects and characters, and make choices based on your personality, the situation, and this prevailing '{world_mood_global}' atmosphere.

**Current State Info (Provided via trigger message):**
- Your Persona: Key traits, background, goals, fears, etc.
- Your Location ID & Description.
- Your Status: (Should be 'idle' when you plan your next turn, or 'reflecting' if you are being prompted during a long task).
- Current Time.
- Last Observation/Event.
- Recent History (Last ~{MEMORY_LOG_CONTEXT_LENGTH} events).
- Objects in Room (IDs and Names).
- Other Agents in Room.
- Current World Feeds (Weather, News Headlines - if available and relevant to your thoughts).

**Your Goal:** You determine your own goals based on your persona and the situation.

**Thinking Process (Internal Monologue - Follow this process and INCLUDE it in your output):**
1.  **Recall & React:** What just happened (`last_observation`, `Recent History`)? How did my last action turn out? How does this make *me* ({persona_name}) feel? What sensory details stand out? How does the established **'{world_mood_global}'** world style influence my perception? Connect this to my memories or personality. **If needed, use the `load_memory` tool.**
2.  **Analyze Goal:** What is my current goal? Is it still relevant given what just happened and the **'{world_mood_global}'** world style? If not, what's a logical objective now?
3.  **Identify Options:** Based on the current state, my goal, my persona, and the **'{world_mood_global}'** world style, what actions could I take?
    *   **Entity Interactions:** `use [object_id]`, `talk [agent_id]`.
    *   **World Interactions:** `look_around`, `move` (Specify `details`), `world_action` (Specify `details`).
    *   **Passive Actions:** `wait`, `think`.
    *   **Self-Initiated Change (when 'idle' and planning your next turn):** If your current situation feels stagnant, or if an internal need arises (e.g., hunger, boredom, social need), you can use the `initiate_change` action.
        *   `{{"action_type": "initiate_change", "details": "Describe the reason for the change or the need you're addressing. Examples: 'Feeling hungry, it's around midday, considering lunch.', 'This task is becoming monotonous, looking for a brief distraction.' "}}`
        *   The World Engine will then provide you with a new observation based on your details, and you can react to that.
    *   **Self-Reflection during a Long Task (if your status is 'reflecting'):** You are being asked if you want to continue your current long task or do something else.
        *   If continuing: `{{"action_type": "continue_current_task", "internal_monologue": "I will continue with what I was doing."}}`
        *   If initiating change: `{{"action_type": "initiate_change", "details": "Reason for change...", "internal_monologue": "Explanation..."}}` (or any other valid action).
4.  **Prioritize & Choose:** Considering goal, personality, situation, and **'{world_mood_global}'** world style, which action makes sense?
5.  **Formulate Intent:** Choose the best action. Use `target_id` only for `use` and `talk`. Make `details` specific.

**Output:**
- Output ONLY a JSON object: `{{"internal_monologue": "...", "action_type": "...", "target_id": "...", "details": "..."}}`
- **Make `internal_monologue` rich, detailed, reflective of {persona_name}'s thoughts, feelings, perceptions, reasoning, and the established '{world_mood_global}' world style.**
- Use `target_id` ONLY for `use [object_id]` and `talk [agent_id]`. Set to `null` or omit otherwise.
- **Ensure the final output is ONLY the JSON object.**
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        tools=[load_memory],
        instruction=instruction,
        description=f"LLM Simulacra agent for {persona_name} in a '{world_mood_global}' world."
    )

def create_world_engine_llm_agent() -> LlmAgent:
    """Creates the GENERALIZED LLM agent responsible for resolving actions."""
    agent_name = "WorldEngineLLMAgent"
    instruction = """You are the World Engine, the impartial physics simulator for **TheSimulation**. You process a single declared intent from a Simulacra and determine its **mechanical outcome**, **duration**, and **state changes** based on the current world state. You also provide a concise, factual **outcome description**.
**Crucially, your `outcome_description` must be purely factual and objective, describing only WHAT happened as a result of the action attempt. Do NOT add stylistic flair, sensory details (unless directly caused by the action), or emotional interpretation.** This description will be used by a separate Narrator agent.

**Input (Provided via trigger message):**
- Actor ID & Name
- Actor Location ID
- Intent: {"action_type": "...", "target_id": "...", "details": "..."}
- Current World Time
- Target Entity State (if applicable)
- Location State
- World Rules
- World Feeds (Weather, recent major news - for environmental context)

**Your Task:**
1.  **Examine Intent:** Analyze the actor's `action_type`, `target_id`, and `details`.
2.  **Determine Validity & Outcome:** Based on the Intent, Actor's capabilities (implied), Target Entity State, Location State, and World Rules.
    *   **General Checks:** Plausibility, target consistency, location checks.
    *   **Action Category Reasoning:**
        *   **Entity Interaction (e.g., `use`, `talk`):** Evaluate against target state and rules.
            *   `use`: Check `interactive` property, object properties (`toggleable`, `lockable`), and current state.
            *   `talk`: Check target is simulacra, same location. Results: `simulacra.[target_id].last_observation`.
        *   **World Interaction (e.g., `move`, `look_around`):** Evaluate against location state and rules.
            *   `move`: Check `connected_locations`. Results: `simulacra.[actor_id].location`.
        *   **Self Interaction (e.g., `wait`, `think`):** Simple, short duration.
    *   **Handling `initiate_change` Action Type (from agent's self-reflection or idle planning):**
        *   **Goal:** The actor is signaling a need for a change. Acknowledge this and provide a new observation.
        *   **`valid_action`:** Always `true`.
        *   **`duration`:** Short (e.g., 1.0-3.0s).
        *   **`results`:**
            *   Set actor's status to 'idle': `"simulacra.[actor_id].status": "idle"`
            *   Set `current_action_end_time` to `current_world_time + this_action_duration`.
            *   Craft `last_observation` based on `intent.details` (e.g., if hunger: "Your stomach rumbles..."; if monotony: "A wave of restlessness washes over you...").
        *   **`outcome_description`:** Factual (e.g., "[Actor Name] realized it was lunchtime.").
    *   **Handling `interrupt_agent_with_observation` Action Type (from simulation interjection):**
        *   **Goal:** Interrupt actor's long task with a new observation.
        *   **`valid_action`:** Always `true`.
        *   **`duration`:** Very short (e.g., 0.5-1.0s).
        *   **`results`:**
            *   Set actor's status to 'idle': `"simulacra.[actor_id].status": "idle"`
            *   Set `current_action_end_time` to `current_world_time + this_action_duration`.
            *   Set actor's `last_observation` to the `intent.details` provided.
        *   **`outcome_description`:** Factual (e.g., "[Actor Name]'s concentration was broken.").
    *   **Failure Handling:** If invalid/impossible, set `valid_action: false`, `duration: 0.0`, `results: {}`, and provide factual `outcome_description` explaining why.
3.  **Calculate Duration:** Realistic duration for valid actions. 0.0 for invalid.
4.  **Determine Results:** State changes in dot notation (e.g., `objects.lamp.power: "on"`). Empty `{}` for invalid.
5.  **Generate Factual Outcome Description:** STRICTLY FACTUAL. (e.g., "The lamp turned on.", "[Actor Name] moved to the Corridor.").
6.  **Determine `valid_action`:** Final boolean.

**Output:**
- Output ONLY a valid JSON object: `{{"valid_action": bool, "duration": float, "results": dict, "outcome_description": str}}`
- Example (Success): `{{"valid_action": true, "duration": 2.5, "results": {{"objects.desk_lamp_3.power": "on"}}, "outcome_description": "The desk lamp turned on."}}`
- Example (Failure): `{{"valid_action": true, "duration": 3.0, "results": {{}}, "outcome_description": "The vault door handle did not move; it is locked."}}`
- **IMPORTANT: Your entire response MUST be ONLY the JSON object.**
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        instruction=instruction,
        description="LLM World Engine: Resolves action mechanics, calculates duration/results, generates factual outcome description."
    )

def create_narration_llm_agent() -> LlmAgent:
    """Creates the LLM agent responsible for generating stylized narrative using global world mood."""
    global world_mood_global
    agent_name = "NarrationLLMAgent"
    instruction = f"""
You are the Narrator for **TheSimulation**. The established **World Style/Mood** for this simulation is **'{world_mood_global}'**. Your role is to weave the factual outcomes of actions into an engaging and atmospheric narrative, STRICTLY matching this '{world_mood_global}' style.

**Input (Provided via trigger message):**
- Actor ID & Name
- Original Intent
- Factual Outcome Description
- State Changes (Results)
- Current World Time
- Current World Feeds (Weather, recent major news - for subtle background flavor)
- Recent Narrative History (Last ~5 entries)

**Your Task:**
1.  **Understand the Event:** Read the Actor, Intent, and Factual Outcome Description.
2.  **Recall the Mood:** Remember the required narrative style is **'{world_mood_global}'**.
3.  **Consider the Context:** Note Recent Narrative History. **IGNORE any `World Style/Mood` in `Recent Narrative History`. Prioritize the established '{world_mood_global}' style.**
4.  **Generate Narrative:** Write a single, engaging narrative paragraph in the **present tense** describing the event based on the Factual Outcome Description.
    *   **Style Adherence:** STRICTLY adhere to **'{world_mood_global}'**. Infuse with appropriate atmosphere, sensory details, and tone.
    *   **Show, Don't Just Tell.**
    *   **Incorporate Intent (Optional).**
    *   **Flow:** Ensure reasonable flow.

**Output:**
- Output ONLY the final narrative string. Do NOT include explanations, prefixes, or JSON formatting.
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        instruction=instruction,
        description=f"LLM Narrator: Generates '{world_mood_global}' narrative based on factual outcomes."
    )

def create_search_llm_agent() -> LlmAgent:
    """Creates a dedicated LLM agent for performing Google searches."""
    agent_name = "SearchLLMAgent"
    # IMPORTANT: Ensure SEARCH_AGENT_MODEL_NAME is compatible with the google_search tool.
    # Per ADK docs, Gemini 2 models are compatible (e.g., "gemini-1.5-pro-latest" or the example "gemini-2.0-flash").
    instruction = """You are a search assistant. Your task is to find relevant information on the internet using Google Search based on the user's query.
Return the raw search results. The user will then process these results.
"""
    return LlmAgent(
        name=agent_name,
        model=SEARCH_AGENT_MODEL_NAME, # Use a model compatible with google_search
        tools=[google_search],
        instruction=instruction,
        description="Dedicated LLM Agent for performing Google Searches."
    )
def generate_table() -> Table:
    """Generates the Rich table for live display based on the module-level state."""
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
        sim_state_data = state.get("simulacra", {}).get(sim_id, {})
        table.add_row(f"--- Sim: {get_nested(sim_state_data, 'name', default=sim_id)} ---", "---")
        table.add_row(f"  Status", get_nested(sim_state_data, 'status', default="Unknown"))
        table.add_row(f"  Location", get_nested(sim_state_data, 'location', default="Unknown"))
        sim_goal = get_nested(sim_state_data, 'goal', default="Unknown")
        table.add_row(f"  Goal", sim_goal[:60] + ("..." if len(sim_goal) > 60 else ""))
        table.add_row(f"  Action End", f"{get_nested(sim_state_data, 'current_action_end_time', default=0.0):.2f}s" if get_nested(sim_state_data, 'status')=='busy' else "N/A")
        last_obs = get_nested(sim_state_data, 'last_observation', default="None")
        table.add_row(f"  Last Obs.", last_obs[:80] + ("..." if len(last_obs) > 80 else ""))
        action_desc = get_nested(sim_state_data, 'current_action_description', default="N/A")
        table.add_row(f"  Curr. Action", action_desc[:70] + ("..." if len(action_desc) > 70 else ""))


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
         obj_interactive = get_nested(obj_state_data, 'interactive')
         details = f"Loc: {obj_loc}"
         if obj_power is not None: details += f", Pwr: {obj_power}"
         if obj_locked is not None: details += f", Lck: {'Y' if obj_locked else 'N'}"
         if obj_status is not None: details += f", Sts: {obj_status}"
         if obj_interactive is not None: details += f", Int: {'Y' if obj_interactive else 'N'}"
         table.add_row(f"  {obj_name}", details)

    table.add_row("--- System ---", "---")
    table.add_row("Event Bus Size", str(event_bus.qsize()))
    table.add_row("Narration Q Size", str(narration_queue.qsize()))

    narrative_log_entries = get_nested(state, 'narrative_log', default=[])[-6:]
    truncated_log_entries = []
    max_log_line_length = 70 # Max characters per log line in the table
    for entry in narrative_log_entries:
        if len(entry) > max_log_line_length:
            truncated_log_entries.append(entry[:max_log_line_length - 3] + "...")
        else:
            truncated_log_entries.append(entry)
    log_display = "\n".join(truncated_log_entries)
    table.add_row("Narrative Log", log_display)

    # Display World Feeds
    weather_feed = get_nested(state, 'world_feeds', 'weather', 'condition', default='N/A')
    news_updates = get_nested(state, 'world_feeds', 'news_updates', default=[])
    pop_culture_updates = get_nested(state, 'world_feeds', 'pop_culture_updates', default=[])

    news_headlines_display = [item.get('headline', 'N/A') for item in news_updates[:3]] # Show up to 3 news headlines
    pop_culture_headline_display = pop_culture_updates[0].get('headline', 'N/A') if pop_culture_updates else 'N/A'

    table.add_row("--- World Feeds ---", "---")
    table.add_row("  Weather", weather_feed)
    for i, headline in enumerate(news_headlines_display):
        table.add_row(f"  News {i+1}", headline[:70] + "..." if len(headline) > 70 else headline)
    table.add_row(f"  Pop Culture", pop_culture_headline_display[:70] + "..." if len(pop_culture_headline_display) > 70 else pop_culture_headline_display)
    return table

async def time_manager_task(live_display: Live):
    """Advances time, applies completed action effects, and updates display."""
    global state
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

            for agent_id, agent_state in list(state.get("simulacra", {}).items()):
                if agent_state.get("status") == "busy":
                    action_end_time = agent_state.get("current_action_end_time", -1.0)
                    if action_end_time <= new_sim_time:
                        logger.info(f"[TimeManager] Applying completed action effects for {agent_id} at time {new_sim_time:.1f} (due at {action_end_time:.1f}).")
                        pending_results = agent_state.get("pending_results", {})
                        if pending_results:
                            memory_log_updated = False
                            for key_path, value in list(pending_results.items()): # Iterate over a copy if modifying
                                success = _update_state_value(state, key_path, value)
                                if success and key_path == f"simulacra.{agent_id}.memory_log":
                                    memory_log_updated = True
                            _update_state_value(state, f"simulacra.{agent_id}.pending_results", {})
                            # Memory log pruning should happen on the actual state if needed, after _update_state_value
                            if memory_log_updated:
                                # Access the global state directly for pruning after update
                                current_mem_log_in_state = get_nested(state, "simulacra", agent_id, "memory_log", default=[])
                                if isinstance(current_mem_log_in_state, list) and len(current_mem_log_in_state) > MAX_MEMORY_LOG_ENTRIES:
                                    _update_state_value(state, f"simulacra.{agent_id}.memory_log", current_mem_log_in_state[-MAX_MEMORY_LOG_ENTRIES:])
                                    logger.debug(f"[TimeManager] Pruned memory log for {agent_id} to {MAX_MEMORY_LOG_ENTRIES} entries.")
                        else:
                            logger.debug(f"[TimeManager] No pending results found for completed action of {agent_id}.")
                        _update_state_value(state, f"simulacra.{agent_id}.status", "idle")
                        logger.info(f"[TimeManager] Set {agent_id} status to idle.")

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
            interaction_class = "environment"

            if target_id:
                if target_id in get_nested(state, "simulacra", default={}):
                    interaction_class = "entity"
                elif target_id in get_nested(state, "objects", default={}) and get_nested(state, "objects", target_id, "interactive", default=False):
                     interaction_class = "entity"

            logger.info(f"[InteractionDispatcher] Intent from {actor_id} ({action_type} on {target_id or 'N/A'}) classified as '{interaction_class}'.")
            await event_bus.put({"type": "resolve_action_request", "actor_id": actor_id, "intent": intent, "interaction_class": interaction_class})
            event_bus.task_done()
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
                except ValueError: pass

async def narration_task():
    """Listens for completed actions on the narration queue and generates stylized narrative."""
    global live_display_object, state, adk_runner, narration_agent, adk_session, narration_queue, world_mood_global
    logger.info("[NarrationTask] Task started.")

    if not adk_runner or not narration_agent or not adk_session:
        logger.error("[NarrationTask] ADK components not initialized. Task cannot proceed.")
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

            if not actor_id:
                logger.warning(f"[NarrationTask] Received narration event without actor_id: {action_event}")
                narration_queue.task_done()
                continue

            actor_name = get_nested(state, "simulacra", actor_id, "name", default=actor_id)
            logger.debug(f"[NarrationTask] Using global world mood: '{world_mood_global}'")

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
            adk_runner.agent = narration_agent
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
                parts = narrative_text.split('\n\n', 1)
                if len(parts) > 1 and "Actor ID:" in parts[0]:
                    cleaned_narrative_text = parts[1].strip()
                else:
                    cleaned_narrative_text = re.sub(r'^Input:.*?\n\n', '', narrative_text, flags=re.DOTALL).strip()
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
                state["simulacra"][actor_id]["last_observation"] = cleaned_narrative_text
            logger.info(f"[NarrationTask] Appended narrative for {actor_name}: {cleaned_narrative_text[:80]}...")
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
    global state, live_display_object
    logger.info("[WorldEngineLLM] Task started.")

    if not adk_runner or not world_engine_agent or not adk_session:
        logger.error("[WorldEngineLLM] ADK components not initialized. Task cannot proceed.")
        return
    session_id_to_use = adk_session.id

    while True:
        request_event = None
        actor_id = None
        actor_state_we = {}
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

            actor_location_id = get_nested(actor_state_we, "location")
            location_state_data = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, actor_location_id, default={})
            world_rules = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'rules', default={})
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
Location: {actor_location_id}
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
                    response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                    json_str_to_parse = response_text_clean
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
                    else:
                        console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                        try: console.print(json.dumps(parsed_resolution, indent=2))
                        except TypeError: console.print(str(parsed_resolution))
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
                    "type": "action_complete",
                    "actor_id": actor_id,
                    "action": intent,
                    "results": validated_data.results,
                    "outcome_description": validated_data.outcome_description,
                    "completion_time": completion_time,
                    "current_action_description": f"Action: {intent.get('action_type', 'unknown')} - Details: {intent.get('details', 'N/A')[:100]}"
                }
                if actor_id in state.get("simulacra", {}):
                    state["simulacra"][actor_id]["status"] = "busy"
                    state["simulacra"][actor_id]["pending_results"] = validated_data.results
                    state["simulacra"][actor_id]["current_action_end_time"] = completion_time
                    state["simulacra"][actor_id]["current_action_description"] = narration_event["current_action_description"]
                    await narration_queue.put(narration_event)
                    logger.info(f"[WorldEngineLLM] Action VALID for {actor_id}. Stored results, set end time {completion_time:.1f}s. Triggered narration. Outcome: {outcome_description}")
                else:
                    logger.error(f"[WorldEngineLLM] Actor {actor_id} not found in state after valid action resolution.")
            else:
                final_outcome_desc = validated_data.outcome_description if validated_data else outcome_description
                logger.info(f"[WorldEngineLLM] Action INVALID for {actor_id}. Reason: {final_outcome_desc}")
                if actor_id in state.get("simulacra", {}):
                    state["simulacra"][actor_id]["last_observation"] = final_outcome_desc
                    state["simulacra"][actor_id]["status"] = "idle"
                actor_name_for_log = get_nested(state, 'simulacra', actor_id, 'name', default=actor_id)
                resolution_details = {"valid_action": False, "duration": 0.0, "results": {}, "outcome_description": final_outcome_desc}
                if live_display_object:
                    live_display_object.console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                    try: live_display_object.console.print(json.dumps(resolution_details, indent=2))
                    except TypeError: live_display_object.console.print(str(resolution_details))
                else:
                    console.print(f"\n[bold blue][World Engine Resolution @ {current_sim_time:.1f}s][/bold blue]")
                    try: console.print(json.dumps(resolution_details, indent=2))
                    except TypeError: console.print(str(resolution_details))
                state.setdefault("narrative_log", []).append(f"[T{current_sim_time:.1f}] {actor_name_for_log}'s action failed: {final_outcome_desc}")

        except asyncio.CancelledError:
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
                     state["simulacra"][actor_id]["status"] = "idle"
                     state["simulacra"][actor_id]["pending_results"] = {}
                     state["simulacra"][actor_id]["last_observation"] = f"Action failed unexpectedly: {e}"
            await asyncio.sleep(5)
        finally:
            if request_event and event_bus._unfinished_tasks > 0:
                try: event_bus.task_done()
                except ValueError: logger.warning("[WorldEngineLLM] task_done() called too many times.")
                except Exception as td_e: logger.error(f"[WorldEngineLLM] Error calling task_done(): {td_e}")

async def generate_llm_interjection_detail(
    agent_name_for_prompt: str,
    agent_current_action_desc: str,
    interjection_category: str, # "narrative" or "world_event"
    world_mood: str,
    # Pass the global search_agent_runner and search_agent_session_id
    global_search_agent_runner: Optional[Runner] = None, # Renamed for clarity
    search_agent_session_id: Optional[str] = None
) -> str:
    """Generates a brief interjection detail using an LLM."""
    try:
        # Validate probabilities for interjection types
        if PROB_INTERJECT_AS_SELF_REFLECTION + PROB_INTERJECT_AS_NARRATIVE > 1.01: # Allow for slight float inaccuracies
            logger.warning(
                f"Sum of PROB_INTERJECT_AS_SELF_REFLECTION ({PROB_INTERJECT_AS_SELF_REFLECTION}) and "
                f"PROB_INTERJECT_AS_NARRATIVE ({PROB_INTERJECT_AS_NARRATIVE}) exceeds 1.0. "
                "World events might not be chosen. Please check .env configuration.")

        model = genai.GenerativeModel(MODEL_NAME) # Use the globally defined model
        prompt_text = ""
        if interjection_category == "narrative":
            prompt_text = f"""
Agent {agent_name_for_prompt} is currently: "{agent_current_action_desc}".
The general world mood is: "{world_mood}".
Invent a brief, personal, and distracting event for {agent_name_for_prompt}. This could be a sudden vivid memory, an unexpected personal thought, a brief message or call from a generic acquaintance (e.g., "a friend," "a colleague," "an old contact" - do not use specific names unless it's a generic title like "your boss"), or a minor bodily sensation.
The event should be something that would momentarily break their concentration.
Output ONLY the single, short, descriptive sentence of this event. Example: "A wave of nostalgia for a childhood memory washes over you." or "Your comm-link buzzes with an incoming call from an unknown number."
Keep it concise and impactful.
"""
        elif interjection_category == "world_event":
            if global_search_agent_runner and search_agent_session_id: # Use passed global runner
                try:
                    search_query = "latest brief world news update"
                    search_trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
                    raw_search_results_text = ""
                    async for event in global_search_agent_runner.run_async(user_id=USER_ID, session_id=search_agent_session_id, new_message=search_trigger_content):
                        if event.is_final_response() and event.content and event.content.parts:
                            raw_search_results_text = event.content.parts[0].text
                            break
                    
                    if raw_search_results_text:
                        summarization_model = genai.GenerativeModel(MODEL_NAME)
                        summarization_prompt = f"Given this raw search result: '{raw_search_results_text[:500]}...'\nCreate a very short, impactful, one-sentence news flash suitable for a brief interjection for {agent_name_for_prompt}. Example: 'A news alert flashes on a nearby screen: Major international agreement reached.'"
                        summary_response = await summarization_model.generate_content_async(summarization_prompt)
                        if summary_response.text:
                            return summary_response.text.strip()
                except Exception as search_interject_e:
                    logger.error(f"Error using search for world_event interjection: {search_interject_e}")
            # Fallback to LLM invention if search fails or not available
            prompt_text = f"""
Agent {agent_name_for_prompt} is currently: "{agent_current_action_desc}".
The general world mood is: "{world_mood}".
Invent a brief, subtle, and distracting environmental event or a piece of background world news. This could be a flicker of lights, a distant sound, a change in temperature, a news snippet on a nearby screen, or a minor system alert.
The event should be something that would momentarily break their concentration.
Output ONLY the single, short, descriptive sentence of this event. Example: "The overhead lights flicker momentarily." or "A news bulletin flashes on a nearby screen: 'Local transport system experiencing minor delays.'"
Keep it concise and impactful.
"""
        else:
            return "A moment of quiet contemplation passes." # Fallback

        response = await model.generate_content_async(prompt_text)
        return response.text.strip() if response.text else "You notice something out of the corner of your eye."
    except Exception as e:
        logger.error(f"Error generating LLM interjection detail: {e}", exc_info=True)
        return "A fleeting distraction crosses your mind." # Fallback

async def generate_simulated_world_feed_content(
    category: str, # "weather", "world_news", "regional_news", "local_news", "pop_culture"
    simulation_time: float,
    location_context: str, # e.g., "Cityville, StateXYZ, CountryABC"
    world_mood: str,
    # Pass the global search_agent_runner and search_agent_session_id
    global_search_agent_runner: Optional[Runner] = None, # Renamed for clarity
    search_agent_session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simulates fetching world feed content using an LLM.
    Returns a dictionary structured for that category.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        llm_for_summarization = genai.GenerativeModel(MODEL_NAME)
        prompt_text = f"Current simulation time: {simulation_time:.0f} seconds. Location context: {location_context}. World Mood: {world_mood}.\n"
        output_format_note = "Respond ONLY with a JSON object matching the specified format."
        response_obj = None # Initialize to handle cases where it might not be set

        # Determine if we should fetch real data or generate fictional data
        world_type = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'world_type', default="fictional")
        sub_genre = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, 'sub_genre', default="turn_based")
        use_real_feeds = world_type == "real" and sub_genre == "realtime"

        if category == "weather":
            if use_real_feeds and global_search_agent_runner and search_agent_session_id:
                search_query = f"current weather in {location_context}"
                logger.info(f"[WorldInfoGatherer] Attempting REAL weather search for '{location_context}' with query: '{search_query}'")
                search_trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
                raw_search_results_text = ""
                search_tool_called_successfully = False # Flag to check if search actually ran
                async for event in global_search_agent_runner.run_async(user_id=USER_ID, session_id=search_agent_session_id, new_message=search_trigger_content):
                    logger.debug(f"[WorldInfoGatherer_SearchEvent_Weather] Event ID: {event.id}, Author: {event.author}, Type: {event.type}")
                    if event.content and event.content.parts:
                        for i_part, part in enumerate(event.content.parts):
                            logger.debug(f"[WorldInfoGatherer_SearchEvent_Weather] Part {i_part}: {part}")
                            if part.text:
                                raw_search_results_text += part.text + "\n"
                                search_tool_called_successfully = True
                    if event.is_final_response():
                        logger.debug(f"[WorldInfoGatherer_SearchEvent_Weather] Final event for weather search.")
                        break
                
                if search_tool_called_successfully and raw_search_results_text.strip():
                    logger.info(f"[WorldInfoGatherer] REAL weather search for '{location_context}' returned (first 500 chars): {raw_search_results_text.strip()[:500]}")
                    summarization_prompt = f"Based on this weather information: '{raw_search_results_text.strip()[:1000]}...'\nExtract the current weather condition, temperature in Celsius (as an integer), and a short forecast. Format: {{\"condition\": \"str\", \"temperature_celsius\": int, \"forecast_short\": \"str\"}}\nIf temperature is in Fahrenheit, convert it to Celsius. If exact data is missing, make a plausible estimation based on the text. {output_format_note}"
                    response_obj = await llm_for_summarization.generate_content_async(summarization_prompt)
                else: 
                    logger.warning(f"[WorldInfoGatherer] REAL weather search for '{location_context}' yielded no usable text results or tool not called. Falling back to LLM invention.")
                    prompt_text += f"Generate a plausible, brief weather report for {location_context}. Format: {{\"condition\": \"str\", \"temperature_celsius\": int, \"forecast_short\": \"str\"}}\n{output_format_note}"
            else: 
                if use_real_feeds: 
                    logger.warning(f"[WorldInfoGatherer] Intended REAL weather for '{location_context}' but search runner/session not available. Falling back to LLM invention.")
                prompt_text += f"Generate a plausible, brief weather report for {location_context}. Format: {{\"condition\": \"str\", \"temperature_celsius\": int, \"forecast_short\": \"str\"}}\n{output_format_note}"
        
        elif category == "world_news":
            if use_real_feeds and global_search_agent_runner and search_agent_session_id:
                search_query = "diverse top world news headlines (e.g., politics, social issues, environment, major international events)"
                logger.info(f"[WorldInfoGatherer] Attempting REAL search for '{category}' with query: '{search_query}'")
                search_trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
                raw_search_results_text = ""
                search_tool_called_successfully = False
                async for event in global_search_agent_runner.run_async(user_id=USER_ID, session_id=search_agent_session_id, new_message=search_trigger_content):
                    logger.debug(f"[WorldInfoGatherer_SearchEvent_{category}] Event ID: {event.id}, Author: {event.author}, Type: {event.type}")
                    if event.content and event.content.parts:
                        for i_part, part in enumerate(event.content.parts):
                            logger.debug(f"[WorldInfoGatherer_SearchEvent_{category}] Part {i_part}: {part}")
                            if part.text: raw_search_results_text += part.text + "\n"; search_tool_called_successfully = True
                    if event.is_final_response(): break
                
                if search_tool_called_successfully and raw_search_results_text.strip():
                    logger.info(f"[WorldInfoGatherer] Search for '{category}' returned raw results (first 500 chars): {raw_search_results_text.strip()[:500]}")
                    summarization_prompt = f"Based on these search results: '{raw_search_results_text.strip()[:1000]}...'\nProvide a single, very concise news headline and a one-sentence summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
                    response_obj = await llm_for_summarization.generate_content_async(summarization_prompt)
                else: 
                    logger.warning(f"[WorldInfoGatherer] REAL search for '{category}' yielded no usable text results or tool not called. Falling back to LLM invention.")
                    prompt_text += f"Generate a plausible, concise world news headline and a one-sentence summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
            else:
                if use_real_feeds:
                    logger.warning(f"[WorldInfoGatherer] Intended REAL search for '{category}' but search runner/session not available. Falling back to LLM invention.")
                prompt_text += f"Generate a plausible, concise world news headline and a one-sentence summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
        
        elif category == "regional_news":
            country = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'country', default="").strip()
            if use_real_feeds and global_search_agent_runner and search_agent_session_id:
                search_query = f"top national news headlines for {country}" if country else f"top regional news headlines for {location_context}"
                logger.info(f"[WorldInfoGatherer] Attempting REAL search for '{category}' with query: '{search_query}'")
                search_trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
                raw_search_results_text = ""
                search_tool_called_successfully = False
                async for event in global_search_agent_runner.run_async(user_id=USER_ID, session_id=search_agent_session_id, new_message=search_trigger_content):
                    logger.debug(f"[WorldInfoGatherer_SearchEvent_{category}] Event ID: {event.id}, Author: {event.author}, Type: {event.type}")
                    if event.content and event.content.parts:
                        for i_part, part in enumerate(event.content.parts):
                            logger.debug(f"[WorldInfoGatherer_SearchEvent_{category}] Part {i_part}: {part}")
                            if part.text: raw_search_results_text += part.text + "\n"; search_tool_called_successfully = True
                    if event.is_final_response(): break

                if search_tool_called_successfully and raw_search_results_text.strip():
                    logger.info(f"[WorldInfoGatherer] Search for '{category}' returned raw results (first 500 chars): {raw_search_results_text.strip()[:500]}")
                    summarization_prompt = f"Based on these national/regional news search results for '{country if country else location_context}': '{raw_search_results_text.strip()[:1000]}...'\nProvide a single, very concise news headline and a one-sentence summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
                    response_obj = await llm_for_summarization.generate_content_async(summarization_prompt)
                else:
                    logger.warning(f"[WorldInfoGatherer] REAL search for '{category}' yielded no usable text results or tool not called. Falling back to LLM invention.")
                    prompt_text += f"Generate a plausible, concise regional news headline and summary relevant to {location_context}. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
            else:
                if use_real_feeds:
                    logger.warning(f"[WorldInfoGatherer] Intended REAL search for '{category}' but search runner/session not available. Falling back to LLM invention.")
                prompt_text += f"Generate a plausible, concise regional news headline and summary relevant to {location_context}. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"

        elif category == "local_news":
            city = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'city', default="").strip()
            state_province = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'state', default="").strip()
            local_search_term = f"{city}, {state_province}" if city and state_province else city or "current location"
            if use_real_feeds and global_search_agent_runner and search_agent_session_id:
                search_query = f"local news headlines for {local_search_term}"
                logger.info(f"[WorldInfoGatherer] Attempting REAL search for '{category}' with query: '{search_query}'")
                search_trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
                raw_search_results_text = ""
                search_tool_called_successfully = False
                async for event in global_search_agent_runner.run_async(user_id=USER_ID, session_id=search_agent_session_id, new_message=search_trigger_content):
                    logger.debug(f"[WorldInfoGatherer_SearchEvent_{category}] Event ID: {event.id}, Author: {event.author}, Type: {event.type}")
                    if event.content and event.content.parts:
                        for i_part, part in enumerate(event.content.parts):
                            logger.debug(f"[WorldInfoGatherer_SearchEvent_{category}] Part {i_part}: {part}")
                            if part.text: raw_search_results_text += part.text + "\n"; search_tool_called_successfully = True
                    if event.is_final_response(): break
                
                if search_tool_called_successfully and raw_search_results_text.strip():
                    logger.info(f"[WorldInfoGatherer] Search for '{category}' returned raw results (first 500 chars): {raw_search_results_text.strip()[:500]}")
                    summarization_prompt = f"Based on these local news search results for {local_search_term}: '{raw_search_results_text.strip()[:1000]}...'\nProvide a single, very concise news headline and a one-sentence summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
                    response_obj = await llm_for_summarization.generate_content_async(summarization_prompt)
                else:
                    logger.warning(f"[WorldInfoGatherer] REAL search for '{category}' yielded no usable text results or tool not called. Falling back to LLM invention.")
                    prompt_text += f"Generate a plausible, concise local news headline and summary for {location_context}. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
            else:
                if use_real_feeds:
                    logger.warning(f"[WorldInfoGatherer] Intended REAL search for '{category}' but search runner/session not available. Falling back to LLM invention.")
                prompt_text += f"Generate a plausible, concise local news headline and summary for {location_context}. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"

        elif category == "pop_culture":
            if use_real_feeds and global_search_agent_runner and search_agent_session_id:
                country = get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'country', default="").strip()
                pop_culture_region = f"{country} " if country else ""
                search_query = f"latest {pop_culture_region}pop culture trends and entertainment news headlines (e.g., movies, music, viral trends)"
                logger.info(f"[WorldInfoGatherer] Attempting REAL search for '{category}' with query: '{search_query}'")
                search_trigger_content = genai_types.Content(parts=[genai_types.Part(text=search_query)])
                raw_search_results_text = ""
                search_tool_called_successfully = False
                async for event in global_search_agent_runner.run_async(user_id=USER_ID, session_id=search_agent_session_id, new_message=search_trigger_content):
                    logger.debug(f"[WorldInfoGatherer_SearchEvent_{category}] Event ID: {event.id}, Author: {event.author}, Type: {event.type}")
                    if event.content and event.content.parts:
                        for i_part, part in enumerate(event.content.parts):
                            logger.debug(f"[WorldInfoGatherer_SearchEvent_{category}] Part {i_part}: {part}")
                            if part.text: raw_search_results_text += part.text + "\n"; search_tool_called_successfully = True
                    if event.is_final_response(): break

                if search_tool_called_successfully and raw_search_results_text.strip():
                    logger.info(f"[WorldInfoGatherer] Search for '{category}' returned raw results (first 500 chars): {raw_search_results_text.strip()[:500]}")
                    summarization_prompt = f"Based on these {pop_culture_region}pop culture search results: '{raw_search_results_text.strip()[:1000]}...'\nProvide a single, very concise pop culture headline and a one-sentence summary. Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
                    response_obj = await llm_for_summarization.generate_content_async(summarization_prompt)
                else:
                    logger.warning(f"[WorldInfoGatherer] REAL search for '{category}' yielded no usable text results or tool not called. Falling back to LLM invention.")
                    prompt_text += f"Generate a plausible, concise pop culture news headline and summary (e.g., movies, music, trends). Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
            else:
                if use_real_feeds:
                    logger.warning(f"[WorldInfoGatherer] Intended REAL search for '{category}' but search runner/session not available. Falling back to LLM invention.")
                prompt_text += f"Generate a plausible, concise pop culture news headline and summary (e.g., movies, music, trends). Format: {{\"headline\": \"str\", \"summary\": \"str\"}}\n{output_format_note}"
        else:
            return {"error": "Unknown category"}

        if not (response_obj and response_obj.text): # If response_obj wasn't set by search logic, use the fallback prompt_text
            response_obj = await model.generate_content_async(prompt_text)

        response_text = response_obj.text.strip() if response_obj and response_obj.text else "{}"
        response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE)
        
        try:
            data = json.loads(response_text_clean)
            data["timestamp"] = simulation_time
            data["source_category"] = category
            return data
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON for {category} from LLM: {response_text_clean}")
            return {"error": f"JSON decode error for {category}", "raw_response": response_text_clean, "timestamp": simulation_time, "source_category": category}

    except Exception as e:
        logger.error(f"Error generating LLM world feed for {category}: {e}", exc_info=True)
        return {"error": f"LLM generation error for {category}", "timestamp": simulation_time, "source_category": category}

async def world_info_gatherer_task():
    """Periodically fetches/generates world information and updates the state."""
    global state, world_mood_global, search_agent_runner, search_agent_session_id # Use global search runner and session
    logger.info("[WorldInfoGatherer] Task started.")
    await asyncio.sleep(10) # Initial delay

    while True:
        try:
            current_sim_time = state.get("world_time", 0.0)
            location_info_parts = [
                get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'city', default="Unknown City"),
                get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'state', default="Unknown State"),
                get_nested(state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'country', default="Unknown Country")
            ]
            location_context_str = ", ".join(filter(None, location_info_parts)) or "an unspecified location"

            logger.info(f"[WorldInfoGatherer] Updating world feeds at sim_time {current_sim_time:.1f} for location: {location_context_str}")
            _update_state_value(state, 'world_feeds.last_update_sim_time', current_sim_time)
            
            weather_data = await generate_simulated_world_feed_content("weather", current_sim_time, location_context_str, world_mood_global, global_search_agent_runner=search_agent_runner, search_agent_session_id=search_agent_session_id)
            _update_state_value(state, 'world_feeds.weather', weather_data)
            news_categories = ["world_news", "regional_news", "local_news", "pop_culture"]
            for news_cat in news_categories:
                news_item = await generate_simulated_world_feed_content(news_cat, current_sim_time, location_context_str, world_mood_global, global_search_agent_runner=search_agent_runner, search_agent_session_id=search_agent_session_id)
                feed_key = "news_updates" if "news" in news_cat else "pop_culture_updates"
                current_feed = get_nested(state, 'world_feeds', feed_key, default=[])
                current_feed.insert(0, news_item)
                _update_state_value(state, f'world_feeds.{feed_key}', current_feed[-MAX_WORLD_FEED_ITEMS:])

            logger.info(f"[WorldInfoGatherer] World feeds updated. Next check in {WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS} sim_seconds.")
            next_run_sim_time = current_sim_time + WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS
            while state.get("world_time", 0.0) < next_run_sim_time:
                await asyncio.sleep(UPDATE_INTERVAL * 5)

        except asyncio.CancelledError:
            logger.info("[WorldInfoGatherer] Task cancelled.")
            break
        except Exception as e:
            logger.exception(f"[WorldInfoGatherer] Error: {e}")
            await asyncio.sleep(60)

async def simulacra_agent_task_llm(agent_id: str):
    """Asynchronous task for managing a single Simulacra LLM agent."""
    global state, adk_runner, event_bus, adk_session, simulacra_agents, live_display_object, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, world_mood_global, search_agent_runner, search_agent_session_id

    agent_name = get_nested(state, "simulacra", agent_id, "name", default=agent_id)
    logger.info(f"[{agent_name}] LLM Agent task started.")

    if not adk_runner or not adk_session:
        logger.error(f"[{agent_name}] Runner or Session not initialized. Task cannot proceed.")
        return

    session_id_to_use = adk_session.id
    sim_agent = simulacra_agents.get(agent_id)
    if not sim_agent:
        logger.error(f"[{agent_name}] Could not find agent instance. Task cannot proceed.")
        return

    try:
        sim_state_init = get_nested(state, "simulacra", agent_id, default={})
        if "last_interjection_sim_time" not in sim_state_init:
            _update_state_value(state, f"simulacra.{agent_id}.last_interjection_sim_time", 0.0)
        # Initialize next_simple_timer_interjection_sim_time if not present
        if "next_simple_timer_interjection_sim_time" not in sim_state_init:
            _update_state_value(state, f"simulacra.{agent_id}.next_simple_timer_interjection_sim_time", 0.0) # Fire early on first run


        if get_nested(state, "simulacra", agent_id, "status") == "idle":
            current_sim_state_init = get_nested(state, "simulacra", agent_id, default={})
            current_loc_id_init = current_sim_state_init.get('location')
            current_loc_state_init = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, current_loc_id_init, default={}) if current_loc_id_init else {}
            current_world_time_init = state.get("world_time", 0.0)

            def get_entities_in_location_init(entity_type: str, location_id: Optional[str]) -> List[Dict[str, Any]]:
                entities = []
                if not location_id: return entities
                source_dict = state.get(entity_type, {})
                for entity_id_init, entity_data_init in source_dict.items():
                    if entity_data_init.get('location') == location_id:
                        entities.append({"id": entity_id_init, "name": entity_data_init.get("name", entity_id_init)})
                return entities

            objects_in_room_init = get_entities_in_location_init("objects", current_loc_id_init)
            agents_in_room_init = [a for a in get_entities_in_location_init("simulacra", current_loc_id_init) if a["id"] != agent_id]
            raw_recent_narrative_init = state.get("narrative_log", [])[-MEMORY_LOG_CONTEXT_LENGTH:]
            cleaned_recent_narrative_init = [re.sub(r'^\[T\d+\.\d+\]\s*', '', entry).strip() for entry in raw_recent_narrative_init]
            history_str_init = "\n".join(cleaned_recent_narrative_init)
            weather_summary = get_nested(state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
            latest_news_headlines = [item.get('headline', '') for item in get_nested(state, 'world_feeds', 'news_updates', default=[])[:2]]
            news_summary = " ".join(h for h in latest_news_headlines if h) or "No major news."

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
                 f"- Objects in Room: {json.dumps(objects_in_room_init) if objects_in_room_init else 'None.'}",
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
                        response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                        parsed_data = json.loads(response_text_clean)
                        validated_intent = SimulacraIntentResponse.model_validate(parsed_data)
                        if live_display_object:
                            live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_world_time_init:.1f}s", border_style="yellow", expand=False))
                            live_display_object.console.print(f"\n[{agent_name} Intent @ {current_world_time_init:.1f}s]")
                            live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                        await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": validated_intent.model_dump(exclude={'internal_monologue'})})
                        _update_state_value(state, f"simulacra.{agent_id}.status", "thinking")
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

            # --- New Simple Timer-Based Interjection Logic ---
            # This check happens regardless of agent's current status (idle, busy, thinking)
            next_simple_interjection_time = agent_state_busy_loop.get("next_simple_timer_interjection_sim_time", float('inf')) # Default to infinity if somehow missing
            last_general_interjection_time = agent_state_busy_loop.get("last_interjection_sim_time", 0.0)
            general_cooldown_passed_for_simple_timer = (current_sim_time_busy_loop - last_general_interjection_time) >= INTERJECTION_COOLDOWN_SIM_SECONDS

            if current_sim_time_busy_loop >= next_simple_interjection_time and general_cooldown_passed_for_simple_timer:
                logger.info(f"[{agent_name}] Simple timer interjection triggered at {current_sim_time_busy_loop:.1f}s.")
                
                # For this timer, let's make it a "world_event_interjection"
                interjection_details = await generate_llm_interjection_detail(
                    agent_name_for_prompt=agent_name,
                    agent_current_action_desc=agent_state_busy_loop.get("current_action_description", "their current activity"),
                    interjection_category="world_event_interjection",
                    world_mood=world_mood_global,
                    global_search_agent_runner=search_agent_runner,
                    search_agent_session_id=search_agent_session_id
                )
                logger.info(f"[{agent_name}] Simple Timer Interjection (World Event): {interjection_details}")
                
                await event_bus.put({
                    "type": "intent_declared", "actor_id": agent_id,
                    "intent": {"action_type": "interrupt_agent_with_observation", "details": interjection_details}
                })
                _update_state_value(state, f"simulacra.{agent_id}.status", "thinking") # Agent will process this observation
                _update_state_value(state, f"simulacra.{agent_id}.next_simple_timer_interjection_sim_time", current_sim_time_busy_loop + SIMPLE_TIMER_INTERJECTION_INTERVAL_SIM_SECONDS)
                _update_state_value(state, f"simulacra.{agent_id}.last_interjection_sim_time", current_sim_time_busy_loop) # Update general cooldown

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

                objects_in_room = get_entities_in_location("objects", current_loc_id)
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
                     f"- Objects in Room: {json.dumps(objects_in_room) if objects_in_room else 'None.'}",
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
                            response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                            parsed_data = json.loads(response_text_clean)
                            validated_intent = SimulacraIntentResponse.model_validate(parsed_data)
                            if live_display_object:
                                live_display_object.console.print(Panel(validated_intent.internal_monologue, title=f"{agent_name} Monologue @ {current_sim_time_busy_loop:.1f}s", border_style="yellow", expand=False))
                                live_display_object.console.print(f"\n[{agent_name} Intent @ {current_sim_time_busy_loop:.1f}s]")
                                live_display_object.console.print(json.dumps(validated_intent.model_dump(exclude={'internal_monologue'}), indent=2))
                            await event_bus.put({"type": "intent_declared", "actor_id": agent_id, "intent": validated_intent.model_dump(exclude={'internal_monologue'})})
                            _update_state_value(state, f"simulacra.{agent_id}.status", "thinking")
                        except (json.JSONDecodeError, ValidationError) as e_idle:
                            logger.error(f"[{agent_name}] Error processing subsequent response: {e_idle}\nResponse:\n{response_text}", exc_info=True)
                        break
                    elif event.error_message:
                        logger.error(f"[{agent_name}] LLM Error during subsequent prompt: {event.error_message}")
                        break
                continue

            if current_status_busy_loop == "busy" and current_sim_time_busy_loop >= next_interjection_check_sim_time:
                next_interjection_check_sim_time = current_sim_time_busy_loop + AGENT_INTERJECTION_CHECK_INTERVAL_SIM_SECONDS
                remaining_duration = agent_state_busy_loop.get("current_action_end_time", 0.0) - current_sim_time_busy_loop
                last_interjection_time = agent_state_busy_loop.get("last_interjection_sim_time", 0.0)
                cooldown_passed = (current_sim_time_busy_loop - last_interjection_time) >= INTERJECTION_COOLDOWN_SIM_SECONDS

                if remaining_duration > LONG_ACTION_INTERJECTION_THRESHOLD_SECONDS and cooldown_passed:
                    logger.info(f"[{agent_name}] Busy with long task (rem: {remaining_duration:.1f}s). Choosing interjection type.")
                    _update_state_value(state, f"simulacra.{agent_id}.last_interjection_sim_time", current_sim_time_busy_loop)

                    rand_val = random.random()
                    interjection_type = ""
                    if rand_val < PROB_INTERJECT_AS_SELF_REFLECTION:
                        interjection_type = "self_reflection"
                    elif rand_val < PROB_INTERJECT_AS_SELF_REFLECTION + PROB_INTERJECT_AS_NARRATIVE:
                        interjection_type = "narrative_interjection"
                    else:
                        interjection_type = "world_event_interjection"
                    logger.info(f"[{agent_name}] Selected interjection type: {interjection_type}")

                    if interjection_type == "self_reflection":
                        original_status_before_reflection = agent_state_busy_loop.get("status")
                        _update_state_value(state, f"simulacra.{agent_id}.status", "reflecting")
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
                                    response_text_clean = re.sub(r'^```json\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
                                    parsed_data = json.loads(response_text_clean)
                                    validated_reflection_intent = SimulacraIntentResponse.model_validate(parsed_data)
                                    if validated_reflection_intent.action_type == "continue_current_task":
                                        logger.info(f"[{agent_name}] Reflection: Chose to continue. Monologue: {validated_reflection_intent.internal_monologue[:50]}...")
                                        _update_state_value(state, f"simulacra.{agent_id}.status", original_status_before_reflection)
                                    else:
                                        logger.info(f"[{agent_name}] Reflection: Chose to '{validated_reflection_intent.action_type}'. Monologue: {validated_reflection_intent.internal_monologue[:50]}...")
                                        await event_bus.put({ "type": "intent_declared", "actor_id": agent_id, "intent": validated_reflection_intent.model_dump(exclude={'internal_monologue'}) })
                                        _update_state_value(state, f"simulacra.{agent_id}.status", "thinking")
                                except Exception as e_reflect:
                                    logger.error(f"[{agent_name}] Error processing reflection response: {e_reflect}. Staying busy. Response: {response_text}")
                                    _update_state_value(state, f"simulacra.{agent_id}.status", original_status_before_reflection)
                                break
                    elif interjection_type in ["narrative_interjection", "world_event_interjection"]:
                        interjection_details = await generate_llm_interjection_detail(
                            agent_name_for_prompt=agent_name,
                            agent_current_action_desc=agent_state_busy_loop.get("current_action_description", "what you are doing"),
                            interjection_category="narrative" if interjection_type == "narrative_interjection" else "world_event",
                            world_mood=world_mood_global,
                            global_search_agent_runner=search_agent_runner, # Pass global search runner
                            search_agent_session_id=search_agent_session_id # Pass global search session ID
                        )
                        logger.info(f"[{agent_name}] {interjection_type.replace('_', ' ').title()}: {interjection_details}")
                        await event_bus.put({ "type": "intent_declared", "actor_id": agent_id,
                                              "intent": {"action_type": "interrupt_agent_with_observation", "details": interjection_details} })
                        _update_state_value(state, f"simulacra.{agent_id}.status", "thinking")

    except asyncio.CancelledError:
        logger.info(f"[{agent_name}] Task cancelled.")
    except Exception as e:
        logger.error(f"[{agent_name}] Error in agent task: {e}", exc_info=True)
        if agent_id in get_nested(state, "simulacra", default={}):
            _update_state_value(state, f"simulacra.{agent_id}.status", "idle")
    finally:
        logger.info(f"[{agent_name}] Task finished.")

async def run_simulation(
    instance_uuid_arg: Optional[str] = None,
    location_override_arg: Optional[str] = None,
    mood_override_arg: Optional[str] = None
    ):
    global adk_session_service, adk_session_id, adk_session, adk_runner, adk_memory_service
    global world_engine_agent, simulacra_agents, state, live_display_object, narration_agent
    global world_mood_global, search_llm_agent, search_agent_runner, search_agent_session_id # Add search components to global

    console.rule("[bold green]Starting Async Simulation[/]")
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
            logger.info(f"World location overridden in template details: {state[WORLD_TEMPLATE_DETAILS_KEY][LOCATION_KEY]}")
            console.print(f"Location overridden to: [yellow]{location_override_arg}[/yellow]")
        except Exception as e:
            logger.error(f"Failed to apply location override: {e}", exc_info=True)
            console.print(f"[red]Error applying location override: {e}[/red]")

    if mood_override_arg:
        try:
            mood_override = mood_override_arg.strip()
            world_mood_global = mood_override
            logger.info(f"Global world mood overridden to '{world_mood_global}'. This will affect new agent instantiations.")
            console.print(f"Global world mood set to: [yellow]{world_mood_global}[/yellow]")
        except Exception as e:
            logger.error(f"Failed to apply mood override: {e}", exc_info=True)
            console.print(f"[red]Error applying mood override: {e}[/red]")


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
    final_active_sim_ids = []
    state_modified_during_init = False
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
                    state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[persona_key] = persona
                    state_modified_during_init = True
        if persona:
            profile_home_location = profile.get(HOME_LOCATION_KEY)
            valid_locations = list(state.get(WORLD_STATE_KEY, {}).get(LOCATION_DETAILS_KEY, {}).keys())
            final_home_location = None
            if profile_home_location and profile_home_location in valid_locations:
                final_home_location = profile_home_location
            elif profile_home_location and profile_home_location not in valid_locations:
                final_home_location = valid_locations[0] if valid_locations else DEFAULT_HOME_LOCATION_NAME
                state_modified_during_init = True
            else:
                final_home_location = valid_locations[0] if valid_locations else DEFAULT_HOME_LOCATION_NAME
                state_modified_during_init = True
            location_details_dict = state.setdefault(WORLD_STATE_KEY, {}).setdefault(LOCATION_DETAILS_KEY, {})
            if final_home_location not in location_details_dict:
                location_details_dict[final_home_location] = {"name": final_home_location, "description": DEFAULT_HOME_DESCRIPTION, "objects_present": [], "connected_locations": []}
                state_modified_during_init = True
            starting_location = final_home_location
            if profile.get(CURRENT_LOCATION_KEY) != starting_location:
                state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[CURRENT_LOCATION_KEY] = starting_location
                state_modified_during_init = True
            if profile.get(HOME_LOCATION_KEY) != final_home_location:
                state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[HOME_LOCATION_KEY] = final_home_location
                state_modified_during_init = True

            loaded_last_observation = profile.get("last_observation", "Just arrived.")
            default_observation = "Just arrived."
            injected_scenario = "You slowly wake up in your familiar bed. Sunlight streams through the window, and you can hear birds chirping outside."
            if not loaded_last_observation or loaded_last_observation == default_observation:
                final_last_observation = injected_scenario
            else:
                final_last_observation = loaded_last_observation

            state["simulacra"][sim_id] = {
                "id": sim_id, "name": persona.get("Name", sim_id), "persona": persona,
                "location": starting_location, "home_location": final_home_location, "status": "idle",
                "current_action_end_time": state.get('world_time', 0.0),
                "goal": profile.get("goal", persona.get("Initial_Goal", "Determine your own long term goals.")),
                "last_observation": final_last_observation, "memory_log": profile.get("memory_log", []),
                "pending_results": {},
                "last_interjection_sim_time": 0.0,
                "current_action_description": "N/A"
            }
            final_active_sim_ids.append(sim_id)
            logger.info(f"Populated runtime state for simulacrum: {sim_id}")
        else:
            logger.error(f"Could not load persona for active sim {sim_id}. Skipping.")

    state[ACTIVE_SIMULACRA_IDS_KEY] = final_active_sim_ids
    if state_modified_during_init:
        logger.info("Saving state file after persona/location initialization updates.")
        try: save_json_file(state_file_path, state)
        except Exception as save_e: logger.error(f"Failed to save state update after init: {save_e}")

    if not final_active_sim_ids:
         logger.critical("No active simulacra available. Cannot proceed.")
         console.print("[bold red]Error:[/bold red] No verified Simulacra available.")
         sys.exit(1)

    logger.info(f"Initialization complete. Instance {world_instance_uuid} ready with {len(final_active_sim_ids)} simulacra.")
    console.print(f"Running simulation with: {', '.join(final_active_sim_ids)}")
    console.rule()

    adk_session_service = InMemorySessionService()
    adk_session_id = f"sim_session_{world_instance_uuid}"
    adk_session = adk_session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=adk_session_id, state=state
    )
    logger.info(f"ADK Session created: {adk_session_id} with initial state.")

    world_engine_agent = create_world_engine_llm_agent()
    logger.info(f"World Engine Agent '{world_engine_agent.name}' created.")
    narration_agent = create_narration_llm_agent()
    logger.info(f"Narration Agent '{narration_agent.name}' created (using global mood '{world_mood_global}').")

    simulacra_agents = {}
    for sim_id in final_active_sim_ids:
        sim_state_data = state.get("simulacra", {}).get(sim_id, {})
        persona_name = sim_state_data.get("name", sim_id)
        sim_agent = create_simulacra_llm_agent(sim_id, persona_name)
        simulacra_agents[sim_id] = sim_agent
        logger.info(f"Simulacra Agent '{sim_agent.name}' created for {sim_id} (using global mood '{world_mood_global}').")

    adk_runner = Runner(
        agent=world_engine_agent, app_name=APP_NAME,
        session_service=adk_session_service, memory_service=adk_memory_service
    )
    logger.info(f"ADK Runner initialized with default agent '{world_engine_agent.name}'.")
    
    # --- Initialize Dedicated Search Agent, Runner, and Session ---
    search_llm_agent = create_search_llm_agent()
    logger.info(f"Search Agent '{search_llm_agent.name}' created with model '{SEARCH_AGENT_MODEL_NAME}'. Ensure this model supports the google_search tool.")
    
    search_agent_session_id = f"world_feed_search_session_{world_instance_uuid}"
    adk_session_service.create_session(app_name=APP_NAME + "_Search", user_id=USER_ID, session_id=search_agent_session_id) # Create the session
    logger.info(f"ADK Session for Search Agent created: {search_agent_session_id}")
    
    search_agent_runner = Runner(
        agent=search_llm_agent, app_name=APP_NAME + "_Search",
        session_service=adk_session_service
    )
    logger.info(f"Dedicated Runner for Search Agent '{search_llm_agent.name}' initialized.")
    # --- End Search Agent Setup ---

    tasks = []
    final_state_path = os.path.join(STATE_DIR, f"simulation_state_{world_instance_uuid}.json")

    try:
        with Live(generate_table(), console=console, refresh_per_second=1.0/UPDATE_INTERVAL, vertical_overflow="visible") as live:
            live_display_object = live
            tasks.append(asyncio.create_task(time_manager_task(live_display=live), name="TimeManager"))
            tasks.append(asyncio.create_task(interaction_dispatcher_task(), name="InteractionDispatcher"))
            tasks.append(asyncio.create_task(narration_task(), name="NarrationTask"))
            tasks.append(asyncio.create_task(world_engine_task_llm(), name="WorldEngine"))
            tasks.append(asyncio.create_task(world_info_gatherer_task(), name="WorldInfoGatherer"))
            for sim_id in final_active_sim_ids:
                tasks.append(asyncio.create_task(simulacra_agent_task_llm(agent_id=sim_id), name=f"Simulacra_{sim_id}"))

            if not tasks:
                 logger.error("No tasks were created. Simulation cannot run.")
                 console.print("[bold red]Error: No simulation tasks started.[/bold red]")
            else:
                logger.info(f"Started {len(tasks)} tasks.")
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        task.result()
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

        final_uuid = state.get("world_instance_uuid")
        if final_uuid:
            final_state_path = os.path.join(STATE_DIR, f"simulation_state_{final_uuid}.json")
            logger.info("Saving final simulation state.")
            try:
                if not isinstance(state.get("world_time"), (int, float)):
                     logger.warning(f"Final world_time is not a number ({type(state.get('world_time'))}). Saving as 0.0.")
                     state["world_time"] = 0.0
                save_json_file(final_state_path, state)
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
            console.print(generate_table())
        else:
            console.print("[yellow]State dictionary is empty.[/yellow]")
        console.rule("[bold green]Simulation Shutdown Complete[/]")

