# async_event_driven_llm_toy_v3_final.py

import asyncio
import json
import logging
import time
import heapq
import os
import sys
import random
import re
from typing import Any, Dict, List, Optional, Tuple

# --- ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table
    from rich.syntax import Syntax
    console = Console()
except ImportError:
    class DummyConsole:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print(f"\n--- {args[0] if args else ''} ---")
    console = DummyConsole()
    print("Rich console not found, using basic print.")

# --- Logging Setup ---
log_filename = "async_toy_v3.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)
logger = logging.getLogger(__name__)
logger.info("--- Starting Async Event-Driven LLM Toy V3 (Generalized) ---")

# --- Configuration ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.info("dotenv not installed, ensure GOOGLE_API_KEY is set in environment.")

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_GEMINI_PRO", "gemini-2.0-flash")
APP_NAME = "AsyncToySimV3"
USER_ID = "user1"

# --- Simulation Parameters ---
SIMULATION_SPEED_FACTOR = 0.5
UPDATE_INTERVAL = 0.1
MAX_SIMULATION_TIME = 180.0

# --- Core Components ---
event_bus = asyncio.Queue()
schedule: List[Tuple[float, int, Dict[str, Any]]] = []
schedule_event_counter = 0

# --- Central State Dictionary (Standardized Objects) ---
state: Dict[str, Any] = {
    "world_time": 0.0,
    "sim_alex": {
        "id": "alex",
        "name": "Alex",
        "location": "MysteriousOffice",
        "status": "idle",
        "current_action_end_time": 0.0,
        "goal": "Unknown.",
        "last_observation": "Woke up in an unfamiliar room.",
        "memory_log": ["Woke up here."] # Simple memory log (can be expanded)
    },
    "objects": {
        "computer_office": {
            "id": "computer_office",
            "name": "Old Computer",
            "location": "MysteriousOffice",
            "power": "off",
            "logged_in": False,
            "password": "password123",
            "current_user": None,
            "description": "An old, bulky desktop computer sits on the desk. Its screen is dark.",
            "properties": ["powerable", "login_required"]
        },
        "desk_office": {
            "id": "desk_office",
            "name": "Wooden Desk",
            "location": "MysteriousOffice",
            "description": "A sturdy wooden desk. The surface is mostly clear except for the computer.",
            "drawers_locked": True,
            "properties": ["has_drawers", "surface", "searchable"] # Make desk searchable
        },
        "door_office": {
            "id": "door_office",
            "name": "Heavy Door",
            "location": "MysteriousOffice",
            "description": "A heavy, dark wooden door. It looks solid and connects to a corridor.",
            "locked": True,
            "status": "closed",
            "key_required": "small_brass_key", # Changed key name
            "destination": "Corridor",
            "properties": ["lockable", "openable"]
        },
         "window_office": {
            "id": "window_office",
            "name": "Barred Window",
            "location": "MysteriousOffice",
            "description": "A single, barred window high up on one wall. It looks out onto a brick wall.",
            "openable": False,
            "properties": ["viewable", "barred"]
        },
        "bookshelf_office": {
             "id": "bookshelf_office",
             "name": "Bookshelf",
             "location": "MysteriousOffice",
             "description": "A tall bookshelf filled with dusty technical manuals and binders.",
             "properties": ["searchable"]
        }
    },
    "locations": {
        "MysteriousOffice": {
            "description": "A windowless room, surprisingly tidy. Walls are painted a neutral grey. A sturdy wooden desk with an old computer sits against one wall. Opposite is a heavy wooden door. High on another wall is a small, barred window showing only a brick wall outside. A bookshelf stands in the corner. The air is still and quiet."
        },
        "Corridor": {
             "description": "A dimly lit, narrow corridor stretches ahead. The air is cooler here."
        }
    },
    "narrative_log": ["Simulation started."]
}

# --- ADK Setup ---
session_service: Optional[InMemorySessionService] = None
session_id: Optional[str] = None
session: Optional[Session] = None
runner: Optional[Runner] = None

# --- Agent Definitions using LlmAgent (Generalized World Engine) ---

def create_simulacra_llm_agent(sim_id: str, persona_name: str) -> LlmAgent:
    """Creates the LLM agent representing the character."""
    agent_name = f"SimulacraLLM_{sim_id}"
    agent_state_ref = state[f'sim_{sim_id}']

    # <<< PROMPT WITH MONOLOGUE OUTPUT >>>
    instruction = f"""
You are {persona_name} ({sim_id}), currently in a mysterious room. Your primary goal is to **{agent_state_ref['goal']}**.
You think step-by-step to decide your next action based on your observations, goal, and internal state.

**Current State Info (Provided via context/trigger):**
- Your Goal: `state['sim_{sim_id}']['goal']`
- Your Location ID: `state['sim_{sim_id}']['location']`
- Your Status: `state['sim_{sim_id}']['status']` (Should be 'idle' when you plan)
- Current Time: `state['world_time']`
- Last Observation/Event: `state['sim_{sim_id}']['last_observation']`
- Recent History (Last ~6 events): Provided in trigger context.
- Objects in Room (IDs and Names): Provided in trigger context.
- Location Description (if you looked): Provided in 'Last Observation' after 'look_around'

**Your Goal:** {agent_state_ref['goal']}

**Thinking Process (Internal Monologue - Follow this process and INCLUDE it in your output):**
1.  **Recall & React:** What was the last thing I observed or did (`last_observation`)? What happened just before that (`Recent History`)? Did my recent actions work? How does it relate to my goal? How am I feeling?
2.  **Analyze Goal:** What is my goal? What do I want to achieve? How does it relate to my current state? What are the obstacles in my way?
3.  **Identify Options:** Based on the current state and my recent history/last observation, what actions could I take *right now* using the available object IDs?
    *   `look_around`: Get a detailed description of the room.
    *   `use [object_id]`: Interact with an object (e.g., `use door_office`, `use computer_office`, `use bookshelf_office`). Specify `details`.
    *   `wait`: If stuck or waiting.
    *   `think`: If needing to pause and reflect.
4.  **Prioritize:** If the door is open, leave! If locked, find a key (check desk, bookshelf?) or alternative (computer?). If computer login failed, try a different password or check elsewhere. If a search failed, try searching somewhere else. Don't repeat the exact same failed action immediately based on recent history.
5.  **Formulate Intent:** Choose the single best action. Use the correct `target_id` from the objects list. Be specific in 'details'.

**Output:**
- Output ONLY a JSON object representing your chosen intent AND your internal monologue.
- Format: `{{"action_type": "...", "target_id": "...", "details": "...", "internal_monologue": "..."}}` # Escaped braces
- Valid `action_type`: "use", "wait", "look_around", "think"
- Use `target_id` from the provided object list (e.g., "door_office", "computer_office").
- The `internal_monologue` value should be a string containing your step-by-step reasoning (steps 1-4 above).
- Example (Try Door Handle): `{{"action_type": "use", "target_id": "door_office", "details": "try handle", "internal_monologue": "1. Last time I looked, the door was closed and locked. Goal is to get out. Feeling determined. 2. Door is main exit, need to unlock it. 3. Options: try handle, check desk, check bookshelf. 4. Trying the handle again is quick confirmation."}}` # Escaped braces
- Example (Look): `{{"action_type": "look_around", "internal_monologue": "1. Just woke up, don't know where I am. Goal is escape. Feeling confused. 2. Need to understand the room layout. 3. Options: look around, try door. 4. Looking around gives the most info first."}}` # Escaped braces
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        instruction=instruction,
        description=f"LLM Simulacra agent for {persona_name}, focused on escaping."
    )

# <<< WORLD ENGINE PROMPT WITH OUTCOME FOCUS >>>
def create_world_engine_llm_agent() -> LlmAgent:
    """Creates the GENERALIZED LLM agent responsible for resolving actions."""
    agent_name = "WorldEngineLLMAgent"

    instruction = f"""
You are the World Engine, simulating the physics and state changes of the environment. You process a single declared intent from a Simulacra and determine its outcome, duration, and narrative description based on the current world state. **Crucially, your narrative must describe the RESULT or CONSEQUENCE of the action attempt, not just the attempt itself.**

**Input (Provided via trigger message):**
- Actor ID: e.g., "alex"
- Actor Name: e.g., "Alex"
- Actor Location ID: e.g., "MysteriousOffice"
- Intent: `{{"action_type": "...", "target_id": "...", "details": "..."}}` # Escaped here
- Current World Time: e.g., 15.3
- Target Object State: The current state dictionary of the object specified in `intent['target_id']` (if applicable).
- Location Description: The description of the actor's current location.

**Your Task:**
1.  Examine the actor's intent (`action_type`, `target_id`, `details`).
2.  **Determine Validity & Outcome based on Intent and Target Object State:**
    *   **Location Check:** Is the actor in the same location as the target object? If not, `valid_action: false`, narrative: "[Actor] tries to use [Object Name] but it's not here." Duration 1s.
    *   **Action Type Check:** Is `action_type` valid ("use", "wait", "look_around", "think")? If not, `valid_action: false`, narrative: "[Actor] attempts to [invalid action], which seems impossible." Duration 1s.
    *   **If `action_type` is "use":**
        *   Look at the `Target Object State` (description, properties, status like power, locked).
        *   Is the intended `details` plausible for this object?
        *   **Determine Outcome & Narrative (Focus on Result):**
            *   **Turning On/Off:** If `details` is "turn on" and object is `powerable` and `power: "off"`. Results: `{{"objects.[target_id].power": "on"}}`. Narrative: "The [Object Name] powers on, [brief description of effect, e.g., screen flickers to life]." Duration 2s. `valid_action: true`. If already on, Narrative: "The [Object Name] is already on." Duration 1s. `valid_action: true`. (Similar logic for "turn off").
            *   **Opening/Closing:** If `details` is "open" and object is `openable`, `locked: false`, `status: "closed"`. Results: `{{"objects.[target_id].status": "open"}}`. Narrative: "The [Object Name] swings open, revealing [brief description of what's revealed, e.g., a dark interior, the corridor]." Duration 2s. `valid_action: true`. If locked, Narrative: "The [Object Name] remains firmly locked." Duration 2s. `valid_action: true` (the attempt was valid). If already open, Narrative: "The [Object Name] is already open." Duration 1s. `valid_action: true`. (Similar logic for "close").
            *   **Locking/Unlocking:** If `details` is "unlock" and object is `lockable`. Check if actor has `key_required` (assume NO for now). Narrative: "[Actor] tries the lock on the [Object Name], but doesn't have the right key." Duration 3s. `valid_action: true`. (If they had key: Results: `{{"objects.[target_id].locked": false}}`. Narrative: "The key turns smoothly, and the [Object Name] unlocks!").
            *   **Going Through (Doors):** If `target_id` is a door (e.g., "door_office") and `details` is "go through" (or similar). Check if `status` is "open". If yes, Results: `{{"sim_[ACTOR_ID].location": "[Destination]"}}`. Narrative: "[Actor] steps through the open [Object Name] into the [Destination Name]." Duration 3s. `valid_action: true`. If closed/locked, `valid_action: false`, narrative: "The [Object Name] blocks the way; it's closed/locked." Duration 1s.
            *   **Computer Login:** If `target_id` is a computer, `power: "on"`, `logged_in: false`. If `details` contain "login" or "password". Extract password attempt. Compare to `Target Object State['password']`.
                *   Success: Results: `{{"objects.[target_id].logged_in": true, "objects.[target_id].current_user": "[ACTOR_ID]", "objects.[target_id].last_login_attempt_failed": false}}`. Narrative: "Login successful! The computer displays a simple desktop interface." Duration 5s. `valid_action: true`.
                *   Failure: Results: `{{"objects.[target_id].last_login_attempt_failed": true}}`. Narrative: "Login failed. The screen displays 'Incorrect password'." Duration 5s. `valid_action: true`.
            *   **Computer Use (Logged In):** If computer `power: "on"`, `logged_in: true`. Handle "search files", "log off". Narrative describes outcome (e.g., "After searching the files, Alex finds an email mentioning...", "Alex logs off the computer, returning it to the login screen."). Duration 10-20s. Update `current_user` on log off.
            *   **Searching:** If `details` involve "search", "look for", "check" and object is `searchable`. **Decide outcome (e.g., 30% chance find).**
                *   **Found:** Narrative: "[Actor] searches the [Object Name] and finds [Specific Item, e.g., a small brass key, a crumpled note]!" Duration 10s. `results: {{}}` (or update inventory later). `valid_action: true`. # Escaped braces
                *   **Not Found:** Narrative: "[Actor] searches the [Object Name] thoroughly but finds nothing useful or out of the ordinary." Duration 10s. `results: {{}}`. `valid_action: true`. # Escaped braces
            *   **Other Plausible 'use':** Narrative describes the *result* of the interaction (e.g., "Alex examines the window closely; the bars are solid and rusted.", "Alex reads the book title: 'Advanced Circuitry'."). Short/medium duration. `valid_action: true`.
            *   **Implausible 'use':** `valid_action: false`, narrative: "[Actor] tries to [details] the [Object Name], but nothing happens / that doesn't seem possible." Duration 1s.
    *   **If `action_type` is "look_around":** Narrative MUST BE the exact `Location Description` provided in the input. Duration 3s. No results. `valid_action: true`.
    *   **If `action_type` is "wait" or "think":** Narrative: "Alex waits, observing the quiet room." or "Alex pauses, considering the situation." Duration 5s or 1s. No results. `valid_action: true`.
3.  Calculate `duration` (float, simulation seconds).
4.  Determine `results` (dict, state changes on completion, using dot notation). Replace '[ACTOR_ID]' and '[target_id]' appropriately.
5.  Generate `narrative` (string, present tense, descriptive, **focusing on the outcome/result**).
6.  Determine `valid_action` (boolean).

**Output:**
- Output ONLY a JSON object with the keys: "duration", "results", "narrative", "valid_action". Ensure results dictionary keys use dot notation.
"""
    return LlmAgent(
        name=agent_name,
        model=MODEL_NAME,
        instruction=instruction,
        description="LLM World Engine: Resolves actions based on target state, calculates duration/results, generates outcome-focused narrative."
    )

# --- Instantiate LLM Agents ---
sim_alex_llm_agent = create_simulacra_llm_agent("alex", "Alex")
world_engine_llm_agent = create_world_engine_llm_agent() # Uses the updated prompt


# --- Agent Task: Simulacra (Alex) using LLM ---
async def simulacra_agent_task_llm(agent_id: str):
    """Represents the thinking and acting loop for a simulacra using LLM."""
    while runner is None:
        await asyncio.sleep(0.01)

    agent_state = state[f"sim_{agent_id}"]
    llm_agent = sim_alex_llm_agent
    logger.info(f"[{agent_state['name']}] LLM Agent task started.")

    while True:
        try:
            while agent_state["status"] == "busy":
                await asyncio.sleep(UPDATE_INTERVAL * 2)

            current_time = state["world_time"]
            current_location_id = agent_state.get("location", "Unknown")
            objects_in_room_full = {
                obj_id: obj_data for obj_id, obj_data in state.get("objects", {}).items()
                if obj_data.get("location") == current_location_id
            }
            objects_in_room_summary = {
                obj_id: obj_data.get("name", obj_data.get("description", "Unnamed Object"))
                for obj_id, obj_data in objects_in_room_full.items()
            }

            # Get recent history
            recent_history_log = state.get("narrative_log", [])[-6:]
            recent_history_str = "\n".join(recent_history_log)

            logger.debug(f"[{agent_state['name']}] Observing at time {current_time:.1f}. Status: {agent_state['status']}. Objects: {list(objects_in_room_summary.keys())}")

            # Add history to trigger_text
            trigger_text = f"""
Current State for {agent_state['name']} ({agent_id}):
Goal: {agent_state.get('goal', 'None')}
Location: {current_location_id}
Status: {agent_state.get('status', 'Unknown')}
Time: {current_time:.1f}
Last Observation/Event: {agent_state.get('last_observation', 'None')}
Recent History (Last ~6 Events):
{recent_history_str}
Objects in Room (ID: Name/Description): {json.dumps(objects_in_room_summary, indent=2)}

Based on this state and your goal, what is your next action intent? Use the object IDs provided above.
"""
            trigger = types.Content(parts=[types.Part(text=trigger_text)])

            # Initialize variables
            intent_json_str = None
            parsed_intent = None

            logger.info(f"[{agent_state['name']}] Requesting intent from LLM...")
            if runner and session_id:
                runner.agent = llm_agent
                async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=trigger):
                    if event.is_final_response() and event.content:
                        intent_json_str = event.content.parts[0].text
                        logger.debug(f"SimLLM ({event.author}) Final Content: {intent_json_str[:100]}...")
                    elif event.error_message:
                        logger.error(f"SimLLM Error ({event.author}): {event.error_message}")
                    else:
                         logger.debug(f"SimLLM Event ({event.author}): Type={type(event.content)}")
            else:
                logger.error(f"[{agent_state['name']}] Runner or session_id not initialized, cannot call LLM.")
                await asyncio.sleep(1)
                continue

            # Parse and Print
            if intent_json_str:
                parsed_intent = None # Ensure reset before try block
                try:
                    match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*\})", intent_json_str, re.DOTALL)
                    if match:
                        json_str_to_parse = match.group(1) or match.group(2)
                        if json_str_to_parse:
                            parsed_intent = json.loads(json_str_to_parse)
                            logger.info(f"[{agent_state['name']}] LLM decided intent: {parsed_intent.get('action_type')}")

                            # Print Monologue and Intent
                            monologue = parsed_intent.get("internal_monologue")
                            if monologue:
                                console.print(Panel(monologue, title=f"{agent_state['name']} Internal Monologue @ {current_time:.1f}s", border_style="dim yellow", expand=False))
                            else:
                                console.print(f"\n[dim yellow][{agent_state['name']} @ {current_time:.1f}s - No Monologue Provided][/dim yellow]")

                            intent_to_print = {k: v for k, v in parsed_intent.items() if k != "internal_monologue"}
                            console.print(f"\n[bold yellow][{agent_state['name']} Intent @ {current_time:.1f}s][/bold yellow]")
                            console.print(json.dumps(intent_to_print, indent=2))

                        else:
                             logger.error(f"[{agent_state['name']}] Regex matched but no JSON content found. Raw: {repr(intent_json_str)}")
                             console.print(f"[red]Error extracting JSON from {agent_state['name']} response.[/red]")
                    else:
                        logger.error(f"[{agent_state['name']}] Could not find JSON block in LLM response. Raw: {repr(intent_json_str)}")
                        console.print(f"[red]Could not find JSON block in {agent_state['name']} response.[/red]")

                except json.JSONDecodeError as e:
                    logger.error(f"[{agent_state['name']}] Failed to parse extracted JSON: {e}. Extracted: {repr(json_str_to_parse if 'json_str_to_parse' in locals() else 'N/A')}. Raw: {repr(intent_json_str)}")
                    console.print(f"[red]Error parsing extracted {agent_state['name']} JSON.[/red]")
                except Exception as e:
                    logger.error(f"[{agent_state['name']}] Error processing intent JSON: {e}. Raw: {repr(intent_json_str)}")
                    console.print(f"[red]Error processing {agent_state['name']} intent JSON.[/red]")
            else:
                logger.warning(f"[{agent_state['name']}] LLM did not return an intent.")
                console.print(f"[yellow]Warning: {agent_state['name']} LLM did not return intent.[/yellow]")

            # Act (Publish Intent)
            if parsed_intent and isinstance(parsed_intent, dict) and "action_type" in parsed_intent:
                event_data = {
                    "type": "intent_declared",
                    "actor_id": agent_id,
                    "intent": parsed_intent, # Publish the full dict including monologue
                    "timestamp": current_time
                }
                await event_bus.put(event_data)
                logger.debug(f"[{agent_state['name']}] Published intent: {parsed_intent['action_type']}")
                agent_state["status"] = "busy"
                agent_state["current_action_end_time"] = float('inf')
            else:
                logger.warning(f"[{agent_state['name']}] No valid intent generated. Waiting.")
                await asyncio.sleep(random.uniform(3.0, 5.0) / SIMULATION_SPEED_FACTOR)

            await asyncio.sleep(random.uniform(1.0, 3.0) / SIMULATION_SPEED_FACTOR)

        except asyncio.CancelledError:
            logger.info(f"[{agent_state['name']}] LLM Task cancelled.")
            break
        except Exception as e:
            logger.exception(f"[{agent_state['name']}] Error in LLM agent loop: {e}")
            await asyncio.sleep(5.0 / SIMULATION_SPEED_FACTOR)

# --- World Engine Task using LLM (Generalized) ---
async def world_engine_task_llm():
    """Listens for intents, calls GENERALIZED LLM to resolve, and schedules completions."""
    global schedule_event_counter
    while runner is None:
        await asyncio.sleep(0.01)

    llm_agent = world_engine_llm_agent
    logger.info("[WorldEngineLLM] Task started, listening for intents.")
    while True:
        try:
            event = await event_bus.get()
            current_time = state["world_time"]
            logger.debug(f"[WorldEngineLLM] Received event: {event['type']} at time {current_time:.1f}")

            if event["type"] == "intent_declared":
                actor_id = event["actor_id"]
                intent = event["intent"]
                actor_state = state.get(f"sim_{actor_id}")
                actor_name = actor_state.get("name", actor_id) if actor_state else actor_id
                actor_location = actor_state.get("location", "Unknown") if actor_state else "Unknown"
                location_desc = state.get("locations", {}).get(actor_location, {}).get("description", "An undescribed place.")

                target_id = intent.get("target_id")
                target_object_state = None
                action_type = intent.get("action_type")

                if action_type == "use" and target_id:
                    target_object_state = state.get("objects", {}).get(target_id)
                    if not target_object_state:
                        logger.warning(f"[WorldEngineLLM] Intent target object '{target_id}' not found in state.")
                        if actor_state:
                            actor_state["status"] = "idle"
                            actor_state["last_observation"] = f"Tried to use '{target_id}', but it doesn't seem to exist here."
                        state["narrative_log"].append(f"[T{current_time:.1f}] {actor_name} tries to use '{target_id}', but it doesn't seem to exist here.")
                        event_bus.task_done()
                        continue

                trigger_text = f"""
Resolve action for {actor_name} ({actor_id}) at time {current_time:.1f}.
Location: {actor_location}
Intent: {json.dumps(intent)}

Current Relevant State:
Target Object State ({target_id}): {json.dumps(target_object_state, indent=2) if target_object_state else "N/A (Action might not target an object)"}
Location Description: {location_desc}

Determine duration, results (state changes on completion), narrative, and validity based on your general instructions for object interaction, movement, looking, waiting, and thinking. Remember to replace '[ACTOR_ID]' and '[target_id]' in results/narrative appropriately.
"""
                trigger = types.Content(parts=[types.Part(text=trigger_text)])
                resolution_json_str = None
                parsed_resolution = None

                logger.info(f"[WorldEngineLLM] Requesting resolution for {actor_id}'s intent: {action_type} on {target_id or 'N/A'}")
                if runner and session_id:
                    runner.agent = llm_agent
                    async for llm_event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=trigger):
                        if llm_event.is_final_response() and llm_event.content:
                            resolution_json_str = llm_event.content.parts[0].text
                            logger.debug(f"WorldLLM ({llm_event.author}) Final Content: {resolution_json_str[:100]}...")
                        elif llm_event.error_message:
                            logger.error(f"WorldLLM Error ({llm_event.author}): {llm_event.error_message}")
                        else:
                            logger.debug(f"WorldLLM Event ({llm_event.author}): Type={type(llm_event.content)}")
                else:
                    logger.error("[WorldEngineLLM] Runner or session_id not initialized, cannot call LLM.")
                    event_bus.task_done()
                    continue

                if resolution_json_str:
                    parsed_resolution = None
                    try:
                        match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*\})", resolution_json_str, re.DOTALL)
                        if match:
                            json_str_to_parse = match.group(1) or match.group(2)
                            if json_str_to_parse:
                                # Replace placeholders before parsing
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

                                parsed_resolution = json.loads(json_str_to_parse)
                                logger.info(f"[WorldEngineLLM] LLM resolved action: Valid={parsed_resolution.get('valid_action')}, Duration={parsed_resolution.get('duration')}")
                                console.print(f"\n[bold blue][World Engine Resolution @ {current_time:.1f}s][/bold blue]")
                                console.print(json.dumps(parsed_resolution, indent=2))
                            else:
                                logger.error(f"[WorldEngineLLM] Regex matched but no JSON content found. Raw: {repr(resolution_json_str)}")
                                console.print(f"[red]Error extracting JSON from World Engine response.[/red]")
                        else:
                            logger.error(f"[WorldEngineLLM] Could not find JSON block in LLM response. Raw: {repr(resolution_json_str)}")
                            console.print(f"[red]Could not find JSON block in World Engine response.[/red]")

                    except json.JSONDecodeError as e:
                        logger.error(f"[WorldEngineLLM] Failed to parse extracted JSON: {e}. Extracted: {repr(json_str_to_parse if 'json_str_to_parse' in locals() else 'N/A')}. Raw: {repr(resolution_json_str)}")
                        console.print(f"[red]Error parsing extracted World Engine JSON.[/red]")
                    except Exception as e:
                        logger.error(f"[WorldEngineLLM] Error processing resolution JSON: {e}. Raw: {repr(resolution_json_str)}")
                        console.print(f"[red]Error processing World Engine resolution JSON.[/red]")
                else:
                    logger.warning("[WorldEngineLLM] LLM did not return a resolution.")
                    console.print(f"[yellow]Warning: World Engine LLM did not return resolution.[/yellow]")

                if parsed_resolution and isinstance(parsed_resolution, dict):
                    duration = parsed_resolution.get("duration", 1.0)
                    results = parsed_resolution.get("results", {})
                    narrative = parsed_resolution.get("narrative", f"{actor_name} did something unclear.")
                    valid_action = parsed_resolution.get("valid_action", False)

                    if valid_action:
                        completion_time = current_time + duration
                        schedule_event_counter += 1
                        completion_event = {
                            "type": "action_complete",
                            "actor_id": actor_id,
                            "action_type": intent.get("action_type"),
                            "results": results,
                            "narrative": narrative,
                            "start_time": current_time,
                            "end_time": completion_time
                        }
                        heapq.heappush(schedule, (completion_time, schedule_event_counter, completion_event))
                        logger.info(f"[WorldEngineLLM] Scheduled '{intent.get('action_type')}' completion for {actor_id} at {completion_time:.1f} (Duration: {duration:.1f}s).")

                        if actor_state:
                            actor_state["status"] = "busy"
                            actor_state["current_action_end_time"] = completion_time

                        await event_bus.put({
                            "type": "action_started", "actor_id": actor_id,
                            "action_details": intent, "end_time": completion_time,
                            "timestamp": current_time
                        })
                    else:
                        logger.warning(f"[WorldEngineLLM] LLM deemed action invalid for {actor_id}: {intent.get('action_type')}. Narrative: {narrative}")
                        if actor_state:
                            actor_state["status"] = "idle"
                            actor_state["current_action_end_time"] = current_time
                            actor_state["last_observation"] = narrative
                        state["narrative_log"].append(f"[T{current_time:.1f}] {narrative}")

                else:
                    logger.error(f"[WorldEngineLLM] Invalid or missing resolution from LLM for {actor_id}'s intent. Setting actor to idle.")
                    if actor_state:
                        actor_state["status"] = "idle"
                        actor_state["current_action_end_time"] = current_time
                        actor_state["last_observation"] = f"Action '{intent.get('action_type')}' could not be resolved."
                    state["narrative_log"].append(f"[T{current_time:.1f}] {actor_name}'s action ({intent.get('action_type')}) could not be resolved.")

            event_bus.task_done()

        except asyncio.CancelledError:
            logger.info("[WorldEngineLLM] Task cancelled.")
            break
        except Exception as e:
            logger.exception(f"[WorldEngineLLM] Error processing event: {e}")
            try: event_bus.task_done()
            except ValueError: pass

# --- Time Manager / Main Loop Task ---
async def time_manager_task():
    """Advances time, processes scheduled events, and updates state."""
    logger.info("[TimeManager] Task started.")
    last_real_time = time.monotonic()

    def generate_table() -> Table:
        """Generates the Rich table for live display."""
        table = Table(title=f"Simulation State @ {state['world_time']:.2f}s",
                      show_header=True,
                      header_style="bold magenta",
                      box=None,
                      padding=(0, 1),
                      expand=True)

        table.add_column("Parameter", style="dim", no_wrap=True)
        table.add_column("Value", overflow="fold", no_wrap=False)

        alex_state = state.get("sim_alex", {})
        computer_state = state.get("objects", {}).get("computer_office", {})
        desk_state = state.get("objects", {}).get("desk_office", {})
        door_state = state.get("objects", {}).get("door_office", {})
        window_state = state.get("objects", {}).get("window_office", {})
        bookshelf_state = state.get("objects", {}).get("bookshelf_office", {})

        table.add_row("Alex Status", alex_state.get("status", "Unknown"))
        table.add_row("Alex Goal", alex_state.get("goal", "Unknown"))
        table.add_row("Alex Location", alex_state.get("location", "Unknown"))
        table.add_row("Alex Action End", f"{alex_state.get('current_action_end_time', 0.0):.2f}s" if alex_state.get('status')=='busy' else "N/A")
        table.add_row("Alex Last Obs.", alex_state.get("last_observation", "None"))
        table.add_row("--- Objects ---", "---")
        table.add_row("Computer Power", computer_state.get("power", "Unknown"))
        table.add_row("Computer Logged In", str(computer_state.get("logged_in", "Unknown")))
        table.add_row("Computer User", str(computer_state.get("current_user")))
        table.add_row("Desk Drawers", "Locked" if desk_state.get("drawers_locked") else "Unlocked")
        table.add_row("Door Locked", "Yes" if door_state.get("locked") else "No")
        table.add_row("Door Status", door_state.get("status", "Unknown"))
        table.add_row("Window Openable", "Yes" if window_state.get("openable") else "No")
        table.add_row("--- System ---", "---")
        table.add_row("Schedule Size", str(len(schedule)))
        table.add_row("Event Bus Size", str(event_bus.qsize()))
        log_display = "\n".join(state.get('narrative_log', [])[-6:])
        table.add_row("Narrative Log", log_display)
        return table

    try:
        with Live(generate_table(), console=console, refresh_per_second=4, vertical_overflow="visible") as live:
            while state["world_time"] < MAX_SIMULATION_TIME:
                current_real_time = time.monotonic()
                real_delta_time = current_real_time - last_real_time
                last_real_time = current_real_time
                sim_delta_time = real_delta_time * SIMULATION_SPEED_FACTOR
                new_sim_time = state["world_time"] + sim_delta_time
                state["world_time"] = new_sim_time

                while schedule and schedule[0][0] <= new_sim_time:
                    completion_time, event_id, event_data = heapq.heappop(schedule)
                    logger.info(f"[TimeManager] Processing scheduled event {event_id} ({event_data.get('type')}) due at {completion_time:.1f}")

                    results = event_data.get("results", {})
                    for key_path, value in results.items():
                        try:
                            keys = key_path.split('.')
                            target = state
                            for k in keys[:-1]: target = target[k]
                            target[keys[-1]] = value
                            logger.info(f"[TimeManager] Applied state update: {key_path} = {value}")
                        except (KeyError, TypeError, IndexError) as e:
                            logger.error(f"[TimeManager] Failed to apply scheduled state update '{key_path}': {e}")

                    narrative = event_data.get("narrative")
                    if narrative:
                        state["narrative_log"].append(f"[T{completion_time:.1f}] {narrative}")

                    actor_id = event_data.get("actor_id")
                    if actor_id:
                        actor_state = state.get(f"sim_{actor_id}")
                        if actor_state and actor_state.get("status") == "busy" and abs(actor_state.get("current_action_end_time", -1) - completion_time) < 0.01:
                            actor_state["status"] = "idle"
                            actor_state["current_action_end_time"] = completion_time
                            logger.info(f"[TimeManager] Set {actor_id} status to idle.")
                            actor_state["last_observation"] = narrative
                            logger.info(f"[TimeManager] Stored action narrative in {actor_id}'s last_observation.")
                        elif actor_state and actor_state.get("status") == "busy":
                             logger.warning(f"[TimeManager] Action completion for {actor_id} at {completion_time:.1f} did not match current busy state end time {actor_state.get('current_action_end_time', -1)}. Status left as busy.")
                        elif actor_state:
                             logger.debug(f"[TimeManager] Action completion for {actor_id} at {completion_time:.1f} occurred while actor was {actor_state.get('status')}.")

                    await event_bus.put(event_data)

                live.update(generate_table())
                await asyncio.sleep(UPDATE_INTERVAL)

    except asyncio.CancelledError:
        logger.info("[TimeManager] Task cancelled.")
    except Exception as e:
        logger.exception(f"[TimeManager] Error in main loop: {e}")
    finally:
        logger.info(f"[TimeManager] Loop finished at sim time {state['world_time']:.1f}")
        console.print(Panel(f"Simulation loop ended at sim time {state['world_time']:.1f}s", style="bold red"))
        console.print("\nFinal State Table:")
        console.print(generate_table())


# --- Main Execution ---
async def run_simulation():
    """Sets up and runs all concurrent tasks."""
    global session_service, session_id, session, runner

    console.rule("[bold green]Starting Event-Driven LLM Simulation V3 (Generalized)[/]")

    if not API_KEY:
        console.print("[bold red]ERROR: GOOGLE_API_KEY environment variable not set.[/bold red]")
        sys.exit(1)
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        logger.info("Google Generative AI configured.")
    except Exception as e:
        logger.critical(f"Failed to configure Google API: {e}", exc_info=True)
        console.print(f"[bold red]ERROR: Failed to configure Google API: {e}[/bold red]")
        sys.exit(1)

    session_service = InMemorySessionService()
    session_id = f"async_toy_v3_{asyncio.get_running_loop().__hash__()}"
    session = session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state=state
    )
    runner = Runner(agent=sim_alex_llm_agent, app_name=APP_NAME, session_service=session_service)
    logger.info(f"ADK Session and Runner initialized. Session ID: {session_id}")

    tasks = []
    try:
        tasks.append(asyncio.create_task(simulacra_agent_task_llm("alex")))
        tasks.append(asyncio.create_task(world_engine_task_llm()))
        tasks.append(asyncio.create_task(time_manager_task()))

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            if task.exception():
                logger.error(f"Task {task.get_name()} raised an exception: {task.exception()}")

        logger.info("One of the main tasks completed or failed. Initiating shutdown.")

    except Exception as e:
        logger.exception(f"Error during simulation setup or execution: {e}")
    finally:
        logger.info("Cancelling remaining tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All tasks cancelled or finished.")
        console.rule("[bold green]Simulation Shutdown Complete[/]")


if __name__ == "__main__":
    try:
        if sys.platform == "win32":
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
        console.print("\n[orange_red1]Simulation interrupted.[/]")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred:[/bold red]")
        console.print_exception(show_locals=False)
    finally:
        logging.shutdown()
