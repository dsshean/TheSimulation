# main.py (Phased Architecture Setup)

import asyncio
import copy
import json
import glob
import logging # Add logging import
import os
import sys
import re # Import re for sanitizing name
from typing import Dict, Tuple, Any # Added Any for type hint
from datetime import datetime, timezone
# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

# ADK Imports
from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent # Added ParallelAgent
from google.adk.runners import Runner
# Choose your session service (InMemory or persistent like DatabaseSessionService)
from google.adk.sessions import InMemorySessionService, BaseSessionService, Session # Added BaseSessionService, Session
# from google.adk.sessions import DatabaseSessionService # Example for persistence

# Rich Imports
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# --- Agent Imports ---
# Import the INSTANCES directly from their modules where defined
# Ensure these modules have the final, correct agent definitions
from src.agents.world_state_agent import world_state_agent # Updater + Executor Tool Provider
from src.agents.world_engine import world_engine_agent # Validator Role
from src.agents.npc import npc_agent                 # Interaction Resolver Role
from src.agents.narration import narration_agent     # Narrator Role (Simple)
from src.agents.world_execution_agent import world_execution_agent # Import the World Execution Agent
# Import the factory for Simulacra
from src.agents.simulacra import create_agent as create_simulacra_agent

# --- Config, State, Generation ---
from src.config import settings
# from src.session.initial_state import default_initial_sim_state # Keep for fallback (if needed)
# from src.generation.life_generator import generate_new_simulacra_background # If needed

# --- Simulation Loop ---
from src.simulation_loop import run_phased_simulation # Import the NEW phased loop

# Setup console
console = Console()

# --- MODIFIED: Configure logging to file ---
log_filename = "simulation.log"
logging.basicConfig(
    level=logging.INFO,  # Capture INFO, WARNING, ERROR, CRITICAL messages
    # level=logging.DEBUG, # Use DEBUG for more verbose tool/event details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'  # 'w' overwrites the file each run, 'a' appends
)
# Optional: Also log to console (in addition to file) at a higher level
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.WARNING) # Only show warnings and errors on console
# console_formatter = logging.Formatter('%(levelname)s: %(message)s')
# console_handler.setFormatter(console_formatter)
# logging.getLogger('').addHandler(console_handler) # Add handler to root logger

logger = logging.getLogger(__name__) # Get logger for main.py
logger.info(f"--- Application Start --- Logging configured to file: {log_filename}")
# --- END MODIFIED ---


# --- Constants ---
APP_NAME = "TheSimulation"
USER_ID = "player1"
SESSION_ID = "session_phased_1" # Unique ID for this run/save
MAX_TURNS = 10 # Default simulation length
NUM_SIMULACRA = 2 # Define how many simulacra to run
STATE_FILE_PATH = "simulation_state.json" # Path for the persistent state
WORLD_CONFIG_PATH = "world_config.json"   # Path for world config

# --- MODIFIED: State Key Constants (Include 'simulacra_' prefix) ---
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_GOAL_KEY_FORMAT = "simulacra_{}_goal"
SIMULACRA_PERSONA_KEY_FORMAT = "simulacra_{}_persona"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
WORLD_STATE_KEY = "current_world_state"
# --- END MODIFIED ---

# --- Helper Function to Sanitize Name for ID ---
def sanitize_name_for_id(name: str) -> str:
    """Converts a name into a suitable lowercase ID with underscores."""
    if not name:
        return "unknown_agent"
    # Remove non-alphanumeric characters (except spaces), replace spaces with underscores
    s = re.sub(r'[^\w\s]', '', name.lower())
    s = re.sub(r'\s+', '_', s)
    return s if s else "unknown_agent" # Ensure not empty


# --- Helper to load first JSON summary ---
def load_first_life_summary():
    """Loads the first life_summary JSON file found in the current directory."""
    console.print("[cyan]Looking for existing life_summary JSON files...[/cyan]")
    search_dir = "."
    try:
        for file in os.listdir(search_dir):
            if file.startswith("life_summary_") and file.endswith(".json"):
                file_path = os.path.join(search_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        console.print(f"[green]Found and using life summary file: {file}[/green]")
                        return json.load(f)
                except Exception as e:
                    console.print(f"[bold red]Error loading {file}:[/bold red] {e}")
                    break # Stop after first attempt fails
    except Exception as e:
         console.print(f"[bold red]Error accessing directory {search_dir}:[/bold red] {e}")
    console.print("[yellow]No usable life_summary JSON file found. Using default state.[/yellow]")
    return None

# --- MODIFIED Helper to setup initial state for MULTIPLE Simulacra ---
def setup_initial_state_multi(num_simulacra: int = 2) -> Dict[str, Any]:
    """
    Sets up the initial state dictionary for multiple Simulacra.
    Uses sanitized character names for IDs (e.g., 'eleanor_vance').
    Loads from simulation_state.json if it exists, otherwise creates initial state
    from world_config.json and life_summary.json, then saves it.
    *** Overwrites world_time with current time upon loading. ***
    """
    console.print(f"Checking for persistent state file: {STATE_FILE_PATH}")
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, 'r') as f:
                initial_state = json.load(f)
            console.print(f"[green]Loaded existing state from {STATE_FILE_PATH}[/green]")

            # --- ADDED: Overwrite world_time after loading ---
            now_utc = datetime.now(timezone.utc)
            current_time_iso = now_utc.isoformat()
            if WORLD_STATE_KEY in initial_state and isinstance(initial_state[WORLD_STATE_KEY], dict):
                initial_state[WORLD_STATE_KEY]["world_time"] = current_time_iso
                logger.info(f"Overwrote world_time in loaded state with current time: {current_time_iso}")
            else:
                logger.warning(f"Could not find '{WORLD_STATE_KEY}' or it's not a dict in loaded state. Time not overwritten.")
            # --- END ADDED ---

            # Optional: Validate loaded state structure here if needed
            return initial_state
        except Exception as e:
            console.print(f"[red]Error loading state from {STATE_FILE_PATH}: {e}. Will attempt to create new state.[/red]")
            initial_state = {} # Reset state if loading failed
    else:
        console.print(f"[yellow]State file not found. Creating initial state...[/yellow]")
        initial_state = {}

    # --- Create Initial State Logic (if file didn't exist or load failed) ---
    active_ids = [] # Will store sanitized names like 'eleanor_vance'
    base_persona = None
    world_config = None

    # 1. Load World Config for initial location
    try:
        with open(WORLD_CONFIG_PATH, 'r') as f:
            world_config = json.load(f)
        console.print(f"Loaded world config from {WORLD_CONFIG_PATH}")
    except Exception as e:
        console.print(f"[bold red]Fatal Error: Could not load world config from {WORLD_CONFIG_PATH}: {e}. Cannot initialize.[/bold red]")
        world_config = {"location": {"city": "Town Square"}} # Minimal default
        console.print("[yellow]Using default location 'Town Square' due to config load error.[/yellow]")

    # --- MODIFIED: Determine initial location from flattened world_config location ---
    location_dict = world_config.get("location", {})
    if location_dict and isinstance(location_dict, dict):
        # Flatten the dictionary into a string, excluding coordinates if present
        location_parts = []
        for key, value in location_dict.items():
            if key != "coordinates" and value: # Exclude coordinates and empty values
                location_parts.append(f"{key.capitalize()}: {value}")
        default_initial_location = ", ".join(location_parts)
        if not default_initial_location: # Handle case where dict only had coordinates or empty values
             default_initial_location = "Town Square" # Fallback
    else:
        default_initial_location = "Town Square" # Fallback if 'location' key is missing or not a dict
    console.print(f"Using default initial location derived from config: '{default_initial_location}'")
    # --- END MODIFIED ---

    # 2. Load Base Persona from life_summary.json
    try:
        json_files = glob.glob("life_summary_*.json")
        if json_files:
            first_file = sorted(json_files)[0]
            console.print(f"Found and using life summary file: {first_file}")
            with open(first_file, 'r') as f:
                base_persona = json.load(f)
            console.print("Using loaded life summary as base.")
        else:
            console.print("[yellow]No life_summary JSON found. Base persona will be minimal.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error loading life summary JSON: {e}. Base persona will be minimal.[/red]")

    # Fallback if base_persona loading failed or no file found
    if base_persona is None:
        base_persona = {
            "persona_details": { # Ensure structure matches expected
                "Name": "Default Person", "Age": 30, "Occupation": "Wanderer",
                "Personality_Traits": ["Curious", "Cautious"],
                "Background": "Appeared with no memory."
                # Add Initial_Location/Goal here if using defaults
                # "Initial_Location": default_initial_location,
                # "Initial_Goal": "Find out where I am."
            }
        }
        console.print("[yellow]Using minimal default base persona.[/yellow]")

    # 3. Populate initial state dictionary
    console.print("Populating initial state...")
    created_ids = set() # To handle potential duplicate sanitized names
    for i in range(1, num_simulacra + 1):
        # Extract key persona details for state (simplified structure)
        base_details = base_persona.get("persona_details", {})
        original_name = base_details.get("Name", f"Person_{i}")

        # --- Derive agent_id from name ---
        agent_id = sanitize_name_for_id(original_name)
        # Handle potential duplicates if multiple simulacra are based on the same persona
        temp_agent_id = agent_id
        suffix_counter = 1
        while temp_agent_id in created_ids:
            suffix_counter += 1
            temp_agent_id = f"{agent_id}_{suffix_counter}"
        agent_id = temp_agent_id # Use the unique ID
        created_ids.add(agent_id)
        active_ids.append(agent_id) # Store the derived ID ('eleanor_vance', 'eleanor_vance_2', etc.)
        # ---

        persona_summary_dict = {
            # Use original name + suffix if needed for display/summary, but ID is sanitized
            "Name": original_name if suffix_counter == 1 else f"{original_name}_{suffix_counter}",
            "Occupation": base_details.get("Occupation", "Unknown"),
            "Personality_Traits": base_details.get("Personality_Traits", [])
        }

        # Get initial location and goal (prefer from persona, fallback to the derived default_initial_location)
        initial_location = base_details.get("Initial_Location", default_initial_location)
        initial_goal = base_details.get("Initial_Goal", "Figure things out.")

        # --- Define state keys using agent_id and updated format strings ---
        location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(agent_id) # e.g., simulacra_eleanor_vance_location
        goal_key = SIMULACRA_GOAL_KEY_FORMAT.format(agent_id)
        persona_key = SIMULACRA_PERSONA_KEY_FORMAT.format(agent_id)
        status_key = SIMULACRA_STATUS_KEY_FORMAT.format(agent_id)
        # ---

        # Set values in initial_state
        initial_state[location_key] = initial_location
        initial_state[goal_key] = initial_goal
        initial_state[persona_key] = persona_summary_dict # Store the *simplified* dict
        initial_state[status_key] = {"condition": "Normal", "mood": "Neutral"}

        console.print(f"  - Prepared state for ID '{agent_id}': Name={persona_summary_dict['Name']}, Loc={initial_location}, Goal={initial_goal}")

    # 4. Add global state elements
    initial_state[ACTIVE_SIMULACRA_IDS_KEY] = active_ids # Stores ['eleanor_vance', 'eleanor_vance_2']
    console.print(f"Active Simulacra IDs set in state: {active_ids}")

    # --- Add WORLD_STATE_KEY if not exists (when creating new) ---
    if WORLD_STATE_KEY not in initial_state:
         # --- MODIFIED: Use current time when creating new state too ---
         now_utc = datetime.now(timezone.utc)
         current_time_iso = now_utc.isoformat()
         logger.info(f"Setting initial world_time for new state: {current_time_iso}")
         # ---
         initial_state[WORLD_STATE_KEY] = {
             "world_time": current_time_iso, # Use current time here
             "location_details": { # Base locations
                 "Town Square": "A bustling square with a fountain.",
                 "Market Square": "Rows of stalls selling various goods.",
                 default_initial_location: world_config.get("description", "The primary setting derived from config.")
             }, "npcs": {}, "objects": {}
         }
         # Add initial locations of simulacra to location_details if missing
         for current_agent_id in active_ids:
             loc = initial_state.get(SIMULACRA_LOCATION_KEY_FORMAT.format(current_agent_id))
             if loc and loc not in initial_state[WORLD_STATE_KEY]["location_details"]:
                  initial_state[WORLD_STATE_KEY]["location_details"][loc] = "A starting location." # Generic description
    # ---

    # 5. Save the newly created state to file
    try:
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(initial_state, f, indent=2)
        console.print(f"[green]Initial state saved to {STATE_FILE_PATH}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving initial state to {STATE_FILE_PATH}: {e}[/red]")

    return initial_state
# --- END MODIFIED Function definition ---


async def main_entry():
    """Main asynchronous entry point for the simulation."""
    # --- API Key Check ---
    if not settings.GOOGLE_API_KEY:
        console.print("[bold red]ERROR: GOOGLE_API_KEY not found.[/bold red]")
        sys.exit(1)

    console.print(Panel("Welcome to the ADK World Simulation (Phased Turn)", title="Welcome", border_style="blue"))

    # --- Determine Start Mode ---
    session = None
    start_mode = "new_or_default"
    if len(sys.argv) > 1 and sys.argv[1].lower() == "continue":
        start_mode = "continue"

    # --- Initialize Services ---
    session_service = InMemorySessionService()
    console.print("[yellow]Using InMemorySessionService ('continue' is session-only).[/yellow]")
    # --- Load or Create Session ---
    sim_agents_config = {} # Will hold config for creating simulacra agents
    if start_mode == "continue":
        console.print(f"[cyan]Attempting to load existing session: {APP_NAME}/{USER_ID}/{SESSION_ID}...[/cyan]")
        try:
            # Use correct args for service (InMemory likely just needs ID)
            session = session_service.get_session(session_id=SESSION_ID)
            if session:
                console.print("[green]Existing session loaded successfully.[/green]")
                # Reconstruct agent config from loaded state
                loaded_active_ids = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
                for sim_id in loaded_active_ids:
                     sim_agents_config[sim_id] = {"name": session.state.get(SIMULACRA_STATUS_KEY_FORMAT.format(sim_id), {}).get("Name", sim_id)}
                console.print(f"Reconstructed agent config for IDs: {list(sim_agents_config.keys())}")
            else:
                console.print(f"[yellow]Warning: Session '{SESSION_ID}' not found. Starting new.[/yellow]")
                start_mode = "new_or_default"
        except Exception as e:
            console.print(f"[red]Error loading session:[/red] {e}. Starting new.")
            start_mode = "new_or_default"

    if start_mode == "new_or_default":
        # Use helper to setup state for multiple simulacra
        initial_state = setup_initial_state_multi(num_simulacra=NUM_SIMULACRA) # Pass only num_simulacra, remove await
        try:
            # Ensure world_config path is in initial state if needed by WSAgent
            initial_state["world_config_path"] = "world_config.json" # Or get from settings
            session = session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID, state=initial_state
            )
            console.print(f"[cyan]New Session created:[/cyan] ID={session.id}")
        except Exception as e:
            console.print(f"[bold red]Error creating new session:[/bold red] {e}")
            session = None

    if not session:
         console.print("[bold red]Failed to load or create a session. Exiting.[/bold red]")
         sys.exit(1)

    # --- Instantiate Agents ---
    console.rule("[cyan]Instantiating Agents[/cyan]")
    # Ensure core agents are loaded correctly (assuming they are module-level instances)
    core_agents = {
        "World State": world_state_agent,
        "World Engine (Validator)": world_engine_agent,
        "NPC Interaction": npc_agent,
        "World Execution": world_execution_agent, # Add World Execution Agent here
        "Narration": narration_agent
    }
    for name, agent_instance in core_agents.items():
         if not agent_instance:
             console.print(f"[bold red]Error: Core agent '{name}' not loaded. Exiting.[/bold red]")
             sys.exit(1)
    console.print("  - Core agents loaded.")

    # --- MODIFIED: Instantiate Simulacra Agents using derived IDs as names ---
    simulacra_agents_dict: Dict[str, BaseAgent] = {}
    active_sim_ids_from_state = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, []) # Get actual IDs used ('eleanor_vance', etc.)

    if not active_sim_ids_from_state:
         console.print("[bold red]Error: No active simulacra IDs found in initial state. Cannot instantiate agents.[/bold red]")
         sys.exit(1)

    for agent_id in active_sim_ids_from_state: # agent_id is now 'eleanor_vance', 'eleanor_vance_2', etc.
        # Use the factory function
        sim_agent = create_simulacra_agent() # Assumes factory doesn't need args specific to ID here
        if sim_agent:
            # --- Set agent name to the derived ID ---
            sim_agent.name = agent_id # <<< Agent name is now 'eleanor_vance', etc.
            # ---

            simulacra_agents_dict[agent_id] = sim_agent # Store using agent_id as key
            console.print(f"  - Instantiated Simulacra: {sim_agent.name} (ID: {agent_id})")
        else:
            console.print(f"[bold red]Error creating Simulacra agent for ID: {agent_id}[/bold red]")

    if len(simulacra_agents_dict) != len(active_sim_ids_from_state):
        console.print("[bold red]Mismatch in created Simulacra agents vs active IDs. Exiting.[/bold red]")
        sys.exit(1)
    # --- END MODIFIED ---
    console.rule() # Use rule() instead of print("─" * 160)

    # --- Create Runner ---
    runner = Runner(
        agent=world_state_agent, # Default agent, loop overrides it
        app_name=APP_NAME,
        session_service=session_service
        # Add artifact_service / memory_service if used
    )
    console.print(f"[cyan]Runner initialized for app '{APP_NAME}'.[/cyan]")

    # --- Run the Phased Simulation Loop ---
    console.rule("[bold magenta]Starting Phased Simulation[/bold magenta]")
    try:
        # Call the phased loop function, passing all agent instances
        await run_phased_simulation(
            runner=runner,
            session_service=session_service,
            session=session, # Pass the loaded or newly created session
            # Pass agent instances
            world_state_agent=world_state_agent,
            simulacra_agents=simulacra_agents_dict, # Pass the dict of instances
            world_engine_agent=world_engine_agent,
            npc_agent=npc_agent,
            world_execution_agent=world_execution_agent, # Pass the World Execution Agent
            narration_agent=narration_agent,
            # Pass config
            max_turns=MAX_TURNS
        )
    except Exception as e:
        console.print(f"[bold red]An error occurred during the simulation execution:[/bold red]")
        console.print_exception(show_locals=True)
    finally:
        console.rule("[bold magenta]Simulation Ended[/bold magenta]")
        # try:
        #     session_service.delete_session(session_id=SESSION_ID) # Correct args for service
        #     console.print(f"Session '{SESSION_ID}' deleted.")
        # except Exception as del_e:
        #      console.print(f"[yellow]Could not delete session '{SESSION_ID}': {del_e}[/yellow]")


if __name__ == "__main__":
    try:
        # Ensure the event loop policy is set correctly for Windows if needed
        if sys.platform == "win32":
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main_entry())
    except KeyboardInterrupt:
        console.print("\n[bold orange_red1]Simulation interrupted by user.[/bold orange_red1]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred in main execution:[/bold red]")
        console.print_exception(show_locals=True)
