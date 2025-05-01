# main3.py (Refactored for Instance Loading and New Loop Structure)

import argparse
import asyncio
import glob
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ADK Imports
from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import (BaseSessionService, InMemorySessionService,
                                 Session)
# Rich Imports
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# --- Agent Imports ---
from src.agents.narration import narration_agent
from src.agents.simulacra_v3 import \
    create_agent as create_simulacra_agent
from src.agents.world_engine import create_world_engine_validator
from src.agents.world_execution_agent import world_execution_agent
from src.agents.world_state_agent import world_state_agent
# --- Core Imports ---
from src.config import settings
from src.initialization import (
    ensure_state_structure, find_latest_simulation_state_file,
    generate_unique_id, load_json_file, save_json_file)
# --- Simulation Loop ---
from src.simulation_loop_v3 import run_phased_simulation
# --- State Keys ---
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
WORLD_STATE_KEY = "current_world_state"
SIMULACRA_PROFILES_KEY = "simulacra_profiles"
CURRENT_LOCATION_KEY = "current_location"
HOME_LOCATION_KEY = "home_location"
# ---

# Setup console
console = Console()

# --- Logging Setup ---
log_filename = "simulation_main3.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)
logger = logging.getLogger(__name__)
logger.info(f"--- Application Start (main3.py) --- Logging configured to file: {log_filename}")

# --- Constants ---
APP_NAME = "TheSimulationV3"
USER_ID = "player1"
MAX_TURNS = settings.MAX_SIMULATION_TURNS if hasattr(settings, 'MAX_SIMULATION_TURNS') else 10

# Define paths
WORLD_CONFIG_DIR = "data"
STATE_DIR = "data/states"
LIFE_SUMMARY_DIR = "data/life_summaries"

# Ensure directories exist
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)
os.makedirs(WORLD_CONFIG_DIR, exist_ok=True)


# --- Main Execution Logic ---
async def main(instance_uuid_arg: Optional[str]):
    """Loads and runs a specific simulation instance."""
    console.print(Panel(f"[[bold yellow]{APP_NAME}[/]] - Initializing...", title="Startup", border_style="blue"))
    logger.info("Starting main execution (main3.py).")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- API Key Check ---
    if not settings.GOOGLE_API_KEY:
        console.print("[bold red]ERROR: GOOGLE_API_KEY not found in environment variables.[/bold red]")
        logger.critical("GOOGLE_API_KEY not found.")
        sys.exit(1)

    # --- Determine Instance UUID and State File Path ---
    world_instance_uuid: Optional[str] = None
    state_file_path: Optional[str] = None

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
        latest_state_file = find_latest_simulation_state_file()
        if latest_state_file:
            state_file_path = latest_state_file
            match = re.search(r"simulation_state_([a-f0-9\-]+)\.json", os.path.basename(latest_state_file))
            if match:
                world_instance_uuid = match.group(1)
                logger.info(f"Found latest state file: {latest_state_file} (UUID from filename: {world_instance_uuid})")
                console.print(f"Loading latest instance state file: {state_file_path}")
            else:
                logger.error(f"Could not extract UUID from latest state file name: {latest_state_file}")
                console.print(f"[bold red]Error:[/bold red] Could not determine UUID from latest state file '{os.path.basename(latest_state_file)}'.")
                sys.exit(1)
        else:
            logger.error("No instance UUID specified and no existing state files found.")
            console.print("[bold red]Error:[/bold red] No instance UUID specified and no simulation state files found in 'data/states/'.")
            console.print("Please run 'setup_simulation.py' first or specify an instance UUID.")
            sys.exit(1)

    # --- Load Simulation State FIRST ---
    simulation_state: Optional[Dict[str, Any]] = None
    try:
        logger.info(f"Attempting to load state file: {state_file_path}")
        simulation_state = load_json_file(state_file_path)
        if simulation_state is None:
             raise FileNotFoundError(f"State file found but failed to load content: {state_file_path}")

        uuid_from_state = simulation_state.get("world_instance_uuid")
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

    except (FileNotFoundError, ValueError, IOError, Exception) as e:
        logger.critical(f"Failed to load simulation instance state from {state_file_path}: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to load simulation state file '{state_file_path}'. Check logs. Error: {e}")
        sys.exit(1)

    # --- Load World Configuration (Optional Fallback) ---
    world_config_path = os.path.join(WORLD_CONFIG_DIR, f"world_config_{world_instance_uuid}.json")
    world_config_data: Optional[Dict[str, Any]] = None
    logger.info(f"Attempting to load world config file: {world_config_path}")
    try:
        world_config_data = load_json_file(world_config_path)
        if world_config_data is None:
            logger.warning(f"World config file '{world_config_path}' not found for instance {world_instance_uuid}. Proceeding using world details from state file.")
            console.print(f"[yellow]Warning:[/yellow] World config file not found. Relying on state file's world details.")
        else:
            if world_config_data.get("world_instance_uuid") != world_instance_uuid:
                 logger.warning(f"UUID mismatch between world config file ({world_config_path}) and instance UUID ({world_instance_uuid}). Prioritizing state file UUID.")
            logger.info(f"Successfully loaded world config for instance {world_instance_uuid}")
            console.print(f"World Config loaded: {world_config_data.get('description', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to load or parse world config file '{world_config_path}': {e}", exc_info=True)
        console.print(f"[yellow]Warning:[/yellow] Failed to load world config file. Error: {e}")

    # --- Ensure State Structure ---
    logger.info("Ensuring essential state structure...")
    state_modified_by_ensure = ensure_state_structure(simulation_state)
    if state_modified_by_ensure:
        logger.info("State structure updated (missing keys added). Saving state file.")
        try:
            save_json_file(state_file_path, simulation_state)
            logger.info(f"State saved to {state_file_path} after ensuring structure.")
        except Exception as save_e:
             logger.error(f"Failed to save state update after ensuring structure: {save_e}")
             console.print(f"[bold red]Error:[/bold red] Failed to save state update to {state_file_path} after ensuring structure. Check logs.")
    else:
        logger.info("State structure is already valid.")

    # --- Initial Location Verification ---
    logger.info("Verifying initial simulacra locations...")
    state_modified_for_location = False
    simulacra_profiles = simulation_state.get(SIMULACRA_PROFILES_KEY, {})
    simulacra_ids_from_profiles = list(simulacra_profiles.keys())

    if not simulacra_ids_from_profiles:
        logger.warning("No simulacra profiles found in state to verify locations for.")
    else:
        logger.info(f"Checking locations for simulacra: {simulacra_ids_from_profiles}")
        for sim_id in simulacra_ids_from_profiles:
            profile = simulacra_profiles.get(sim_id)
            if not profile:
                logger.warning(f"Profile not found for simulacrum ID: {sim_id}. Skipping location check.")
                continue

            current_location = profile.get(CURRENT_LOCATION_KEY)
            home_location = profile.get(HOME_LOCATION_KEY)

            if not current_location or not str(current_location).strip():
                if home_location and str(home_location).strip():
                    logger.info(f"Simulacrum '{sim_id}' missing or has empty current_location. Setting to home_location: '{home_location}'.")
                    simulation_state[SIMULACRA_PROFILES_KEY][sim_id][CURRENT_LOCATION_KEY] = home_location
                    state_modified_for_location = True
                else:
                    logger.error(f"Simulacrum '{sim_id}' missing current_location AND valid home_location. Cannot set default location!")
            else:
                 logger.debug(f"Simulacrum '{sim_id}' already has current_location: '{current_location}'.")

    if state_modified_for_location:
        logger.info("Initial locations updated. Saving state file.")
        try:
            save_json_file(state_file_path, simulation_state)
            logger.info(f"State saved to {state_file_path} after location initialization.")
        except Exception as save_e:
             logger.error(f"Failed to save state update after location init: {save_e}")
             console.print(f"[bold red]Error:[/bold red] Failed to save state update to {state_file_path}. Check logs.")
    elif not state_modified_by_ensure: # Only log this if no other save happened
        logger.info("No state changes required for initial locations.")


    # --- Verify Simulacra and Load Personas ---
    console.rule("[cyan]Verifying Simulacra[/cyan]")
    state_sim_ids = simulation_state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
    active_sim_ids: List[str] = []

    # Find ALL summaries for the instance first
    life_summary_pattern_instance = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_*_{world_instance_uuid}.json")
    available_summary_files = glob.glob(life_summary_pattern_instance)
    logger.info(f"Checking for life summaries matching pattern: {life_summary_pattern_instance}")
    logger.info(f"Found {len(available_summary_files)} potential summary files for instance {world_instance_uuid}.")

    # Determine which sim_ids are actually available based on file content
    available_sim_ids_from_files = set()
    valid_summary_files_map = {} # Store path by sim_id for later use
    for filepath in available_summary_files:
        summary = load_json_file(filepath)
        # Check if the file content matches the current world instance UUID
        if summary and summary.get("world_instance_uuid") == world_instance_uuid:
            sim_id_from_file = summary.get("simulacra_id")
            if sim_id_from_file:
                available_sim_ids_from_files.add(sim_id_from_file)
                valid_summary_files_map[sim_id_from_file] = filepath # Map sim_id to its file path
            else:
                logger.warning(f"Life summary file {filepath} is missing 'simulacra_id'.")
        elif summary:
            logger.warning(f"Life summary file {filepath} has mismatched world_instance_uuid. Skipping.")

    logger.info(f"Found {len(available_sim_ids_from_files)} valid life summaries by content for instance {world_instance_uuid}: {available_sim_ids_from_files}")

    # Determine active sims based on state AND available files
    if state_sim_ids:
        active_sim_ids = [sid for sid in state_sim_ids if sid in available_sim_ids_from_files]
        missing_ids = set(state_sim_ids) - set(active_sim_ids)
        if missing_ids:
            logger.warning(f"Simulacra from state ({missing_ids}) missing valid summary files. Ignoring.")
            console.print(f"[yellow]Warning:[/yellow] Some Simulacra from state ({', '.join(missing_ids)}) missing valid summary files. Ignoring.")
    else:
        logger.warning(f"No Simulacra IDs found in the loaded state file ({state_file_path}).")
        console.print(f"[yellow]Warning:[/yellow] No active Simulacra found in state.")


    # --- Load Personas for Active Sims ---
    personas_loaded_count = 0
    sim_profiles = simulation_state.get(SIMULACRA_PROFILES_KEY, {})
    for sim_id in active_sim_ids: # Iterate through the verified active IDs
        profile = sim_profiles.get(sim_id, {})
        persona_key = "persona_details"
        persona = profile.get(persona_key)

        if not persona:
            logger.warning(f"Persona for {sim_id} not found in state profile. Attempting fallback load.")
            # Use the pre-found file path
            fallback_file_path = valid_summary_files_map.get(sim_id) # Get the path found earlier

            if fallback_file_path:
                logger.info(f"Attempting fallback load for {sim_id} from: {fallback_file_path}")
                try:
                    life_data = load_json_file(fallback_file_path)
                    if life_data and persona_key in life_data:
                        persona = life_data[persona_key]
                        logger.info(f"Successfully loaded persona for {sim_id} from fallback file: {fallback_file_path}")
                        # Update the profile in the main state dictionary
                        simulation_state.setdefault(SIMULACRA_PROFILES_KEY, {}).setdefault(sim_id, {})[persona_key] = persona
                    else:
                        logger.error(f"Fallback summary file {fallback_file_path} for {sim_id} missing '{persona_key}' key.")
                except Exception as load_e:
                    logger.error(f"Error loading fallback summary file {fallback_file_path} for {sim_id}: {load_e}", exc_info=True)
            else:
                # This error means the sim_id was active but somehow didn't have a file mapped earlier
                logger.error(f"Internal Error: No valid summary file path found for active sim_id {sim_id} during fallback.")

        # Check if persona exists now (either from initial state or fallback)
        if persona:
             if simulation_state.get(SIMULACRA_PROFILES_KEY, {}).get(sim_id, {}).get(persona_key):
                 personas_loaded_count += 1
        else:
            # This error log now correctly reflects that both initial load and fallback failed
            logger.error(f"Could not load persona for active sim {sim_id} from state or fallback file.")
            console.print(f"[red]Error:[/red] Failed to load persona for {sim_id}.")

    # --- Final Check if Simulation Can Proceed ---
    if not active_sim_ids:
        logger.critical("No active simulacra available or verified for this instance. Cannot proceed.")
        console.print("[bold red]Error:[/bold red] No verified Simulacra available for the simulation instance.")
        sys.exit(1)

    if personas_loaded_count != len(active_sim_ids):
        logger.warning(f"Loaded personas ({personas_loaded_count}) does not match active simulacra ({len(active_sim_ids)}). Some agents might use default personas.")
        # Decide if this is critical
        # console.print("[bold red]Error:[/bold red] Failed to load personas for all active simulacra. Exiting.")
        # sys.exit(1)

    # Update state with the final list of verified active IDs (if changed)
    if simulation_state.get(ACTIVE_SIMULACRA_IDS_KEY) != active_sim_ids:
        logger.info(f"Updating state's active simulacra list to verified IDs: {active_sim_ids}")
        simulation_state[ACTIVE_SIMULACRA_IDS_KEY] = active_sim_ids
        try:
            save_json_file(state_file_path, simulation_state)
        except Exception as save_e: logger.error(f"Failed to save updated active ID list to state: {save_e}")

    logger.info(f"Simulation instance {world_instance_uuid} will run with {len(active_sim_ids)} verified simulacra: {active_sim_ids}")
    console.print(f"Running simulation with: {', '.join(active_sim_ids)}")
    console.rule()


    # --- Initialize ADK Session ---
    session_service = InMemorySessionService()
    adk_session_id = f"adk_session_{world_instance_uuid}_{timestamp}"
    simulation_state["session_id"] = adk_session_id

    try:
        session = session_service.create_session(
            user_id=USER_ID, app_name=APP_NAME,
            session_id=adk_session_id, state=simulation_state
        )
        logger.info(f"ADK Session created: {adk_session_id} with initial state.")
    except Exception as e:
        logger.critical(f"Failed to create ADK session with initial state: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to create ADK session. Check logs.")
        sys.exit(1)

    # --- Instantiate Agents ---
    console.rule("[cyan]Instantiating Agents[/cyan]")
    try:
        core_agents_map = {
            "World State": world_state_agent,
            "World Execution": world_execution_agent,
            "Narration": narration_agent
        }
        logger.info(f"Using core agents: {[name for name in core_agents_map.keys()]}")
        console.print(f"  - {len(core_agents_map)} Core agents ready.")
    except Exception as e:
        logger.critical(f"Failed to prepare core agents: {e}", exc_info=True)
        console.print(f"[bold red]Error preparing core agents:[/bold red] {e}")
        sys.exit(1)

    simulacra_agents_dict: Dict[str, BaseAgent] = {}
    validator_agents_dict: Dict[str, BaseAgent] = {}

    console.print("Instantiating Validator Agents...")
    for sim_id in active_sim_ids:
        try:
            validator_instance = create_world_engine_validator(sim_id=sim_id)
            if validator_instance and isinstance(validator_instance, BaseAgent):
                validator_agents_dict[sim_id] = validator_instance
                logger.info(f"Created validator agent for {sim_id} ({validator_instance.name})")
                console.print(f"  - Validator: {validator_instance.name}")
            else: logger.error(f"Factory failed for validator {sim_id}.")
        except Exception as e: logger.error(f"Error creating validator {sim_id}: {e}", exc_info=True)

    if len(validator_agents_dict) != len(active_sim_ids):
         logger.warning("Mismatch creating validator agents.")
         if not validator_agents_dict:
              console.print("[bold red]No validator agents created. Exiting.[/bold red]")
              sys.exit(1)

    console.print("Instantiating Simulacra Agents (V3)...")
    for sim_id in active_sim_ids:
        profile = simulation_state.get(SIMULACRA_PROFILES_KEY, {}).get(sim_id, {})
        persona_key = "persona_details"
        persona = profile.get(persona_key)
        if not persona:
             logger.error(f"Persona missing for {sim_id} during agent creation. Using default.")
             console.print(f"[red]Warning: Persona missing for {sim_id}. Using default.[/red]")
             persona = {"Name": sim_id, "Background": "Default"}

        try:
            agent_instance = create_simulacra_agent(
                sim_id=sim_id, persona=persona, session=session
            )
            if agent_instance and isinstance(agent_instance, BaseAgent):
                simulacra_agents_dict[sim_id] = agent_instance
                logger.info(f"Created V3 simulacra agent for {sim_id} ({agent_instance.name})")
                console.print(f"  - Simulacra: {agent_instance.name}")
            else: logger.error(f"Factory failed for V3 simulacra {sim_id}.")
        except Exception as e: logger.error(f"Error creating V3 simulacra {sim_id}: {e}", exc_info=True)

    if len(simulacra_agents_dict) != len(active_sim_ids):
         logger.warning("Mismatch creating V3 simulacra agents.")
         if not simulacra_agents_dict:
              console.print("[bold red]No V3 simulacra agents created. Exiting.[/bold red]")
              sys.exit(1)
    console.rule()

    # --- Create Runner ---
    try:
        runner = Runner(
            agent=world_state_agent,
            app_name=APP_NAME, session_service=session_service
        )
        logger.info(f"Runner initialized for app '{APP_NAME}'.")
    except Exception as e:
        logger.critical(f"Failed to initialize ADK Runner: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to initialize ADK Runner.")
        sys.exit(1)

    # --- Run Phased Simulation (Using New Loop) ---
    console.rule("[bold magenta]Starting Phased Simulation (V3 Loop)[/bold magenta]")
    final_state = None
    try:
        final_state = await run_phased_simulation(
            runner=runner,
            session_service=session_service,
            session=session,
            world_state_agent=world_state_agent,
            simulacra_agents=simulacra_agents_dict,
            validator_agents=validator_agents_dict,
            world_execution_agent=world_execution_agent,
            narration_agent=narration_agent,
            max_turns=MAX_TURNS
        )
        logger.info("Phased simulation loop (V3) completed.")
        console.rule("[bold green]Simulation Complete[/]")

    except Exception as e:
        logger.critical(f"Error during simulation loop (V3): {e}", exc_info=True)
        console.print(f"\n[bold red]Error during simulation:[/bold red] {e}")
        try:
            error_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=adk_session_id)
            if error_session and hasattr(error_session, 'state'): final_state = error_session.state
            elif hasattr(session, 'state'): final_state = session.state
            else: final_state = simulation_state

            error_state_filename = f"simulation_state_error_{world_instance_uuid}_{timestamp}.json"
            error_state_path = os.path.join(STATE_DIR, error_state_filename)
            save_json_file(error_state_path, final_state)
            logger.info(f"Saved error state to {error_state_path}")
            console.print(f"[yellow]Saved state at time of error to {error_state_path}[/yellow]")
        except Exception as get_state_e:
            logger.error(f"Could not retrieve or save session state after error: {get_state_e}")
            final_state = simulation_state

    # --- Save Final State ---
    if final_state:
        logger.info("Saving final simulation state.")
        try:
            final_session_to_save = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=adk_session_id)
            if final_session_to_save and hasattr(final_session_to_save, 'state'):
                state_to_save = final_session_to_save.state
            else: state_to_save = final_state

            save_json_file(state_file_path, state_to_save)
            logger.info(f"Final simulation state saved to {state_file_path}")
            console.print(f"Final state saved to {state_file_path}")
        except Exception as save_e:
             logger.error(f"Failed to save final state to {state_file_path}: {save_e}", exc_info=True)
             console.print(f"[red]Error saving final state to {state_file_path}: {save_e}[/red]")
    else:
        logger.warning("No final state available to save (likely due to early error).")


# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run {APP_NAME} simulation instance.")
    parser.add_argument(
        "--instance-uuid", type=str,
        help="Specify the UUID of the simulation instance to load. If omitted, the latest instance is loaded.",
        default=None
    )
    args = parser.parse_args()

    try:
        if sys.platform == "win32":
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main(instance_uuid_arg=args.instance_uuid))
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
        console.print("\n[bold orange_red1]Simulation interrupted by user.[/bold orange_red1]")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main execution block: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred:[/bold red]")
        console.print_exception(show_locals=False)
    finally:
        logger.info("--- Application End (main3.py) ---")
        logging.shutdown()
        console.print("Application finished.")
