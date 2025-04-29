# main3.py (Refactored for Instance Loading and New Loop Structure)

import argparse  # <<< Added for command-line arguments
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
# Import agent INSTANCES or factory functions
from src.agents.narration import narration_agent
# from src.agents.npc import npc_agent # Will be replaced by interaction resolver
from src.agents.simulacra_v3 import \
    create_agent as create_simulacra_agent  # Use V3 agent
from src.agents.world_engine import create_world_engine_validator  # Factory
from src.agents.world_execution_agent import world_execution_agent
from src.agents.world_state_agent import world_state_agent
# --- Core Imports ---
from src.config import settings
# Removed life_generator import as generation is moved out
from src.initialization import (  # Added find_latest
    find_latest_simulation_state_file, generate_unique_id, load_json_file,
    save_json_file) # Removed load_or_create_simulation_instance
# --- Simulation Loop ---
# <<< Point to the new simulation loop file >>>
from src.simulation_loop_v3 import run_phased_simulation
# --- State Keys (Import centrally if defined elsewhere) ---
from src.tools.world_state_tools import (ACTIVE_SIMULACRA_IDS_KEY,
                                         WORLD_STATE_KEY)

# Setup console
console = Console()

# --- Logging Setup ---
log_filename = "simulation_main3.log" # Use a different log file
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
# NUM_SIMULACRA is now determined by the loaded instance state

# Define paths
WORLD_CONFIG_DIR = "data" # Directory where world configs are saved by setup_simulation.py
STATE_DIR = "data/states"
LIFE_SUMMARY_DIR = "data/life_summaries"

# Ensure directories exist (optional, as loading assumes they exist)
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
            # Assume UUID from arg is correct for now, will verify against file content
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
            # Extract UUID from filename to use as the initial guess
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
             # This case should ideally not happen if the file exists, but handle defensively
             raise FileNotFoundError(f"State file found but failed to load content: {state_file_path}")

        # --- Verify UUID consistency ---
        uuid_from_state = simulation_state.get("world_instance_uuid")
        if not uuid_from_state:
            logger.critical(f"State file {state_file_path} is missing 'world_instance_uuid'. Cannot proceed.")
            console.print(f"[bold red]Error:[/bold red] State file is missing the 'world_instance_uuid' key.")
            sys.exit(1)
        if uuid_from_state != world_instance_uuid:
            logger.critical(f"UUID mismatch! Filename/Arg suggested '{world_instance_uuid}', but state file contains '{uuid_from_state}'.")
            console.print(f"[bold red]Error:[/bold red] UUID mismatch between state file content ('{uuid_from_state}') and expected UUID ('{world_instance_uuid}').")
            sys.exit(1)
        # UUID is confirmed
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
            # Log warning but continue, relying on state file's world details
            logger.warning(f"World config file '{world_config_path}' not found for instance {world_instance_uuid}. Proceeding using world details from state file.")
            console.print(f"[yellow]Warning:[/yellow] World config file not found. Relying on state file's world details.")
        else:
            # Optional: Keep the UUID mismatch check if relevant
            if world_config_data.get("world_instance_uuid") != world_instance_uuid:
                 logger.warning(f"UUID mismatch between world config file ({world_config_path}) and instance UUID ({world_instance_uuid}). Prioritizing state file UUID.")
            logger.info(f"Successfully loaded world config for instance {world_instance_uuid}")
            console.print(f"World Config loaded: {world_config_data.get('description', 'N/A')}")

    except Exception as e: # Catch other potential errors during loading/parsing
        logger.error(f"Failed to load or parse world config file '{world_config_path}': {e}", exc_info=True)
        console.print(f"[yellow]Warning:[/yellow] Failed to load world config file. Error: {e}")
        # Continue without world_config_data

    # --- Verify Simulacra and Load Personas ---
    console.rule("[cyan]Verifying Simulacra[/cyan]")
    state_sim_ids = simulation_state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
    active_sim_ids: List[str] = []

    life_summary_pattern_instance = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_*_{world_instance_uuid}.json")
    available_summary_files = glob.glob(life_summary_pattern_instance)
    available_sim_ids_from_files = set()

    logger.info(f"Checking for life summaries matching pattern: {life_summary_pattern_instance}")
    for filepath in available_summary_files:
        summary = load_json_file(filepath)
        # Double-check UUID match within the file content
        if summary and summary.get("world_instance_uuid") == world_instance_uuid:
            sim_id = summary.get("simulacra_id")
            if sim_id:
                available_sim_ids_from_files.add(sim_id)
            else:
                 logger.warning(f"Life summary file {filepath} is missing 'simulacra_id'.")
        elif summary:
             logger.warning(f"Life summary file {filepath} has mismatched world_instance_uuid (Expected: {world_instance_uuid}, Found: {summary.get('world_instance_uuid')}). Skipping.")

    logger.info(f"Found {len(available_sim_ids_from_files)} valid life summaries for instance {world_instance_uuid}: {available_sim_ids_from_files}")

    # Filter IDs from state against available summaries
    if state_sim_ids:
        active_sim_ids = [sid for sid in state_sim_ids if sid in available_sim_ids_from_files]
        missing_ids = set(state_sim_ids) - set(active_sim_ids)
        if missing_ids:
            logger.warning(f"Simulacra from state ({missing_ids}) are missing valid summary files for instance {world_instance_uuid}. They will not be activated.")
            console.print(f"[yellow]Warning:[/yellow] Some Simulacra from state ({', '.join(missing_ids)}) missing summary files. Ignoring.")
    else:
        logger.warning(f"No Simulacra IDs found in the loaded state file ({state_file_path}).")
        console.print(f"[yellow]Warning:[/yellow] No active Simulacra found in state. Ensure setup was completed.")

    # --- Load Personas for Active Sims ---
    personas_loaded_count = 0
    for sim_id in active_sim_ids:
        persona_key = f"simulacra_{sim_id}_persona"
        persona = simulation_state.get(persona_key)

        if not persona:
            logger.warning(f"Persona for {sim_id} not found in state dict. Attempting fallback load from summary.")
            summary_filename_glob = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_{sim_id}_*_{world_instance_uuid}.json")
            found_files = glob.glob(summary_filename_glob)
            if found_files:
                try:
                    latest_summary_file = max(found_files, key=os.path.getctime)
                    life_data = load_json_file(latest_summary_file)
                    if life_data and "persona" in life_data:
                        persona = life_data["persona"]
                        logger.info(f"Successfully loaded persona for {sim_id} from fallback file: {latest_summary_file}")
                        simulation_state[persona_key] = persona
                        personas_loaded_count += 1
                    else:
                        logger.error(f"Fallback summary file {latest_summary_file} for {sim_id} missing 'persona' key.")
                except Exception as load_e:
                    logger.error(f"Error loading fallback summary file for {sim_id}: {load_e}", exc_info=True)
            else:
                logger.error(f"No fallback summary file found matching pattern for {sim_id}.")

        if persona:
             if persona_key in simulation_state:
                 personas_loaded_count += 1
        else:
            logger.error(f"Could not load persona for active sim {sim_id}. Agent behavior might be impaired.")
            console.print(f"[red]Error:[/red] Failed to load persona for {sim_id}.")

    # --- Final Check if Simulation Can Proceed ---
    if not active_sim_ids:
        logger.critical("No active simulacra available or verified for this instance. Cannot proceed.")
        console.print("[bold red]Error:[/bold red] No verified Simulacra available for the simulation instance.")
        console.print("Please ensure 'setup_simulation.py' was run successfully for this instance UUID.")
        sys.exit(1)

    # Update state with the final list of verified active IDs (if changed)
    if simulation_state.get(ACTIVE_SIMULACRA_IDS_KEY) != active_sim_ids:
        logger.info(f"Updating state's active simulacra list to verified IDs: {active_sim_ids}")
        simulation_state[ACTIVE_SIMULACRA_IDS_KEY] = active_sim_ids
        try:
            save_json_file(state_file_path, simulation_state)
        except Exception as save_e:
             logger.error(f"Failed to save updated active ID list to state: {save_e}")

    logger.info(f"Simulation instance {world_instance_uuid} will run with {len(active_sim_ids)} verified simulacra: {active_sim_ids}")
    console.print(f"Running simulation with: {', '.join(active_sim_ids)}")
    console.rule()

    # --- Initialize ADK Session ---
    session_service = InMemorySessionService()
    # Use a consistent ADK session ID derived from the world instance for potential resume?
    # Or keep unique per run? Let's keep unique per run for now.
    adk_session_id = f"adk_session_{world_instance_uuid}_{timestamp}"
    simulation_state["session_id"] = adk_session_id # Store the ADK session ID in state

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
    # Core agents that don't depend on Simulacra ID
    try:
        # Note: npc_agent is removed as interaction is handled by the resolver in the loop
        core_agents_map = {
            "World State": world_state_agent,
            "World Execution": world_execution_agent,
            "Narration": narration_agent
        }
        # Instantiate if they are factories, or just use if they are instances
        # Assuming they are instances for now based on imports
        logger.info(f"Using core agents: {[name for name in core_agents_map.keys()]}")
        console.print(f"  - {len(core_agents_map)} Core agents ready.")
    except Exception as e:
        logger.critical(f"Failed to prepare core agents: {e}", exc_info=True)
        console.print(f"[bold red]Error preparing core agents:[/bold red] {e}")
        sys.exit(1)

    # Simulacra-specific agents (Validators and Simulacra V3)
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
        persona = simulation_state.get(f"simulacra_{sim_id}_persona")
        if not persona:
             logger.error(f"Persona missing for {sim_id} during agent creation. Using default.")
             console.print(f"[red]Warning: Persona missing for {sim_id}. Using default.[/red]")
             persona = {"Name": sim_id, "Background": "Default"}

        try:
            # Call the V3 factory
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
            agent=world_state_agent, # Default agent, will be overridden in loop
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
        # <<< Call the new simulation loop function >>>
        # Note: npc_agent is removed from the call
        final_state = await run_phased_simulation(
            runner=runner,
            session_service=session_service,
            session=session,
            world_state_agent=world_state_agent,
            simulacra_agents=simulacra_agents_dict,
            validator_agents=validator_agents_dict,
            # npc_agent=npc_agent, # Removed
            world_execution_agent=world_execution_agent,
            narration_agent=narration_agent,
            max_turns=MAX_TURNS
        )
        logger.info("Phased simulation loop (V3) completed.")
        console.rule("[bold green]Simulation Complete[/]")

    except Exception as e:
        logger.critical(f"Error during simulation loop (V3): {e}", exc_info=True)
        console.print(f"\n[bold red]Error during simulation:[/bold red] {e}")
        # Attempt to get and save error state
        try:
            # Fetch the latest state from the service if possible
            error_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=adk_session_id)
            if error_session and hasattr(error_session, 'state'):
                 final_state = error_session.state
            elif hasattr(session, 'state'):
                 final_state = session.state # Use state from last known good session object
            else:
                 final_state = simulation_state # Fallback to initial loaded state

            error_state_filename = f"simulation_state_error_{world_instance_uuid}_{timestamp}.json"
            error_state_path = os.path.join(STATE_DIR, error_state_filename)
            save_json_file(error_state_path, final_state)
            logger.info(f"Saved error state to {error_state_path}")
            console.print(f"[yellow]Saved state at time of error to {error_state_path}[/yellow]")
        except Exception as get_state_e:
            logger.error(f"Could not retrieve or save session state after error: {get_state_e}")
            # Use the initially loaded state as the last resort for saving
            final_state = simulation_state

    # --- Save Final State ---
    if final_state:
        logger.info("Saving final simulation state.")
        try:
            # Ensure the state being saved is the most recent one
            final_session_to_save = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=adk_session_id)
            if final_session_to_save and hasattr(final_session_to_save, 'state'):
                state_to_save = final_session_to_save.state
            else:
                state_to_save = final_state # Fallback if session fetch fails

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
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description=f"Run {APP_NAME} simulation instance.")
    parser.add_argument(
        "--instance-uuid",
        type=str,
        help="Specify the UUID of the simulation instance to load. If omitted, the latest instance is loaded.",
        default=None
    )
    args = parser.parse_args()
    # ---

    try:
        if sys.platform == "win32":
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main(instance_uuid_arg=args.instance_uuid)) # Pass the argument to main
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
