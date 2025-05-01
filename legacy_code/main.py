# main.py (Revised for Stability and New Features)

import asyncio
import glob
import logging
import os
import re  # Keep re if sanitize_name_for_id is used (though less likely now with UUIDs)
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ADK Imports
from google.adk.agents import BaseAgent  # Use BaseAgent for type hints
# from google.adk.agents import Agent # If needed directly
from google.adk.runners import Runner
from google.adk.sessions import (BaseSessionService, InMemorySessionService,
                                 Session)
# Rich Imports
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.agents.narration import narration_agent
from src.agents.npc import npc_agent
from src.agents.simulacra import \
    create_agent as create_simulacra_agent  # Factory
from src.agents.world_engine import create_world_engine_validator
from src.agents.world_execution_agent import world_execution_agent
# --- Agent Imports ---
# Import agent INSTANCES or factory functions
from src.agents.world_state_agent import world_state_agent
from src.config import \
    settings  # Use for API Key, Model, potentially MAX_TURNS/NUM_SIMULACRA
from src.generation.life_generator import generate_new_simulacra_background
from src.initialization import \
    generate_unique_id  # Assuming this is now defined in initialization.py
from src.initialization import (load_json_file,
                                load_or_create_simulation_instance,
                                load_world_template, save_json_file)
# --- Simulation Loop ---
from src.simulation_loop import run_phased_simulation
from src.tools.world_state_tools import (  # Import state keys if used directly; Add other keys if main.py manipulates them directly
    ACTIVE_SIMULACRA_IDS_KEY, WORLD_STATE_KEY)

# Setup console
console = Console()

# --- Logging Setup ---
log_filename = "simulation.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)
logger = logging.getLogger(__name__)
logger.info(f"--- Application Start --- Logging configured to file: {log_filename}")

# --- Constants ---
APP_NAME = "TheSimulation"
USER_ID = "player1"
# Get simulation parameters (use settings or define directly)
MAX_TURNS = settings.MAX_SIMULATION_TURNS if hasattr(settings, 'MAX_SIMULATION_TURNS') else 10
NUM_SIMULACRA = settings.NUM_SIMULACRA if hasattr(settings, 'NUM_SIMULACRA') else 2

# Define paths directly in main.py
WORLD_TEMPLATE_PATH = "world_config.json"   # Path for world config template
STATE_DIR = "data/states"                     # Directory for state files
LIFE_SUMMARY_DIR = "data/life_summaries"        # Directory for life summaries

# Ensure directories exist
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True)


# --- Main Execution Logic ---
async def main():
    console.print(Panel(f"[[bold yellow]{APP_NAME}[/]] - Initializing...", title="Startup", border_style="blue"))
    logger.info("Starting main execution.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- API Key Check ---
    if not settings.GOOGLE_API_KEY:
        console.print("[bold red]ERROR: GOOGLE_API_KEY not found in environment variables.[/bold red]")
        logger.critical("GOOGLE_API_KEY not found.")
        sys.exit(1)

    # 1. Load World Template
    try:
        world_template_data = load_world_template(WORLD_TEMPLATE_PATH)
        logger.info(f"Using World Template: {WORLD_TEMPLATE_PATH}")
        console.print(f"World Template loaded: {world_template_data.get('description', 'N/A')}")
    except (FileNotFoundError, Exception) as e:
        logger.critical(f"Failed to load world template: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to load world template '{WORLD_TEMPLATE_PATH}'. Error: {e}")
        sys.exit(1)

    # 2. Load or Create Simulation Instance State
    current_session_id = f"session_{generate_unique_id()}" # Generate a unique ID for this run's ADK session
    instance_uuid_to_load = None # Set this based on args later if needed to load specific instance
    simulation_state = None
    world_instance_uuid = None
    state_file_path = None

    try:
        # Call with loaded template data and the generated session ID
        simulation_state, world_instance_uuid, state_file_path = load_or_create_simulation_instance(
            world_template=world_template_data,
            session_id=current_session_id, # Pass the generated session ID
            instance_uuid_to_load=instance_uuid_to_load
        )
        logger.info(f"Using Simulation Instance UUID: {world_instance_uuid}")
        logger.info(f"Using State File Path: {state_file_path}")
        console.print(f"Simulation Instance loaded/created (UUID: {world_instance_uuid})")
        console.print(f"State File: {state_file_path}")
    except (FileNotFoundError, ValueError, IOError, RuntimeError, Exception) as e:
        logger.critical(f"Failed to load or create simulation instance state: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to initialize simulation state. Check logs. Error: {e}")
        sys.exit(1)

    # 3. Manage Simulacra (Check existing, generate if needed)
    # Get active IDs from the loaded state
    active_sim_ids = simulation_state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
    was_state_loaded_with_ids = bool(active_sim_ids) # Track if state file had IDs initially

    # --- Refine active_sim_ids based on existing summary files for this instance ---
    # This ensures we don't try to activate IDs from state if their summary is missing
    life_summary_pattern_instance = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_*_{world_instance_uuid}.json")
    all_known_sim_ids_for_instance = []
    for filepath in glob.glob(life_summary_pattern_instance):
        summary = load_json_file(filepath)
        if summary and summary.get("world_instance_uuid") == world_instance_uuid:
            sim_id = summary.get("simulacra_id")
            if sim_id:
                all_known_sim_ids_for_instance.append(sim_id)

    if was_state_loaded_with_ids:
        original_state_ids = list(active_sim_ids) # Copy before filtering
        active_sim_ids = [sid for sid in active_sim_ids if sid in all_known_sim_ids_for_instance]
        if len(active_sim_ids) < len(original_state_ids):
             missing_ids = set(original_state_ids) - set(active_sim_ids)
             logger.warning(f"Simulacra from state ({missing_ids}) missing summary files for instance {world_instance_uuid}. Continuing with {active_sim_ids}")
        logger.info(f"Confirmed {len(active_sim_ids)} simulacra from state with existing summaries: {active_sim_ids}")
    else:
        # If state had no IDs, activate available ones up to the limit
        active_sim_ids = all_known_sim_ids_for_instance[:NUM_SIMULACRA]
        logger.info(f"No IDs in state. Activating up to {NUM_SIMULACRA} existing simulacra from files: {active_sim_ids}")
        # Update state if we activated some from files
        if active_sim_ids:
             simulation_state[ACTIVE_SIMULACRA_IDS_KEY] = active_sim_ids

    # --- Generate New Simulacra if Needed ---
    num_existing_active = len(active_sim_ids)
    num_to_generate = NUM_SIMULACRA - num_existing_active
    simulacra_generated_this_run = False

    console.print(f"Target Simulacra: {NUM_SIMULACRA}. Found Active: {num_existing_active}. Need to Generate: {max(0, num_to_generate)}")

    if num_to_generate > 0:
        logger.info(f"Attempting to generate {num_to_generate} new simulacra for instance {world_instance_uuid}.")
        console.rule(f"[bold yellow]Generating {num_to_generate} New Simulacra[/bold yellow]")

        # Get world details needed for generation FROM the loaded state
        world_details_for_gen = simulation_state.get("world_template_details", {})
        world_type_from_state = world_details_for_gen.get("world_type", "real")
        world_desc_from_state = world_details_for_gen.get("description", "Standard real world simulation.")

        newly_generated_count = 0
        for i in range(num_to_generate):
            console.print(f"Generating simulacrum {i+1}/{num_to_generate}...")
            new_sim_id = generate_unique_id("sim") # Generate a unique ID

            try:
                # Call generation function
                persona_life_data = await generate_new_simulacra_background(
                    sim_id=new_sim_id,
                    world_instance_uuid=world_instance_uuid,
                    world_type=world_type_from_state,
                    world_description=world_desc_from_state
                )

                if persona_life_data and "persona_details" in persona_life_data:
                    persona_data = persona_life_data["persona_details"]

                    # Update the main simulation state dictionary (in memory)
                    simulation_state.setdefault(ACTIVE_SIMULACRA_IDS_KEY, []).append(new_sim_id)
                    simulation_state[f"simulacra_{new_sim_id}_persona"] = persona_data

                    # Add other default state entries (location, goal, status)
                    world_state_details = simulation_state.get(WORLD_STATE_KEY, {})
                    location_keys = list(world_state_details.get("location_details", {}).keys())
                    default_location = location_keys[0] if location_keys else "Default Location"
                    simulation_state[f"simulacra_{new_sim_id}_location"] = persona_data.get("Initial_Location", default_location)
                    simulation_state[f"simulacra_{new_sim_id}_goal"] = persona_data.get("Initial_Goal", "Explore the world.")
                    simulation_state[f"simulacra_{new_sim_id}_status"] = {"condition": "Normal", "mood": "Neutral"}
                    # simulation_state[ACTION_VALIDATION_KEY_FORMAT.format(new_sim_id)] = None # If using this key

                    active_sim_ids.append(new_sim_id) # Add to our current list
                    simulacra_generated_this_run = True
                    newly_generated_count += 1
                    logger.info(f"Successfully generated and added {new_sim_id} to state.")
                    console.print(f"  -> Generated: {persona_data.get('Name', new_sim_id)}")

                    # Save the generated life summary separately
                    summary_filename = f"life_summary_{new_sim_id}_{timestamp}_{world_instance_uuid}.json"
                    summary_filepath = os.path.join(LIFE_SUMMARY_DIR, summary_filename)
                    save_json_file(summary_filepath, persona_life_data) # Save the full data from generator
                    logger.info(f"Saved life summary for {new_sim_id} to {summary_filepath}")

                else:
                    logger.error(f"Failed to generate valid persona data for {new_sim_id}. Skipping.")
                    console.print(f"[red]  -> Failed to generate {new_sim_id}.[/red]")

            except Exception as gen_e:
                logger.error(f"Error during generation for new simulacra {new_sim_id}: {gen_e}", exc_info=True)
                console.print(f"[red]  -> Error generating {new_sim_id}: {gen_e}[/red]")
        console.rule()
        logger.info(f"Finished generation attempt. Generated {newly_generated_count} new simulacra.")

    # --- Save State Update if Needed ---
    # Save if new sims were generated OR if we activated sims from files into an empty state
    if simulacra_generated_this_run or (not was_state_loaded_with_ids and active_sim_ids):
         logger.info("Saving updated simulation state (new/activated simulacra) to instance file.")
         try:
            simulation_state[ACTIVE_SIMULACRA_IDS_KEY] = active_sim_ids # Ensure list is up-to-date
            save_json_file(state_file_path, simulation_state)
            logger.info(f"Saved updated state to {state_file_path}")
         except Exception as e:
             logger.error(f"Failed to save state update to {state_file_path}: {e}", exc_info=True)
             console.print(f"[bold red]Error:[/bold red] Failed to save state update to {state_file_path}. Check logs.")
             # Decide if this is critical and should stop the simulation
             # sys.exit(1)
    else:
         logger.info("No state changes requiring save before agent creation.")

    # --- Final Check if Simulation Can Proceed ---
    if not active_sim_ids:
        logger.critical("No active simulacra available or generated for this instance. Cannot proceed.")
        console.print("[bold red]Error:[/bold red] No simulacra available for the simulation.")
        sys.exit(1)

    logger.info(f"Simulation instance {world_instance_uuid} will run with {len(active_sim_ids)} simulacra: {active_sim_ids}")
    console.print(f"Running simulation with: {', '.join(active_sim_ids)}")

    # 4. Initialize ADK Session
    session_service = InMemorySessionService()
    # Use the session ID already set in the state by load_or_create_simulation_instance
    adk_session_id = simulation_state.get("session_id", current_session_id) # Fallback just in case

    try:
        # Create session, passing the final simulation_state
        session = session_service.create_session(
            user_id=USER_ID,
            app_name=APP_NAME,
            session_id=adk_session_id,
            state=simulation_state  # Pass the final state here
        )
        logger.info(f"ADK Session created/loaded: {adk_session_id} with initial state.")
    except Exception as e:
        logger.critical(f"Failed to create ADK session with initial state: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to create ADK session. Check logs.")
        sys.exit(1)

    # 5. Instantiate Agents
    console.rule("[cyan]Instantiating Agents[/cyan]")
    core_agents_list: List[BaseAgent] = []
    simulacra_agents_dict: Dict[str, BaseAgent] = {} # Use dict for easier lookup
    validator_agents_dict: Dict[str, BaseAgent] = {} # <<< ADD: Initialize validator dict

    try:
        # Instantiate or reference core agents
        core_agents_map = {
            "World State": world_state_agent,
            "NPC Interaction": npc_agent,
            "World Execution": world_execution_agent,
            "Narration": narration_agent
        }
        for name, agent_instance in core_agents_map.items():
             if isinstance(agent_instance, BaseAgent):
                 core_agents_list.append(agent_instance)
                 logger.info(f"Core agent '{name}' ({agent_instance.name}) added.")
             else:
                 raise TypeError(f"Core agent '{name}' is not a valid BaseAgent instance.")
        console.print(f"  - {len(core_agents_list)} Core agents loaded.")
    except Exception as e:
        logger.critical(f"Failed to load core agents: {e}", exc_info=True)
        console.print(f"[bold red]Error loading core agents:[/bold red] {e}")
        sys.exit(1)

    # <<< ADD: Instantiate Validator Agents >>>
    console.print("Instantiating Validator Agents...")
    for sim_id in active_sim_ids: # Iterate through the same active IDs
        try:
            validator_instance = create_world_engine_validator(sim_id=sim_id)
            if validator_instance and isinstance(validator_instance, BaseAgent):
                validator_agents_dict[sim_id] = validator_instance
                logger.info(f"Created and added validator agent instance for {sim_id} ({validator_instance.name})")
                console.print(f"  - Instantiated Validator: {validator_instance.name}")
            else:
                logger.error(f"Factory failed to create a valid BaseAgent validator instance for {sim_id}. Skipping.")
                console.print(f"[red]  - Failed to instantiate Validator: {sim_id}[/red]")
        except Exception as create_val_e:
            logger.error(f"Error calling create_world_engine_validator for {sim_id}: {create_val_e}", exc_info=True)
            console.print(f"[red]  - Error instantiating Validator {sim_id}: {create_val_e}[/red]")

    # Check if enough validators were created
    if len(validator_agents_dict) != len(active_sim_ids):
         logger.warning(f"Mismatch between active IDs ({len(active_sim_ids)}) and successfully created validator agents ({len(validator_agents_dict)}).")
         # Decide if this is critical, maybe exit if none were created
         if not validator_agents_dict:
              console.print("[bold red]No validator agents were successfully instantiated. Exiting.[/bold red]")
              sys.exit(1)

    # Instantiate Simulacra Agents
    console.print("Instantiating Simulacra Agents...")
    for sim_id in active_sim_ids:
        persona = simulation_state.get(f"simulacra_{sim_id}_persona")

        # Fallback to loading from summary file if not in state dict
        if not persona:
            logger.warning(f"Persona for {sim_id} not found in state dict. Attempting fallback load.")
            summary_filename_glob = os.path.join(LIFE_SUMMARY_DIR, f"life_summary_{sim_id}_*_{world_instance_uuid}.json")
            found_files = glob.glob(summary_filename_glob)
            if found_files:
                try:
                    latest_summary_file = max(found_files, key=os.path.getctime)
                    life_data = load_json_file(latest_summary_file)
                    if life_data and "persona_details" in life_data:
                        persona = life_data["persona_details"]
                        logger.info(f"Successfully loaded persona for {sim_id} from fallback file: {latest_summary_file}")
                        # Update the main state dict with the loaded persona
                        simulation_state[f"simulacra_{sim_id}_persona"] = persona
                        # Potentially update session state too if needed immediately by other agents
                        # await session.set_state({f"simulacra_{sim_id}_persona": persona}) # Requires set_state method
                    else:
                        logger.error(f"Fallback summary file {latest_summary_file} for {sim_id} missing 'persona_details'.")
                except Exception as load_e:
                    logger.error(f"Error loading fallback summary file for {sim_id}: {load_e}", exc_info=True)
            else:
                logger.error(f"No fallback summary file found matching pattern for {sim_id}.")

        # If still no persona, create a default/minimal one
        if not persona:
            logger.error(f"Could not load or find persona for {sim_id}. Creating minimal default.")
            persona = { # Minimal default persona structure
                "Name": sim_id, "Age": "Unknown", "Occupation": "Unknown",
                "Personality_Traits": ["Generic"], "Background": "Minimal default persona."
            }
            simulation_state[f"simulacra_{sim_id}_persona"] = persona # Add default to state

        # Create agent using the factory
        try:
            # --- MODIFIED: Remove world_config argument ---
            agent_instance = create_simulacra_agent(
                sim_id=sim_id,
                persona=persona,
                # world_config=world_config_for_agents, # Removed
                session=session # Pass the ADK session object
            )
            # --- END MODIFICATION ---
            if agent_instance and isinstance(agent_instance, BaseAgent):
                simulacra_agents_dict[sim_id] = agent_instance # Store in dict
                logger.info(f"Created and added simulacra agent instance for {sim_id} ({agent_instance.name})")
                console.print(f"  - Instantiated Simulacra: {agent_instance.name}")
            else:
                 logger.error(f"Factory failed to create a valid BaseAgent instance for {sim_id}. Skipping.")
                 console.print(f"[red]  - Failed to instantiate Simulacra: {sim_id}[/red]")
        except Exception as create_e:
             logger.error(f"Error calling create_simulacra_agent for {sim_id}: {create_e}", exc_info=True)
             console.print(f"[red]  - Error instantiating Simulacra {sim_id}: {create_e}[/red]")

    if len(simulacra_agents_dict) != len(active_sim_ids):
         logger.warning(f"Mismatch between active IDs ({len(active_sim_ids)}) and successfully created simulacra agents ({len(simulacra_agents_dict)}).")
         if not simulacra_agents_dict:
              console.print("[bold red]No simulacra agents were successfully instantiated. Exiting.[/bold red]")
              sys.exit(1)

    console.rule()

    # 6. Create Runner
    try:
        # Initialize Runner correctly (using a default agent like world_state_agent)
        runner = Runner(
            agent=world_state_agent, # Default/Orchestrator agent
            app_name=APP_NAME,
            session_service=session_service
        )
        logger.info(f"Runner initialized for app '{APP_NAME}' with default agent '{world_state_agent.name}'.")
    except Exception as e:
        logger.critical(f"Failed to initialize ADK Runner: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to initialize ADK Runner. Check logs.")
        sys.exit(1)

    # 7. Run Phased Simulation
    console.rule("[bold magenta]Starting Phased Simulation[/bold magenta]")
    final_state = None
    try:
        # Call the phased loop function, passing the dictionary of simulacra agents
        final_state = await run_phased_simulation(
            runner=runner,
            session_service=session_service, # Pass service if loop needs it
            session=session, # Pass the ADK session
            # Pass agent instances individually as expected by run_phased_simulation
            world_state_agent=world_state_agent,
            simulacra_agents=simulacra_agents_dict, # Pass the dict
            validator_agents=validator_agents_dict, # <<< ADD: Pass the validator dict
            npc_agent=npc_agent,
            world_execution_agent=world_execution_agent,
            narration_agent=narration_agent,
            # Pass config
            max_turns=MAX_TURNS
        )
        logger.info("Phased simulation loop completed.")
        console.rule("[bold green]Simulation Complete[/]")

    except Exception as e:
        logger.critical(f"Error during simulation loop: {e}", exc_info=True)
        console.print(f"\n[bold red]Error during simulation:[/bold red] {e}")
        # Attempt to get state at time of error
        try:
            # Assuming session object holds the latest state or can retrieve it
            # If session.state is updated live, use that. Otherwise, try get_state.
            if hasattr(session, 'get_state') and asyncio.iscoroutinefunction(session.get_state):
                 current_state = await session.get_state()
                 final_state = current_state # Use this as the state to save
                 logger.info("Retrieved session state via get_state() at time of error.")
            elif hasattr(session, 'state'):
                 final_state = session.state # Use the state attribute directly
                 logger.info("Using session.state attribute at time of error.")
            else:
                 logger.warning("Could not retrieve session state after error. Using state before loop.")
                 final_state = simulation_state # Fallback

            # Save error state with instance UUID in the name
            error_state_filename = f"simulation_state_error_{world_instance_uuid}_{timestamp}.json"
            error_state_path = os.path.join(STATE_DIR, error_state_filename)
            save_json_file(error_state_path, final_state)
            logger.info(f"Saved error state to {error_state_path}")
            console.print(f"[yellow]Saved state at time of error to {error_state_path}[/yellow]")

        except Exception as get_state_e:
            logger.error(f"Could not retrieve or save session state after error: {get_state_e}")
            final_state = simulation_state # Fallback to state before loop started

    # 8. Save Final State (if loop completed normally or state was retrieved after error)
    if final_state:
        logger.info("Saving final simulation state.")
        try:
            save_json_file(state_file_path, final_state) # Use the specific state_file_path
            logger.info(f"Final simulation state saved to {state_file_path}")
            console.print(f"Final state saved to {state_file_path}")
        except Exception as save_e:
             logger.error(f"Failed to save final state to {state_file_path}: {save_e}", exc_info=True)
             console.print(f"[red]Error saving final state to {state_file_path}: {save_e}[/red]")
    else:
        logger.warning("No final state available to save (Simulation might have failed early).")


# --- Entry Point ---
if __name__ == "__main__":
    try:
        # Ensure the event loop policy is set correctly for Windows if needed
        if sys.platform == "win32":
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
        console.print("\n[bold orange_red1]Simulation interrupted by user.[/bold orange_red1]")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main execution block: {e}", exc_info=True)
        console.print(f"[bold red]An unexpected error occurred:[/bold red]")
        console.print_exception(show_locals=False) # Keep locals off generally
    finally:
        logger.info("--- Application End ---")
        logging.shutdown() # Ensure logs are flushed
        console.print("Application finished.")