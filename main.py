# main.py - Main entry point (Updated for Sequential Execution)
import asyncio
import copy  # Import copy for deep copying initial state
import json
import logging
import os
import sys
import traceback

# Import ADK components
from google.adk.agents import SequentialAgent # Import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# Import Rich for console output
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule # Import Rule for separators

# Import agents - MAKE SURE world_state_agent is imported
from src.agents import narration as narration_agent_factory
from src.agents import npc as npc_agent_factory
from src.agents import simulacra as simulacra_agent_factory
# Import the redefined world_engine instance directly
from src.agents.world_engine import world_engine as world_engine_agent_instance
# Import the world_state_agent instance directly
from src.agents.world_state_agent import world_state_agent # <<< ENSURE THIS IMPORT WORKS

# Import configuration, initial state, agent factories, and main loop
from src.config import settings
from src.generation.life_generator import generate_new_simulacra_background
from src.session.initial_state import default_initial_sim_state
# Assuming simulation_loop contains the detailed logging function
from src.simulation_loop import run_simulation_turn # Keep using detailed logging

# Suppress INFO and DEBUG logs globally
logging.basicConfig(level=logging.WARNING)
logging.getLogger("google.adk").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Instantiate Rich Console
console = Console()

def load_first_life_summary():
    """Loads the first life_summary JSON file found in the current directory."""
    console.print("[cyan]Looking for existing life_summary JSON files...[/cyan]")
    for file in os.listdir("."):
        if file.startswith("life_summary_") and file.endswith(".json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    console.print(f"[green]Found life summary file: {file}[/green]")
                    return json.load(f)
            except Exception as e:
                console.print(f"[bold red]Error loading {file}:[/bold red] {e}")
    console.print("[yellow]No life_summary JSON files found. Using default state.[/yellow]")
    return None

def setup_simulation(initial_state_to_use: dict):  # Accept initial state
    """Creates all agents, session, and runner using the provided initial state."""
    console.rule("[bold cyan]Setting up Simulation[/bold cyan]")
    console.print("[cyan]Using Initial State:[/cyan]")
    console.print(f"Simulacra Name: [i]{initial_state_to_use.get('simulacra_status', {}).get('Name', 'N/A')}[/i]")
    console.print(f"Simulacra Age: [i]{initial_state_to_use.get('simulacra_status', {}).get('Age', 'N/A')}[/i]")
    console.print(f"Simulacra Location: [i]{initial_state_to_use.get('simulacra_location', 'N/A')}[/i]")
    console.print(f"Simulacra Goal: [i]{initial_state_to_use.get('simulacra_goal', 'N/A')}[/i]")

    # Create Agent Instances (assuming factory functions or direct imports work)
    simulacra_agent = simulacra_agent_factory.create_agent()
    npc_agent = npc_agent_factory.create_agent()

    # Create Narration agent, passing other agent instances it needs to interact with
    # Note: world_engine_agent_instance is imported directly above
    narration_agent = narration_agent_factory.create_agent(
        simulacra_agent_instance=simulacra_agent,
        world_engine_agent_instance=world_engine_agent_instance, # Pass the correct WE instance
        npc_agent_instance=npc_agent
    )

    # --- Check if all required agent instances are valid ---
    # world_state_agent and world_engine_agent_instance are imported directly
    if not all([world_state_agent, narration_agent, simulacra_agent, world_engine_agent_instance, npc_agent]):
        console.print("[bold red]One or more essential agents failed to initialize. Exiting.[/bold red]")
        return None, None

    # --- MODIFICATION START ---
    # Create SequentialAgent Workflow including world_state_agent first
    simulation_workflow_agent = SequentialAgent(
        name="SimulationTurnWorkflow", # Give the workflow a name
        sub_agents=[
            world_state_agent,  # <<< Run world_state_agent first
            narration_agent     # <<< Then run narration_agent
        ],
        description="Executes one turn of the simulation: first updates world state, then narrates and orchestrates actions."
    )
    console.print(f"Workflow agent '[bold purple]{simulation_workflow_agent.name}[/bold purple]' created (World State -> Narration).")
    # --- MODIFICATION END ---


    # Create Session Service
    session_service = InMemorySessionService()
    console.print("Session Service created ([i]InMemorySessionService[/i]).")

    # Create Session using the provided initial state
    try:
        session = session_service.create_session(
            app_name=settings.APP_NAME,
            user_id=settings.USER_ID,
            session_id=settings.SESSION_ID,
            state=initial_state_to_use  # Use the passed-in state
        )
        console.print(f"Session created: App='{settings.APP_NAME}', User='{settings.USER_ID}', Session='{settings.SESSION_ID}'")
    except Exception as e:
        console.print(f"[bold red]Error creating session:[/bold red] {e}")
        console.print_exception(show_locals=True)
        return None, None

    # Create Runner with the new workflow agent
    try:
        runner = Runner(
            agent=simulation_workflow_agent, # <<< Use the sequential workflow agent
            app_name=settings.APP_NAME,
            session_service=session_service
        )
        console.print(f"Runner created for agent '[bold purple]{runner.agent.name}[/bold purple]'.")
    except Exception as e:
        console.print(f"[bold red]Error creating runner:[/bold red] {e}")
        console.print_exception(show_locals=True)
        return None, None

    console.rule("[bold cyan]Simulation Setup Complete[/bold cyan]")
    return runner, session_service

async def run_simulation_main(sim_runner, sim_session_service):
    """Main simulation loop to process turns using the detailed logger."""
    console.rule("[bold magenta]Starting Simulation[/bold magenta]")
    turn_number = 1  # Start with the first turn

    # --- ADDED: Pass config file path to initial state ---
    # This makes the config file path available to the world_state_agent
    # when it reads the state at the beginning of its run.
    current_session = sim_session_service.get_session(
        app_name=settings.APP_NAME,
        user_id=settings.USER_ID,
        session_id=settings.SESSION_ID
    )
    if current_session:
        # Add or update the config file path in the session state
        current_session.state['world_config_path'] = 'world_config.json' # Or get from settings/args
        console.print(f"Set 'world_config_path' in session state: {current_session.state['world_config_path']}")
        # Note: For InMemorySessionService, this modification might need
        # to happen directly on the service's internal storage if get_session
        # returns a deep copy. Check ADK behavior or update state via an event if needed.
        # For simplicity here, assuming direct modification works for InMemory.
    else:
        console.print("[bold red]Error: Could not retrieve session to set config path.[/bold red]")
        return
    # --- END ADDED SECTION ---


    try:
        while True:  # Loop for simulation turns
            # Use the detailed turn runner from simulation_loop.py
            # This function now runs the SequentialAgent (world_state -> narration)
            await run_simulation_turn(turn_number, sim_runner, sim_session_service)

            # Increment turn number
            turn_number += 1

            # Prompt to continue
            continue_simulation = console.input("[bold yellow]Continue simulation? (yes/no): [/bold yellow]").strip().lower()
            if continue_simulation == "no":
                console.print("[bold red]Ending Simulation.[/bold red]")
                break

    except Exception as e:
        console.print(f"[bold red]An error occurred during the simulation:[/bold red] {e}")
        # console.print_exception(show_locals=True) # Uncomment for full traceback
    finally:
        console.rule("[bold magenta]Simulation Ended[/bold magenta]")


async def main_entry():
    """Handles user choice, generation (if needed), setup, and runs simulation."""
    # Check API key first
    if not settings.GOOGLE_API_KEY:
        console.print("[bold red]ERROR: GOOGLE_API_KEY not found in environment variables or .env file.[/bold red]")
        console.print("Please set the GOOGLE_API_KEY environment variable.")
        # Attempt to load from .env as a fallback
        try:
            from dotenv import load_dotenv
            load_dotenv()
            if not os.getenv("GOOGLE_API_KEY"):
                 raise ValueError("Still no GOOGLE_API_KEY after loading .env")
            settings.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            console.print("[yellow]Loaded GOOGLE_API_KEY from .env file.[/yellow]")
        except Exception as load_err:
             console.print(f"[bold red]Could not load .env or find key: {load_err}[/bold red]")
             sys.exit(1)


    console.print(Panel("Welcome to the ADK World Simulation!", title="Welcome", border_style="blue"))
    choice = console.input("Generate a [bold cyan]new[/bold cyan] Simulacra background or use [bold yellow]default[/bold yellow]? (Type 'new' or 'default', press Enter for default): ").lower().strip()

    current_initial_state = None
    if not choice:  # Default to 'default' if the user presses Enter
        choice = "default"

    if choice == 'new':
        console.print("\nStarting background generation...", style="italic yellow")
        generated_life_data = await generate_new_simulacra_background() # Ensure this function is async if it uses await

        if generated_life_data and generated_life_data.get("persona_details"):
            console.print("Background generation finished.", style="green")
            current_initial_state = copy.deepcopy(default_initial_sim_state)
            persona = generated_life_data["persona_details"]
            # Update state based on generation - adjust keys as necessary
            current_initial_state["simulacra_location"] = persona.get("Current_location", current_initial_state["simulacra_location"])
            current_initial_state["simulacra_goal"] = persona.get("Life_Goals", f"Settle into {persona.get('Current_location', 'the area')}")
            current_initial_state["simulacra_status"]["Name"] = persona.get("Name", "Generated Simulacra")
            current_initial_state["simulacra_status"]["Age"] = persona.get("Age", 30)
            current_initial_state["simulacra_status"]["Occupation"] = persona.get("Occupation", "Unknown")
            current_initial_state["simulacra_status"]["Personality_Traits"] = persona.get("Personality_Traits", [])
            current_initial_state["generated_background"] = generated_life_data
            console.print(f"Using generated persona: [bold]{persona.get('Name')}[/bold], Age {persona.get('Age')}")
        else:
            console.print("Background generation failed or produced no data. Using default state.", style="bold red")
            current_initial_state = copy.deepcopy(default_initial_sim_state)
    else: # Default or load existing
        console.print("Attempting to load existing life summary or use default state...", style="yellow")
        life_summary_data = load_first_life_summary()
        if life_summary_data:
            current_initial_state = copy.deepcopy(default_initial_sim_state)
            persona = life_summary_data.get("persona_details", {})
            # Populate state from loaded file
            current_initial_state["simulacra_location"] = persona.get("Current_location", 'Unknown')
            current_initial_state["simulacra_goal"] = persona.get('Life_Goals', 'Explore the area.')
            current_initial_state["simulacra_status"]["Name"] = persona.get("Name", "Loaded Simulacra")
            current_initial_state["simulacra_status"]["Age"] = persona.get("Age", 30)
            current_initial_state["simulacra_status"]["Occupation"] = persona.get("Occupation", "Unknown")
            current_initial_state["simulacra_status"]["Personality_Traits"] = persona.get("Personality_Traits", [])
            current_initial_state["generated_background"] = life_summary_data
            console.print(f"Using loaded persona: [bold]{persona.get('Name')}[/bold], Age {persona.get('Age')}")
        else:
            console.print("Using default initial state.", style="yellow")
            current_initial_state = copy.deepcopy(default_initial_sim_state)

    # Ensure initial state is not None before proceeding
    if current_initial_state is None:
         console.print("[bold red]Initial state could not be determined. Exiting.[/bold red]")
         sys.exit(1)

    # Setup simulation components using the chosen initial state
    sim_runner, sim_session_service = setup_simulation(current_initial_state)

    # Run the main simulation loop if setup was successful
    if sim_runner and sim_session_service:
        try:
            await run_simulation_main(sim_runner, sim_session_service)
        except Exception as e:
            console.print(f"[bold red]An error occurred during simulation execution:[/bold red]")
            console.print_exception(show_locals=True)
    else:
        console.print("[bold red]Exiting due to setup failure.[/bold red]")


if __name__ == "__main__":
    try:
        asyncio.run(main_entry())
    except KeyboardInterrupt:
        console.print("\n[bold orange_red1]Simulation interrupted by user.[/bold orange_red1]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red]")
        console.print_exception(show_locals=True)

