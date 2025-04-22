# main.py - Main entry point
import asyncio
import copy  # Import copy for deep copying initial state
import json
import logging
import os
import sys
import traceback

from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types  # Ensure this import is present
from rich.console import Console
from rich.panel import Panel

from src.agents import narration as narration_agent_factory
from src.agents import npc as npc_agent_factory
from src.agents import simulacra as simulacra_agent_factory
from src.agents import world_engine as world_engine_agent_factory
# Import configuration, initial state, agent factories, and main loop
from src.config import settings
from src.generation.life_generator import generate_new_simulacra_background
from src.session.initial_state import default_initial_sim_state

# Suppress INFO and DEBUG logs globally
logging.basicConfig(level=logging.WARNING)

# Suppress specific libraries (e.g., google.adk, google.genai, urllib3)
logging.getLogger("google.adk").setLevel(logging.ERROR)  # Suppress INFO and WARNING logs
logging.getLogger("google.genai").setLevel(logging.ERROR)  # Suppress INFO and WARNING logs
logging.getLogger("urllib3").setLevel(logging.ERROR)  # Suppress HTTP request logs

# Suppress logs from all other libraries
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
    # Print key initial state details for confirmation
    console.print(f"Simulacra Name: [i]{initial_state_to_use.get('simulacra_status', {}).get('Name', 'N/A')}[/i]")
    console.print(f"Simulacra Age: [i]{initial_state_to_use.get('simulacra_status', {}).get('Age', 'N/A')}[/i]")
    console.print(f"Simulacra Location: [i]{initial_state_to_use.get('simulacra_location', 'N/A')}[/i]")
    console.print(f"Simulacra Goal: [i]{initial_state_to_use.get('simulacra_goal', 'N/A')}[/i]")

    # Create Agents using factory functions
    simulacra_agent = simulacra_agent_factory.create_agent()
    world_engine_agent = world_engine_agent_factory.create_agent()
    npc_agent = npc_agent_factory.create_agent()

    # Create Narration agent, passing sub-agents
    narration_agent = narration_agent_factory.create_agent(
        simulacra_agent_instance=simulacra_agent,
        world_engine_agent_instance=world_engine_agent,
        npc_agent_instance=npc_agent
    )

    if not narration_agent:
        console.print("[bold red]Failed to create the Narration agent. Exiting.[/bold red]")
        return None, None

    # Create SequentialAgent Workflow with only the Narration agent
    simulation_workflow_agent = SequentialAgent(
        name="SimulationWorkflowAgent",
        sub_agents=[
            narration_agent  # Narration agent orchestrates other agents
        ]
    )

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

    # Create Runner
    try:
        runner = Runner(
            agent=simulation_workflow_agent,
            app_name=settings.APP_NAME,
            session_service=session_service
        )
        console.print(f"Runner created for agent '[bold yellow]{runner.agent.name}[/bold yellow]'.")
    except Exception as e:
        console.print(f"[bold red]Error creating runner:[/bold red] {e}")
        console.print_exception(show_locals=True)
        return None, None

    console.rule("[bold cyan]Simulation Setup Complete[/bold cyan]")
    return runner, session_service

async def run_simulation_main(sim_runner, sim_session_service):
    """Main simulation loop to process turns."""
    console.rule("[bold magenta]Starting Simulation[/bold magenta]")
    turn_number = 1  # Start with the first turn

    try:
        while True:  # Infinite loop for simulation turns
            console.print(f"[bold cyan]Running Turn {turn_number}[/bold cyan]")
            user_input_text = f"Continue simulation (Turn {turn_number})."

            if not user_input_text.strip():
                user_input_text = "Default input text for simulation."

            # Create a Content object with the user input
            content = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_input_text)]  # Ensure `text` is not empty
            )

            # Run the simulation turn
            async for event in sim_runner.run_async(
                user_id=settings.USER_ID,
                session_id=settings.SESSION_ID,
                new_message=content
            ):
                author = getattr(event, "author", "Unknown Agent")
                console.print(f'{author} - Is current agent Event')
                if event.is_final_response():
                    # Tag the final response with the agent's name
                    console.print(f"[bold green]{author}-Final Response for Turn {turn_number}:[/bold green] {event.content.parts[0].text}")
                else:
                    # Handle intermediate events
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                # Tag intermediate events with the agent's name
                                console.print(f"[dim]{author}-Intermediate Event:[/dim] {part.text.replace("\n", "")}")
                            elif part.function_call:
                                # Tag function calls with the agent's name
                                console.print(f"[dim]{author}-Function Call: {part.function_call.name} with args {part.function_call.args}[/dim]")
                            elif part.function_response:
                                # Tag function responses with the agent's name
                                console.print(f"[dim]{author}-Function Response: {part.function_response.name} with response {part.function_response.response}[/dim]")

            # Increment turn number
            turn_number += 1

            # Optional: Add a break condition or user input to stop the simulation

            continue_simulation = console.input("[bold yellow]Continue simulation? (yes/no): [/bold yellow]").strip().lower()
            if continue_simulation == "no":
                console.print("[bold red]Ending Simulation.[/bold red]")
                break

    except Exception as e:
        console.print(f"[bold red]An error occurred during the simulation:[/bold red] {e}")
        console.print_exception(show_locals=True)
    finally:
        console.rule("[bold magenta]Simulation Ended[/bold magenta]")

async def main_entry():
    """Handles user choice, generation (if needed), setup, and runs simulation."""
    # Check API key first
    if not settings.GOOGLE_API_KEY:
        sys.exit(1)

    console.print(Panel("Welcome to the ADK World Simulation!", title="Welcome", border_style="blue"))
    choice = console.input("Generate a [bold cyan]new[/bold cyan] Simulacra background or use [bold yellow]default[/bold yellow]? (Type 'new' or 'default'): ").lower().strip()

    current_initial_state = None
    if not choice:  # Default to 'default' if the user presses Enter
        choice = "default"
        
    if choice == 'new':
        console.print("\nStarting background generation...", style="italic yellow")
        # Run the generation function
        generated_life_data = await generate_new_simulacra_background()

        if generated_life_data and generated_life_data.get("persona_details"):
            console.print("Background generation finished.", style="green")
            # Adapt the default state with generated data
            current_initial_state = copy.deepcopy(default_initial_sim_state)  # Start fresh
            persona = generated_life_data["persona_details"]

            # Update relevant fields - adapt keys as needed based on your state structure
            current_initial_state["simulacra_location"] = persona.get("Current_location", current_initial_state["simulacra_location"])
            current_initial_state["simulacra_goal"] = f"Settle into {persona.get('Current_location', 'the area')} and pursue being a {persona.get('Occupation', 'person')}."
            current_initial_state["simulacra_status"]["Name"] = persona.get("Name", "Generated Simulacra")
            current_initial_state["simulacra_status"]["Age"] = persona.get("Age", 30)
            current_initial_state["simulacra_status"]["Occupation"] = persona.get("Occupation", "Unknown")
            current_initial_state["simulacra_status"]["Personality_Traits"] = persona.get("Personality_Traits", [])
            # Optionally store the full generated data if needed later in the simulation
            current_initial_state["generated_background"] = generated_life_data
            console.print(f"Using generated persona: [bold]{persona.get('Name')}[/bold], Age {persona.get('Age')}")

        else:
            console.print("Background generation failed or produced no data. Using default state.", style="bold red")
            current_initial_state = copy.deepcopy(default_initial_sim_state)
    else:
        console.print("Using default initial state.", style="yellow")
        # Load the first life_summary JSON file if available
        life_summary_data = load_first_life_summary()
        if life_summary_data:
            current_initial_state = copy.deepcopy(default_initial_sim_state)
            persona = life_summary_data.get("persona_details", {})
            current_initial_state["simulacra_location"] = persona.get("Current_location", 'Unknown')
            current_initial_state["simulacra_goal"] = persona.get('Life_Goals', 'the area')
            current_initial_state["simulacra_status"]["Name"] = persona.get("Name", "Default Simulacra")
            current_initial_state["simulacra_status"]["Age"] = persona.get("Age", 30)
            current_initial_state["simulacra_status"]["Occupation"] = persona.get("Occupation", "Unknown")
            current_initial_state["simulacra_status"]["Personality_Traits"] = persona.get("Personality_Traits", [])
            current_initial_state["generated_background"] = life_summary_data
        else:
            current_initial_state = copy.deepcopy(default_initial_sim_state)

    # Setup simulation components using the chosen initial state
    sim_runner, sim_session_service = setup_simulation(current_initial_state)

    # Run the main simulation loop if setup was successful
    if sim_runner and sim_session_service:
        try:
            # Pass runner and service to the main simulation loop function
            await run_simulation_main(sim_runner, sim_session_service)
        except Exception as e:
            console.print(f"[bold red]An error occurred during simulation execution:[/bold red]")
            console.print_exception(show_locals=True)
    else:
        console.print("[bold red]Exiting due to setup failure.[/bold red]")


if __name__ == "__main__":
    try:
        # Run the main entry function that handles setup and simulation run
        asyncio.run(main_entry())
    except KeyboardInterrupt:
        console.print("\n[bold orange_red1]Simulation interrupted by user.[/bold orange_red1]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red]")
        console.print_exception(show_locals=True)