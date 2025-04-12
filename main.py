# main.py
import asyncio
import datetime as dt # Use alias
import json
import logging
import os
import time
import argparse
from collections import deque # Import deque for efficient history tracking
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Literal, Optional, Union

# import yaml # REMOVED
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule # Keep Rule import
# from rich.layout import Group # <<< REMOVE THIS IMPORT >>>

# --- Import Core Components from src ---
try:
    from src.llm_service import LLMService
    from src.engines.world_engine import WorldEngine
    from src.simulacra import Simulacra
    from src.models import WorldReactionProfile # Only need this for argument choices
except ImportError as e:
     print(f"CRITICAL Error importing core modules: {e}")
     print("Please ensure src directory and required files exist and are importable.")
     exit(1)

# --- Basic Setup ---
os.makedirs("logs", exist_ok=True)
LOG_FILE_PATH = "logs/simulation_main.log"
logging.Formatter.converter = time.gmtime
file_handler = RotatingFileHandler(
    LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s.%(msecs)03dZ - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
))
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(file_handler)
logging.getLogger('google.api_core').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# --- End Setup ---


# --- API Key Check ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    startup_console = Console()
    startup_console.print("\n[bold red]🛑 ERROR: GOOGLE_API_KEY environment variable not set.[/bold red]")
    startup_console.print("   Please set the GOOGLE_API_KEY environment variable with your API key.")
    logger.critical("GOOGLE_API_KEY environment variable not set.")
    exit(1)
if not os.environ.get("OPENAI_API_KEY"): logger.warning("OPENAI_API_KEY environment variable not set (legacy check).")
# --- End API Key Check ---

async def run_simulation(cycles: int,
                       world_state_file: str,
                       delta_state_path: str,
                       life_summary_path: Optional[str],
                       new_simulacra: bool,
                       console: Console,
                       reaction_profile_arg: Union[str, Dict],
                       step_duration_minutes: int):
    """
    Orchestrates the simulation run, initializing components and managing the cycle loop.
    Accepts the step duration and handles recent history context.
    """
    start_run_time = time.time()
    logger.info(f"Starting simulation run with step duration: {step_duration_minutes} minutes...")
    console.rule("[bold blue]🚀 LAUNCHING SIMULATION 🚀[/bold blue]")
    # ... (startup info display remains the same) ...
    startup_info = Text.assemble(
        ("Cycles Requested: ", "dim"), (f"{cycles}\n", "bold yellow"),
        ("Step Duration: ", "dim"), (f"{step_duration_minutes} minutes\n", "bold yellow"),
        ("New Simulacra: ", "dim"), (f"{new_simulacra}\n", "bold yellow"),
        ("World State File: ", "dim"), (f"'{world_state_file}'\n", "cyan"),
        ("Simulacra Delta File: ", "dim"), (f"'{delta_state_path}'\n", "cyan"),
        ("Simulacra Life Summary: ", "dim"), (f"'{life_summary_path}'\n", "cyan"),
        ("Reaction Profile: ", "dim"), (f"{reaction_profile_arg}", "magenta")
    )
    console.print(Panel(startup_info, title="[bold green]Run Configuration[/bold green]", border_style="green", expand=True))

    # --- Handle File Deletion ---
    if new_simulacra:
         if os.path.exists(delta_state_path):
              try:
                   os.remove(delta_state_path); console.print(f"[yellow]🗑️ Removed {delta_state_path}[/yellow]"); logger.info(f"Removed {delta_state_path} due to --new flag.")
              except OSError as e: console.print(f"[red]⚠️ Error removing {delta_state_path}: {e}[/red]"); logger.error(f"Error removing {delta_state_path}: {e}")

    # --- Initialize World and Simulacra ---
    # ... (Initialization remains the same) ...
    world = None; simulacra = None
    try:
        console.print("\n[cyan]----- INITIALIZING WORLD ----- [/cyan]")
        with console.status("[bold cyan]⚙️ Creating World Engine instance...", spinner="dots"):
            world = await WorldEngine.create( state_file_path=world_state_file, load_state=not new_simulacra, life_summary_path=life_summary_path, console=console, reaction_profile_arg=reaction_profile_arg )
        if not world or not world.world_state or not world.immediate_environment: raise RuntimeError("WorldEngine failed to initialize state.")
        console.print("[green]✅ World Engine Initialized.[/green]")
        console.print("\n[cyan]----- INITIALIZING SIMULACRA ----- [/cyan]")
        with console.status("[bold cyan]⚙️ Creating Simulacra instance...", spinner="dots"):
            simulacra = Simulacra( life_summary_path=life_summary_path, delta_state_path=delta_state_path, console=console, new_simulacra=new_simulacra )
        if not simulacra or not hasattr(simulacra, 'persona') or not simulacra.persona: raise RuntimeError("Simulacra failed to initialize properly.")
        console.print(f"[green]✅ Simulacra Initialized: '[bold magenta]{simulacra.persona.get('name', 'Unknown')}[/bold magenta]'[/green]")
        world.register_agent(simulacra)
    except Exception as init_err: logger.critical("Fatal error during World/Simulacra initialization.", exc_info=True); console.print(f"\n[bold red]❌ Fatal Error during initialization:[/bold red]\n   {init_err}"); return

    # --- Display Initial Info ---
    # ... (Initial display remains the same) ...
    console.rule("[bold green]✨ SIMULATION STARTING ✨[/bold green]")
    reaction_profile_obj = world.reaction_profile; profile_description = reaction_profile_obj.get_description()
    if isinstance(reaction_profile_arg, str): profile_name = reaction_profile_arg.capitalize()
    elif isinstance(reaction_profile_arg, dict): profile_name = "Custom"
    else: profile_name = "Balanced"
    profile_text = Text.assemble( ("Profile Active: ", "bold cyan"), (f"{profile_name}\n\n", "bold magenta"), (profile_description, "dim cyan") )
    console.print(Panel(profile_text, title="[bold cyan]World Reaction Profile[/bold cyan]", border_style="cyan", expand=True))
    console.print("\n[bold]Initial World State:[/bold]"); console.print(world.get_world_summary()); console.print(world.get_environment_summary())
    if world.initial_narrative_context: console.print(Panel( Markdown(world.initial_narrative_context), title="[bold blue]📖 Initial Narrative Context[/bold blue]", border_style="blue", expand=True ))

    # --- Prepare Initial Perception & History Tracking ---
    world_update_for_simulacra = {}; last_action_taken_str: Optional[str] = None
    # Use deque for efficient fixed-size history
    recent_narratives: deque[str] = deque(maxlen=3)
    if world.initial_narrative_context and "[Narrative" not in world.initial_narrative_context: # Add initial narrative if valid
        recent_narratives.append(world.initial_narrative_context)
    try:
        if not world.world_state or not world.immediate_environment: raise RuntimeError("World state invalid before first cycle.")
        world_update_for_simulacra = {
            "world_state": world.world_state,
            "immediate_environment": world.immediate_environment,
            "observations": [ f"You find yourself at {world.immediate_environment.get('current_location_name', 'an unknown location')}.", f"Current time: {world.world_state.get('current_time', '?')} on {world.world_state.get('current_date', '?')}.", f"The atmosphere is {world.immediate_environment.get('social_atmosphere', 'neutral')}.", f"Weather: {world.world_state.get('weather_condition', 'normal')}." ],
             "consequences": ["Started simulation."]
        }
        logger.info("Prepared initial perception data for Simulacra.")
    except Exception as prep_err: logger.error("Error preparing initial perception data.", exc_info=True); console.print(f"[bold red]❌ Error preparing initial state for Simulacra: {prep_err}. Aborting.[/bold red]"); return

    # --- Simulation Loop ---
    logger.info("Starting simulation cycles...")
    console.rule(f"[bold yellow]🔄 STARTING SIMULATION LOOP ({cycles} cycles) 🔄[/bold yellow]")
    for cycle in range(cycles):
        start_cycle_time = time.time()
        console.rule(f"[bold cyan]⏳ CYCLE {cycle+1} / {cycles} START ⏳[/bold cyan]")
        logger.info(f"Beginning Cycle {cycle+1}")

        # <<< Time Advancement Logic >>>
        try:
            if world.world_state and 'current_date' in world.world_state and 'current_time' in world.world_state:
                current_dt_str = f"{world.world_state['current_date']} {world.world_state['current_time']}"
                current_dt_obj = dt.datetime.strptime(current_dt_str, "%Y-%m-%d %H:%M")
                time_delta = dt.timedelta(minutes=step_duration_minutes)
                new_dt_obj = current_dt_obj + time_delta
                world.world_state['current_date'] = new_dt_obj.strftime("%Y-%m-%d")
                world.world_state['current_time'] = new_dt_obj.strftime("%H:%M")
                new_hour = new_dt_obj.hour
                if 5 <= new_hour < 12: world.world_state['day_phase'] = "Morning"
                elif 12 <= new_hour < 17: world.world_state['day_phase'] = "Afternoon"
                elif 17 <= new_hour < 21: world.world_state['day_phase'] = "Evening"
                else: world.world_state['day_phase'] = "Night"
                logger.info(f"Advanced time by {step_duration_minutes} min. New time: {world.world_state['current_time']}, Phase: {world.world_state['day_phase']}")
                console.print(f"[dim]Updated Time: {world.world_state['current_date']} {world.world_state['current_time']} ({world.world_state['day_phase']})[/dim]")
            else: logger.warning("Could not advance time: world_state or time fields missing."); console.print("[yellow]⚠️ Could not advance time: world_state invalid.[/yellow]")
        except Exception as time_err: logger.error(f"Error advancing time: {time_err}. Time state might be incorrect.", exc_info=True); console.print(f"[bold red]⚠️ Error advancing simulation time: {time_err}[/bold red]")
        # <<< End Time Advancement >>>

        try:
            # --- Simulacra Turn ---
            console.print("\n[bold magenta]----- 🤔 SIMULACRA TURN ----- [/bold magenta]")
            logger.debug(f"Cycle {cycle+1}: Simulacra processing perception...")
            # --- <<< MODIFIED: Pass last_action_taken_str to decide_next_action >>> ---
            await simulacra.process_perception(world_update_for_simulacra)
            # Get the full dictionary including internal state
            simulacra_full_response = await simulacra.decide_next_action(
                world_update_for_simulacra,
                step_duration_minutes=step_duration_minutes,
                last_action_taken=last_action_taken_str # Pass the previous action string
            )
            logger.debug(f"Cycle {cycle+1}: Simulacra full response generated: Action='{simulacra_full_response.get('action')}', Reflection='{simulacra_full_response.get('reflection', '')[:30]}...'")

            # Extract action parts for world processing and logging
            simulacra_action_part = {
                "action": simulacra_full_response.get('action', 'wait'),
                "action_details": simulacra_full_response.get('action_details')
            }
            # Extract internal state for passing to world engine
            initiator_reflection = simulacra_full_response.get('reflection')
            initiator_thought_process = simulacra_full_response.get('thought_process')

            # --- Update last_action_taken_str for next cycle ---
            action_verb = simulacra_action_part['action']
            action_details = simulacra_action_part.get('action_details')
            # ... (formatting last_action_taken_str remains the same) ...
            if action_verb == 'wait' and not action_details: # Check if it was a forced wait
                 logger.error(f"Cycle {cycle+1}: Simulacra failed to produce a valid action. Response: {simulacra_full_response}")
                 console.print("[bold red]⚠️ Simulacra response invalid or failed. Skipping cycle remainder.[/bold red]")
                 last_action_taken_str = None # Reset if action failed
                 continue # Skip to next cycle

            details_str = ""
            if isinstance(action_details, dict):
                details_parts = [f"{k}={v}" for k, v in action_details.items() if v is not None]
                if details_parts: details_str = f" ({', '.join(details_parts)})"
            elif isinstance(action_details, str): details_str = f" ({action_details})"
            last_action_taken_str = f"{action_verb}{details_str}" # Store formatted action string

            console.print(f"\n[bold magenta]Cycle {cycle+1}: Simulacra decided action: {last_action_taken_str}[/bold magenta]")
            # Optional: Print reflection/thought process here too if desired for debugging
            # console.print(f"[dim magenta]  Reflection: {initiator_reflection}[/dim magenta]")
            # console.print(f"[dim magenta]  Thought: {initiator_thought_process}[/dim magenta]")

            # --- World Turn ---
            console.print("\n[bold blue]----- 🌍 WORLD TURN ----- [/bold blue]")
            action_panel_content = Text.assemble(
                 ("Action Received: ", "dim blue"), (f"{simulacra_action_part['action']}\n", "bold"),
                 ("Details: ", "dim blue"), (f"{json.dumps(simulacra_action_part.get('action_details', {}))}", "")
            )
            console.print(Panel(action_panel_content, title="[bold blue]Processing Action[/bold blue]", border_style="blue", expand=True))
            logger.debug(f"Cycle {cycle+1}: World processing action: {simulacra_action_part['action']}")

            with console.status("[bold cyan]⚙️ World Engine processing update...", spinner="dots"):
                 # Pass the extracted action AND internal state to process_update
                 world_update_result = await world.process_update(
                     initiator_action=simulacra_action_part, # Pass only the action part here
                     initiator_persona=simulacra.persona,
                     step_duration_minutes=step_duration_minutes,
                     recent_narrative_updates=list(recent_narratives),
                     initiator_reflection=initiator_reflection,       # Pass reflection
                     initiator_thought_process=initiator_thought_process # Pass thought process
                 )
                 # --- <<< END MODIFICATION >>> ---
            logger.debug(f"Cycle {cycle+1}: World update processed.")

            if not world_update_result or 'world_state' not in world_update_result:
                logger.error(f"Cycle {cycle+1}: World Engine failed to produce a valid update. Response: {world_update_result}")
                console.print("[bold red]⚠️ World Engine update invalid. Skipping cycle remainder.[/bold red]")
                continue
            console.print("[green]✅ World update processed.[/green]")
            # --- End World Turn ---

            # --- Prepare Perception for Next Cycle & Update History ---
            world_update_for_simulacra = world_update_result # Pass the full result
            # --- <<< ADDED: Update recent_narratives history >>> ---
            new_narrative = world_update_result.get("narrative_update")
            if new_narrative and isinstance(new_narrative, str) and "[Narrative" not in new_narrative:
                recent_narratives.append(new_narrative) # Add to deque (automatically handles maxlen)
                logger.debug(f"Added to narrative history (size {len(recent_narratives)}): {new_narrative[:80]}...")
            # --- <<< END ADDED >>> ---
            # --- End Prep & History ---

            # --- Display Cycle Results ---
            # ... (Display logic remains the same) ...
            console.print("\n[bold purple]----- 📊 CYCLE RESULTS ----- [/bold purple]")
            panel_content = []
            narrative_update_text = world_update_result.get("narrative_update", "[italic dim]No narrative update generated.[/italic dim]")
            panel_content.append(Rule("[bold blue]📖 Narrative Update[/bold blue]", style="blue"))
            panel_content.append(Markdown(narrative_update_text, justify="left"))
            panel_content.append("")
            panel_content.append(Rule("[bold green]📊 World Status Update[/bold green]", style="green"))
            panel_content.append(world.get_world_summary())
            panel_content.append(world.get_environment_summary())
            panel_content.append("")
            panel_content.append(Rule("[bold yellow]⚡👀 Perception Details[/bold yellow]", style="yellow"))
            consequences = world_update_result.get("consequences", [])
            observations = world_update_result.get("observations", [])
            if consequences: panel_content.append("[bold green]Consequences:[/bold green]"); [panel_content.append(f"- {item}") for item in consequences]
            else: panel_content.append("[italic green]No notable consequences.[/italic green]")
            panel_content.append("")
            if observations:
                panel_content.append("[bold yellow]Observations:[/bold yellow]")
                for item in observations:
                    if isinstance(item, dict) and item.get("type") == "dialogue": panel_content.append(Text.assemble(("- ", "dim"), (f"{item.get('from', '?')}: ", "bold"), f'"{item.get("utterance", "...")}"'))
                    elif isinstance(item, str): panel_content.append(f"- {item}")
                    else: panel_content.append(f"- [dim](Unrecognized observation format: {type(item)})[/dim]")
            else: panel_content.append("[italic yellow]Nothing specific observed.[/italic yellow]")
            console.print(Panel( Group(*panel_content), title=f"[bold purple]Cycle {cycle+1} Summary[/bold purple]", border_style="purple", expand=True ))
            # --- End Display ---

        except Exception as cycle_err:
             logger.error(f"Error during cycle {cycle+1}", exc_info=True)
             console.print(f"\n[bold red]❌ An unexpected error occurred during cycle {cycle+1}:[/bold red]\n   {cycle_err}")
             console.print("   Attempting to continue to the next cycle. Check logs.")

        finally:
            end_cycle_time = time.time()
            cycle_duration = end_cycle_time - start_cycle_time
            logger.info(f"Cycle {cycle+1} finished in {cycle_duration:.2f} seconds.")
            console.rule(f"[bold dim cyan] CYCLE {cycle+1} END ({cycle_duration:.2f}s) [/bold dim cyan]")
        time.sleep(1)
    # --- End Simulation Loop ---
    # ... (Final save and conclusion remain the same) ...
    console.rule("[bold yellow]🏁 SIMULATION LOOP COMPLETE 🏁[/bold yellow]")
    logger.info("Simulation loop finished. Performing final state save.")
    try:
        with console.status("[bold cyan]💾 Saving final states...", spinner="dots"):
            if world: world.save_state()
            if simulacra: simulacra.save_state()
        logger.info("Final states saved successfully.")
        console.print("[green]✅ Final states saved.[/green]")
    except Exception as save_err:
        logger.error("Error during final state save.", exc_info=True)
        console.print(f"[bold red]⚠️ Error saving final simulation state: {save_err}[/bold red]")
    final_narrative = world_update_for_simulacra.get("narrative_update", "[No final narrative update available]") if 'world_update_for_simulacra' in locals() else "[Simulation ended abruptly]"
    console.print(Panel( Markdown(f"**Final Narrative State:**\n\n{final_narrative}"), title="[bold blue]📜 STORY CONCLUSION[/bold blue]", border_style="blue", expand=True ))
    console.print(Panel("[bold green]🏁 SIMULATION ENDED NORMALLY 🏁[/bold green]", border_style="green", expand=True))
    end_run_time = time.time()
    total_duration = end_run_time - start_run_time
    logger.info(f"Simulation run finished. Total duration: {total_duration:.2f} seconds.")
    console.print(f"[dim]Total runtime: {total_duration:.2f} seconds.[/dim]")
    # --- End Final Save & Conclusion ---


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    # ... (argument parsing remains the same, includes --step-duration) ...
    parser = argparse.ArgumentParser( description='Run a modular simulacra simulation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( '-c', '--cycles', type=int, default=10, help='Number of simulation cycles to run' )
    parser.add_argument( '--new', action='store_true', help='Start with a new Simulacra persona (ignores delta file) and ignore existing world state.' )
    parser.add_argument( '--world-state-file', type=str, default='simulacra_state.json', help='Path to load/save WorldEngine state & config' )
    parser.add_argument( '--delta-state', type=str, default='simulacra_deltas.json', help='Path to load/save Simulacra runtime state (persona + history)' )
    parser.add_argument( '--life-summary', type=str, default='life_summary_Eleanor_Vance_32_generated.json', help='Path to a generated life summary JSON to initialize Simulacra persona from (if delta file is missing or --new is used)' )
    parser.add_argument( '--step-duration', type=int, default=60, help='Duration of each simulation step in minutes.' )
    profile_choices = ['balanced']
    try:
        profile_choices = list(WorldReactionProfile.create_profile.__func__.__globals__.get('profile_overrides', {}).keys())
        profile_choices.append('balanced')
        if not profile_choices or profile_choices == ['balanced']: profile_choices = ['balanced', 'protagonist', 'realistic']
    except Exception as e: logger.warning(f"Could not dynamically determine reaction profile choices: {e}. Using defaults."); profile_choices = ['balanced', 'protagonist', 'realistic', 'bustling_city', 'quiet_village']
    parser.add_argument( '--profile', type=str, default='balanced', choices=sorted(list(set(profile_choices))), help='World reaction profile preset name' )
    args = parser.parse_args()

    # --- Console Setup ---
    main_console = Console(record=True)
    # --- End Console Setup ---

    # --- Run Simulation ---
    # ... (run_simulation call remains the same, passing args.step_duration) ...
    exit_code = 0
    try:
        asyncio.run(run_simulation(
            cycles=args.cycles,
            world_state_file=args.world_state_file,
            delta_state_path=args.delta_state,
            life_summary_path=args.life_summary,
            new_simulacra=args.new,
            console=main_console,
            reaction_profile_arg=args.profile,
            step_duration_minutes=args.step_duration # Pass the parsed argument
        ))
    except KeyboardInterrupt:
         main_console.print("\n[bold orange3]🚦 Simulation interrupted by user.[/bold orange3]")
         logger.warning("Simulation interrupted by user (KeyboardInterrupt).")
         exit_code = 130
    except Exception as main_err:
         logger.critical("Main simulation execution crashed.", exc_info=True)
         main_console.print(f"\n[bold red]❌ A critical error occurred during simulation execution:[/bold red]")
         main_console.print_exception(show_locals=False, width=120)
         main_console.print(f"[bold red]Check '{LOG_FILE_PATH}' for detailed logs.[/bold red]")
         exit_code = 1
    finally:
        logger.info("Main script execution finished.")
        # ... (optional console saving logic) ...
        print(f"\nExiting simulation. (Exit Code: {exit_code})")
        exit(exit_code)