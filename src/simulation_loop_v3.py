# src/simulation_loop_v3.py

import asyncio
import json # <<< Added for parsing agent output
import logging
from typing import Any, Dict, List, Optional

from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, Session
from google.genai import types
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax # <<< Added for pretty printing JSON

# --- Agent Imports ---
# <<< Import the new interaction resolver factory >>>
from src.agents.interaction_resolver_v3 import \
    create_agent as create_interaction_resolver_agent

# --- State Keys (Import centrally if defined elsewhere) ---
# Assuming these keys are defined and accessible
WORLD_STATE_KEY = "current_world_state"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
TURN_VALIDATION_RESULTS_KEY = "turn_validation_results"
WORLD_TIME_KEY = "world_time"
TURN_INTERACTION_LOG_KEY = "turn_interaction_log" # Key to store narrative log
# Assuming intent key format is needed for display
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent"

logger = logging.getLogger(__name__)
console = Console()

async def run_phased_simulation(
    runner: Runner,
    session_service: BaseSessionService,
    session: Session,
    world_state_agent: BaseAgent,
    simulacra_agents: Dict[str, BaseAgent], # Expects SequentialAgents now
    validator_agents: Dict[str, BaseAgent],
    # npc_agent: BaseAgent, # <<< Removed old npc_agent parameter >>>
    world_execution_agent: BaseAgent,
    narration_agent: BaseAgent,
    max_turns: int
) -> Optional[Dict[str, Any]]:
    """
    Runs the main simulation loop with distinct phases for V3 architecture.

    Args:
        runner: The ADK Runner instance.
        session_service: The session service managing state.
        session: The initial ADK Session object.
        world_state_agent: Agent responsible for updating world state (time, etc.).
        simulacra_agents: Dict mapping sim_id to their sequential planning agent.
        validator_agents: Dict mapping sim_id to their validation agent.
        world_execution_agent: Agent handling execution of physical actions (moves).
        narration_agent: Agent generating narrative summaries.
        max_turns: Maximum number of simulation turns to run.

    Returns:
        The final simulation state dictionary, or None if an error occurred.
    """
    current_state = session.state
    session_id = session.id
    user_id = session.user_id

    # <<< Instantiate the new interaction resolver agent >>>
    try:
        interaction_resolver_agent = create_interaction_resolver_agent()
        logger.info(f"Instantiated Interaction Resolver Agent: {interaction_resolver_agent.name}")
    except Exception as e:
        logger.critical(f"Failed to instantiate Interaction Resolver Agent: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to create Interaction Resolver Agent.")
        return current_state # Return current state before loop

    for turn in range(1, max_turns + 1):
        console.rule(f"[bold cyan]Turn {turn}/{max_turns}[/]", style="cyan")
        logger.info(f"--- Starting Turn {turn} ---")

        # Initialize turn-specific state keys if they don't exist
        if TURN_INTERACTION_LOG_KEY not in current_state:
            current_state[TURN_INTERACTION_LOG_KEY] = []
        if TURN_VALIDATION_RESULTS_KEY not in current_state:
             current_state[TURN_VALIDATION_RESULTS_KEY] = {}


        try:
            # --- Phase 1: World State Update ---
            console.print(Panel("Phase 1: World State Update", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 1: Running World State Agent.")
            runner.agent = world_state_agent
            trigger_msg = f"Update world state for turn {turn}."
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg)])):
                if event.actions and event.actions.state_delta:
                    logger.debug(f"World State Delta: {event.actions.state_delta}")
                if event.error_message:
                    logger.error(f"Error in World State Agent: {event.error_message}")
                    console.print(f"[red]Error in World State Agent: {event.error_message}[/red]")
            session = session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
            if not session: raise RuntimeError("Session lost after World State update.")
            current_state = session.state
            logger.info(f"World Time updated to: {current_state.get(WORLD_STATE_KEY, {}).get(WORLD_TIME_KEY, 'N/A')}")
            console.print(f"  [dim]World Time:[/dim] {current_state.get(WORLD_STATE_KEY, {}).get(WORLD_TIME_KEY, 'N/A')}")

            # --- Phase 2: Simulacra Internal Planning ---
            console.print(Panel("Phase 2: Simulacra Planning/Introspection", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 2: Running Simulacra Planning Agents.")
            active_sim_ids = current_state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
            if not active_sim_ids:
                logger.warning("No active simulacra found in state. Skipping planning phase.")
                console.print("[yellow]Warning:[/yellow] No active simulacra. Skipping planning.")
            else:
                planning_sub_agents = [simulacra_agents[sim_id] for sim_id in active_sim_ids if sim_id in simulacra_agents]
                if planning_sub_agents:
                    parallel_planning_agent = ParallelAgent(
                        name=f"ParallelSimPlanning_Turn{turn}", sub_agents=planning_sub_agents,
                        description="Runs internal planning for all active simulacra concurrently."
                    )
                    runner.agent = parallel_planning_agent
                    trigger_msg = f"Begin internal planning for turn {turn}."
                    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg)])):
                        # Optional: Log intermediate planning steps if needed
                        # if event.content and not event.is_final_response():
                        #    logger.debug(f"Planning Step {event.author}: {event.content.parts[0].text[:100]}...")
                        if event.actions and event.actions.state_delta:
                             logger.info(f"Planning State Delta from {event.author}: {event.actions.state_delta}")
                        if event.error_message:
                            logger.error(f"Error in Planning Agent {event.author}: {event.error_message}")
                            console.print(f"[red]Error in Planning ({event.author}): {event.error_message}[/red]")
                    session = session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
                    if not session: raise RuntimeError("Session lost after Simulacra Planning.")
                    current_state = session.state
                    logger.info("Simulacra planning phase complete.")
                    console.print("  [green]Simulacra planning complete.[/green]")
                    # <<< Print generated intents >>>
                    console.print("  [bold]Generated Intents:[/bold]")
                    for sim_id in active_sim_ids:
                        intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id)
                        intent = current_state.get(intent_key)
                        if intent:
                            try:
                                # Pretty print JSON intent
                                intent_str = json.dumps(intent, indent=2)
                                syntax = Syntax(intent_str, "json", theme="default", line_numbers=False)
                                console.print(f"    [cyan]{sim_id}:[/cyan]")
                                console.print(syntax)
                            except Exception: # Fallback if not valid JSON or other error
                                console.print(f"    [cyan]{sim_id}:[/cyan] {intent}")
                        else:
                            console.print(f"    [cyan]{sim_id}:[/cyan] [dim](No intent generated)[/dim]")
                    # <<< End Print >>>
                else:
                    logger.warning("No valid simulacra agents found for planning phase.")
                    console.print("[yellow]Warning:[/yellow] No valid simulacra agents for planning.")

            # --- Phase 3: Intent Validation ---
            console.print(Panel("Phase 3: Intent Validation", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 3: Running Validation Agents.")
            validation_sub_agents = [validator_agents[sim_id] for sim_id in active_sim_ids if sim_id in validator_agents]
            if validation_sub_agents:
                parallel_validation_agent = ParallelAgent(
                    name=f"ParallelValidation_Turn{turn}", sub_agents=validation_sub_agents,
                    description="Runs validation for all active simulacra intents concurrently."
                )
                runner.agent = parallel_validation_agent
                trigger_msg = f"Validate intents for turn {turn}."
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg)])):
                    if event.actions and event.actions.state_delta:
                         logger.info(f"Validation State Delta from {event.author}: {event.actions.state_delta}")
                    if event.error_message:
                        logger.error(f"Error in Validation Agent {event.author}: {event.error_message}")
                        console.print(f"[red]Error in Validation ({event.author}): {event.error_message}[/red]")
                session = session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
                if not session: raise RuntimeError("Session lost after Validation.")
                current_state = session.state
                logger.info("Validation phase complete.")
                console.print("  [green]Intent validation complete.[/green]")
                # <<< Print validation summary >>>
                validation_results = current_state.get(TURN_VALIDATION_RESULTS_KEY, {})
                console.print("  [bold]Validation Results:[/bold]")
                if validation_results:
                    for sim_id, result in validation_results.items():
                        status = "[green]Valid[/green]" if result.get('is_valid') else "[red]Invalid[/red]"
                        reason = f" ([dim]{result.get('reasoning', 'No reason given')}[/dim])" if not result.get('is_valid') else ""
                        intent_desc = result.get('validated_intent', {}).get('action_type', 'Unknown Action')
                        console.print(f"    [cyan]{sim_id}:[/cyan] {status} - {intent_desc}{reason}")
                else:
                    console.print("    [dim](No validation results found)[/dim]")
                # <<< End Print >>>
            else:
                logger.warning("No validator agents found for validation phase.")
                console.print("[yellow]Warning:[/yellow] No validator agents found.")
                current_state[TURN_VALIDATION_RESULTS_KEY] = {}

            # --- Phase 4: Interaction Resolution (Talk/Use Actions) ---
            console.print(Panel("Phase 4: Interaction Resolution", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 4: Running Interaction Resolution.")
            validation_results = current_state.get(TURN_VALIDATION_RESULTS_KEY, {})
            interaction_intents_exist = any(
                result.get('is_valid') and result.get('validated_intent', {}).get('action_type') in ['talk', 'use']
                for result in validation_results.values()
            )

            if interaction_intents_exist:
                logger.info("Valid 'talk' or 'use' intents found, running Interaction Resolver.")
                runner.agent = interaction_resolver_agent # Use the new agent
                trigger_msg = f"Resolve 'talk' and 'use' interactions for turn {turn} based on validation results."

                resolver_output_json = None
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg)])):
                    if event.is_final_response() and event.content:
                        resolver_output_json = event.content.parts[0].text
                        logger.info(f"Interaction Resolver Output: {resolver_output_json}")
                    if event.error_message:
                        logger.error(f"Error in Interaction Resolver Agent {event.author}: {event.error_message}")
                        console.print(f"[red]Error in Interaction Resolver ({event.author}): {event.error_message}[/red]")

                # --- Process the Resolver's Output ---
                if resolver_output_json:
                    # <<< Add cleaning step here too, just in case >>>
                    cleaned_output = resolver_output_json.strip()
                    if cleaned_output.startswith("```json"): cleaned_output = cleaned_output[7:].strip()
                    elif cleaned_output.startswith("```"): cleaned_output = cleaned_output[3:].strip()
                    if cleaned_output.endswith("```"): cleaned_output = cleaned_output[:-3].strip()
                    # <<< End cleaning step >>>
                    try:
                        output_data = json.loads(cleaned_output) # Use cleaned output
                        state_updates = output_data.get("state_updates", {})
                        narrative_log = output_data.get("narrative_log", [])

                        # Apply state updates
                        if state_updates:
                            logger.info(f"Applying {len(state_updates)} state updates from Interaction Resolver.")
                            console.print("  [bold]Interaction State Updates:[/bold]") # <<< Print Updates >>>
                            for key, value in state_updates.items():
                                console.print(f"    [dim]Updating {key}:[/dim] {value}") # <<< Print Update Detail >>>
                                if isinstance(current_state.get(key), dict) and isinstance(value, dict):
                                    current_state[key].update(value)
                                else:
                                    current_state[key] = value
                            session_service.update_session(session_id=session_id, state=current_state)
                            logger.info("Session state updated after applying interaction results.")
                        else:
                            logger.info("Interaction Resolver provided no state updates.")
                            console.print("    dim[/dim]") # <<< Print No Updates >>>

                        # Append narrative log
                        if narrative_log:
                            logger.info(f"Appending {len(narrative_log)} entries to interaction log.")
                            console.print("  [bold]Interaction Narrative Log:[/bold]") # <<< Print Log >>>
                            if TURN_INTERACTION_LOG_KEY not in current_state:
                                current_state[TURN_INTERACTION_LOG_KEY] = []
                            current_state[TURN_INTERACTION_LOG_KEY].extend(narrative_log)
                            for entry in narrative_log:
                                console.print(f"    - {entry}") # <<< Print Log Entry >>>
                        else:
                             logger.info("Interaction Resolver provided no narrative log entries.")
                             console.print("    dim[/dim]") # <<< Print No Log >>>

                    except json.JSONDecodeError as json_err: # <<< Use updated error handling >>>
                        error_message = f"Failed to decode JSON output from Interaction Resolver. Error: {json_err}. repr(cleaned_output):\n>>>\n{repr(cleaned_output)}\n<<<"
                        logger.error(error_message)
                        console.print(f"[bold red]Error:[/bold red] Failed to parse Interaction Resolver output.\nError: {json_err}\nAttempted to parse (repr):\n>>>\n{repr(cleaned_output)}\n<<<")
                    except Exception as proc_e:
                         logger.error(f"Error processing Interaction Resolver output: {proc_e}", exc_info=True)
                         console.print(f"[red]Error:[/red] Failed processing Interaction Resolver output: {proc_e}")
                else:
                    logger.warning("No final output received from Interaction Resolver agent.")
                    console.print("[yellow]Warning:[/yellow] No output from Interaction Resolver.")

                # Fetch the potentially updated state again
                session = session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
                if not session: raise RuntimeError("Session lost after Interaction Resolution processing.")
                current_state = session.state
                logger.info("Interaction resolution phase complete.")
                console.print("  [green]Interaction resolution complete.[/green]")

            else:
                logger.info("No valid 'talk' or 'use' intents found for this turn.")
                console.print("  [dim]No interactions to resolve this turn.[/dim]")


            # --- Phase 5: Execution (Move Actions) ---
            console.print(Panel("Phase 5: Action Execution (Moves)", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 5: Running World Execution Agent.")
            validation_results = current_state.get(TURN_VALIDATION_RESULTS_KEY, {})
            # <<< Corrected filtering for move intents >>>
            move_intents = {
                sim_id: result['validated_intent']
                for sim_id, result in validation_results.items()
                if result.get('is_valid') and result.get('validated_intent', {}).get('action_type') == 'move'
            }

            if move_intents: # Check if the dictionary is not empty
                logger.info(f"Valid 'move' intents found ({len(move_intents)}), running Execution Agent.")
                runner.agent = world_execution_agent
                trigger_msg = f"Execute 'move' actions for turn {turn} based on validation results."
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg)])):
                    if event.actions and event.actions.state_delta:
                        logger.info(f"Execution State Delta from {event.author}: {event.actions.state_delta}")
                    if event.error_message:
                        logger.error(f"Error in Execution Agent {event.author}: {event.error_message}")
                        console.print(f"[red]Error in Execution ({event.author}): {event.error_message}[/red]")
                session = session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
                if not session: raise RuntimeError("Session lost after Execution.")
                current_state = session.state
                logger.info("Execution phase complete.")
                console.print("  [green]Action execution complete.[/green]")
                # <<< Print executed moves >>>
                console.print("  [bold]Executed Moves:[/bold]")
                for sim_id, intent in move_intents.items():
                    dest = intent.get('destination', 'Unknown')
                    console.print(f"    - [cyan]{sim_id}[/cyan] moved towards {dest}")
                # <<< End Print >>>
            else:
                logger.info("No valid move intents found for this turn.")
                console.print("  [dim]No moves to execute this turn.[/dim]")


            # --- Phase 6: Narration ---
            console.print(Panel("Phase 6: Narration", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 6: Running Narration Agent.")
            runner.agent = narration_agent
            trigger_msg = f"Narrate the events of turn {turn}."
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg)])):
                if event.is_final_response() and event.content:
                    narrative = event.content.parts[0].text
                    logger.info(f"Turn {turn} Narrative: {narrative}")
                    # <<< Print narrative using Panel >>>
                    console.print(Panel(narrative, title=f"Turn {turn} Summary", border_style="green", expand=False))
                if event.error_message:
                    logger.error(f"Error in Narration Agent: {event.error_message}")
                    console.print(f"[red]Error in Narration: {event.error_message}[/red]")
            session = session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
            if not session: raise RuntimeError("Session lost after Narration.")
            current_state = session.state
            logger.info(f"--- End of Turn {turn} ---")

            # Optional delay between turns
            # await asyncio.sleep(1)

        except Exception as e:
            logger.critical(f"Critical error during turn {turn}: {e}", exc_info=True)
            console.print(f"[bold red]\n--- Critical Error during Turn {turn} ---[/]")
            console.print_exception(show_locals=False)
            return current_state # Return state at time of error

    logger.info(f"Simulation finished after {max_turns} turns.")
    return current_state
