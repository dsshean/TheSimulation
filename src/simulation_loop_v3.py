# src/simulation_loop_v3.py

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

# ADK Imports
from google.adk.events import Event, EventActions # Ensure these are imported
from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, Session
from google.genai import types

# Rich Imports
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax

# --- Agent Imports ---
from src.agents.interaction_resolver_v3 import \
    create_agent as create_interaction_resolver_agent

# --- Utility Imports ---
from src.loop_utils import (format_localized_timestamp,
                            get_timezone_from_location, parse_json_output, parse_json_output_last)
from timezonefinder import TimezoneFinder # Keep import here if instantiating tf here

# --- State Keys ---
WORLD_STATE_KEY = "current_world_state"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
# <<< Use the CONSISTENT format string for validation results >>>
VALIDATION_RESULT_KEY_FORMAT = "simulacra_{sim_id}_validation_result" # Correct format
# <<< REMOVE combined key definition (not used in this approach) >>>
# TURN_VALIDATION_RESULTS_KEY = "turn_validation_results"
WORLD_TIME_KEY = "world_time"
TURN_INTERACTION_LOG_KEY = "turn_interaction_log"
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{sim_id}_intent"
WORLD_TEMPLATE_DETAILS_KEY = "world_template_details"
LOCATION_KEY = "location"
CITY_KEY = "city"
STATE_KEY = "state"
COUNTRY_KEY = "country"
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
TURN_EXECUTION_NARRATIVES_KEY = "turn_execution_narratives"
SIMULACRA_NARRATION_KEY_FORMAT = "simulacra_{}_last_narration"

logger = logging.getLogger(__name__)
console = Console()
tf = TimezoneFinder()

# --- Main Simulation Loop Function ---
async def run_phased_simulation(
    runner: Runner,
    session_service: BaseSessionService,
    session: Session,
    world_state_agent: BaseAgent,
    simulacra_agents: Dict[str, BaseAgent],
    validator_agents: Dict[str, BaseAgent],
    world_execution_agent: BaseAgent,
    narration_agent: BaseAgent,
    max_turns: int
) -> Optional[Dict[str, Any]]:
    """
    Runs the main simulation loop with distinct phases for V3 architecture.
    Uses individual output_keys for saving validation results.
    Attempts to parse results when reading from state.
    """
    current_state = session.state
    session_id = session.id
    user_id = session.user_id
    app_name = runner.app_name

    try:
        interaction_resolver_agent = create_interaction_resolver_agent()
        logger.info(f"Instantiated Interaction Resolver Agent: {interaction_resolver_agent.name}")
    except Exception as e:
        logger.critical(f"Failed to instantiate Interaction Resolver Agent: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] Failed to create Interaction Resolver Agent.")
        return current_state

    for turn in range(1, max_turns + 1):
        console.rule(f"[bold cyan]Turn {turn}/{max_turns}[/]", style="cyan")
        logger.info(f"--- Starting Turn {turn} ---")

        # Initialize turn-specific state keys
        if TURN_INTERACTION_LOG_KEY not in current_state:
            current_state[TURN_INTERACTION_LOG_KEY] = []
        # <<< No need to initialize a combined key anymore >>>
        # if TURN_VALIDATION_RESULTS_KEY not in current_state:
        #      current_state[TURN_VALIDATION_RESULTS_KEY] = {}
        if TURN_EXECUTION_NARRATIVES_KEY not in current_state:
             current_state[TURN_EXECUTION_NARRATIVES_KEY] = {}

        primary_location_details: Optional[Dict[str, Any]] = None
        sim_timezone_str: str = 'UTC' # Default fallback

        try:
            # --- Phase 1: World State Update ---
            # (Keep Phase 1 logic as is)
            console.print(Panel("Phase 1: World State Update", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 1: Running World State Agent.")
            world_details = current_state.get(WORLD_TEMPLATE_DETAILS_KEY, {})
            primary_location_details = world_details.get(LOCATION_KEY)
            if not primary_location_details:
                 logger.warning("Could not determine primary location details for Phase 1.")
            runner.agent = world_state_agent
            trigger_msg_p1 = f"Update world state for turn {turn}."
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg_p1)])):
                console.print(event)
                if event.actions and event.actions.state_delta: logger.debug(f"P1 Delta: {event.actions.state_delta}")
                if event.error_message: logger.error(f"P1 Error: {event.error_message}")
            session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
            if not session: raise RuntimeError("Session lost after World State update.")
            current_state = session.state
            world_time_iso = current_state.get(WORLD_STATE_KEY, {}).get(WORLD_TIME_KEY)
            found_tz = get_timezone_from_location(primary_location_details, tf_instance=tf)
            if found_tz: sim_timezone_str = found_tz
            else: logger.warning(f"Could not determine timezone for primary location. Falling back to UTC.")
            localized_time_str = format_localized_timestamp(world_time_iso, sim_timezone_str)
            logger.info(f"World Time updated to: {world_time_iso} (Localized: {localized_time_str})")
            console.print(f"  [dim]World Time:[/dim] {localized_time_str}")
            # --- End Phase 1 ---

            # --- Phase 2: Simulacra Internal Planning ---
            # (Keep Phase 2 logic as is, including intent parsing and synthetic event for intents)
            console.print(Panel("Phase 2: Simulacra Planning/Introspection", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 2: Running Simulacra Planning Agents.")
            active_sim_ids = current_state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
            if not active_sim_ids:
                logger.warning("No active simulacra found. Skipping planning.")
                console.print("[yellow]Warning:[/yellow] No active simulacra. Skipping planning.")
            else:
                planning_sub_agents = [simulacra_agents[sim_id] for sim_id in active_sim_ids if sim_id in simulacra_agents]
                if planning_sub_agents:
                    parallel_planning_agent = ParallelAgent(name=f"ParallelSimPlanning_Turn{turn}", sub_agents=planning_sub_agents)
                    runner.agent = parallel_planning_agent
                    trigger_msg_p2 = f"Begin internal planning for turn {turn}. Current time context: {localized_time_str}."
                    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg_p2)])):
                        console.print(event)
                        if event.actions and event.actions.state_delta: logger.info(f"P2 Delta from {event.author}: {event.actions.state_delta}")
                        if event.error_message: logger.error(f"P2 Error ({event.author}): {event.error_message}")
                    session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
                    if not session: raise RuntimeError("Session lost after Simulacra Planning.")
                    current_state = session.state
                    logger.info("Simulacra planning phase complete.")
                    console.print("  [green]Simulacra planning complete.[/green]")

                    console.print("  [bold]Processing Generated Intents:[/bold]")
                    parsed_intents_delta: Dict[str, Any] = {}
                    for sim_id in active_sim_ids:
                        intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id=sim_id)
                        raw_intent_value = current_state.get(intent_key)
                        if isinstance(raw_intent_value, str):
                            logger.info(f"Attempting to parse intent string for {sim_id}...")
                            parsed_intent = parse_json_output(raw_intent_value, "P2-IntentParse", f"Sim_{sim_id}_DecideIntent", console, logger)
                            if parsed_intent is not None:
                                parsed_intents_delta[intent_key] = parsed_intent
                                logger.info(f"Successfully parsed intent for {sim_id} (to be updated).")
                                try:
                                    syntax = Syntax(json.dumps(parsed_intent, indent=2), "json", theme="default", line_numbers=False)
                                    console.print(f"    [cyan]{sim_id} (Parsed):[/cyan]"); console.print(syntax)
                                except Exception: console.print(f"    [cyan]{sim_id} (Parsed):[/cyan] {parsed_intent}")
                            else: console.print(f"    [cyan]{sim_id}:[/cyan] [red]Failed to parse intent string (see logs).[/red]")
                        elif isinstance(raw_intent_value, dict):
                            logger.debug(f"Intent for {sim_id} is already a dictionary.")
                            try:
                                syntax = Syntax(json.dumps(raw_intent_value, indent=2), "json", theme="default", line_numbers=False)
                                console.print(f"    [cyan]{sim_id} (Existing Dict):[/cyan]"); console.print(syntax)
                            except Exception: console.print(f"    [cyan]{sim_id} (Existing Dict):[/cyan] {raw_intent_value}")
                        elif raw_intent_value is None: console.print(f"    [cyan]{sim_id}:[/cyan] [dim](No intent generated)[/dim]")
                        else: console.print(f"    [cyan]{sim_id}:[/cyan] [yellow]Unexpected intent type:[/yellow] {type(raw_intent_value)}")

                    if parsed_intents_delta:
                        logger.info("Applying parsed intent updates via synthetic event.")
                        update_event = Event(author="IntentParser", actions=EventActions(state_delta=parsed_intents_delta))
                        session_service.append_event(session=session, event=update_event)
                        logger.info("Synthetic event appended to update state with parsed intents.")
                        session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
                        if not session: raise RuntimeError("Session lost after applying parsed intent delta.")
                        current_state = session.state
                        logger.info("State refreshed after applying parsed intent delta.")
                    else: logger.info("No intents needed parsing or updating.")
                else: logger.warning("No valid simulacra agents found for planning phase.")
            # --- End Phase 2 ---

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
                trigger_msg_p3 = f"Validate intents for turn {turn}."

                # <<< Run the parallel agent - state updates handled by output_key >>>
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg_p3)])):
                    # console.print(event) # Keep console print if desired
                    # Log delta if present (should contain the individual validation result as a string)
                    if event.actions and event.actions.state_delta:
                         # Log the raw delta being applied automatically by output_key
                         logger.info(f"P3 Raw State Delta Applied (via output_key) from {event.author}: {event.actions.state_delta}")
                    if event.error_message:
                        logger.error(f"Error in Validation Agent {event.author}: {event.error_message}")
                        console.print(f"[red]Error in Validation ({event.author}): {event.error_message}[/red]")

                # <<< START: Post-processing to parse and update state >>>
                logger.info("Phase 3: Post-processing validation results...")
                # Fetch the state again AFTER the parallel run completes and output_key deltas are applied
                session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
                if not session: raise RuntimeError("Session lost after Validation.")
                current_state = session.state

                parsed_validation_updates: Dict[str, Optional[Dict[str, Any]]] = {}
                for sim_id in active_sim_ids:
                    result_key = VALIDATION_RESULT_KEY_FORMAT.format(sim_id=sim_id)
                    raw_result = current_state.get(result_key) # Get individual result (should be string now)

                    if isinstance(raw_result, str):
                        # Attempt to parse the string saved by output_key
                        logger.info(f"Attempting to parse validation result string for {sim_id} from key '{result_key}'")
                        parsed_dict = parse_json_output_last(raw_result, "P3-PostParse", f"Validator_{sim_id}", console, logger)
                        if parsed_dict is not None:
                            # Store the parsed dictionary for update
                            parsed_validation_updates[result_key] = parsed_dict
                            logger.info(f"Successfully parsed validation result for {sim_id}. Queued for state update.")
                        else:
                            # Parsing failed, store None to clear potentially bad data
                            parsed_validation_updates[result_key] = None
                            logger.warning(f"Failed to parse validation result string for {sim_id} from key '{result_key}'. Queued None for state update.")
                            console.print(f"    [cyan]{sim_id}:[/cyan] [red]Failed to parse result string from state key '{result_key}'. Raw: {raw_result[:100]}...[/red]")
                    elif isinstance(raw_result, dict):
                        # Already a dict, no update needed unless we want to ensure it's the *last* JSON if multiple existed
                        logger.debug(f"Validation result for {sim_id} is already a dictionary. No post-parsing needed.")
                        # Optionally, you could re-parse even dicts if you suspect nested JSON strings, but usually not needed here.
                    elif raw_result is None:
                         logger.debug(f"No validation result found for {sim_id} under key '{result_key}'.")
                         # No update needed if it's already None
                    else:
                        # Unexpected type, store None to clear
                        parsed_validation_updates[result_key] = None
                        logger.warning(f"Unexpected validation result format for {sim_id} under key '{result_key}': {type(raw_result)}. Queued None for state update.")
                        console.print(f"    [cyan]{sim_id}:[/cyan] [red]Unexpected result format under key '{result_key}': {type(raw_result)}[/red]")

                # Apply the parsed updates via a synthetic event
                if parsed_validation_updates:
                    logger.info(f"Applying {len(parsed_validation_updates)} parsed validation updates via synthetic event.")
                    update_event = Event(
                        author="ValidationPostProcessor", # Identify the source
                        actions=EventActions(state_delta=parsed_validation_updates)
                    )
                    session_service.append_event(session=session, event=update_event)
                    logger.info("Synthetic event appended to update state with parsed validation results.")

                    # CRITICAL: Refresh state again after applying the update
                    session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
                    if not session: raise RuntimeError("Session lost after applying parsed validation delta.")
                    current_state = session.state
                    logger.info("State refreshed after applying parsed validation delta.")
                else:
                    logger.info("No validation results needed parsing or updating.")
                # <<< END: Post-processing >>>

                logger.info("Validation phase complete (including post-processing).")
                console.print("  [green]Intent validation complete.[/green]")

                # Print validation summary (now reads dictionaries directly from state)
                console.print("  [bold]Validation Results:[/bold]")
                results_found_count = 0
                for sim_id in active_sim_ids:
                    result_key = VALIDATION_RESULT_KEY_FORMAT.format(sim_id=sim_id)
                    # <<< Now expect a dictionary or None >>>
                    parsed_result = current_state.get(result_key)

                    if isinstance(parsed_result, dict):
                        results_found_count += 1
                        status_val = parsed_result.get('validation_status', 'unknown').lower()
                        is_valid = status_val in ['approved', 'modified']
                        status_display = f"[green]{status_val.capitalize()}[/green]" if is_valid else f"[red]{status_val.capitalize()}[/red]"
                        reason = f" ([dim]{parsed_result.get('reasoning', 'No reason given')}[/dim])" if not is_valid else ""
                        original_intent = parsed_result.get('original_intent', {})
                        if not isinstance(original_intent, dict): original_intent = {}
                        intent_desc = original_intent.get('action_type', 'Unknown Action')
                        console.print(f"    [cyan]{sim_id}:[/cyan] {status_display} - {intent_desc}{reason}")
                    elif parsed_result is None:
                         # This means either no result was generated, or parsing failed and it was set to None
                         console.print(f"    [cyan]{sim_id}:[/cyan] [dim](No valid result found/parsed)[/dim]")
                    # else: Should not happen if post-processing worked correctly

                if results_found_count == 0:
                    console.print("    [dim](No valid validation results found in state)[/dim]")

            else:
                logger.warning("No validator agents found for validation phase.")
                console.print("[yellow]Warning:[/yellow] No validator agents found.")
                # Ensure subsequent phases don't fail if keys are missing
                for sim_id in active_sim_ids:
                    result_key = VALIDATION_RESULT_KEY_FORMAT.format(sim_id=sim_id)
                    if result_key not in current_state: current_state[result_key] = None
            # --- End Phase 3 ---

            # --- Phase 4: Interaction Resolution ---
            console.print(Panel("Phase 4: Interaction Resolution", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 4: Running Interaction Resolution.")
            interaction_intents_exist = False
            for sim_id in active_sim_ids: # Iterate through active IDs
                result_key = VALIDATION_RESULT_KEY_FORMAT.format(sim_id=sim_id)
                # <<< SIMPLIFIED: Directly get the result (should be dict or None) >>>
                validation_result = current_state.get(result_key)

                # <<< SIMPLIFIED: Check if it's a valid dictionary and meets criteria >>>
                if (isinstance(validation_result, dict) and
                    validation_result.get('validation_status') in ['approved', 'modified'] and
                    isinstance(validation_result.get('original_intent'), dict) and
                    validation_result.get('original_intent', {}).get('action_type') in ['talk', 'use']):
                    interaction_intents_exist = True
                    break # Found one, no need to check further

            if interaction_intents_exist:
                # ... (Rest of Phase 4 logic remains largely the same, as it now expects dictionaries) ...
                logger.info("Valid 'talk' or 'use' intents found, running Interaction Resolver.")
                runner.agent = interaction_resolver_agent
                trigger_msg_p4 = f"Resolve 'talk' and 'use' interactions for turn {turn} based on individual validation results (keys like '{VALIDATION_RESULT_KEY_FORMAT.format(sim_id='<sim_id>')}')."
                resolver_output_json_str = None
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg_p4)])):
                    console.print(event)
                    if event.is_final_response() and event.content: resolver_output_json_str = event.content.parts[0].text
                    if event.error_message: logger.error(f"P4 Error ({event.author}): {event.error_message}")

                if resolver_output_json_str:
                    # Parsing the resolver's output is still needed
                    output_data = parse_json_output(resolver_output_json_str, "P4-ResolveParse", interaction_resolver_agent.name, console, logger)
                    if output_data:
                        # Applying resolver's state updates is still needed
                        state_updates = output_data.get("state_updates", {})
                        narrative_log = output_data.get("narrative_log", [])
                        if state_updates:
                            logger.info(f"Applying {len(state_updates)} state updates from Interaction Resolver.")
                            console.print("  [bold]Interaction State Updates:[/bold]")
                            resolver_update_event = Event(author=interaction_resolver_agent.name, actions=EventActions(state_delta=state_updates))
                            session_service.append_event(session=session, event=resolver_update_event) # Use append_event
                            logger.info("Session state updated after applying interaction results.")
                            for key, value in state_updates.items(): console.print(f"    [dim]Updating {key}:[/dim] {value}")
                        else: console.print("    [dim](No state updates)[/dim]")
                        # Handling narrative log is still needed
                        if narrative_log:
                            logger.info(f"Appending {len(narrative_log)} entries to interaction log.")
                            console.print("  [bold]Interaction Narrative Log:[/bold]")
                            if TURN_INTERACTION_LOG_KEY not in current_state: current_state[TURN_INTERACTION_LOG_KEY] = []
                            if isinstance(narrative_log, list):
                                current_state[TURN_INTERACTION_LOG_KEY].extend(narrative_log) # Direct append ok
                                for entry in narrative_log: console.print(f"    - {entry}")
                            else: logger.warning(f"Interaction Resolver 'narrative_log' was not a list: {type(narrative_log)}")
                        else: console.print("    [dim](No narrative log entries)[/dim]")
                    else: console.print(f"[bold red]Error:[/bold red] Failed to parse Interaction Resolver output (see logs).")
                else: logger.warning("No final output received from Interaction Resolver agent.")

                # Refresh state after potential append_event
                session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
                if not session: raise RuntimeError("Session lost after Interaction Resolution processing.")
                current_state = session.state
                logger.info("Interaction resolution phase complete.")
                console.print("  [green]Interaction resolution complete.[/green]")
            else:
                logger.info("No valid 'talk' or 'use' intents found for this turn.")
                console.print("  [dim]No interactions to resolve this turn.[/dim]")
            # --- End Phase 4 ---

            # --- Phase 5: Execution (Move Actions) ---
            # <<< SIMPLIFY Phase 5 input check >>>
            console.print(Panel("Phase 5: Action Execution (Moves)", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 5: Running World Execution Agent.")
            move_intents = {}
            for sim_id in active_sim_ids: # Iterate through active IDs
                result_key = VALIDATION_RESULT_KEY_FORMAT.format(sim_id=sim_id)
                # <<< SIMPLIFIED: Directly get the result (should be dict or None) >>>
                validation_result = current_state.get(result_key)

                # <<< SIMPLIFIED: Check if it's a valid dictionary and a move action >>>
                if (isinstance(validation_result, dict) and
                    validation_result.get('validation_status') in ['approved', 'modified'] and
                    isinstance(validation_result.get('original_intent'), dict) and
                    validation_result.get('original_intent', {}).get('action_type') == 'move'):
                    move_intents[sim_id] = validation_result['original_intent'] # Pass original intent

            if move_intents:
                # ... (Rest of Phase 5 logic remains the same) ...
                logger.info(f"Valid 'move' intents found ({len(move_intents)}), running Execution Agent.")
                runner.agent = world_execution_agent
                trigger_msg_p5 = f"Execute 'move' actions for turn {turn} based on individual validation results (keys like '{VALIDATION_RESULT_KEY_FORMAT.format(sim_id='<sim_id>')}')."
                execution_output_json_str = None
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg_p5)])):
                    console.print(event)
                    if event.is_final_response() and event.content: execution_output_json_str = event.content.parts[0].text
                    if event.error_message: logger.error(f"P5 Error ({event.author}): {event.error_message}")

                if execution_output_json_str:
                    parsed_execution_results = parse_json_output(execution_output_json_str, "P5-ExecParse", world_execution_agent.name, console, logger)
                    if parsed_execution_results:
                        location_updates_delta = {}
                        execution_narratives_this_turn = {} # Collect narratives here
                        console.print("  [bold]Executed Moves:[/bold]")
                        for sim_id_exec, exec_result in parsed_execution_results.items():
                            if isinstance(exec_result, dict):
                                new_location = exec_result.get("new_location")
                                narrative = exec_result.get("narrative", "(No narrative provided)")
                                execution_narratives_this_turn[sim_id_exec] = narrative # Store narrative
                                console.print(f"    - [cyan]{sim_id_exec}[/cyan] moved to '{new_location}'. Narrative: {narrative}")
                                if new_location:
                                    location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id=sim_id_exec)
                                    location_updates_delta[location_key] = new_location
                            else: logger.warning(f"Invalid execution result format for {sim_id_exec}: {type(exec_result)}")

                        # Apply location updates via synthetic event
                        if location_updates_delta:
                            logger.info(f"Applying location updates from execution: {location_updates_delta}")
                            exec_update_event = Event(author=world_execution_agent.name, actions=EventActions(state_delta=location_updates_delta))
                            session_service.append_event(session=session, event=exec_update_event) # Use append_event
                        else: logger.info("No location updates derived from execution results.")

                        # Store execution narratives in state (direct update ok for this intermediate key)
                        current_state[TURN_EXECUTION_NARRATIVES_KEY] = execution_narratives_this_turn
                        logger.info(f"Stored {len(execution_narratives_this_turn)} execution narratives.")

                    else:
                        console.print(f"[bold red]Error:[/bold red] Failed to parse World Execution Agent output (see logs).")
                        current_state[TURN_EXECUTION_NARRATIVES_KEY] = {} # Ensure empty on parse error
                else:
                    logger.warning("No final output received from World Execution agent.")
                    current_state[TURN_EXECUTION_NARRATIVES_KEY] = {} # Ensure empty if no output

                # Refresh state after potential append_event
                session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
                if not session: raise RuntimeError("Session lost after Execution.")
                current_state = session.state
                logger.info("Execution phase complete.")
                console.print("  [green]Action execution complete.[/green]")
            else:
                logger.info("No valid move intents found for this turn.")
                console.print("  [dim]No moves to execute this turn.[/dim]")
                current_state[TURN_EXECUTION_NARRATIVES_KEY] = {} # Ensure empty if no moves

            # --- Phase 6: Narration ---
            # (Keep Phase 6 logic as is - it reads individual keys via its tool)
            console.print(Panel("Phase 6: Narration", title="Simulation Phase", border_style="dim"))
            logger.info("Phase 6: Running Narration Agent.")
            runner.agent = narration_agent
            trigger_msg_p6 = f"Narrate the events of turn {turn}."
            narration_output_json_str = None
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=types.Content(parts=[types.Part(text=trigger_msg_p6)])):
                console.print(event)
                if event.actions and event.actions.state_delta: logger.info(f"P6 Delta from {event.author}: {event.actions.state_delta}")
                if event.is_final_response() and event.content: narration_output_json_str = event.content.parts[0].text
                if event.error_message: logger.error(f"P6 Error: {event.error_message}")

            session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
            if not session: raise RuntimeError("Session lost after Narration.")
            current_state = session.state

            console.print(Panel("Turn Summary", border_style="green", expand=False))
            narratives_found = False
            for sim_id in active_sim_ids:
                 # <<< CORRECTED LINE >>>
                 narration_key = SIMULACRA_NARRATION_KEY_FORMAT.format(sim_id)
                 # <<< END CORRECTION >>>
                 narrative_text = current_state.get(narration_key)
                 if narrative_text and isinstance(narrative_text, str):
                     console.print(f"[bold cyan]{sim_id}:[/bold cyan] {narrative_text}")
                     narratives_found = True
            if not narratives_found:
                 console.print("[dim](No narratives saved to state this turn)[/dim]")

            logger.info(f"--- End of Turn {turn} ---")
            # --- End Phase 6 ---

        except Exception as e:
            logger.critical(f"Critical error during turn {turn}: {e}", exc_info=True)
            console.print(f"[bold red]\n--- Critical Error during Turn {turn} ---[/]")
            console.print_exception(show_locals=False)
            return current_state # Return state at time of error

    logger.info(f"Simulation finished after {max_turns} turns.")
    return current_state
