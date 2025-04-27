# src/simulation_loop.py (Phased Turn Orchestration Loop)
import asyncio
import json
import logging
from rich.console import Console
from rich.rule import Rule # Import Rule for visual separators
from rich.padding import Padding # Import Padding for indentation
from typing import Dict, List, Any, Optional, Set
# from pydantic import ValidationError # Remove if only used here
# ADK Imports
from google.adk.agents import BaseAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, Session
# Use the existing types import for Content, Part, ToolConfig etc.
from google.genai import types
from datetime import datetime, timedelta

from src.loop_utils import print_event_details, parse_json_output, format_iso_timestamp

console = Console()
logger = logging.getLogger(__name__)

ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent"
ACTION_VALIDATION_KEY_FORMAT = "simulacra_{}_validation_result"
INTERACTION_RESULT_KEY_FORMAT = "simulacra_{}_interaction_result"
WORLD_STATE_KEY = "current_world_state"
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_GOAL_KEY_FORMAT = "simulacra_{}_goal"
SIMULACRA_PERSONA_KEY_FORMAT = "simulacra_{}_persona"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
SIMULACRA_MONOLOGUE_KEY_FORMAT = "last_simulacra_{}_monologue"
SIMULACRA_NARRATION_KEY_FORMAT = "simulacra_{}_last_narration"
TURN_VALIDATION_RESULTS_KEY = "turn_validation_results"
TURN_EXECUTION_NARRATIVES_KEY = "turn_execution_narratives"

async def run_phased_simulation(
    runner: Runner,
    session_service: BaseSessionService,
    session: Session,
    # Agent Instances
    world_state_agent: BaseAgent,
    simulacra_agents: Dict[str, BaseAgent],
    world_engine_agent: BaseAgent,
    npc_agent: BaseAgent,
    world_execution_agent: BaseAgent, # Make sure this is passed in
    narration_agent: BaseAgent,
    # Config
    max_turns: int = 5
):
    """
    Runs the simulation loop using a phased turn structure.
    Orchestrates:
    1. World Update (WorldStateAgent)
    2. Parallel Intent Generation (SimulacraAgents via ParallelAgent)
    3. Physical Validation (WorldEngineAgent)
    4. Interaction Resolution (NpcAgent)
    5. Physical Execution (WorldStateAgent tool)
    6. Narration (NarrationAgent)
    """
    console.print(Padding(f"\n[bold cyan]--- Starting Phased Simulation (Max {max_turns} Turns) ---[/bold cyan]", (1, 0, 0, 0)))
    console.print(f"Session ID: {session.id}")

    user_id = session.user_id
    session_id = session.id

    initial_primary_location = session.state.get("simulation_primary_location")
    if not initial_primary_location:
        initial_primary_location = "New York City, NY" # Fallback location
        logger.warning(f"Could not find 'simulation_primary_location' in session state. Falling back to default: {initial_primary_location}")
    else:
        logger.info(f"Using initial primary location from session state: {initial_primary_location}")

    default_trigger = types.Content(parts=[types.Part(text="Proceed with turn phase.")])

    for turn in range(max_turns):
        console.rule(f"Turn {turn + 1}/{max_turns}", style="bold cyan")

        active_sim_ids_for_turn = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
        sim_location_map: Dict[str, str] = {}

        if not active_sim_ids_for_turn:
            logger.warning("No active simulacra found at start of turn. Will use initial primary location for query.")
        else:
            for sim_id in active_sim_ids_for_turn:
                loc_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
                current_loc = session.state.get(loc_key)
                if current_loc:
                    sim_location_map[sim_id] = current_loc
                else:
                    logger.warning(f"Could not find location for active sim {sim_id} using key {loc_key}. Using initial primary location as fallback.")
                    sim_location_map[sim_id] = initial_primary_location

        if sim_location_map:
            unique_locations_for_tool = list(dict.fromkeys(sim_location_map.values()))
        else:
            unique_locations_for_tool = [initial_primary_location]

        logger.info(f"Unique locations to query details for: {unique_locations_for_tool}")

        # --- Phase 1: World State Update & Sync (World State Agent) ---
        try:
            console.print(Rule("Phase 1: World State Update & Sync", style="yellow"))
            runner.agent = world_state_agent
            console.print(Padding(f"Running [bold]{world_state_agent.name}[/bold] to sync real-world details and update time...", (0, 0, 0, 2)))

            locations_str = ", ".join([f"'{loc}'" for loc in unique_locations_for_tool])
            phase1_trigger_text = (
                 f"Perform the start-of-turn world state update. "
                 f"First, for each unique location in this list: [{locations_str}], use the 'get_setting_details' tool to get its current details. "
                 f"Then, use the 'update_and_get_world_state' tool ONCE to advance world time and get the overall state."
            )
            phase1_trigger = types.Content(parts=[types.Part(text=phase1_trigger_text)])
            console.print(Padding(f"[dim]Trigger: {phase1_trigger_text}[/dim]", (0, 0, 0, 4)))

            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=phase1_trigger):
                # Assuming print_event_details adds its own padding/prefix
                print_event_details(event, "P1", console, logger)
            console.print(Padding("[green]Phase 1 Complete.[/green]", (0, 0, 1, 2)))
        except Exception as e:
             logger.exception("Error during World State Update & Sync (Phase 1).")
             console.print(Padding(f"[bold red]Error in Phase 1: {e}. State may be inconsistent. Skipping turn.[/bold red]", (0, 0, 1, 2)))
             await asyncio.sleep(1)
             continue

        # --- Phase 2: Parallel Simulacra Intent Generation ---
        active_sim_ids = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
        active_sim_instances = [sim_agent for sim_id, sim_agent in simulacra_agents.items() if sim_id in active_sim_ids]

        if not active_sim_instances:
             console.print(Padding("[yellow]Phase 2: No active simulacra found. Skipping Intent/Validation/Interaction/Execution.[/yellow]", (1, 0, 1, 0)))
             session.state[TURN_VALIDATION_RESULTS_KEY] = {}
             session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
        else:
            try:
                console.print(Rule(f"Phase 2: Parallel Simulacra Intent Generation ({len(active_sim_instances)} acting)", style="yellow"))

                for agent_instance in active_sim_instances:
                    if hasattr(agent_instance, 'parent_agent'):
                        agent_instance.parent_agent = None

                parallel_sim_agent = ParallelAgent(name=f"ParallelSimulacra_Turn{turn+1}", sub_agents=active_sim_instances)
                runner.agent = parallel_sim_agent
                console.print(Padding(f"Running [bold]{parallel_sim_agent.name}[/bold] for simulacra reflection & intent tools...", (0, 0, 0, 2)))
                phase2_trigger_text = (
                    "Based on your current status, goal, recent monologue, and the overall world state, "
                    "decide on your primary action/intent for this turn. "
                    "Use your available tools (e.g., 'check_self_status', 'generate_internal_monologue', 'attempt_move_to', 'attempt_interact') "
                    "to reflect and set your intent in the session state."
                )
                phase2_trigger = types.Content(parts=[types.Part(text=phase2_trigger_text)])
                console.print(Padding(f"[dim]Trigger: {phase2_trigger_text}[/dim]", (0, 0, 0, 4)))

                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=phase2_trigger):
                    # Assuming print_event_details adds its own padding/prefix
                    print_event_details(event, "P2", console, logger)
                console.print(Padding("[green]Phase 2 Complete.[/green]", (0, 0, 1, 2)))
            except Exception as e:
                 logger.exception("Error during Parallel Simulacra execution (Phase 2).")
                 console.print(Padding(f"[bold red]Error in Phase 2: {e}. Skipping rest of turn phases.[/bold red]", (0, 0, 1, 2)))
                 session.state[TURN_VALIDATION_RESULTS_KEY] = {}
                 session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
                 await asyncio.sleep(1)
                 continue

        # --- Phase 3 (Validation) ---
        try:
            console.print(Rule("Phase 3: Physical Validation", style="yellow"))
            runner.agent = world_engine_agent
            validation_trigger_text = "Validate all pending actions based on current world state and rules."
            validation_trigger = types.Content(parts=[types.Part(text=validation_trigger_text)])
            console.print(Padding(f"Running [bold]{world_engine_agent.name}[/bold] to validate intents...", (0, 0, 0, 2)))
            console.print(Padding(f"[dim]Trigger: {validation_trigger_text}[/dim]", (0, 0, 0, 4)))

            parsed_validation_results = None
            final_validation_text = None

            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=validation_trigger):
                # Assuming print_event_details adds its own padding/prefix
                print_event_details(event, "P3", console, logger)
                if event.is_final_response() and event.content and event.content.parts:
                    part = event.content.parts[0]
                    if hasattr(part, 'text'):
                        final_validation_text = part.text
                        # --- MODIFIED: Print full final text before parsing ---
                        console.print(Padding("[bold]Final Validation Output (Raw):[/bold]", (1, 0, 0, 4)))
                        console.print(Padding(final_validation_text, (0, 0, 1, 4)))
                        # ---

            parsed_validation_results = parse_json_output(
                final_validation_text, "P3", world_engine_agent.name, console, logger
            )

            session.state[TURN_VALIDATION_RESULTS_KEY] = parsed_validation_results if parsed_validation_results is not None else {}
            if parsed_validation_results is not None:
                logger.info(f"Stored combined validation results under key '{TURN_VALIDATION_RESULTS_KEY}'.")
                console.print(Padding(f"Validation results parsed and stored.", (0, 0, 0, 4)))
            else:
                logger.warning(f"No valid validation results received/parsed from WorldEngine. Storing empty dict under '{TURN_VALIDATION_RESULTS_KEY}'.")
                console.print(Padding(f"[yellow]Could not parse validation results.[/yellow]", (0, 0, 0, 4)))

            console.print(Padding("[green]Phase 3 Complete.[/green]", (0, 0, 1, 2)))
        except Exception as e:
            logger.exception("Error during World Engine Validation (Phase 3).")
            console.print(Padding(f"[bold red]Error in Phase 3: {e}. Skipping Interaction/Execution.[/bold red]", (0, 0, 1, 2)))
            session.state[TURN_VALIDATION_RESULTS_KEY] = {}
            session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
            # Fall through to Narration (Phase 5)

        # --- Phase 4a (Interaction) ---
        if session.state.get(TURN_VALIDATION_RESULTS_KEY):
            try:
                console.print(Rule("Phase 4a: Interaction Resolution", style="yellow"))
                runner.agent = npc_agent
                interaction_trigger_text = "Resolve pending 'talk' and 'interact' actions based on validation status."
                interaction_trigger = types.Content(parts=[types.Part(text=interaction_trigger_text)])
                console.print(Padding(f"Running [bold]{npc_agent.name}[/bold] to resolve interactions using its tools...", (0, 0, 0, 2)))
                console.print(Padding(f"[dim]Trigger: {interaction_trigger_text}[/dim]", (0, 0, 0, 4)))

                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=interaction_trigger):
                    # Assuming print_event_details adds its own padding/prefix
                    print_event_details(event, "P4a", console, logger)
                console.print(Padding("[green]Phase 4a Complete.[/green]", (0, 0, 1, 2)))
            except Exception as e:
                 logger.exception("Error during NPC Interaction Resolution (Phase 4a).")
                 console.print(Padding(f"[bold red]Error in Phase 4a: {e}.[/bold red]", (0, 0, 1, 2)))

        # --- Phase 4b: Physical Action Execution & Narration ---
        all_validation_results = session.state.get(TURN_VALIDATION_RESULTS_KEY, {})
        if all_validation_results:
            try:
                console.print(Rule("Phase 4b: Physical Execution & Narration", style="yellow"))
                approved_physical_actions_for_batch = []
                active_sim_ids_for_physical = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])

                logger.debug("--- Starting Phase 4b: Checking for approved 'move' actions ---")
                for sim_id in active_sim_ids_for_physical:
                    validation_result = all_validation_results.get(sim_id)
                    if validation_result:
                        original_intent = validation_result.get("original_intent")
                        action_type = original_intent.get("action_type") if original_intent else None
                        val_status = validation_result.get("validation_status")
                        estimated_duration = validation_result.get("estimated_duration_seconds")

                        if val_status in ["approved", "modified"] and action_type == "move":
                            logger.info(f"  Action Approved/Modified for {sim_id}: Type={action_type}, Status={val_status}. Adding to batch.")
                            action_detail = {
                                "sim_id": sim_id,
                                "action_type": action_type,
                                "details": original_intent,
                                "estimated_duration_seconds": estimated_duration
                            }
                            approved_physical_actions_for_batch.append(action_detail)
                        elif action_type != "move":
                             logger.debug(f"  Skipping action for {sim_id} in Phase 4b: Type '{action_type}' handled elsewhere (e.g., Phase 4a).")
                        else:
                             logger.debug(f"  Action for {sim_id} not added to physical batch (Status: {val_status}, Type: {action_type})")
                    else:
                        logger.debug(f"  Skipping {sim_id}: Missing validation_result in combined dict.")

                if approved_physical_actions_for_batch:
                    console.print(Padding(f"Found {len(approved_physical_actions_for_batch)} approved 'move' actions. Running [bold]{world_execution_agent.name}[/bold] for execution & narration...", (0, 0, 0, 2)))

                    runner.agent = world_execution_agent

                    current_world_time_iso = session.state.get(WORLD_STATE_KEY, {}).get("world_time")
                    current_world_time_str = format_iso_timestamp(current_world_time_iso)

                    try:
                        actions_json_string = json.dumps(approved_physical_actions_for_batch, indent=2)
                    except TypeError:
                        logger.error("Could not serialize actions batch to JSON for prompt.")
                        actions_json_string = str(approved_physical_actions_for_batch)

                    execution_trigger_text = (
                        "You are the world execution engine. Process the following approved 'move' actions. "
                        f"The current world time is approximately {current_world_time_str}.\n"
                        "For EACH action:\n"
                        "1. **Infer Specific Locations:** If the 'origin' or 'destination' in the details are general (e.g., a city name, 'Library'), use your knowledge to infer plausible, specific starting and ending locations (e.g., 'Downtown Asheville near Pack Square', 'Pack Memorial Library'). If they are already specific, use them.\n"
                        "2. **Determine Mode of Transport:** Based on the inferred specific locations and general context, decide the most likely mode of transport (e.g., walking, driving, public transit).\n"
                        "3. **Estimate Realistic Duration:** Calculate a realistic travel time in seconds based on the specific locations and chosen mode of transport.\n"
                        "4. **Generate Detailed Narrative:** Create an engaging, descriptive, present-tense narrative of the journey. Include the specific starting point, the mode of transport, key landmarks or experiences during the travel (if plausible), the specific destination, and the estimated duration.\n"
                        "Respond ONLY with a valid JSON object mapping each processed `sim_id` to a dictionary containing: "
                        "{'narrative': 'string', 'new_location': 'string (the specific inferred destination)', 'duration_seconds': integer (your realistic estimate)}. "
                        f"Actions to process:\n```json\n{actions_json_string}\n```"
                    )
                    execution_trigger = types.Content(parts=[types.Part(text=execution_trigger_text)])
                    console.print(Padding(f"[dim]Trigger includes {len(approved_physical_actions_for_batch)} actions.[/dim]", (0, 0, 0, 4))) # Simplified trigger print

                    parsed_execution_results = None
                    final_execution_text = None

                    async for event in runner.run_async(
                        user_id=user_id, session_id=session_id, new_message=execution_trigger
                    ):
                        # Assuming print_event_details adds its own padding/prefix
                        print_event_details(event, "P4b", console, logger)
                        if event.is_final_response() and event.content and event.content.parts:
                             part = event.content.parts[0]
                             if hasattr(part, 'text'):
                                 final_execution_text = part.text
                                 # --- MODIFIED: Print full final text before parsing ---
                                 console.print(Padding("[bold]Final Execution Output (Raw):[/bold]", (1, 0, 0, 4)))
                                 console.print(Padding(final_execution_text, (0, 0, 1, 4)))
                                 # ---

                    parsed_execution_results = parse_json_output(
                        final_execution_text, "P4b", world_execution_agent.name, console, logger
                    )

                    execution_narratives = {}
                    max_duration = 0
                    if parsed_execution_results:
                        console.print(Padding("Processing execution results...", (0, 0, 0, 4)))
                        for sim_id, result_data in parsed_execution_results.items():
                            if isinstance(result_data, dict):
                                narrative = result_data.get("narrative")
                                new_location = result_data.get("new_location")
                                duration = result_data.get("duration_seconds", 0)

                                if narrative:
                                    execution_narratives[sim_id] = narrative
                                    logger.info(f"Stored execution narrative for {sim_id}.")

                                if new_location:
                                    location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
                                    session.state[location_key] = new_location
                                    logger.info(f"Updated location for {sim_id} to '{new_location}'.")
                                    console.print(Padding(f"Updated location for {sim_id} -> '{new_location}'", (0, 0, 0, 6)))
                                else:
                                    logger.warning(f"No 'new_location' provided by agent for move action of {sim_id}.")

                                if duration > max_duration:
                                    max_duration = duration
                            else:
                                logger.warning(f"Invalid result format for {sim_id} in execution JSON: {result_data}")

                        if max_duration > 0:
                            try:
                                world_state = session.state.get(WORLD_STATE_KEY, {})
                                current_time_iso = world_state.get("world_time")
                                if current_time_iso:
                                    current_dt = datetime.fromisoformat(current_time_iso)
                                    new_dt = current_dt + timedelta(seconds=max_duration)
                                    if WORLD_STATE_KEY not in session.state:
                                        session.state[WORLD_STATE_KEY] = {}
                                    session.state[WORLD_STATE_KEY]["world_time"] = new_dt.isoformat()
                                    formatted_new_time = format_iso_timestamp(new_dt.isoformat())
                                    logger.info(f"Advanced world time by {max_duration} seconds to {formatted_new_time}.")
                                    console.print(Padding(f"Advanced world time by {max_duration}s -> {formatted_new_time}", (0, 0, 0, 6)))
                                else:
                                    logger.warning("Could not find world_time in state to advance.")
                            except Exception as time_e:
                                logger.exception(f"Error advancing world time: {time_e}")
                        console.print(Padding("Execution results processed.", (0, 0, 0, 4)))
                    else:
                        console.print(Padding("[yellow]Could not parse execution results.[/yellow]", (0, 0, 0, 4)))


                    session.state[TURN_EXECUTION_NARRATIVES_KEY] = execution_narratives
                    console.print(Padding("[green]Phase 4b Complete.[/green]", (0, 0, 1, 2)))

                else:
                    console.print(Padding("[yellow]Phase 4b: No approved 'move' actions found to execute.[/yellow]", (0, 0, 1, 2)))
                    session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}

            except Exception as e:
                logger.exception("Error during Physical Execution & Narration (Phase 4b).")
                console.print(Padding(f"[bold red]Error in Phase 4b: {e}. Skipping phase.[/bold red]", (0, 0, 1, 2)))
                session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
                await asyncio.sleep(1)

        else:
             console.print(Padding("[yellow]Phase 4b: Skipping due to missing validation results from Phase 3.[/yellow]", (0, 0, 1, 2)))
             session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}

        # --- Phase 5: Narration ---
        try:
            console.print(Rule("Phase 5: Narration", style="yellow"))
            runner.agent = narration_agent
            narration_trigger_text = (
                "Generate the final narrative summary for this turn. "
                f"Use the 'get_narration_context' tool to gather necessary information "
                f"(simulacra status, goals, monologues from '{SIMULACRA_MONOLOGUE_KEY_FORMAT}', "
                f"interaction results from '{INTERACTION_RESULT_KEY_FORMAT}', "
                f"execution narratives from '{TURN_EXECUTION_NARRATIVES_KEY}', "
                f"and world state details from '{WORLD_STATE_KEY}' and 'location_details')."
            )
            narration_trigger = types.Content(parts=[types.Part(text=narration_trigger_text)])
            console.print(Padding(f"Running [bold]{narration_agent.name}[/bold] to generate turn summary...", (0, 0, 0, 2)))
            console.print(Padding(f"[dim]Trigger: {narration_trigger_text}[/dim]", (0, 0, 0, 4)))

            parsed_narration_results = None
            final_narration_text = None

            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=narration_trigger):
                 # Assuming print_event_details adds its own padding/prefix
                 print_event_details(event, "P5", console, logger)
                 if event.is_final_response() and event.content and event.content.parts:
                      part = event.content.parts[0]
                      if hasattr(part, 'text'):
                          final_narration_text = part.text
                          # --- MODIFIED: Print full final text before parsing ---
                          console.print(Padding("[bold]Final Narration Output (Raw):[/bold]", (1, 0, 0, 4)))
                          console.print(Padding(final_narration_text, (0, 0, 1, 4)))
                          # ---

            parsed_narration_results = parse_json_output(
                final_narration_text, "P5", narration_agent.name, console, logger
            )

            if parsed_narration_results:
                console.print(Padding("Storing and displaying final narratives...", (0, 0, 0, 4)))
                for sim_id, narration_text in parsed_narration_results.items():
                    if isinstance(narration_text, str):
                        narration_key = SIMULACRA_NARRATION_KEY_FORMAT.format(sim_id)
                        session.state[narration_key] = narration_text
                        logger.info(f"Stored final narration for {sim_id} under key '{narration_key}'.")
                        # Print the final narration for this character
                        console.print(Rule(f"Narrative for {sim_id}", style="dim white"))
                        console.print(Padding(f"{narration_text}", (0, 0, 1, 2))) # Add padding to narrative
                    else:
                        logger.warning(f"Invalid narration format for {sim_id}: Expected string, got {type(narration_text)}")
            else:
                logger.warning("No valid narration results received/parsed from NarrationAgent.")
                console.print(Padding("[yellow]Could not parse narration results.[/yellow]", (0, 0, 0, 4)))

            console.print(Padding("[green]Phase 5 Complete.[/green]", (0, 0, 1, 2)))
        except Exception as e:
            logger.exception("Error during Narration (Phase 5).")
            console.print(Padding(f"[bold red]Error in Phase 5: {e}.[/bold red]", (0, 0, 1, 2)))

        # Pause slightly between turns
        console.print(Padding(f"--- End of Turn {turn + 1} ---", (1, 0, 1, 0)))
        await asyncio.sleep(1)

    console.rule("[bold cyan]Simulation Finished[/bold cyan]", style="bold cyan")