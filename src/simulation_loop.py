# src/simulation_loop.py (Phased Turn Orchestration Loop)
import asyncio
import json
import logging
from rich.console import Console
from rich.rule import Rule # Import Rule for visual separators
from typing import Dict, List, Any, Optional, Set # Keep List/Set if used elsewhere
# from pydantic import ValidationError # Remove if only used here
# ADK Imports
from google.adk.agents import BaseAgent, ParallelAgent # Removed LlmAgent as it's not directly used here
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, Session
# Use the existing types import for Content, Part, ToolConfig etc.
from google.genai import types
from datetime import datetime, timedelta

# --- MODIFIED: Import the new formatter ---
from src.loop_utils import print_event_details, parse_json_output, format_iso_timestamp
# ---

console = Console()
logger = logging.getLogger(__name__)

ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
# Key format where Simulacra write intent (read by Validator)
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent"
# Key format where WE Validator writes validation status (read in Phase 4b)
ACTION_VALIDATION_KEY_FORMAT = "simulacra_{}_validation_result"
# Key format where NpcAgent writes interaction results
INTERACTION_RESULT_KEY_FORMAT = "simulacra_{}_interaction_result"
# Key for the comprehensive world state dict
WORLD_STATE_KEY = "current_world_state"
# Key format for Simulacra state needed by tools/execution
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_GOAL_KEY_FORMAT = "simulacra_{}_goal"
SIMULACRA_PERSONA_KEY_FORMAT = "simulacra_{}_persona"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
SIMULACRA_MONOLOGUE_KEY_FORMAT = "last_simulacra_{}_monologue"
SIMULACRA_NARRATION_KEY_FORMAT = "simulacra_{}_last_narration"
TURN_VALIDATION_RESULTS_KEY = "turn_validation_results"
# --- ADDED: Key for execution narratives from Phase 4b ---
TURN_EXECUTION_NARRATIVES_KEY = "turn_execution_narratives"
# ---


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
    console.print(f"\n[bold cyan]--- Starting Phased Simulation (Max {max_turns} Turns) ---[/bold cyan]")
    console.print(f"Session ID: {session.id}")

    user_id = session.user_id
    session_id = session.id

    # --- MODIFIED: Determine initial primary location ---
    initial_primary_location = session.state.get("simulation_primary_location")
    if not initial_primary_location:
        initial_primary_location = "New York City, NY" # Fallback location
        logger.warning(f"Could not find 'simulation_primary_location' in session state. Falling back to default: {initial_primary_location}")
    else:
        logger.info(f"Using initial primary location from session state: {initial_primary_location}")
    # --- END MODIFIED ---

    default_trigger = types.Content(parts=[types.Part(text="Proceed with turn phase.")])

    for turn in range(max_turns):
        console.rule(f"Turn {turn + 1}/{max_turns}", style="cyan")

        # --- REFACTORED: Determine current locations using only Dict ---
        active_sim_ids_for_turn = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
        sim_location_map: Dict[str, str] = {}

        if not active_sim_ids_for_turn:
            logger.warning("No active simulacra found at start of turn. Will use initial primary location for query.")
            # Map remains empty, fallback handled below
        else:
            for sim_id in active_sim_ids_for_turn:
                loc_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
                current_loc = session.state.get(loc_key)
                if current_loc:
                    sim_location_map[sim_id] = current_loc
                else:
                    logger.warning(f"Could not find location for active sim {sim_id} using key {loc_key}. Using initial primary location as fallback.")
                    sim_location_map[sim_id] = initial_primary_location # Map fallback location

        # Derive unique locations from the dictionary values
        if sim_location_map:
            unique_locations_for_tool = list(dict.fromkeys(sim_location_map.values()))
        else:
            # Fallback if map is empty (no active sims or no locations found)
            unique_locations_for_tool = [initial_primary_location]

        logger.info(f"Phase 1: Unique locations to query details for: {unique_locations_for_tool}")
        # logger.info(f"Phase 1: Simulacra location map: {sim_location_map}") # Keep if useful for debugging
        # --- END REFACTORED ---

        # --- Phase 1: World State Update & Sync (World State Agent) ---
        try:
            console.print("[reverse yellow] Phase 1: World State Update & Sync [/reverse yellow]")
            runner.agent = world_state_agent
            console.print(f"[dim]Running WorldStateAgent ({world_state_agent.name}) to sync real-world details and update time...[/dim]")

            # Use the derived unique locations
            locations_str = ", ".join([f"'{loc}'" for loc in unique_locations_for_tool])
            phase1_trigger_text = (
                 f"Perform the start-of-turn world state update. "
                 f"First, for each unique location in this list: [{locations_str}], use the 'get_setting_details' tool to get its current details. "
                 f"Then, use the 'update_and_get_world_state' tool ONCE to advance world time and get the overall state."
            )
            phase1_trigger = types.Content(parts=[types.Part(text=phase1_trigger_text)])

            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=phase1_trigger):
                print_event_details(event, "P1", console, logger)
            console.print("[green]Phase 1 Complete.[/green]")
        except Exception as e:
             logger.exception("Error during World State Update & Sync (Phase 1).")
             console.print(f"[bold red]Error in Phase 1: {e}. State may be inconsistent.[/bold red]")
             await asyncio.sleep(1)
             continue # Skip to next turn on Phase 1 error

        # --- Phase 2: Parallel Simulacra Intent Generation ---
        active_sim_ids = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, []) # Gets ['eleanor_vance', 'eleanor_vance_2']
        # Uses the correct IDs to fetch agent instances from the simulacra_agents dict
        active_sim_instances = [sim_agent for sim_id, sim_agent in simulacra_agents.items() if sim_id in active_sim_ids]

        if not active_sim_instances:
             console.print("[yellow]Phase 2: No active simulacra found. Skipping Intent/Validation/Interaction/Execution.[/yellow]")
             # Ensure keys expected by later phases are initialized if skipping
             session.state[TURN_VALIDATION_RESULTS_KEY] = {}
             session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
        else:
            try:
                console.print(f"[reverse yellow] Phase 2: Parallel Simulacra Intent Generation ({len(active_sim_instances)} acting) [/reverse yellow]")

                # --- ADDED: Clear parent agent before reuse ---
                for agent_instance in active_sim_instances:
                    if hasattr(agent_instance, 'parent_agent'):
                        # logger.debug(f"Clearing parent agent for {agent_instance.name} (was {getattr(agent_instance.parent_agent, 'name', 'Unknown')})")
                        agent_instance.parent_agent = None
                # --- END ADDED ---

                # ParallelAgent uses sub-agents named 'eleanor_vance', etc.
                parallel_sim_agent = ParallelAgent(name=f"ParallelSimulacra_Turn{turn+1}", sub_agents=active_sim_instances)
                runner.agent = parallel_sim_agent
                console.print(f"[dim]Running ParallelAgent ({parallel_sim_agent.name}) for simulacra reflection & intent tools...[/dim]")
                phase2_trigger_text = (
                    "Based on your current status, goal, recent monologue, and the overall world state, "
                    "decide on your primary action/intent for this turn. "
                    "Use your available tools (e.g., 'check_self_status', 'generate_internal_monologue', 'attempt_move_to', 'attempt_interact') "
                    "to reflect and set your intent in the session state."
                )
                phase2_trigger = types.Content(parts=[types.Part(text=phase2_trigger_text)])
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=phase2_trigger):
                    # --- REPLACED with helper function ---
                    # Note: event.author will be the sub-agent ID (e.g., 'eleanor_vance')
                    print_event_details(event, "P2", console, logger)
                    # ---
                console.print("[green]Phase 2 Complete.[/green]")
            except Exception as e:
                 logger.exception("Error during Parallel Simulacra execution (Phase 2).")
                 console.print(f"[bold red]Error in Phase 2: {e}. Skipping rest of turn phases.[/bold red]")
                 session.state[TURN_VALIDATION_RESULTS_KEY] = {} # Ensure keys are initialized on error
                 session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
                 await asyncio.sleep(1)
                 continue # Skip to next turn

        # --- Phase 3 (Validation) ---
        try:
            console.print("[reverse yellow] Phase 3: Physical Validation [/reverse yellow]")
            runner.agent = world_engine_agent
            validation_trigger = types.Content(parts=[types.Part(text="Validate all pending actions based on current world state and rules.")])
            console.print(f"[dim]Running WorldEngineAgent ({world_engine_agent.name}) to validate intents...")
            # --- ADDED: Variable to store parsed results ---
            parsed_validation_results = None
            final_validation_text = None # Store final text to parse after loop

            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=validation_trigger):
                # --- Print Event Details --- START
                if event.content and event.content.parts:
                    part = event.content.parts[0]
                    if hasattr(part, 'function_call') and part.function_call:
                         tool_call = part.function_call
                         console.print(f"[dim]  P3 ({event.author}) -> Tool Call: {tool_call.name} args: {dict(tool_call.args)}[/dim]")
                    elif hasattr(part, 'function_response') and part.function_response:
                         tool_response = part.function_response
                         console.print(f"[dim]  P3 ({event.author}) <- Tool Response: {tool_response.name} -> {str(tool_response.response)}...[/dim]")
                    elif event.is_final_response() and hasattr(part, 'text'):
                         final_validation_text = part.text
                # --- Print Event Details --- END

            # --- Parse the final text AFTER the loop ---
            parsed_validation_results = parse_json_output(
                final_validation_text, "P3", world_engine_agent.name, console, logger
            )

            # Store results (handle None case from parsing failure)
            session.state[TURN_VALIDATION_RESULTS_KEY] = parsed_validation_results if parsed_validation_results is not None else {}
            if parsed_validation_results is not None:
                logger.info(f"Stored combined validation results under key '{TURN_VALIDATION_RESULTS_KEY}'.")
            else:
                logger.warning(f"No valid validation results received/parsed from WorldEngine. Storing empty dict under '{TURN_VALIDATION_RESULTS_KEY}'.")

            console.print("[green]Phase 3 Complete.[/green]")
        except Exception as e:
            logger.exception("Error during World Engine Validation (Phase 3).")
            console.print(f"[bold red]Error in Phase 3: {e}. Skipping Interaction/Execution.[/bold red]")
            session.state[TURN_VALIDATION_RESULTS_KEY] = {} # Ensure key exists but is empty on error
            session.state[TURN_EXECUTION_NARRATIVES_KEY] = {} # Also init this one
            # Fall through to Narration (Phase 5)

        # --- Phase 4a (Interaction) ---
        # Only run if validation succeeded (or partially succeeded)
        if session.state.get(TURN_VALIDATION_RESULTS_KEY):
            try:
                console.print("[reverse yellow] Phase 4a: Interaction Resolution [/reverse yellow]")
                runner.agent = npc_agent
                interaction_trigger = types.Content(parts=[types.Part(text="Resolve pending 'talk' and 'interact' actions based on validation status.")])
                console.print(f"[dim]Running NPCAgent ({npc_agent.name}) to resolve interactions using its tools...[/dim]")
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=interaction_trigger):
                    # --- Print Event Details --- START
                    if event.content and event.content.parts:
                        part = event.content.parts[0]
                        if hasattr(part, 'function_call') and part.function_call: # Check field exists AND is not None
                             tool_call = part.function_call
                             console.print(f"[dim]  P4a ({event.author}) -> Tool Call: {tool_call.name} with args: {dict(tool_call.args)}[/dim]")
                        elif hasattr(part, 'function_response') and part.function_response: # Check field exists AND is not None
                             tool_response = part.function_response
                             console.print(f"[dim]  P4a ({event.author}) <- Tool Response: {tool_response.name} -> {str(tool_response.response)}...[/dim]")
                        elif event.is_final_response() and hasattr(part, 'text'): # Check for final text
                             console.print(f"[dim]  P4a ({event.author}) Final Output: {part.text if part.text else '[No text output]'}[/dim]")
                    # --- Print Event Details --- END
                console.print("[green]Phase 4a Complete.[/green]")
            except Exception as e:
                 logger.exception("Error during NPC Interaction Resolution (Phase 4a).")
                 console.print(f"[bold red]Error in Phase 4a: {e}.[/bold red]")
                 # Physical execution might still proceed

        # --- Phase 4b: Physical Action Execution & Narration ---
        # Only run if validation succeeded (or partially succeeded)
        all_validation_results = session.state.get(TURN_VALIDATION_RESULTS_KEY, {})
        if all_validation_results:
            try:
                console.print("[reverse yellow] Phase 4b: Physical Execution & Narration [/reverse yellow]")
                approved_physical_actions_for_batch = []
                active_sim_ids_for_physical = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])

                # --- Build the batch (FILTERED for 'move') ---
                logger.debug("--- Starting Phase 4b: Checking for approved 'move' actions ---")
                for sim_id in active_sim_ids_for_physical:
                    validation_result = all_validation_results.get(sim_id)
                    if validation_result:
                        original_intent = validation_result.get("original_intent")
                        action_type = original_intent.get("action_type") if original_intent else None
                        val_status = validation_result.get("validation_status")
                        estimated_duration = validation_result.get("estimated_duration_seconds")

                        # --- MODIFIED FILTER: Only include 'move' actions ---
                        if val_status in ["approved", "modified"] and action_type == "move":
                        # --- END MODIFIED FILTER ---
                            logger.info(f"  Action Approved/Modified for {sim_id}: Type={action_type}, Status={val_status}. Adding to batch.")
                            action_detail = {
                                "sim_id": sim_id,
                                "action_type": action_type,
                                "details": original_intent,
                                "estimated_duration_seconds": estimated_duration # Agent might recalculate
                            }
                            approved_physical_actions_for_batch.append(action_detail)
                        # Optional: Log why other types are skipped
                        elif action_type != "move":
                             logger.debug(f"  Skipping action for {sim_id} in Phase 4b: Type '{action_type}' handled elsewhere (e.g., Phase 4a).")
                        else:
                             logger.debug(f"  Action for {sim_id} not added to physical batch (Status: {val_status}, Type: {action_type})")
                    else:
                        logger.debug(f"  Skipping {sim_id}: Missing validation_result in combined dict.")
                # --- End Building Batch ---

                if approved_physical_actions_for_batch:
                    # --- Updated Console Message ---
                    console.print(f"[dim]Found {len(approved_physical_actions_for_batch)} approved 'move' actions. Running WorldExecutionAgent ({world_execution_agent.name}) for execution & narration...[/dim]")
                    # ---

                    # --- Set the agent ---
                    runner.agent = world_execution_agent

                    # --- MODIFIED: Format time for trigger text ---
                    current_world_time_iso = session.state.get(WORLD_STATE_KEY, {}).get("world_time")
                    current_world_time_str = format_iso_timestamp(current_world_time_iso) # Use formatter
                    # ---

                    try:
                        actions_json_string = json.dumps(approved_physical_actions_for_batch, indent=2)
                    except TypeError:
                        logger.error("Could not serialize actions batch to JSON for prompt.")
                        actions_json_string = str(approved_physical_actions_for_batch) # Fallback

                    # --- Trigger text now uses formatted time ---
                    execution_trigger_text = (
                        "You are the world execution engine. Process the following approved 'move' actions. "
                        f"The current world time is approximately {current_world_time_str}.\n" # Formatted time here
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

                    # --- Variable to store parsed results ---
                    parsed_execution_results = None
                    final_execution_text = None # Store final text

                    # --- Run the agent (NO tool_config) ---
                    async for event in runner.run_async(
                        user_id=user_id, session_id=session_id, new_message=execution_trigger
                    ):
                        # --- REPLACED with helper function ---
                        print_event_details(event, "P4b", console, logger)
                        # ---
                        # Capture final text
                        if event.is_final_response() and event.content and event.content.parts:
                             part = event.content.parts[0]
                             if hasattr(part, 'text'):
                                 final_execution_text = part.text

                    # --- Parse the final text AFTER the loop ---
                    parsed_execution_results = parse_json_output(
                        final_execution_text, "P4b", world_execution_agent.name, console, logger
                    )

                    # --- Store Narratives and Apply State Updates ---
                    execution_narratives = {}
                    max_duration = 0
                    if parsed_execution_results:
                        for sim_id, result_data in parsed_execution_results.items():
                            if isinstance(result_data, dict):
                                narrative = result_data.get("narrative")
                                new_location = result_data.get("new_location") # Gets the SPECIFIC location from agent
                                duration = result_data.get("duration_seconds", 0) # Gets the REALISTIC duration from agent

                                if narrative:
                                    execution_narratives[sim_id] = narrative # Store narrative for Phase 5
                                    logger.info(f"Stored execution narrative for {sim_id}.")

                                # Apply state updates manually
                                if new_location: # Check if new_location is provided
                                    location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
                                    session.state[location_key] = new_location # Updates state with SPECIFIC location
                                    logger.info(f"Updated location for {sim_id} to '{new_location}'.")
                                else:
                                    logger.warning(f"No 'new_location' provided by agent for move action of {sim_id}.")

                                if duration > max_duration:
                                    max_duration = duration
                            else:
                                logger.warning(f"Invalid result format for {sim_id} in execution JSON: {result_data}")

                        # Update world time using the agent's calculated max_duration
                        if max_duration > 0:
                            try:
                                world_state = session.state.get(WORLD_STATE_KEY, {})
                                current_time_iso = world_state.get("world_time") # Get ISO time
                                if current_time_iso:
                                    current_dt = datetime.fromisoformat(current_time_iso)
                                    new_dt = current_dt + timedelta(seconds=max_duration)
                                    # Store time in ISO format
                                    if WORLD_STATE_KEY not in session.state:
                                        session.state[WORLD_STATE_KEY] = {}
                                    session.state[WORLD_STATE_KEY]["world_time"] = new_dt.isoformat()
                                    # Log using formatted time
                                    formatted_new_time = format_iso_timestamp(new_dt.isoformat())
                                    logger.info(f"Advanced world time by {max_duration} seconds to {formatted_new_time}.") # Log formatted time
                                else:
                                    logger.warning("Could not find world_time in state to advance.")
                            except Exception as time_e:
                                logger.exception(f"Error advancing world time: {time_e}")

                    session.state[TURN_EXECUTION_NARRATIVES_KEY] = execution_narratives

                    # --- WORKAROUND: Write current locations to temp file ---
                    try:
                        temp_location_file = "temp_sim_locations.json" # Define temp file name
                        current_locations = {}
                        active_sim_ids_for_write = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
                        for sim_id in active_sim_ids_for_write:
                            loc_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
                            # Read the location that *should* have just been updated in session.state
                            loc = session.state.get(loc_key)
                            if loc:
                                current_locations[sim_id] = loc
                            else:
                                # Fallback if somehow missing after update
                                current_locations[sim_id] = session.state.get(SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id), "Unknown - Error in Write")

                        with open(temp_location_file, "w") as f:
                            json.dump(current_locations, f)
                        logger.info(f"WORKAROUND: Wrote current locations to {temp_location_file}: {current_locations}")
                    except Exception as file_e:
                        logger.error(f"WORKAROUND: Failed to write temp location file: {file_e}")
                    # --- END WORKAROUND ---

                    console.print("[green]Phase 4b Complete.[/green]")
                    # --- End Store/Apply ---

                else:
                    # --- Updated Console Message ---
                    console.print("[yellow]Phase 4b: No approved 'move' actions found to execute.[/yellow]")
                    # --- Ensure key exists but is empty ---
                    session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
                    # ---

            except Exception as e:
                logger.exception("Error during Physical Execution & Narration (Phase 4b).")
                console.print(f"[bold red]Error in Phase 4b: {e}. Skipping phase.[/bold red]")
                # --- Ensure key exists but is empty on error ---
                session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
                # ---
                await asyncio.sleep(1)

        else: # If no validation results from Phase 3
             console.print("[yellow]Phase 4b: Skipping due to missing validation results from Phase 3.[/yellow]")
             session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}

        # --- Phase 5: Narration ---
        try:
            console.print("[reverse yellow] Phase 5: Narration [/reverse yellow]")
            runner.agent = narration_agent
            # Trigger needs context keys (adjust based on get_narration_context tool)
            narration_trigger_text = (
                "Generate the final narrative summary for this turn. "
                f"Use the 'get_narration_context' tool to gather necessary information "
                f"(simulacra status, goals, monologues from '{SIMULACRA_MONOLOGUE_KEY_FORMAT}', "
                f"interaction results from '{INTERACTION_RESULT_KEY_FORMAT}', "
                f"execution narratives from '{TURN_EXECUTION_NARRATIVES_KEY}', "
                f"and world state details from '{WORLD_STATE_KEY}' and 'location_details')."
            )
            narration_trigger = types.Content(parts=[types.Part(text=narration_trigger_text)])
            console.print(f"[dim]Running NarrationAgent ({narration_agent.name}) to generate turn summary...[/dim]")

            parsed_narration_results = None
            final_narration_text = None # Store final text

            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=narration_trigger):
                 # --- REPLACED with helper function ---
                 print_event_details(event, "P5", console, logger)
                 # ---
                 # Capture final text
                 if event.is_final_response() and event.content and event.content.parts:
                      part = event.content.parts[0]
                      if hasattr(part, 'text'):
                          final_narration_text = part.text

            # --- Parse the final text AFTER the loop ---
            # Assuming narration output is also JSON mapping sim_id to narration string
            parsed_narration_results = parse_json_output(
                final_narration_text, "P5", narration_agent.name, console, logger
            )

            # Store results (handle None case)
            if parsed_narration_results:
                for sim_id, narration_text in parsed_narration_results.items():
                    if isinstance(narration_text, str):
                        narration_key = SIMULACRA_NARRATION_KEY_FORMAT.format(sim_id)
                        session.state[narration_key] = narration_text
                        logger.info(f"Stored final narration for {sim_id} under key '{narration_key}'.")
                        # Print the final narration for this character
                        console.print(Rule(f"Narrative for {sim_id}", style="dim"))
                        console.print(f"{narration_text}\n")
                    else:
                        logger.warning(f"Invalid narration format for {sim_id}: Expected string, got {type(narration_text)}")
            else:
                logger.warning("No valid narration results received/parsed from NarrationAgent.")

            console.print("[green]Phase 5 Complete.[/green]")
        except Exception as e:
            logger.exception("Error during Narration (Phase 5).")
            console.print(f"[bold red]Error in Phase 5: {e}.[/bold red]")
            # Continue to end of turn

        # Pause slightly between turns
        console.print(f"--- End of Turn {turn + 1} ---")
        await asyncio.sleep(1) # Adjust as needed

    console.rule("[bold cyan]Simulation Finished[/bold cyan]", style="cyan")