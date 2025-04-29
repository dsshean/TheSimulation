# src/simulation_loop.py (Phased Turn Orchestration Loop)
import asyncio
import json
import logging
from rich.console import Console
from rich.rule import Rule # Import Rule for visual separators
from rich.padding import Padding # Import Padding for indentation
from rich.pretty import pretty_repr # For pretty printing dicts/lists
from typing import Dict, List, Any, Optional, Set
from google.adk.agents import BaseAgent, ParallelAgent # LLMAgent no longer needed directly here
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, Session
from google.adk.events import Event, EventActions # <<< ADD Event, EventActions import
from google.genai import types
from datetime import datetime, timedelta
from src.loop_utils import print_event_details, parse_json_output, format_iso_timestamp
from src.agents.world_engine import create_world_engine_validator

console = Console(width=120) # Adjust width as needed

logger = logging.getLogger(__name__)

ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent"
CREATE_WORLD_ENGINE_VALIDATOR_KEY = "create_world_engine_validator"
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
    # Agent Instances/Dicts
    world_state_agent: BaseAgent,
    simulacra_agents: Dict[str, BaseAgent],
    # --- ADDED: Accept validator agents dict ---
    validator_agents: Dict[str, BaseAgent],
    # ---
    npc_agent: BaseAgent,
    world_execution_agent: BaseAgent,
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
    5. Physical Execution (WorldExecutionAgent)
    6. Narration (NarrationAgent)
    """
    console.print(Padding(f"\n[bold cyan]--- Starting Phased Simulation (Max {max_turns} Turns) ---[/bold cyan]", (1, 0, 1, 0)))
    console.print(f"Session ID: {session.id}")
    console.print("-" * 30) # Simple separator

    user_id = session.user_id
    session_id = session.id

    initial_primary_location = session.state.get("simulation_primary_location")
    if not initial_primary_location:
        initial_primary_location = "New York City, NY" # Fallback location
        logger.warning(f"Could not find 'simulation_primary_location' in session state. Falling back to default: {initial_primary_location}")
    else:
        logger.info(f"Using initial primary location from session state: {initial_primary_location}")

    for turn in range(max_turns):
        console.rule(f"Turn {turn + 1}/{max_turns}", style="bold blue")
        console.print() # Add vertical space

        active_sim_ids_for_turn = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
        sim_location_map: Dict[str, str] = {}

        if not active_sim_ids_for_turn:
            logger.warning("No active simulacra found at start of turn. Will use initial primary location for query.")
            console.print(Padding("[yellow]No active simulacra this turn.[/yellow]", (0, 0, 1, 2)))
        else:
            console.print(Padding(f"Active Simulacra: {', '.join(active_sim_ids_for_turn)}", (0, 0, 0, 2)))
            for sim_id in active_sim_ids_for_turn:
                loc_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
                current_loc = session.state.get(loc_key)
                if current_loc:
                    sim_location_map[sim_id] = current_loc
                else:
                    logger.warning(f"Could not find location for active sim {sim_id} using key {loc_key}. Using initial primary location as fallback.")
                    sim_location_map[sim_id] = initial_primary_location # Use fallback for tool query

        if sim_location_map:
            unique_locations_for_tool = list(dict.fromkeys(sim_location_map.values()))
        else:
            unique_locations_for_tool = [initial_primary_location] # Use fallback if no active sims

        logger.info(f"Unique locations to query details for: {unique_locations_for_tool}")
        console.print(Padding(f"Querying world details for locations: {', '.join(unique_locations_for_tool)}", (0, 0, 1, 2)))

        # --- Phase 1: World State Update & Sync (World State Agent) ---
        try:
            console.print(Rule("Phase 1: World State Update & Sync", style="dim yellow"))
            runner.agent = world_state_agent
            console.print(Padding(f"Running [bold]{world_state_agent.name}[/bold]...", (1, 0, 0, 2)))

            # --- Determine Primary Location for get_setting_details ---
            primary_location_for_tool = initial_primary_location # Default fallback
            if active_sim_ids_for_turn and sim_location_map:
                 first_active_sim_id = active_sim_ids_for_turn[0]
                 if first_active_sim_id in sim_location_map:
                     primary_location_for_tool = sim_location_map[first_active_sim_id]
            logger.info(f"Primary location for Phase 1 setting details: {primary_location_for_tool}")
            # ---

            # --- Trigger Text (Instructs agent to use both tools) ---
            phase1_trigger_text = (
                 f"Perform the start-of-turn world state update. "
                 f"First, use the 'get_setting_details' tool for the primary location '{primary_location_for_tool}'. "
                #  f"Then, use the 'update_and_get_world_state' tool to advance world time. "
                 f"Respond ONLY with a brief confirmation message like 'World state synced and time updated.'"
            )
            phase1_trigger = types.Content(parts=[types.Part(text=phase1_trigger_text)])
            console.print(Padding(f"[dim]Trigger: {phase1_trigger_text}[/dim]", (0, 0, 1, 4)))

            # --- MODIFICATION: Process state_delta from event.actions ---
            state_updated_via_delta = False
            final_confirmation_message = None

            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=phase1_trigger):
                print_event_details(event, "P1", console, logger) # Log event details first
                console.print(event) # Or console.log(event)

                # --- CORRECTED State Update Logic using state_delta ---
                if hasattr(event, 'actions') and event.actions:
                    # Directly check state_delta on the event.actions object
                    if hasattr(event.actions, 'state_delta') and isinstance(event.actions.state_delta, dict):
                        delta = event.actions.state_delta
                        if delta: # Check if the delta dictionary is not empty
                            logger.info(f"Applying state_delta: {list(delta.keys())}")
                            # Merge the delta into the main session state
                            session.state.update(delta)
                            state_updated_via_delta = True
                        else:
                            logger.debug("Received an event with an empty state_delta.")
                    # else: logger.debug("Event actions found, but no valid state_delta attribute.")
                # --- END CORRECTION ---

                # Capture final confirmation text (as per instructions)
                elif event.is_final_response() and event.content and event.content.parts:
                    part = event.content.parts[0]
                    if hasattr(part, 'text'):
                        final_confirmation_message = part.text
                        logger.info(f"World State Agent confirmation: {final_confirmation_message}")
                        # break # Optional: break after final confirmation if needed
            # --- End of async for loop ---

            # --- Display final world state (reading from potentially updated session.state) ---
            if not state_updated_via_delta:
                 # This might be okay if the initial state was already correct and tools made no changes,
                 # but usually we expect updates in Phase 1.
                 logger.warning(f"No state_delta updates were applied during Phase 1 loop. State may be unchanged or stale.")
                 console.print(Padding(f"[yellow]Warning: No state updates applied via state_delta in Phase 1.[/yellow]", (1, 0, 0, 4)))

            current_world_state = session.state.get(WORLD_STATE_KEY, {}) # Read the potentially updated state
            console.print(Padding("[bold]World State (After Phase 1):[/bold]", (1, 0, 0, 4)))
            if current_world_state:
                 console.print(Padding(pretty_repr(current_world_state), (0, 0, 0, 4)))
            else:
                 # This case should be less likely if the update logic works
                 console.print(Padding("[yellow]World state key not found or empty after Phase 1 execution.[/yellow]", (0, 0, 0, 4)))
            # ---
            console.print(Padding("[green]Phase 1 Complete.[/green]", (1, 0, 1, 2)))

        except Exception as e:
             logger.exception("Error during World State Update & Sync (Phase 1).")
             console.print(Padding(f"[bold red]Error in Phase 1: {e}. State may be inconsistent. Skipping turn.[/bold red]", (0, 0, 1, 2)))
             await asyncio.sleep(1)
             continue # Skip to next turn

        console.print() # Add vertical space

        # --- Phase 2: Parallel Simulacra Intent Generation ---
        active_sim_ids = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
        active_sim_instances = [sim_agent for sim_id, sim_agent in simulacra_agents.items() if sim_id in active_sim_ids]

        if not active_sim_instances:
             console.print(Padding("[yellow]Phase 2: No active simulacra found. Skipping Intent/Validation/Interaction/Execution.[/yellow]", (1, 0, 1, 0)))
             session.state[TURN_VALIDATION_RESULTS_KEY] = {}
             session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
        else:
            try:
                console.print(Rule(f"Phase 2: Parallel Simulacra Intent Generation ({len(active_sim_instances)} acting)", style="dim yellow"))

                # Ensure parent_agent is None before running in parallel
                for agent_instance in active_sim_instances:
                    if hasattr(agent_instance, 'parent_agent'):
                        agent_instance.parent_agent = None

                parallel_sim_agent = ParallelAgent(name=f"ParallelSimulacra_Turn{turn+1}", sub_agents=active_sim_instances)
                runner.agent = parallel_sim_agent
                console.print(Padding(f"Running [bold]{parallel_sim_agent.name}[/bold]...", (1, 0, 0, 2)))
                phase2_trigger_text = (
                    "Based on your current status, goal, recent monologue, and the overall world state, "
                    "decide on your primary action/intent for this turn. "
                    "Use your available tools (e.g., 'check_self_status', 'generate_internal_monologue', 'attempt_move_to', 'attempt_interact') "
                    "to reflect and set your intent in the session state."
                )
                phase2_trigger = types.Content(parts=[types.Part(text=phase2_trigger_text)])
                console.print(Padding(f"[dim]Trigger: {phase2_trigger_text}[/dim]", (0, 0, 1, 4)))

                # --- ADDED: Flag for state updates ---
                state_updated_via_delta_phase2 = False
                # ---

                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=phase2_trigger):
                    print_event_details(event, "P2", console, logger) # Handles its own formatting
                    console.print(event)
                    # --- ADDED: State Update Logic using state_delta ---
                    if hasattr(event, 'actions') and event.actions:
                        # Directly check state_delta on the event.actions object
                        if hasattr(event.actions, 'state_delta') and isinstance(event.actions.state_delta, dict):
                            delta = event.actions.state_delta
                            if delta: # Check if the delta dictionary is not empty
                                logger.info(f"P2 Applying state_delta: {list(delta.keys())}")
                                # Merge the delta into the main session state
                                session.state.update(delta)
                                state_updated_via_delta_phase2 = True
                            else:
                                logger.debug("P2 Received an event with an empty state_delta.")
                        # else: logger.debug("P2 Event actions found, but no valid state_delta attribute.")
                    # --- END State Update Logic ---

                    # Optional: Add logic here if you need to handle final response from ParallelAgent specifically
                    elif event.is_final_response():
                         logger.info(f"P2 Received final response from ParallelAgent.")
                         # break # Usually safe to break after final response from ParallelAgent

                # --- MODIFIED: Display generated intents and monologues (if saved by agent/tool) ---
                console.print(Padding("[bold]Simulacra Outputs (from Session State after Phase 2):[/bold]", (1, 0, 0, 4)))
                # --- ADDED: Warning if no updates occurred ---
                if not state_updated_via_delta_phase2:
                    logger.warning("P2 No state_delta updates were applied during Phase 2 loop. Monologue/Intent state may be missing.")
                    console.print(Padding(f"[yellow]Warning: No state updates applied via state_delta in Phase 2.[/yellow]", (0, 0, 0, 4)))
                # ---
                for sim_id in active_sim_ids: # Use active_sim_ids defined earlier in the phase
                    intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id)
                    monologue_key = SIMULACRA_MONOLOGUE_KEY_FORMAT.format(sim_id)
                    intent = session.state.get(intent_key)
                    monologue = session.state.get(monologue_key)
                    console.print(Padding(f"[bold]{sim_id}:[/bold]", (0, 0, 0, 6)))
                    console.print(Padding(f"  Intent: {pretty_repr(intent) if intent else '[No Intent Set]'}", (0, 0, 0, 6)))
                    console.print(Padding(f"  Monologue: {monologue if monologue else '[No Monologue Saved]'}", (0, 0, 1, 6)))
                # ---
                console.print(Padding("[green]Phase 2 Complete.[/green]", (1, 0, 1, 2)))

            except Exception as e:
                 # --- MODIFIED: Ensure full exception logging and skip turn ---
                 logger.exception("Error during Parallel Simulacra execution (Phase 2).")
                 console.print(Padding(f"[bold red]Error in Phase 2: {e}. Skipping rest of turn phases.[/bold red]", (0, 0, 1, 2)))
                 session.state[TURN_VALIDATION_RESULTS_KEY] = {} # Ensure subsequent phases know validation didn't happen
                 session.state[TURN_EXECUTION_NARRATIVES_KEY] = {} # Ensure subsequent phases know execution didn't happen
                 await asyncio.sleep(1) # Brief pause before skipping
                 continue # Skip to next turn
            # ---

        console.print() # Add vertical space

        # --- Phase 3: Parallel Physical Validation ---
        active_sim_ids_for_validation = active_sim_ids # Use same IDs as Phase 2
        session.state[TURN_VALIDATION_RESULTS_KEY] = {} # Clear/initialize results key

        if not active_sim_ids_for_validation:
            console.print(Padding("[yellow]Phase 3: No active simulacra found. Skipping Validation.[/yellow]", (1, 0, 1, 0)))
        else:
            try:
                console.print(Rule(f"Phase 3: Parallel Physical Validation ({len(active_sim_ids_for_validation)} targets)", style="dim yellow"))

                validation_sub_agents = []
                missing_validator_ids = []
                for sim_id in active_sim_ids_for_validation:
                    # --- MODIFIED: Get pre-instantiated agent ---
                    sub_agent = validator_agents.get(sim_id)
                    if sub_agent and isinstance(sub_agent, BaseAgent):
                        # Ensure parent_agent is None before adding to ParallelAgent
                        if hasattr(sub_agent, 'parent_agent'):
                            sub_agent.parent_agent = None
                        validation_sub_agents.append(sub_agent)
                        # logger.info(f"Using pre-instantiated validator agent for {sim_id}") # Optional log
                    else:
                        # Handle case where agent wasn't created/passed correctly
                        logger.error(f"Validator agent for {sim_id} not found in passed dictionary. Skipping.")
                        console.print(Padding(f"[red]Error: Validator agent for {sim_id} missing. Skipping this agent.[/red]", (0, 0, 1, 4)))
                        missing_validator_ids.append(sim_id)
                    # --- END MODIFICATION ---

                if not validation_sub_agents:
                     console.print(Padding("[yellow]Phase 3: No valid validator sub-agents found to run. Skipping phase.[/yellow]", (1, 0, 1, 0)))
                     session.state[TURN_VALIDATION_RESULTS_KEY] = {}
                     session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
                else:
                    # Proceed with the successfully found agents
                    parallel_validation_agent = ParallelAgent(name=f"ParallelValidation_Turn{turn+1}", sub_agents=validation_sub_agents)
                    runner.agent = parallel_validation_agent
                    console.print(Padding(f"Running [bold]{parallel_validation_agent.name}[/bold] with {len(validation_sub_agents)} sub-agents...", (1, 0, 0, 2))) # Reduced vertical space

                    # --- ADD: Trigger message for validation ---
                    phase3_trigger_text = (
                        "Validate the intent for your assigned simulacrum based on the current world state and rules. "
                        "Use the 'save_single_validation_result' tool to store the outcome."
                    )
                    phase3_trigger = types.Content(parts=[types.Part(text=phase3_trigger_text)])
                    console.print(Padding(f"[dim]Trigger: {phase3_trigger_text}[/dim]", (0, 0, 1, 4)))
                    # --- END ADD ---

                    state_updated_via_delta_phase3 = False

                    # --- MODIFIED: Add new_message argument ---
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=phase3_trigger # Pass the trigger message
                    ):
                    # --- END MODIFICATION ---
                        print_event_details(event, "P3", console, logger)
                        console.print(event)
                        # --- State Update Logic for Merging Deltas ---
                        if hasattr(event, 'actions') and event.actions and hasattr(event.actions, 'state_delta') and isinstance(event.actions.state_delta, dict):
                            delta = event.actions.state_delta
                            if delta:
                                logger.info(f"P3 Received state_delta: {list(delta.keys())}")
                                for key, value in delta.items():
                                    # Specifically merge results into the TURN_VALIDATION_RESULTS_KEY dictionary
                                    if key == TURN_VALIDATION_RESULTS_KEY and isinstance(value, dict):
                                        if TURN_VALIDATION_RESULTS_KEY not in session.state or not isinstance(session.state[TURN_VALIDATION_RESULTS_KEY], dict):
                                            session.state[TURN_VALIDATION_RESULTS_KEY] = {} # Initialize if missing/wrong type
                                        session.state[TURN_VALIDATION_RESULTS_KEY].update(value)
                                        logger.info(f"Merged validation result for keys: {list(value.keys())}")
                                        state_updated_via_delta_phase3 = True
                                    else:
                                        # Apply other unexpected delta keys directly (though ideally only TURN_VALIDATION_RESULTS_KEY comes from these agents)
                                        logger.warning(f"P3 Received unexpected key in state_delta: {key}. Applying directly.")
                                        session.state[key] = value
                                        state_updated_via_delta_phase3 = True
                            else:
                                logger.debug("P3 Received an event with an empty state_delta.")
                        # --- END State Update Logic ---

                        elif event.is_final_response() and event.content and event.content.parts:
                            # Log the final confirmation message from sub-agents (as per their instructions)
                            part = event.content.parts[0]
                            if hasattr(part, 'text') and hasattr(event, 'invoking_agent_name'):
                                logger.info(f"P3 Final confirmation from {event.invoking_agent_name}: {part.text}")

                    # --- Display combined validation results from state ---
                    console.print(Padding("[bold]Validation Results (from Session State after Phase 3):[/bold]", (1, 0, 0, 4)))
                    if not state_updated_via_delta_phase3:
                        logger.warning("P3 No state_delta updates were applied during Phase 3 loop. Validation results may be missing.")
                        console.print(Padding(f"[yellow]Warning: No state updates applied via state_delta in Phase 3.[/yellow]", (0, 0, 0, 4)))

                    validation_results_from_state = session.state.get(TURN_VALIDATION_RESULTS_KEY, {})
                    if validation_results_from_state:
                        console.print(Padding(pretty_repr(validation_results_from_state), (0, 0, 0, 4)))
                    else:
                        console.print(Padding(f"[yellow]No validation results found in session state under key '{TURN_VALIDATION_RESULTS_KEY}'.[/yellow]", (0, 0, 0, 4)))
                        # Ensure the key exists even if empty, for subsequent phases
                        if TURN_VALIDATION_RESULTS_KEY not in session.state:
                            session.state[TURN_VALIDATION_RESULTS_KEY] = {}

                    console.print(Padding("[green]Phase 3 Complete.[/green]", (1, 0, 1, 2)))

            except Exception as e:
                # Catch errors during the overall Phase 3 execution (e.g., ParallelAgent run)
                logger.exception("Error during Parallel Validation (Phase 3).")
                console.print(Padding(f"[bold red]Error in Phase 3: {e}. Skipping Interaction/Execution.[/bold red]", (0, 0, 1, 2)))
                session.state[TURN_VALIDATION_RESULTS_KEY] = {} # Ensure empty if error
                session.state[TURN_EXECUTION_NARRATIVES_KEY] = {}
                # Fall through to Narration (Phase 5) which should handle missing context

        console.print() # Add vertical space

        # --- Phase 4a (Interaction) ---
        # Only run if validation happened (even if results were empty)
        if TURN_VALIDATION_RESULTS_KEY in session.state:
            try:
                console.print(Rule("Phase 4a: Interaction Resolution", style="dim yellow"))
                runner.agent = npc_agent
                # Modified trigger slightly to suggest state update via tool/context
                interaction_trigger_text = "Resolve pending 'talk' and 'interact' actions based on validation status. Store results in state."
                interaction_trigger = types.Content(parts=[types.Part(text=interaction_trigger_text)])
                console.print(Padding(f"Running [bold]{npc_agent.name}[/bold]...", (1, 0, 0, 2)))
                console.print(Padding(f"[dim]Trigger: {interaction_trigger_text}[/dim]", (0, 0, 1, 4)))

                # --- ADDED: Flag for state updates ---
                state_updated_via_delta_phase4a = False
                # ---
                final_interaction_summary_text = None # Optional: Capture final text for summary/logging

                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=interaction_trigger):
                    print_event_details(event, "P4a", console, logger) # Handles its own formatting
                    console.print(event)
                    # --- ADDED: State Update Logic using state_delta ---
                    if hasattr(event, 'actions') and event.actions:
                        # Directly check state_delta on the event.actions object
                        if hasattr(event.actions, 'state_delta') and isinstance(event.actions.state_delta, dict):
                            delta = event.actions.state_delta
                            if delta: # Check if the delta dictionary is not empty
                                logger.info(f"P4a Applying state_delta: {list(delta.keys())}")
                                # Merge the delta into the main session state
                                session.state.update(delta)
                                state_updated_via_delta_phase4a = True
                            else:
                                logger.debug("P4a Received an event with an empty state_delta.")
                        # else: logger.debug("P4a Event actions found, but no valid state_delta attribute.")
                    # --- END State Update Logic ---

                    # Optional: Capture final text
                    elif event.is_final_response() and event.content and event.content.parts:
                        part = event.content.parts[0]
                        if hasattr(part, 'text'):
                            final_interaction_summary_text = part.text
                            logger.info(f"P4a Final summary text: {final_interaction_summary_text}")
                            # break # Optional break

                # --- MODIFIED: Display interaction results (reading from state updated via delta) ---
                console.print(Padding("[bold]Interaction Results (from Session State after Phase 4a):[/bold]", (1, 0, 0, 4)))
                # --- ADDED: Warning if no updates occurred ---
                if not state_updated_via_delta_phase4a:
                    # This might be normal if no interactions were validated/occurred
                    logger.info("P4a No state_delta updates were applied during Phase 4a loop (may be normal).")
                    console.print(Padding(f"[dim]No state updates applied via state_delta in Phase 4a (no interactions?).[/dim]", (0, 0, 0, 4)))
                # ---
                found_interactions = False
                # Read active sims again in case state changed, though unlikely here
                active_sim_ids_post_interaction = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
                for sim_id in active_sim_ids_post_interaction:
                    # Assuming the agent/tool saved results under INTERACTION_RESULT_KEY_FORMAT via state_delta
                    interaction_key = INTERACTION_RESULT_KEY_FORMAT.format(sim_id)
                    interaction_result = session.state.get(interaction_key)
                    if interaction_result: # Only print if there was a result saved
                         console.print(Padding(f"[bold]{sim_id}:[/bold] {pretty_repr(interaction_result)}", (0, 0, 1, 6)))
                         found_interactions = True
                if not found_interactions:
                     console.print(Padding("[dim]No interaction results found in session state.[/dim]", (0, 0, 1, 6)))
                # ---
                console.print(Padding("[green]Phase 4a Complete.[/green]", (1, 0, 1, 2)))

            except Exception as e:
                 logger.exception("Error during NPC Interaction Resolution (Phase 4a).")
                 console.print(Padding(f"[bold red]Error in Phase 4a: {e}.[/bold red]", (0, 0, 1, 2)))
                 # Continue to 4b/5, interaction results might just be missing

        else:
            # This case should ideally not happen if Phase 3 runs, but good for safety
            console.print(Padding("[yellow]Phase 4a: Skipping because Phase 3 validation results key is missing.[/yellow]", (1, 0, 1, 2)))

        console.print() # Add vertical space

        # --- Phase 4b: Physical Action Execution & Narration ---
        all_validation_results = session.state.get(TURN_VALIDATION_RESULTS_KEY, {})
        if all_validation_results: # Check if validation results exist (even if empty)
            try:
                console.print(Rule("Phase 4b: Physical Execution (Moves)", style="dim yellow"))
                approved_physical_actions_for_batch = []
                active_sim_ids_for_physical = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])

                logger.debug("--- Starting Phase 4b: Checking for approved 'move' actions ---")
                for sim_id in active_sim_ids_for_physical:
                    validation_result = all_validation_results.get(sim_id)
                    if validation_result:
                        original_intent = validation_result.get("original_intent")
                        action_type = original_intent.get("action_type") if original_intent else None
                        val_status = validation_result.get("validation_status")
                        estimated_duration = validation_result.get("estimated_duration_seconds") # From Validator

                        if val_status in ["approved", "modified"] and action_type == "move":
                            logger.info(f"  Action Approved/Modified for {sim_id}: Type={action_type}, Status={val_status}. Adding to batch.")
                            # Pass only necessary info to execution agent
                            action_detail = {
                                "sim_id": sim_id,
                                "action_type": action_type, # Should always be 'move' here
                                "details": original_intent, # Pass original intent for context
                                # Note: Execution agent will recalculate duration based on specifics
                            }
                            approved_physical_actions_for_batch.append(action_detail)
                        elif action_type != "move":
                             logger.debug(f"  Skipping action for {sim_id} in Phase 4b: Type '{action_type}' handled elsewhere (e.g., Phase 4a).")
                        else: # Failed or invalid move
                             logger.debug(f"  Action for {sim_id} not added to physical batch (Status: {val_status}, Type: {action_type})")
                    else:
                        logger.debug(f"  Skipping {sim_id}: Missing validation_result in combined dict.")

                if approved_physical_actions_for_batch:
                    console.print(Padding(f"Found {len(approved_physical_actions_for_batch)} approved 'move' actions.", (1, 0, 0, 2)))
                    console.print(Padding(f"Running [bold]{world_execution_agent.name}[/bold]...", (0, 0, 0, 2)))

                    runner.agent = world_execution_agent

                    current_world_time_iso = session.state.get(WORLD_STATE_KEY, {}).get("world_time")
                    current_world_time_str = format_iso_timestamp(current_world_time_iso) if current_world_time_iso else "[Unknown Time]"

                    try:
                        # Limit size for console display, log full data
                        actions_json_string_short = json.dumps(approved_physical_actions_for_batch[:2]) + ("..." if len(approved_physical_actions_for_batch) > 2 else "")
                        actions_json_string_full = json.dumps(approved_physical_actions_for_batch, indent=2)
                    except TypeError:
                        logger.error("Could not serialize actions batch to JSON for prompt.")
                        actions_json_string_short = str(approved_physical_actions_for_batch)
                        actions_json_string_full = actions_json_string_short # Fallback

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
                        f"Actions to process:\n```json\n{actions_json_string_full}\n```" # Pass full JSON to agent
                    )
                    execution_trigger = types.Content(parts=[types.Part(text=execution_trigger_text)])
                    # Print shorter version to console
                    console.print(Padding(f"[dim]Trigger includes {len(approved_physical_actions_for_batch)} actions. Example: {actions_json_string_short}[/dim]", (0, 0, 1, 4)))

                    parsed_execution_results = None
                    final_execution_text = None
                    state_updated_via_delta_phase4b = False # Flag for agent's own delta

                    async for event in runner.run_async(
                        user_id=user_id, session_id=session_id, new_message=execution_trigger
                    ):
                        print_event_details(event, "P4b", console, logger)
                        console.print(event)
                        # --- Process state_delta FROM THE AGENT/TOOL itself ---
                        if hasattr(event, 'actions') and event.actions:
                            if hasattr(event.actions, 'state_delta') and isinstance(event.actions.state_delta, dict):
                                delta = event.actions.state_delta
                                if delta:
                                    logger.info(f"P4b Applying state_delta from agent: {list(delta.keys())}")
                                    session.state.update(delta)
                                    state_updated_via_delta_phase4b = True
                                else:
                                    logger.debug("P4b Received an event with an empty state_delta from agent.")
                        # ---

                        # Capture final text
                        if event.is_final_response() and event.content and event.content.parts:
                             part = event.content.parts[0]
                             if hasattr(part, 'text'):
                                 final_execution_text = part.text

                    # Parse the result from the agent's final text
                    parsed_execution_results = parse_json_output(
                        final_execution_text, "P4b", world_execution_agent.name, console, logger
                    )

                    execution_narratives = {}
                    max_duration = 0
                    location_updates_delta = {} # <<< Initialize dict to collect location updates

                    console.print(Padding("[bold]Processing Execution Results:[/bold]", (1, 0, 0, 4)))
                    if parsed_execution_results:
                        for sim_id, result_data in parsed_execution_results.items():
                            if isinstance(result_data, dict):
                                narrative = result_data.get("narrative")
                                new_location = result_data.get("new_location")
                                duration = result_data.get("duration_seconds", 0)

                                console.print(Padding(f"[bold]{sim_id}:[/bold]", (0, 0, 0, 6)))
                                if narrative:
                                    execution_narratives[sim_id] = narrative
                                    logger.info(f"Stored execution narrative for {sim_id}.")
                                else:
                                     console.print(Padding(f"  Narrative: [yellow][Missing][/yellow]", (0, 0, 0, 6)))

                                if new_location:
                                    location_key = SIMULACRA_LOCATION_KEY_FORMAT.format(sim_id)
                                    # --- REMOVED Direct Update ---
                                    # session.state[location_key] = new_location
                                    # --- ADDED Collection for Delta ---
                                    location_updates_delta[location_key] = new_location
                                    # ---
                                    logger.info(f"Collected location update for {sim_id} to '{new_location}'.")
                                    console.print(Padding(f"  New Location: '{new_location}' (pending state update)", (0, 0, 0, 6))) # Indicate pending
                                else:
                                    logger.warning(f"No 'new_location' provided by agent for move action of {sim_id}.")
                                    console.print(Padding(f"  New Location: [yellow][Missing][/yellow]", (0, 0, 0, 6)))

                                if isinstance(duration, int) and duration >= 0:
                                     console.print(Padding(f"  Duration: {duration}s", (0, 0, 1, 6)))
                                     if duration > max_duration:
                                         max_duration = duration
                                else:
                                     logger.warning(f"Invalid or missing 'duration_seconds' for {sim_id}: {duration}")
                                     console.print(Padding(f"  Duration: [yellow][Invalid: {duration}][/yellow]", (0, 0, 1, 6)))

                            else:
                                logger.warning(f"Invalid result format for {sim_id} in execution JSON: {result_data}")
                                console.print(Padding(f"[bold]{sim_id}:[/bold] [red]Invalid result format[/red]", (0, 0, 1, 6)))

                        # --- ADDED: Apply collected location updates via a new event ---
                        if location_updates_delta:
                            logger.info(f"Applying collected location updates via separate event: {list(location_updates_delta.keys())}")
                            location_update_event = Event(
                                # Generate a unique ID or use a consistent pattern
                                invocation_id=f"inv_p4b_loc_update_{turn+1}",
                                author="system_orchestrator", # Indicate it's from the loop logic
                                actions=EventActions(state_delta=location_updates_delta),
                                timestamp=datetime.now().timestamp() # Use current time
                            )
                            try:
                                # Use the session service to append this event, which handles state update
                                session_service.append_event(session, location_update_event)
                                console.print(Padding("[green]Location updates applied via state_delta event.[/green]", (1, 0, 0, 4)))
                            except Exception as append_e:
                                logger.exception(f"Failed to append location update event: {append_e}")
                                console.print(Padding(f"[bold red]Error applying location updates via event: {append_e}[/bold red]", (1, 0, 0, 4)))
                                # Note: State might be inconsistent if this fails
                        else:
                            logger.info("No location updates collected to apply.")
                        # --- END ADDED ---

                        # --- World time update remains direct (necessary orchestration step) ---
                        if max_duration > 0:
                            try:
                                world_state = session.state.get(WORLD_STATE_KEY, {})
                                current_time_iso = world_state.get("world_time")
                                if current_time_iso:
                                    current_dt = datetime.fromisoformat(current_time_iso)
                                    new_dt = current_dt + timedelta(seconds=max_duration)
                                    if WORLD_STATE_KEY not in session.state:
                                        session.state[WORLD_STATE_KEY] = {}
                                    session.state[WORLD_STATE_KEY]["world_time"] = new_dt.isoformat() # Direct update kept
                                    formatted_new_time = format_iso_timestamp(new_dt.isoformat())
                                    logger.info(f"Advanced world time by {max_duration} seconds to {formatted_new_time}.")
                                    console.print(Padding(f"[bold]World Time Advanced:[/bold] +{max_duration}s -> {formatted_new_time}", (1, 0, 0, 4)))
                                else:
                                    logger.warning("Could not find world_time in state to advance.")
                                    console.print(Padding("[yellow]Could not advance world time (missing current time).[/yellow]", (1, 0, 0, 4)))
                            except Exception as time_e:
                                logger.exception(f"Error advancing world time: {time_e}")
                                console.print(Padding(f"[red]Error advancing world time: {time_e}[/red]", (1, 0, 0, 4)))
                        else:
                             console.print(Padding("[dim]No time advancement (max duration was 0).[/dim]", (1, 0, 0, 4)))
                        # --- End World time update ---

                    else: # parsed_execution_results was None
                        console.print(Padding("[yellow]Could not parse execution results.[/yellow]", (0, 0, 0, 4)))

                    # Store the collected narratives (even if empty)
                    session.state[TURN_EXECUTION_NARRATIVES_KEY] = execution_narratives # Direct update ok for intermediate results used later in same turn
                    console.print(Padding("[green]Phase 4b Complete.[/green]", (1, 0, 1, 2)))

                else: # No approved moves
                    console.print(Padding("[yellow]Phase 4b: No approved 'move' actions found to execute.[/yellow]", (1, 0, 1, 2)))
                    session.state[TURN_EXECUTION_NARRATIVES_KEY] = {} # Ensure key exists and is empty

            except Exception as e:
                logger.exception("Error during Physical Execution (Phase 4b).")
                console.print(Padding(f"[bold red]Error in Phase 4b: {e}. Skipping phase.[/bold red]", (0, 0, 1, 2)))
                session.state[TURN_EXECUTION_NARRATIVES_KEY] = {} # Ensure empty on error
                await asyncio.sleep(1)

        else: # Validation results key missing (shouldn't happen if P3 ran)
             console.print(Padding("[yellow]Phase 4b: Skipping due to missing validation results from Phase 3.[/yellow]", (1, 0, 1, 2)))
             session.state[TURN_EXECUTION_NARRATIVES_KEY] = {} # Ensure empty

        console.print() # Add vertical space

        # --- Phase 5: Narration ---
        try:
            console.print(Rule("Phase 5: Narration", style="dim yellow"))
            runner.agent = narration_agent
            # Modified trigger slightly to suggest state update via tool/context
            narration_trigger_text = (
                "Generate the final narrative summary for this turn. "
                f"Use the 'get_narration_context' tool to gather necessary information. "
                "Store the final narrative for each relevant simulacra in the session state using the appropriate keys "
                f"(e.g., '{SIMULACRA_NARRATION_KEY_FORMAT.format('<sim_id>')}') via tool context."
            )
            narration_trigger = types.Content(parts=[types.Part(text=narration_trigger_text)])
            console.print(Padding(f"Running [bold]{narration_agent.name}[/bold]...", (1, 0, 0, 2)))
            console.print(Padding(f"[dim]Trigger: {narration_trigger_text}[/dim]", (0, 0, 1, 4)))

            # --- ADDED: Flag for state updates ---
            state_updated_via_delta_phase5 = False
            # ---
            final_narration_summary_text = None # Optional: Capture final text for summary/logging

            # Run agent and process state delta
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=narration_trigger):
                print_event_details(event, "P5", console, logger) # Handles its own formatting
                console.print(event)
                # --- ADDED: State Update Logic using state_delta ---
                if hasattr(event, 'actions') and event.actions:
                    # Directly check state_delta on the event.actions object
                    if hasattr(event.actions, 'state_delta') and isinstance(event.actions.state_delta, dict):
                        delta = event.actions.state_delta
                        if delta: # Check if the delta dictionary is not empty
                            logger.info(f"P5 Applying state_delta: {list(delta.keys())}")
                            # Merge the delta into the main session state
                            session.state.update(delta)
                            state_updated_via_delta_phase5 = True
                        else:
                            logger.debug("P5 Received an event with an empty state_delta.")
                    # else: logger.debug("P5 Event actions found, but no valid state_delta attribute.")
                # --- END State Update Logic ---

                # Optional: Capture final text
                elif event.is_final_response() and event.content and event.content.parts:
                    part = event.content.parts[0]
                    if hasattr(part, 'text'):
                        final_narration_summary_text = part.text
                        logger.info(f"P5 Final summary text: {final_narration_summary_text}")
                          # break # Optional break

            # --- REMOVED: Parsing final text for state update ---
            # parsed_narration_results = parse_json_output(...)

            # --- MODIFIED: Print final narratives directly from state ---
            console.print(Rule("[bold magenta]Final Turn Narratives (from Session State)[/bold magenta]", style="magenta"))
            # --- ADDED: Warning if no updates occurred ---
            if not state_updated_via_delta_phase5:
                logger.warning("P5 No state_delta updates were applied during Phase 5 loop. Final narratives may be missing.")
                console.print(Padding(f"[yellow]Warning: No state updates applied via state_delta in Phase 5.[/yellow]", (0, 0, 0, 4)))
            # ---
            found_narratives = False
            # Read active sims again, as state might have changed (though unlikely here)
            active_sim_ids_for_narration = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
            for sim_id in active_sim_ids_for_narration:
                # Assuming the agent/tool saved results under SIMULACRA_NARRATION_KEY_FORMAT via state_delta
                narration_key = SIMULACRA_NARRATION_KEY_FORMAT.format(sim_id)
                narration_text = session.state.get(narration_key)
                if isinstance(narration_text, str):
                    # Print the final narration for this character with clear labeling
                    console.print(Padding(f"[bold]{sim_id}:[/bold]", (1, 0, 0, 2)))
                    console.print(Padding(f"{narration_text}", (0, 0, 1, 4)))
                    found_narratives = True
                # else: logger.debug(f"No narration found for {sim_id} under key {narration_key}") # Optional debug

            if not found_narratives:
                console.print(Padding("[yellow]No final narratives found in session state.[/yellow]", (1, 0, 1, 4)))

            console.print(Rule(style="magenta")) # Footer rule for the narrative block
            # ---
            console.print(Padding("[green]Phase 5 Complete.[/green]", (1, 0, 1, 2)))

        except Exception as e:
            logger.exception("Error during Narration (Phase 5).")
            console.print(Padding(f"[bold red]Error in Phase 5: {e}.[/bold red]", (0, 0, 1, 2)))

        # --- End of Turn ---
        console.print(Padding(f"--- End of Turn {turn + 1} ---", (1, 0, 1, 0)))
        console.print() # Add vertical space before next turn's rule
        await asyncio.sleep(1) # Keep slight pause

    console.rule("[bold cyan]Simulation Finished[/bold cyan]", style="bold cyan")

# --- Ensure loop_utils.print_event_details uses Padding appropriately ---
# (This function is not shown, but assume it adds indentation/padding itself)