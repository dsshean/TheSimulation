# src/simulation_loop.py (Phased Turn Orchestration Loop)
import asyncio
import json
import logging
from rich.console import Console
from rich.rule import Rule # Import Rule for visual separators
from typing import Dict, List, Any, Optional

# ADK Imports
from google.adk.agents import BaseAgent, ParallelAgent # Removed LlmAgent as it's not directly used here
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, Session
# Use the existing types import for Content, Part, ToolConfig etc.
from google.genai import types

console = Console()
logger = logging.getLogger(__name__)

# --- State key formats/names used consistently across agents and loop ---
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
# --- ADDED: Key format for storing individual narration ---
SIMULACRA_NARRATION_KEY_FORMAT = "simulacra_{}_last_narration"
# ---

async def run_phased_simulation(
    runner: Runner,
    session_service: BaseSessionService,
    session: Session,
    # Agent Instances (needed for setting runner.agent)
    world_state_agent: BaseAgent,
    simulacra_agents: Dict[str, BaseAgent], # Dict: {'sim_id': agent_instance}
    world_engine_agent: BaseAgent, # Validator role
    npc_agent: BaseAgent,          # Interaction Resolver role
    narration_agent: BaseAgent,    # Narrator role
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

    # Default trigger message for agents run via runner.run_async
    default_trigger = types.Content(parts=[types.Part(text="Proceed with turn phase.")])

    for turn in range(max_turns):

        console.rule(f"Turn {turn + 1}/{max_turns}", style="cyan")
        # Optional: Reload session at start of turn if using persistent service
        # session = session_service.get_session(...) # Use appropriate args
        # if not current_session: break
        # session = current_session # Update local reference

        # --- Phase 1: World Update ---
        try:
            console.print("[reverse yellow] Phase 1: World State Update [/reverse yellow]")
            runner.agent = world_state_agent
            console.print(f"[dim]Running WorldStateAgent ({world_state_agent.name}) to call 'update_and_get_world_state' tool...[/dim]")
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=default_trigger):
                # Use event.content.parts to check for tool calls/responses
                logger.debug(f"P1 Event: {event.author} - Final: {event.is_final_response()} - Actions: {event.actions} - Content: {str(event.content)}...")
                # --- Print Event Details --- START
                if event.content and event.content.parts:
                    part = event.content.parts[0]
                    if hasattr(part, 'function_call') and part.function_call: # Check field exists AND is not None
                         tool_call = part.function_call
                         console.print(f"[dim]  P1 ({event.author}) -> Tool Call: {tool_call.name} with args: {dict(tool_call.args)}[/dim]")
                    elif hasattr(part, 'function_response') and part.function_response: # Check field exists AND is not None
                         tool_response = part.function_response
                         console.print(f"[dim]  P1 ({event.author}) <- Tool Response: {tool_response.name} -> {str(tool_response.response)[:100]}...[/dim]")
                    elif event.is_final_response() and hasattr(part, 'text'): # Check for final text
                         console.print(f"[dim]  P1 ({event.author}) Final Output: {part.text if part.text else '[No text output]'}[/dim]")
                # --- Print Event Details --- END
            console.print("[green]Phase 1 Complete.[/green]")
        except Exception as e:
            logger.exception("Error during World State Update (Phase 1).")
            console.print(f"[bold red]Error in Phase 1: {e}. Skipping turn.[/bold red]")
            await asyncio.sleep(1) # Pause before potentially retrying next turn
            continue

        # --- Phase 2 ---
        active_sim_ids = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, []) # Gets ['eleanor_vance', 'eleanor_vance_2']
        # Uses the correct IDs to fetch agent instances from the simulacra_agents dict
        active_sim_instances = [sim_agent for sim_id, sim_agent in simulacra_agents.items() if sim_id in active_sim_ids]

        if not active_sim_instances:
             console.print("[yellow]Phase 2: No active simulacra found. Skipping Intent/Validation/Interaction/Execution.[/yellow]")
             # No 'pass' needed here, just don't enter the else block
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
                async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=default_trigger):
                    agent_id = event.author # Will be 'eleanor_vance', etc.
                    logger.debug(f"P2 Event ({agent_id}): Final: {event.is_final_response()} - Actions: {event.actions} - Content: {str(event.content)}...")
                    # --- Print Event Details --- START
                    if event.content and event.content.parts:
                        part = event.content.parts[0]
                        if hasattr(part, 'function_call') and part.function_call: # Check field exists AND is not None
                             tool_call = part.function_call
                             console.print(f"[dim]  P2 ({agent_id}) -> Tool Call: {tool_call.name} with args: {dict(tool_call.args)}[/dim]")
                        elif hasattr(part, 'function_response') and part.function_response: # Check field exists AND is not None
                             tool_response = part.function_response
                             response_str = str(tool_response.response) + ('...' if len(str(tool_response.response)) > 150 else '')
                             console.print(f"[dim]  P2 ({agent_id}) <- Tool Response: {tool_response.name} -> {response_str}[/dim]")
                        elif event.is_final_response() and hasattr(part, 'text'): # Check for final text
                             console.print(f"[dim]  P2 ({agent_id}) Final Output: {part.text if part.text else '[No text output]'}[/dim]")
                    # --- Print Event Details --- END
                console.print("[green]Phase 2 Complete.[/green]")
            except Exception as e:
                 logger.exception("Error during Parallel Simulacra execution (Phase 2).")
                 console.print(f"[bold red]Error in Phase 2: {e}. Skipping rest of turn phases.[/bold red]")
                 await asyncio.sleep(1)
                 continue

        # --- Phase 3 (Validation) ---
        try:
            console.print("[reverse yellow] Phase 3: Physical Validation [/reverse yellow]")
            runner.agent = world_engine_agent
            validation_trigger = types.Content(parts=[types.Part(text="Validate all pending actions based on current world state and rules.")])
            console.print(f"[dim]Running WorldEngineAgent ({world_engine_agent.name}) to validate intents based on instructions...[/dim]")
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=validation_trigger):
                logger.debug(f"P3 Event: {event.author} - Final: {event.is_final_response()} - Actions: {event.actions} - Content: {str(event.content)}...")
                # --- Print Event Details --- START
                if event.content and event.content.parts:
                    part = event.content.parts[0]
                    if hasattr(part, 'function_call') and part.function_call: # Check field exists AND is not None
                         tool_call = part.function_call
                         console.print(f"[dim]  P3 ({event.author}) -> Tool Call: {tool_call.name} with args: {dict(tool_call.args)}[/dim]")
                    elif hasattr(part, 'function_response') and part.function_response: # Check field exists AND is not None
                         tool_response = part.function_response
                         console.print(f"[dim]  P3 ({event.author}) <- Tool Response: {tool_response.name} -> {str(tool_response.response)[:100]}...[/dim]")
                    elif event.is_final_response() and hasattr(part, 'text'): # Check for final text
                         console.print(f"[dim]  P3 ({event.author}) Final Output (Validation): {part.text if part.text else '[No text output]'}[/dim]")
                # --- Print Event Details --- END
            console.print("[green]Phase 3 Complete.[/green]")
        except Exception as e:
            logger.exception("Error during World Engine Validation (Phase 3).")
            console.print(f"[bold red]Error in Phase 3: {e}. Skipping Interaction/Execution.[/bold red]")
             # Fall through to Narration

        # --- Phase 4a (Interaction) ---
        try:
            console.print("[reverse yellow] Phase 4a: Interaction Resolution [/reverse yellow]")
            runner.agent = npc_agent
            interaction_trigger = types.Content(parts=[types.Part(text="Resolve pending 'talk' and 'interact' actions based on validation status.")])
            console.print(f"[dim]Running NPCAgent ({npc_agent.name}) to resolve interactions using its tools...[/dim]")
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=interaction_trigger):
                logger.debug(f"P4a Event: {event.author} - Final: {event.is_final_response()} - Actions: {event.actions} - Content: {str(event.content)}...")
                # --- Print Event Details --- START
                if event.content and event.content.parts:
                    part = event.content.parts[0]
                    if hasattr(part, 'function_call') and part.function_call: # Check field exists AND is not None
                         tool_call = part.function_call
                         console.print(f"[dim]  P4a ({event.author}) -> Tool Call: {tool_call.name} with args: {dict(tool_call.args)}[/dim]")
                    elif hasattr(part, 'function_response') and part.function_response: # Check field exists AND is not None
                         tool_response = part.function_response
                         console.print(f"[dim]  P4a ({event.author}) <- Tool Response: {tool_response.name} -> {str(tool_response.response)[:100]}...[/dim]")
                    elif event.is_final_response() and hasattr(part, 'text'): # Check for final text
                         console.print(f"[dim]  P4a ({event.author}) Final Output: {part.text if part.text else '[No text output]'}[/dim]")
                # --- Print Event Details --- END
            console.print("[green]Phase 4a Complete.[/green]")
        except Exception as e:
             logger.exception("Error during NPC Interaction Resolution (Phase 4a).")
             console.print(f"[bold red]Error in Phase 4a: {e}.[/bold red]")
             # Physical execution might still proceed

        # --- Phase 4b: Physical Action Execution (World State Agent) ---
        try:
            console.print("[reverse yellow] Phase 4b: Physical Action Execution [/reverse yellow]")
            approved_physical_actions_for_batch = []
            active_sim_ids_for_physical = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, []) # Gets ['eleanor_vance', ...]

            logger.debug("--- Starting Phase 4b: Checking for approved physical actions ---")
            for sim_id in active_sim_ids_for_physical: # sim_id is 'eleanor_vance', etc.
                # --- Uses updated format strings with the correct sim_id ---
                intent_key = SIMULACRA_INTENT_KEY_FORMAT.format(sim_id)
                validation_key = ACTION_VALIDATION_KEY_FORMAT.format(sim_id)
                # ---

                intent_data = session.state.get(intent_key)
                validation_result = session.state.get(validation_key)

                logger.debug(f"Checking {sim_id}: Intent Key='{intent_key}', Validation Key='{validation_key}'")
                logger.debug(f"  Intent Data: {intent_data}")
                logger.debug(f"  Validation Result: {validation_result}")

                if intent_data and validation_result:
                    action_type = intent_data.get("action_type")
                    # Ensure validation_result is a dictionary before accessing keys
                    val_status = validation_result.get("validation_status") if isinstance(validation_result, dict) else None
                    estimated_duration = validation_result.get("estimated_duration_seconds") if isinstance(validation_result, dict) else None

                    logger.debug(f"  Extracted: action_type='{action_type}', validation_status='{val_status}'")

                    # Check if it's an approved/modified PHYSICAL action (move, interact)
                    if val_status in ["approved", "modified"] and action_type in ["move", "interact"]: # Add other physical types if needed
                        logger.info(f"  Action Approved/Modified for {sim_id}: Type={action_type}, Status={val_status}. Adding to batch.")
                        # Prepare the action details for the batch tool
                        action_detail = {
                            "sim_id": sim_id, # Pass the correct ID ('eleanor_vance')
                            "action_type": action_type,
                            "details": intent_data, # Pass the original intent details
                            "estimated_duration_seconds": estimated_duration # Pass duration if needed by executor
                        }
                        approved_physical_actions_for_batch.append(action_detail)
                    else:
                         logger.debug(f"  Action for {sim_id} not added to physical batch (Status: {val_status}, Type: {action_type})")
                else:
                    logger.debug(f"  Skipping {sim_id}: Missing intent_data or validation_result.")

            if approved_physical_actions_for_batch:
                console.print(f"[dim]Found {len(approved_physical_actions_for_batch)} approved physical actions. Running WorldStateAgent executor tool...[/dim]") # Corrected Agent Name

                # --- *** ENSURE THIS LINE USES world_state_agent *** ---
                runner.agent = world_state_agent
                # --- *** END ENSURE *** ---

                execution_trigger = types.Content(parts=[types.Part(text=f"Execute the following batch of {len(approved_physical_actions_for_batch)} approved physical actions using your 'execute_physical_actions_batch' tool.")]) # Explicit trigger
                # Pass the batch data via function call arguments if the tool expects it
                tool_call_args = {"actions_batch": approved_physical_actions_for_batch}

                # Run the World State Agent to execute the batch
                async for event in runner.run_async(
                    user_id=user_id, session_id=session_id, new_message=execution_trigger,
                    # Force the execution tool call
                    tool_config=types.ToolConfig(
                        tool_calling_config=types.ToolCallingConfig(
                            mode=types.ToolCallingConfig.Mode.ANY, # Or FUNCTION
                            allowed_tool_names=["execute_physical_actions_batch"] # Name of the executor tool
                        )
                    ),
                    # Pass arguments if needed by the flow/tool structure
                    # request_metadata={"execute_physical_actions_batch": tool_call_args} # Check ADK docs
                ):
                    # ... (event logging remains the same) ...
                    logger.debug(f"P4b Event: {event.author} - Final: {event.is_final_response()} - Actions: {event.actions} - Content: {str(event.content)}...")
                    # --- Print Event Details --- START
                    if event.content and event.content.parts:
                        part = event.content.parts[0]
                        if hasattr(part, 'function_call') and part.function_call:
                             tool_call = part.function_call
                             # Log args carefully if they are large
                             args_str = str(dict(tool_call.args))[:200] + ('...' if len(str(dict(tool_call.args))) > 200 else '')
                             console.print(f"[dim]  P4b ({event.author}) -> Tool Call: {tool_call.name} with args: {args_str}[/dim]")
                        elif hasattr(part, 'function_response') and part.function_response:
                             tool_response = part.function_response
                             response_str = str(tool_response.response) + ('...' if len(str(tool_response.response)) > 150 else '')
                             console.print(f"[dim]  P4b ({event.author}) <- Tool Response: {tool_response.name} -> {response_str}[/dim]")
                        elif event.is_final_response() and hasattr(part, 'text'):
                             console.print(f"[dim]  P4b ({event.author}) Final Output: {part.text if part.text else '[No text output]'}[/dim]")
                    # --- Print Event Details --- END
                console.print("[green]Phase 4b Complete.[/green]")
            else:
                console.print("[yellow]Phase 4b: No approved physical actions found to execute.[/yellow]")

        except Exception as e:
            logger.exception("Error during Physical Action Execution (Phase 4b).")
            console.print(f"[bold red]Error in Phase 4b: {e}. Skipping phase.[/bold red]")
            await asyncio.sleep(1)

        # --- Phase 5: Narration ---
        try:
            console.print("[reverse yellow] Phase 5: Turn Narration [/reverse yellow]") # Changed title slightly
            runner.agent = narration_agent # Ensure the correct agent is set
            active_sim_ids_for_narration = session.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])

            if not active_sim_ids_for_narration:
                console.print("[yellow]Phase 5: No active simulacra to narrate.[/yellow]")
            else:
                # Generic trigger asking the agent to handle all active simulacra
                narration_trigger = types.Content(parts=[types.Part(text=f"Generate a narrative summary for EACH of the active simulacra based on their actions and results this turn. Active IDs are in session state key '{ACTIVE_SIMULACRA_IDS_KEY}'.")])
                console.print(f"[dim]Running NarrationAgent ({narration_agent.name}) to generate summaries for {len(active_sim_ids_for_narration)} simulacra...[/dim]")

                # *** Single call to the Narration Agent ***
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=narration_trigger
                ):
                    logger.debug(f"P5 Event: {event.author} - Final: {event.is_final_response()} - Actions: {event.actions} - Content: {str(event.content)}...")
                    # --- Print Event Details --- START
                    if event.content and event.content.parts:
                        part = event.content.parts[0]
                        if hasattr(part, 'function_call') and part.function_call:
                             tool_call = part.function_call
                             # Agent might call the tool multiple times internally now
                             console.print(f"[dim]  P5 ({event.author}) -> Tool Call: {tool_call.name} with args: {dict(tool_call.args)}[/dim]")
                        elif hasattr(part, 'function_response') and part.function_response:
                             tool_response = part.function_response
                             console.print(f"[dim]  P5 ({event.author}) <- Tool Response: {tool_response.name} -> {str(tool_response.response)[:100]}...[/dim]")
                        elif event.is_final_response() and hasattr(part, 'text'):
                             final_narrative_text = part.text
                             if final_narrative_text:
                                 # Print the combined narrative output from the agent
                                 console.print(f"\n[bold magenta]Narrator Output:[/bold magenta]\n[magenta]{final_narrative_text}[/magenta]\n")
                                 # Optional: Store the combined text if needed elsewhere
                                 # session.state["turn_narration_summary"] = final_narrative_text
                             else:
                                 console.print(f"[yellow]Narrator returned empty final response.[/yellow]")
                    # --- Print Event Details --- END

            console.print("[green]Phase 5 Complete.[/green]")
        except Exception as e:
            # This catches errors during the single agent run or setup
            logger.exception("Error during Narration (Phase 5).")
            console.print(f"[bold red]Error in Phase 5: {e}.[/bold red]")


        # Pause slightly between turns
        console.print(f"--- End of Turn {turn + 1} ---")
        await asyncio.sleep(1) # Adjust as needed

    console.rule("[bold cyan]Simulation Finished[/bold cyan]", style="cyan")