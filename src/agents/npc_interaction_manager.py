# c:\Users\dshea\Desktop\TheSimulation\src\agents\npc_interaction_manager.py
import asyncio
import json
import logging
import random
import re # Import regex for JSON extraction
import heapq # Import heapq for heappush
from typing import Dict, Any, List, Tuple, Optional

# --- ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.genai import types as genai_types # Renamed to avoid conflict

logger = logging.getLogger(__name__)

# --- NPC Manager Task ---
async def npc_manager_task(
    event_bus: asyncio.Queue, # Keep event_bus if needed for other events, or remove if unused
    npc_interaction_queue: asyncio.Queue, # <<< MINIMAL CHANGE 1: Add parameter
    state: Dict[str, Any],
    schedule: List[Tuple[float, int, Dict[str, Any]]],
    runner: Runner,
    session_id: str,
    USER_ID: str,
    npc_interaction_llm_agent: LlmAgent
):
    """
    Manages interactions involving NPCs. Listens for INTERACTION_REQUESTED events
    on the npc_interaction_queue. Uses an LLM to resolve the interaction outcome.
    """
    # --- MODIFIED: Check includes the new queue ---
    if not all([npc_interaction_queue, state, heapq, runner, session_id, npc_interaction_llm_agent]):
        logger.error("NPC Manager: Core components not available via arguments (npc_interaction_queue, state, heapq, runner, session_id, npc_interaction_llm_agent).")
        return

    logger.info("[NpcInteractionManager] Task started, listening for interaction requests on npc_interaction_queue.")
    while True:
        try:
            # Use asyncio.wait_for with a timeout to allow periodic checks/actions
            # even if no interaction events arrive immediately.
            # <<< MINIMAL CHANGE 2: Get from npc_interaction_queue >>>
            event = await asyncio.wait_for(npc_interaction_queue.get(), timeout=1.0)

            if event.get("type") == "INTERACTION_REQUESTED":
                logger.info(f"[NpcInteractionManager] Received interaction request: {event}")

                actor_id = event.get("actor_id")
                target_npc_id = event.get("target_id")
                details = event.get("interaction_details")
                timestamp = event.get("timestamp", state.get("world_time", 0.0))

                # Use helper function for safe nested access
                actor_state = state.get("simulacra", {}).get(actor_id) # Assuming actor is simulacra
                npc_state = state.get("npcs", {}).get(target_npc_id)

                if not actor_state or not npc_state:
                    logger.warning(f"[NpcInteractionManager] Actor '{actor_id}' or NPC '{target_npc_id}' not found in state for interaction.")
                    # <<< MINIMAL CHANGE 3: Mark npc_interaction_queue done >>>
                    npc_interaction_queue.task_done()
                    continue

                # --- LLM Interaction Resolution ---
                actor_name = actor_state.get("name", actor_id)
                # Prepare context for LLM
                trigger_context = f"""
Resolve NPC interaction at time {timestamp:.1f}.

Actor ID: {actor_id}
Actor Name: {actor_name}
Actor State: {{ "status": "{actor_state.get('status', 'unknown')}" }} # Add more relevant actor state if needed

Target NPC ID: {target_npc_id}
Target NPC State:
{json.dumps(npc_state, indent=2)}

Interaction Details: "{details}"

Determine the NPC's reaction based on your instructions (provided during agent creation). Output ONLY the JSON result.
"""
                trigger = genai_types.Content(parts=[genai_types.Part(text=trigger_context)])
                resolution_json_str = None
                parsed_resolution = None

                logger.info(f"[NpcInteractionManager] Requesting LLM resolution for {actor_name} interacting with {npc_state.get('name', target_npc_id)} ('{details}')")
                try:
                    runner.agent = npc_interaction_llm_agent # Set the correct agent for the runner
                    async for llm_event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=trigger):
                        if llm_event.is_final_response() and llm_event.content:
                            resolution_json_str = llm_event.content.parts[0].text
                            logger.debug(f"NpcLLM ({llm_event.author}) Final Content: {resolution_json_str[:100]}...")
                        elif llm_event.error_message:
                            logger.error(f"NpcLLM Error ({llm_event.author}): {llm_event.error_message}")

                    # Parse the LLM response
                    if resolution_json_str:
                        match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*\})", resolution_json_str, re.DOTALL)
                        if match:
                            json_str_to_parse = match.group(1) or match.group(2)
                            if json_str_to_parse:
                                parsed_resolution = json.loads(json_str_to_parse)
                            else: logger.error(f"[NpcInteractionManager] Regex matched but no JSON content found. Raw: {repr(resolution_json_str)}")
                        else: logger.error(f"[NpcInteractionManager] Could not find JSON block in LLM response. Raw: {repr(resolution_json_str)}")
                    else:
                        logger.warning("[NpcInteractionManager] LLM did not return a resolution string.")

                except json.JSONDecodeError as e:
                    logger.error(f"[NpcInteractionManager] Failed to parse LLM JSON: {e}. Extracted: {repr(json_str_to_parse if 'json_str_to_parse' in locals() else 'N/A')}. Raw: {repr(resolution_json_str)}")
                except Exception as e:
                    logger.exception(f"[NpcInteractionManager] Error during LLM call or parsing: {e}")

                # --- Use LLM Resolution or Fallback ---
                if parsed_resolution and isinstance(parsed_resolution, dict):
                    narrative_outcome = parsed_resolution.get("narrative_outcome", f"{actor_name} interacts with {npc_state.get('name', target_npc_id)}, but the reaction is unclear.")
                    npc_mood_change = parsed_resolution.get("npc_mood_change", npc_state.get("mood", "neutral")) # Keep old mood if not specified
                    npc_status_change = parsed_resolution.get("npc_status_change", "reacting") # Default to reacting
                    npc_reaction_duration = float(parsed_resolution.get("reaction_duration", random.uniform(1.0, 3.0))) # Use LLM duration or random fallback
                    logger.info(f"[NpcInteractionManager] LLM Resolution: Mood='{npc_mood_change}', Status='{npc_status_change}', Duration={npc_reaction_duration:.1f}s")
                else:
                    # Fallback if LLM failed
                    logger.warning("[NpcInteractionManager] Using fallback interaction resolution.")
                    narrative_outcome = f"{actor_name} interacts with {npc_state.get('name', target_npc_id)} ({details}). The reaction is generic."
                    npc_mood_change = npc_state.get("mood", "neutral")
                    npc_status_change = "reacting"
                    npc_reaction_duration = random.uniform(1.0, 3.0)

                # --- Update State (Immediate and Scheduled) ---
                current_time = state.get("world_time", 0.0)
                state.setdefault("narrative_log", []).append(f"[T{current_time:.1f}] {narrative_outcome}")

                if actor_state:
                    actor_state["status"] = "idle"
                    actor_state["last_observation"] = narrative_outcome
                    actor_state["current_action_end_time"] = current_time

                if npc_state:
                    npc_state["mood"] = npc_mood_change
                    npc_state["status"] = npc_status_change
                    npc_state["last_observation"] = f"Reacted to {actor_name}'s interaction ({details})."
                    npc_completion_time = current_time + npc_reaction_duration
                    npc_state["current_action_end_time"] = npc_completion_time

                    completion_event = {
                        "type": "npc_action_complete",
                        "actor_id": target_npc_id,
                        "action_type": npc_status_change,
                        "results": {f"npcs.{target_npc_id}.status": "idle"},
                        "narrative": f"{npc_state.get('name', target_npc_id)} finishes reacting.",
                        "start_time": current_time,
                        "end_time": npc_completion_time
                    }
                    # Ensure schedule_event_counter is handled correctly if used elsewhere
                    # Using a placeholder 0 for now if counter isn't managed here
                    global schedule_event_counter # Assuming it's global like in world_engine
                    try:
                        schedule_event_counter += 1
                        heapq.heappush(schedule, (npc_completion_time, schedule_event_counter, completion_event))
                    except NameError: # Fallback if schedule_event_counter isn't global
                         logger.warning("[NpcInteractionManager] schedule_event_counter not found globally, using 0 for heapq.")
                         heapq.heappush(schedule, (npc_completion_time, 0, completion_event)) # Using 0 for secondary sort key

                    logger.info(f"[NpcInteractionManager] Scheduled '{target_npc_id}' reaction completion at {npc_completion_time:.1f}s.")

                # <<< MINIMAL CHANGE 3: Mark npc_interaction_queue done >>>
                npc_interaction_queue.task_done()

            else: # Event type not INTERACTION_REQUESTED
                 logger.warning(f"[NpcInteractionManager] Received unexpected event type '{event.get('type')}' on interaction queue.")
                 # <<< MINIMAL CHANGE 3: Mark npc_interaction_queue done >>>
                 npc_interaction_queue.task_done() # Mark other events as done to prevent blocking

        except asyncio.TimeoutError:
            # TODO: Add periodic NPC behavior checks/triggers here if desired.
            pass # Continue loop on timeout

        except asyncio.CancelledError:
            logger.info("[NpcInteractionManager] Task cancelled.")
            break
        except Exception as e:
            logger.exception(f"[NpcInteractionManager] Error processing event or timeout: {e}")
            try: # Ensure task_done is called if an event was retrieved before exception
                 # <<< MINIMAL CHANGE 3: Check and mark npc_interaction_queue done >>>
                if 'event' in locals() and npc_interaction_queue.unfinished_tasks > 0: npc_interaction_queue.task_done()
            except ValueError: pass
            except Exception as inner_e: logger.error(f"[NpcInteractionManager] Error calling task_done after exception: {inner_e}")
            await asyncio.sleep(1) # Avoid tight loop on error