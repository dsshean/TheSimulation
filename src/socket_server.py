import json
import logging
import asyncio
import socket
import re
from typing import Dict, Any, Optional, Union

from .simulation_utils import _update_state_value
from .loop_utils import get_nested # Keep this
from .config import SIMULACRA_KEY, ACTIVE_SIMULACRA_IDS_KEY, CURRENT_LOCATION_KEY, WORLD_STATE_KEY, LOCATION_DETAILS_KEY

# Configuration constants
SOCKET_SERVER_HOST = "127.0.0.1"  # localhost
SOCKET_SERVER_PORT = 8765  # Choose any available port
SOCKET_BUFFER_SIZE = 4096

logger = logging.getLogger(__name__)

async def socket_server_task(
    state: Dict[str, Any],
    narration_queue: asyncio.Queue,
    world_mood: str,
    simulation_time_getter: callable,  # Function that returns current sim time
    live_display_object_ref: Optional[Any] = None # Added for cleaner access
):
    """Socket server that allows external connections to inject content into the simulation"""
    logger.info("[SocketServer] Starting socket server...")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((SOCKET_SERVER_HOST, SOCKET_SERVER_PORT))
        server.listen(5)
        server.setblocking(False)
        logger.info(f"[SocketServer] Listening on {SOCKET_SERVER_HOST}:{SOCKET_SERVER_PORT}")
        
        while True:
            try:
                # Use asyncio to check for new connections without blocking
                await asyncio.sleep(0.1)
                
                # Check for new connection
                try:
                    client_socket, client_address = await asyncio.to_thread(server.accept)
                    logger.info(f"[SocketServer] New connection from {client_address}")
                    
                    # Handle client in a separate task
                    asyncio.create_task(handle_client(
                        client_socket, 
                        client_address, 
                        state, 
                        narration_queue, 
                        world_mood,
                        simulation_time_getter,
                        live_display_object_ref # Pass down
                    ))
                    
                except BlockingIOError:
                    # No new connection, continue
                    pass
                    
            except asyncio.CancelledError:
                logger.info("[SocketServer] Socket server task cancelled")
                raise
            except Exception as e:
                logger.error(f"[SocketServer] Error in main server loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    except Exception as e:
        logger.error(f"[SocketServer] Failed to start socket server: {e}", exc_info=True)
    finally:
        server.close()
        logger.info("[SocketServer] Socket server closed")

async def handle_client(
    client_socket,
    address, 
    state: Dict[str, Any],
    narration_queue: asyncio.Queue,
    world_mood: str,
    simulation_time_getter: callable,
    live_display_object_ref: Optional[Any] = None # Added for cleaner access
):
    """Handle a connected client"""
    logger.info(f"[SocketServer] Handling client: {address}")
    client_socket.setblocking(False)
    
    try:
        buffer = b""
        while True:
            try:
                # Read data with timeout
                data = await asyncio.to_thread(client_socket.recv, SOCKET_BUFFER_SIZE)
                
                if not data:
                    # Client disconnected
                    break
                    
                buffer += data
                
                # Process complete messages
                while b'\n' in buffer:
                    message, buffer = buffer.split(b'\n', 1)
                    result = await process_message(
                        message.decode('utf-8'),
                        state,
                        narration_queue,
                        world_mood,
                        simulation_time_getter,
                        live_display_object_ref # Pass down
                    )
                    
                    # Send the result (which should now be a standardized dictionary) back to client
                    response_data = json.dumps(result).encode('utf-8') + b'\n'
                    await asyncio.to_thread(client_socket.sendall, response_data)
                    
            except BlockingIOError:
                # No data available right now
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[SocketServer] Error reading from client {address}: {e}", exc_info=True)
                break
    
    finally:
        client_socket.close()
        logger.info(f"[SocketServer] Closed connection from {address}")

async def process_message(
    message_str: str,
    state: Dict[str, Any],
    narration_queue: asyncio.Queue,
    world_mood: str,
    simulation_time_getter: callable,
    live_display_object_ref: Optional[Any] = None # Added for cleaner access
) -> Dict[str, Any]: # Always return a dictionary
    """Process a received message and route it to the appropriate component"""
    try:
        message = json.loads(message_str)
        command = message.get("command", "").lower()
        current_sim_time = simulation_time_getter()
        
        if command == "narrate":
            # Inject a direct narration event
            narrative_text = message.get("text", "")
            if narrative_text:
                logger.info(f"[SocketServer] Injecting narrative: {narrative_text[:50]}...")
                
                # DIRECT APPROACH: Bypass the narration queue for external narratives
                # This ensures they appear immediately and update all agents
                
                # 1. Create the formatted narrative with timestamp
                final_narrative_entry = f"[T{current_sim_time:.1f}] {narrative_text}"
                
                # 2. Add to the narrative log directly
                state.setdefault("narrative_log", []).append(final_narrative_entry)
                max_narrative_log = 20
                if len(state["narrative_log"]) > max_narrative_log:
                    state["narrative_log"] = state["narrative_log"][-max_narrative_log:]
                
                # 3. Display in console if available
                from rich.panel import Panel  # Import within function
                from rich.console import Console
                console = Console()
                panel_to_display = Panel( # Create panel once
                    narrative_text, 
                    title=f"External Narrative @ {current_sim_time:.1f}s",
                    border_style="cyan", 
                    expand=False
                )
                try:
                    if live_display_object_ref: # Use the passed reference
                        live_display_object_ref.console.print(panel_to_display)
                    else:
                        # Fallback to direct console print
                        # Create a new Console instance here for fallback if not using live_display
                        Console().print(panel_to_display)
                except Exception as e:
                    logger.error(f"[SocketServer] Error displaying narrative: {e}")
                
                # 4. CRITICAL: Update ALL agents with this observation
                # This makes them aware of the external narrative
                active_sim_ids = state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
                for agent_id in active_sim_ids:
                    _update_state_value(
                        state, 
                        f"{SIMULACRA_KEY}.{agent_id}.last_observation", 
                        narrative_text, 
                        logger
                    )
                    # Also set agents to idle so they'll process this new observation
                    if get_nested(state, SIMULACRA_KEY, agent_id, "status") == "busy":
                        _update_state_value(
                            state,
                            f"{SIMULACRA_KEY}.{agent_id}.status",
                            "idle", 
                            logger
                        )
                
                # Also queue a narration event for consistency
                narration_event = {
                    "type": "action_complete",
                    "actor_id": "EXTERNAL_NARRATOR",
                    "action": {"action_type": "narrate", "details": "External narrative"},
                    "results": {},
                    "outcome_description": narrative_text,
                    "completion_time": current_sim_time,
                    "current_action_description": "External narrative injection",
                    "actor_current_location_id": "global",
                    "world_mood": world_mood
                }
                # Use temporal ordering system
                from .simulation_async import add_narration_event_with_ordering
                if add_narration_event_with_ordering is not None:
                    await add_narration_event_with_ordering(narration_event, task_name="SocketServer_ExternalNarrative")
                else:
                    await narration_queue.put(narration_event)  # Fallback
                
                # Log the event
                logger.info(f"[SocketServer] Successfully injected narrative and updated {len(active_sim_ids)} agents")
                
                return {
                    "success": True, 
                    "message": f"Narrative injected and {len(active_sim_ids)} agents updated",
                    "data": {"timestamp": current_sim_time} 
                }
                
        elif command == "inject_event":
            # Inject an event for a specific agent
            agent_id = message.get("agent_id")
            event_description = message.get("description", "")
            
            if agent_id and agent_id in get_nested(state, SIMULACRA_KEY, default={}) and event_description:
                logger.info(f"[SocketServer] Injecting event for agent {agent_id}: {event_description[:50]}...")
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.last_observation", event_description, logger)
                return {
                    "success": True, 
                    "message": "Event injected successfully",
                    "data": {"timestamp": current_sim_time}
                }
        
        elif command == "world_info":
            # Update world info like weather or news
            category = message.get("category", "")
            info = message.get("info", "")
            
            if category and info:
                logger.info(f"[SocketServer] Updating world info - {category}: {info[:50]}...")
                if category == "weather":
                    _update_state_value(state, "world_feeds.weather.condition", info, logger)
                elif category == "news":
                    news_item = {"headline": info, "timestamp": current_sim_time}
                    state.setdefault("world_feeds", {}).setdefault("news_updates", []).insert(0, news_item)
                return {
                    "success": True, 
                    "message": f"World info ({category}) updated successfully"
                }
        
        elif command == "fix_json":
            # Special command to fix JSON output from the narrator
            raw_json = message.get("text", "")
            if raw_json:
                try:
                    # Apply multiple fixes to handle different quoting issues
                    fixed_json = raw_json
                    
                    # Fix 1: Replace unescaped quotes within json string values
                    def fix_quotes_in_json(json_str):
                        # Pattern to find quotes inside string values
                        pattern = r'(": *")([^"\\]*?)(")((?:[^"\\])*?)(")'
                        
                        # Keep applying the replacement until no more changes
                        prev_json = None
                        while prev_json != json_str:
                            prev_json = json_str
                            json_str = re.sub(pattern, r'\1\2\\\3\4\5', json_str)
                        return json_str
                    
                    # Apply the fix
                    fixed_json = fix_quotes_in_json(fixed_json)
                    
                    # Additional fixes for other common issues
                    # Fix 2: Replace double quotes used as apostrophes
                    fixed_json = re.sub(r'(?<=\w)"(?=\w)', r"'", fixed_json)
                    
                    # Fix 3: Handle missing commas between objects
                    fixed_json = re.sub(r'}\s*{', r'},{', fixed_json)
                    
                    # Validate the fixed JSON
                    try:
                        json.loads(fixed_json)
                        valid_json = True
                    except json.JSONDecodeError:
                        valid_json = False
                        
                    return {
                        "success": valid_json, 
                        "message": "JSON successfully fixed" if valid_json else "Warning: JSON may still have issues",
                        "data": {
                            "fixed_json": fixed_json
                        }
                    }
                except Exception as e:
                    logger.error(f"[SocketServer] Error fixing JSON: {e}")
                    return {"success": False, "message": f"Error fixing JSON: {str(e)}"}
        
        elif command == "teleport_agent":
            agent_id = message.get("agent_id")
            new_location_id = message.get("new_location_id")

            if not agent_id or not new_location_id:
                logger.warning(f"[SocketServer] Teleport command missing agent_id or new_location_id: {message}")
                return {"success": False, "message": "Missing agent_id or new_location_id for teleport."}

            if agent_id not in get_nested(state, SIMULACRA_KEY, default={}):
                logger.warning(f"[SocketServer] Cannot teleport agent {agent_id}: Agent ID not found.")
                return {"success": False, "message": f"Agent {agent_id} not found."}

            # Ensure the new location exists in the world's defined locations, or at least log a warning
            if new_location_id not in get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, default={}):
                logger.warning(f"[SocketServer] Teleporting agent {agent_id} to new location '{new_location_id}' which is not yet defined in world_state.location_details. Agent might perceive an undescribed place until look_around.")

            old_location_id = get_nested(state, SIMULACRA_KEY, agent_id, CURRENT_LOCATION_KEY, default="Unknown")
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.{CURRENT_LOCATION_KEY}", new_location_id, logger)
            
            new_loc_name = get_nested(state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, new_location_id, "name", default=new_location_id)
            agent_loc_details_update = f"You are now in {new_loc_name}."
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.location_details", agent_loc_details_update, logger)
            
            # Also update the deprecated 'location' field for broader compatibility if it exists
            if "location" in get_nested(state, SIMULACRA_KEY, agent_id, default={}):
                    _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.location", new_location_id, logger)

            # Set agent to idle and update last_observation
            teleport_observation = f"You have been instantly teleported from {old_location_id} to {new_location_id} ({new_loc_name})."
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.last_observation", teleport_observation, logger)
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)

            # Add to narrative log
            agent_name_for_log = get_nested(state, SIMULACRA_KEY, agent_id, "persona_details", "Name", default=agent_id)
            narrative_teleport_entry = f"[T{current_sim_time:.1f}] [System Teleport] {agent_name_for_log} vanished from {old_location_id} and instantly reappeared in {new_location_id} ({new_loc_name})."
            state.setdefault("narrative_log", []).append(narrative_teleport_entry)

            logger.info(f"[SocketServer] Teleported agent {agent_id} from {old_location_id} to {new_location_id} via dedicated command.")
            return {
                "success": True, 
                "message": f"Agent {agent_id} teleported to {new_location_id}.",
                "data": {
                    "agent_id": agent_id, "new_location_id": new_location_id # Include this in data for client
                }
            }
        
        elif command == "get_state":
            # Get current simulation state with better simulacra detection
            current_sim_time = simulation_time_getter()
            
            # Primary method: Use ACTIVE_SIMULACRA_IDS_KEY
            active_sim_ids = []
            if ACTIVE_SIMULACRA_IDS_KEY in state:
                active_sim_ids = list(state.get(ACTIVE_SIMULACRA_IDS_KEY, [])) # Ensure it's a list copy
            
            # Fallback: If ACTIVE_SIMULACRA_IDS_KEY is empty or missing, try to infer from SIMULACRA_KEY (profiles)
            if not active_sim_ids and SIMULACRA_KEY in state:
                logger.info(f"[SocketServer] ACTIVE_SIMULACRA_IDS_KEY was empty/missing. Inferring active IDs from {SIMULACRA_KEY}.")
                simulacra_profiles = state.get(SIMULACRA_KEY, {})
                active_sim_ids = [
                    sim_id for sim_id, sim_data in simulacra_profiles.items()
                    if isinstance(sim_data, dict) and sim_data.get("status") != "terminated" # Basic check for active status
                ]

            logger.info(f"[SocketServer] Found {len(active_sim_ids)} active simulacra IDs: {active_sim_ids}")
            return {
                "success": True,
                "message": "State retrieved successfully.",
                "data": {
                    "agent_ids": active_sim_ids,  # This is the primary format client looks for for agent selection
                    "state_summary": { # Keep the detailed state under a sub-key if needed by client
                        "simulacra_profiles": {
                            agent_id: {
                                "id": agent_id,
                                "location": get_nested(state, SIMULACRA_KEY, agent_id, "location", default="unknown"),
                                "status": get_nested(state, SIMULACRA_KEY, agent_id, "status", default="unknown")
                            }
                            for agent_id in active_sim_ids
                        }
                    },
                    "time": current_sim_time,
                    "world_mood": world_mood,
                    # Keep these for client compatibility if it uses them directly from response.data
                    "agents": active_sim_ids, 
                    "simulacra_profiles": { 
                        agent_id: {"id": agent_id} for agent_id in active_sim_ids
                    }
                }
            }
        
        elif command == "start_interaction_mode":
            agent_id = message.get("agent_id")
            
            if agent_id and agent_id in get_nested(state, SIMULACRA_KEY, default={}):
                logger.info(f"[SocketServer] Starting interaction mode with {agent_id}")
                
                # Pause simulation activities for this agent
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.interaction_mode", True, logger)
                
                # Store the current status to restore later
                current_status = get_nested(state, SIMULACRA_KEY, agent_id, "status", default="idle")
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status_before_interaction", current_status, logger)
                
                # Set to idle so it will respond to our events
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
                
                return {"success": True, "message": f"Interaction mode started with {agent_id}"}
            else:
                logger.warning(f"[SocketServer] Agent {agent_id} not found for interaction mode")
                return {"success": False, "message": f"Agent {agent_id} not found"}
        
        elif command == "end_interaction_mode":
            agent_id = message.get("agent_id")
            
            if agent_id and agent_id in get_nested(state, SIMULACRA_KEY, default={}):
                logger.info(f"[SocketServer] Ending interaction mode with {agent_id}")
                
                # Remove interaction mode flag
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.interaction_mode", False, logger)
                
                # Restore previous status
                previous_status = get_nested(state, SIMULACRA_KEY, agent_id, "status_before_interaction", default="idle")
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", previous_status, logger)
                
                return {"success": True, "message": f"Interaction mode ended with {agent_id}"}
            else:
                return {"success": False, "message": f"Agent {agent_id} not found"}
        
        elif command == "interaction_event":
            agent_id = message.get("agent_id")
            event_type = message.get("event_type", "text_message")
            content = message.get("content", "")
            
            # Add validation
            if not agent_id:
                return {"success": False, "message": "Missing agent_id parameter"}
            
            if not content:
                return {"success": False, "message": "Missing content parameter"}
            
            # Check if agent exists
            if agent_id not in get_nested(state, SIMULACRA_KEY, default={}):
                return {"success": False, "message": f"Agent {agent_id} not found in simulation"}
            
            # Format the event based on the communication channel
            formatted_event = ""
            
            if event_type == "text_message":
                formatted_event = f"You receive a text message: \"{content}\""
            elif event_type == "phone_call":
                formatted_event = f"You receive a phone call. The caller says: \"{content}\""
            elif event_type == "voice":
                formatted_event = f"A voice speaks to you: \"{content}\""
            elif event_type == "doorbell":
                formatted_event = f"The doorbell rings. {content}"
            elif event_type == "knock":
                formatted_event = f"There's a knock at the door. {content}"
            elif event_type == "noise":
                formatted_event = f"You hear a noise. {content}"
            elif event_type == "custom":
                formatted_event = content
            else:
                formatted_event = f"Something happens: {content}"

            logger.info(f"[SocketServer] Sending '{event_type}' event to {agent_id}: {formatted_event[:50]}...")
            
            # This uses the standard agent event injection mechanism
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.last_observation", formatted_event, logger)
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
            
            # After formatting the event:
            formatted_event = f"You receive a text message: \"{content}\"" # or other event types
    
            # Instead of trying to use the Rich console directly, send it through the narration system:
            current_sim_time = simulation_time_getter()
            
            # Create a special narration event that will display in the main simulation console
            display_narration_event = {
                "type": "action_complete",  # This is what the narration system expects
                "actor_id": "INTERACTION_MODE",  # Special actor ID for display formatting
                "action": {
                    "action_type": "interact",
                    "details": f"Interaction with {agent_id}"
                },
                "results": {},
                "outcome_description": f"[Interactive Mode] {agent_id} experiences: {formatted_event}",
                "completion_time": current_sim_time,
                "current_action_description": "Interactive event",
                "actor_current_location_id": get_nested(state, SIMULACRA_KEY, agent_id, "location", default="unknown"),
                "world_mood": world_mood
            }
            
            # Add to the narration queue - this will be displayed by the main narration loop
            # Use temporal ordering system
            from .simulation_async import add_narration_event_with_ordering
            if add_narration_event_with_ordering is not None:
                await add_narration_event_with_ordering(display_narration_event, task_name="SocketServer_InteractiveEvent")
            else:
                await narration_queue.put(display_narration_event)  # Fallback
            
            # Still update the agent and add to narrative log as before
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.last_observation", formatted_event, logger)
            _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
            
            final_narrative_entry = f"[T{current_sim_time:.1f}] [InteractionMode] {agent_id} experiences: {formatted_event}"
            state.setdefault("narrative_log", []).append(final_narrative_entry)
            
            return {
                "success": True,
                "message": "Interaction event sent to agent",
                "data": {
                    "timestamp": current_sim_time
                }
            }
        elif command == "get_agent_responses":
            # Get all narrative entries since a timestamp
            since_timestamp = message.get("since_timestamp", 0.0)
            agent_id = message.get("agent_id", None)
            
            responses = []
            for entry in state.get("narrative_log", []):
                # Parse timestamp from entries like "[T123.4] Some content"
                match = re.match(r"\[T(\d+\.?\d*)\](.+)", entry)
                if match:
                    entry_time = float(match.group(1))
                    content = match.group(2).strip()
                    
                    # Only include entries newer than last check and relevant to our agent
                    if entry_time > since_timestamp and (agent_id is None or agent_id in content):
                        responses.append({
                            "timestamp": entry_time,
                            "content": content
                        })
            
            # Process responses to extract agent actions/speech
            processed_responses = []
            for resp in responses:
                content = resp.get("content", "")
                timestamp = resp.get("timestamp", 0.0)
                
                # Add parsed information to help client display the response
                response_type = "narrative"
                extracted_text = content
                
                # Try to extract direct speech or actions
                if agent_id and agent_id in content:
                    if "says" in content and agent_id in content.split("says")[0]:
                        response_type = "speech"
                        parts = content.split("says", 1)
                        extracted_text = parts[1].strip() if len(parts) > 1 else content
                    elif "decides to" in content:
                        response_type = "action" 
                        parts = content.split("decides to", 1)
                        extracted_text = parts[1].strip() if len(parts) > 1 else content
                    elif ":" in content and content.index(":") > content.index(agent_id):
                        response_type = "speech"
                        parts = content.split(":", 1)
                        extracted_text = parts[1].strip() if len(parts) > 1 else content
                
                processed_responses.append({
                    "timestamp": timestamp,
                    "content": content,
                    "type": response_type,
                    "extracted_text": extracted_text
                })

            return {
                "success": True,
                "message": "Agent responses retrieved.",
                "data": {
                    "responses": processed_responses,
                    "current_time": simulation_time_getter()
                }
            }
        
        logger.warning(f"[SocketServer] Unknown command or invalid parameters: {message_str}")
        return {"success": False, "message": "Unknown command or invalid parameters"}
    
    except json.JSONDecodeError:
        logger.error(f"[SocketServer] Invalid JSON received: {message_str}")
        return {"success": False, "message": "Invalid JSON format"}
    except Exception as e:
        logger.error(f"[SocketServer] Error processing message: {e}", exc_info=True)
        return {"success": False, "message": f"Server error: {str(e)}"}