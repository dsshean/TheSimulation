import json
import logging
import asyncio
import socket
import re
from typing import Dict, Any, Optional, Union

from .simulation_utils import _update_state_value
from .loop_utils import get_nested
from .config import SIMULACRA_KEY, ACTIVE_SIMULACRA_IDS_KEY

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
                        simulation_time_getter
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
    simulation_time_getter: callable
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
                        simulation_time_getter
                    )
                    
                    # Send response back to client
                    response = {
                        "success": bool(result),
                        "message": "Command processed successfully" if result else "Failed to process command",
                        "data": result if isinstance(result, dict) else None
                    }
                    response_data = json.dumps(response).encode('utf-8') + b'\n'
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
    simulation_time_getter: callable
) -> Union[Dict[str, Any], bool]:
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
                max_narrative_log = 50
                if len(state["narrative_log"]) > max_narrative_log:
                    state["narrative_log"] = state["narrative_log"][-max_narrative_log:]
                
                # 3. Display in console if available
                from rich.panel import Panel  # Import within function
                from rich.console import Console
                console = Console()
                try:
                    # Try to use the live display if it exists
                    live_display = None
                    if 'live_display_object' in globals():
                        live_display = globals()['live_display_object']
                    
                    if live_display:
                        live_display.console.print(Panel(
                            narrative_text, 
                            title=f"External Narrative @ {current_sim_time:.1f}s",
                            border_style="cyan", 
                            expand=False
                        ))
                    else:
                        # Fallback to direct console print
                        console.print(Panel(
                            narrative_text, 
                            title=f"External Narrative @ {current_sim_time:.1f}s",
                            border_style="cyan", 
                            expand=False
                        ))
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
                await narration_queue.put(narration_event)
                
                # Log the event
                logger.info(f"[SocketServer] Successfully injected narrative and updated {len(active_sim_ids)} agents")
                
                return {
                    "success": True, 
                    "timestamp": current_sim_time, 
                    "message": f"Narrative injected and {len(active_sim_ids)} agents updated"
                }
                
        elif command == "inject_event":
            # Inject an event for a specific agent
            agent_id = message.get("agent_id")
            event_description = message.get("description", "")
            
            if agent_id and agent_id in get_nested(state, SIMULACRA_KEY, default={}) and event_description:
                logger.info(f"[SocketServer] Injecting event for agent {agent_id}: {event_description[:50]}...")
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.last_observation", event_description, logger)
                return {"success": True, "timestamp": current_sim_time, "message": "Event injected successfully"}
        
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
                return {"success": True, "message": f"World info ({category}) updated successfully"}
        
        elif command == "fix_json":
            # Special command to fix JSON output from the narrator
            raw_json = message.get("text", "")
            if raw_json:
                try:
                    # Apply multiple fixes to handle different quoting issues
                    fixed_json = raw_json
                    # Fix 1: Replace patterns of "" within string values with \"
                    fixed_json = re.sub(r'(": *")([^"]*?)""([^"]*?)""([^"]*?)(")', r'\1\2\"\3\"\4\5', fixed_json)
                    # Fix 2: More aggressive - replaces all double double-quotes with escaped quotes
                    fixed_json = re.sub(r'""', r'\\"', fixed_json)
                    
                    # Return the fixed JSON to the client
                    return {"success": True, "fixed_json": fixed_json}
                except Exception as e:
                    logger.error(f"[SocketServer] Error fixing JSON: {e}")
                    return {"success": False, "message": f"Error fixing JSON: {str(e)}"}
        
        elif command == "get_state":
            # Get current simulation state with better simulacra detection
            current_sim_time = simulation_time_getter()
            
            # More robust detection of active simulacra
            active_sim_ids = []
            
            # Method 1: Try getting from SIMULACRA_KEY directly (most reliable)
            if SIMULACRA_KEY in state:
                simulacra_dict = state.get(SIMULACRA_KEY, {})
                # Get all simulacra that aren't terminated
                active_sim_ids = [
                    sim_id for sim_id, sim_data in simulacra_dict.items()
                    if isinstance(sim_data, dict) and sim_data.get("status") != "terminated"
                ]
            
            # Method 2: Try getting from ACTIVE_SIMULACRA_IDS_KEY as fallback
            if not active_sim_ids and ACTIVE_SIMULACRA_IDS_KEY in state:
                active_sim_list = state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
                # Sometimes it's a list of strings, sometimes a list of objects
                if active_sim_list and len(active_sim_list) > 0:
                    if isinstance(active_sim_list[0], str):
                        active_sim_ids = active_sim_list
                    elif isinstance(active_sim_list[0], dict) and "id" in active_sim_list[0]:
                        active_sim_ids = [sim["id"] for sim in active_sim_list]
            
            # Log what we found for debugging
            logger.info(f"[SocketServer] Found {len(active_sim_ids)} active simulacra IDs: {active_sim_ids}")
            
            return {
                "success": True,
                "data": {
                    "time": current_sim_time,
                    "world_mood": world_mood,
                    "agent_ids": active_sim_ids,
                    "current_locations": {
                        agent_id: get_nested(state, SIMULACRA_KEY, agent_id, "location", default="unknown")
                        for agent_id in active_sim_ids
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
            
            if agent_id and agent_id in get_nested(state, SIMULACRA_KEY, default={}) and content:
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
                await narration_queue.put(display_narration_event)
                
                # Still update the agent and add to narrative log as before
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.last_observation", formatted_event, logger)
                _update_state_value(state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
                
                final_narrative_entry = f"[T{current_sim_time:.1f}] [InteractionMode] {agent_id} experiences: {formatted_event}"
                state.setdefault("narrative_log", []).append(final_narrative_entry)
                
                return {
                    "success": True,
                    "message": "Interaction event sent to agent",
                    "timestamp": current_sim_time
                }
            else:
                return {"success": False, "message": "Failed to send interaction event"}
        
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
            
            return {
                "success": True,
                "responses": responses,
                "current_time": simulation_time_getter()
            }
        
        logger.warning(f"[SocketServer] Unknown command or invalid parameters: {message_str}")
        return {"success": False, "message": "Unknown command or invalid parameters"}
    
    except json.JSONDecodeError:
        logger.error(f"[SocketServer] Invalid JSON received: {message_str}")
        return {"success": False, "message": "Invalid JSON format"}
    except Exception as e:
        logger.error(f"[SocketServer] Error processing message: {e}", exc_info=True)
        return {"success": False, "message": f"Server error: {str(e)}"}