"""
Command handlers for Redis communication.
Handles commands sent from Tauri frontend to Python simulation.
"""

import asyncio
import logging
from typing import Dict, Any

from .simulation_utils import _update_state_value
from .loop_utils import get_nested
from .config import SIMULACRA_KEY, ACTIVE_SIMULACRA_IDS_KEY, CURRENT_LOCATION_KEY

logger = logging.getLogger(__name__)

class RedisCommandHandlers:
    def __init__(self, state: Dict[str, Any], narration_queue: asyncio.Queue, simulation_time_getter: callable):
        self.state = state
        self.narration_queue = narration_queue
        self.simulation_time_getter = simulation_time_getter
        self.world_mood = ""
    
    def set_world_mood(self, mood: str):
        """Set current world mood"""
        self.world_mood = mood
    
    async def inject_narrative(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle narrative injection command"""
        try:
            narrative_text = message_data.get("text", "")
            if not narrative_text:
                return {"success": False, "message": "Missing narrative text"}
            
            current_sim_time = self.simulation_time_getter()
            logger.info(f"[RedisCommands] Injecting narrative: {narrative_text[:50]}...")
            
            # Add to narrative log
            final_narrative_entry = f"[T{current_sim_time:.1f}] {narrative_text}"
            self.state.setdefault("narrative_log", []).append(final_narrative_entry)
            
            # Update all agents
            active_sim_ids = self.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
            for agent_id in active_sim_ids:
                _update_state_value(
                    self.state,
                    f"{SIMULACRA_KEY}.{agent_id}.last_observation",
                    narrative_text,
                    logger
                )
                # Set agents to idle so they process the observation
                if get_nested(self.state, SIMULACRA_KEY, agent_id, "status") == "busy":
                    _update_state_value(
                        self.state,
                        f"{SIMULACRA_KEY}.{agent_id}.status",
                        "idle",
                        logger
                    )
            
            # Queue narration event
            if self.narration_queue:
                narration_event = {
                    "type": "action_complete",
                    "actor_id": "EXTERNAL_NARRATOR",
                    "action": {"action_type": "narrate", "details": "External narrative"},
                    "results": {},
                    "outcome_description": narrative_text,
                    "completion_time": current_sim_time,
                    "current_action_description": "External narrative injection",
                    "actor_current_location_id": "global",
                    "world_mood": self.world_mood
                }
                await self.narration_queue.put(narration_event)
            
            return {
                "success": True,
                "message": f"Narrative injected and {len(active_sim_ids)} agents updated",
                "data": {"timestamp": current_sim_time, "affected_agents": len(active_sim_ids)}
            }
            
        except Exception as e:
            logger.error(f"[RedisCommands] Error in inject_narrative: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def send_agent_event(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent event command"""
        try:
            agent_id = message_data.get("agent_id")
            event_description = message_data.get("description", "")
            
            if not agent_id or not event_description:
                return {"success": False, "message": "Missing agent_id or description"}
            
            simulacra = get_nested(self.state, SIMULACRA_KEY, default={})
            if agent_id not in simulacra:
                return {"success": False, "message": f"Agent {agent_id} not found"}
            
            current_sim_time = self.simulation_time_getter()
            logger.info(f"[RedisCommands] Sending event to agent {agent_id}: {event_description[:50]}...")
            
            _update_state_value(self.state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
            _update_state_value(self.state, f"{SIMULACRA_KEY}.{agent_id}.last_observation", event_description, logger)
            
            return {
                "success": True,
                "message": "Event sent to agent successfully",
                "data": {"agent_id": agent_id, "timestamp": current_sim_time}
            }
            
        except Exception as e:
            logger.error(f"[RedisCommands] Error in send_agent_event: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def update_world_info(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle world info update command"""
        try:
            category = message_data.get("category", "")
            info = message_data.get("info", "")
            
            if not category or not info:
                return {"success": False, "message": "Missing category or info"}
            
            current_sim_time = self.simulation_time_getter()
            logger.info(f"[RedisCommands] Updating world info - {category}: {info[:50]}...")
            
            if category == "weather":
                _update_state_value(self.state, "world_feeds.weather.condition", info, logger)
            elif category == "news":
                news_item = {"headline": info, "timestamp": current_sim_time}
                self.state.setdefault("world_feeds", {}).setdefault("news_updates", []).insert(0, news_item)
            
            return {
                "success": True,
                "message": f"World info ({category}) updated successfully",
                "data": {"category": category, "timestamp": current_sim_time}
            }
            
        except Exception as e:
            logger.error(f"[RedisCommands] Error in update_world_info: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def teleport_agent(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent teleportation command"""
        try:
            agent_id = message_data.get("agent_id")
            new_location_id = message_data.get("location")
            
            if not agent_id or not new_location_id:
                return {"success": False, "message": "Missing agent_id or location"}
            
            simulacra = get_nested(self.state, SIMULACRA_KEY, default={})
            if agent_id not in simulacra:
                return {"success": False, "message": f"Agent {agent_id} not found"}
            
            current_sim_time = self.simulation_time_getter()
            old_location = get_nested(self.state, SIMULACRA_KEY, agent_id, CURRENT_LOCATION_KEY, default="Unknown")
            
            _update_state_value(self.state, f"{SIMULACRA_KEY}.{agent_id}.{CURRENT_LOCATION_KEY}", new_location_id, logger)
            _update_state_value(self.state, f"{SIMULACRA_KEY}.{agent_id}.teleport_triggered", True, logger)
            
            logger.info(f"[RedisCommands] Agent {agent_id} teleported from {old_location} to {new_location_id}")
            
            return {
                "success": True,
                "message": f"Agent {agent_id} teleported to {new_location_id}",
                "data": {
                    "agent_id": agent_id,
                    "from_location": old_location,
                    "to_location": new_location_id,
                    "timestamp": current_sim_time
                }
            }
            
        except Exception as e:
            logger.error(f"[RedisCommands] Error in teleport_agent: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def interactive_chat(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interactive chat command"""
        try:
            agent_id = message_data.get("agent_id")
            message = message_data.get("message", "")
            
            if not agent_id or not message:
                return {"success": False, "message": "Missing agent_id or message"}
            
            simulacra = get_nested(self.state, SIMULACRA_KEY, default={})
            if agent_id not in simulacra:
                return {"success": False, "message": f"Agent {agent_id} not found"}
            
            current_sim_time = self.simulation_time_getter()
            logger.info(f"[RedisCommands] Interactive chat with {agent_id}: {message[:50]}...")
            
            # Update agent with the message as an observation
            chat_observation = f"[TELEPATHIC MESSAGE] {message}"
            _update_state_value(self.state, f"{SIMULACRA_KEY}.{agent_id}.last_observation", chat_observation, logger)
            _update_state_value(self.state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger)
            
            return {
                "success": True,
                "message": f"Message sent to {agent_id}",
                "data": {"agent_id": agent_id, "timestamp": current_sim_time}
            }
            
        except Exception as e:
            logger.error(f"[RedisCommands] Error in interactive_chat: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

def create_command_handlers(state: Dict[str, Any], narration_queue: asyncio.Queue, simulation_time_getter: callable) -> Dict[str, callable]:
    """Create command handler functions for Redis integration"""
    
    handlers = RedisCommandHandlers(state, narration_queue, simulation_time_getter)
    
    return {
        "inject_narrative": handlers.inject_narrative,
        "send_agent_event": handlers.send_agent_event,
        "update_world_info": handlers.update_world_info,
        "teleport_agent": handlers.teleport_agent,
        "interactive_chat": handlers.interactive_chat,
    }