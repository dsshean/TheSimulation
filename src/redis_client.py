"""
Redis client for real-time communication with Tauri frontend.
Provides pub/sub channels for state updates and command handling.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
import redis.asyncio as redis
from datetime import datetime

from .config import SIMULACRA_KEY, ACTIVE_SIMULACRA_IDS_KEY

# Redis configuration
REDIS_HOST = "127.0.0.1" 
REDIS_PORT = 6379
REDIS_DB = 0

# Channel names
STATE_CHANNEL = "simulation:state"
COMMAND_CHANNEL = "simulation:commands"
EVENT_CHANNEL = "simulation:events"

logger = logging.getLogger(__name__)

class SimulationRedisClient:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.state: Optional[Dict[str, Any]] = None
        self.simulation_time_getter: Optional[Callable] = None
        self.command_handlers: Dict[str, Callable] = {}
        self.running = False
        self.recent_events: list = []  # Buffer for recent events
        self.max_recent_events = 50  # Keep last 50 events
        
    async def connect(self):
        """Connect to Redis server"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info(f"[RedisClient] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            
            # Set up pub/sub for commands
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe(COMMAND_CHANNEL)
            
            return True
            
        except Exception as e:
            logger.error(f"[RedisClient] Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis server"""
        self.running = False
        if self.pubsub:
            await self.pubsub.unsubscribe(COMMAND_CHANNEL)
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("[RedisClient] Disconnected from Redis")
    
    def set_state_reference(self, state: Dict[str, Any], simulation_time_getter: Callable):
        """Set references to simulation state and time getter"""
        self.state = state
        self.simulation_time_getter = simulation_time_getter
    
    def register_command_handler(self, command: str, handler: Callable):
        """Register a command handler function"""
        self.command_handlers[command] = handler
        logger.info(f"[RedisClient] Registered handler for command: {command}")
    
    async def publish_state_update(self):
        """Publish current simulation state to Redis"""
        if not self.redis_client or not self.state or not self.simulation_time_getter:
            return
            
        try:
            current_time = self.simulation_time_getter()
            simulacra = self.state.get(SIMULACRA_KEY, {})
            active_ids = self.state.get(ACTIVE_SIMULACRA_IDS_KEY, [])
            
            state_data = {
                "world_time": current_time,
                "world_instance_uuid": self.state.get("world_instance_uuid", ""),
                "simulacra_profiles": simulacra,
                "active_simulacra_ids": active_ids,
                "current_world_state": self.state.get("current_world_state", {}),
                "world_state": self.state.get("world_state", {}),
                "world_feeds": self.state.get("world_feeds", {}),
                "world_template_details": self.state.get("world_template_details", {}),
                "narrative_log": self.state.get("narrative_log", []),
                "recent_events": self.recent_events,
                "timestamp": datetime.now().isoformat()
            }
            
            message = json.dumps(state_data)
            await self.redis_client.publish(STATE_CHANNEL, message)
            
        except Exception as e:
            logger.error(f"[RedisClient] Error publishing state update: {e}")
    
    async def publish_event(self, event_data: Dict[str, Any]):
        """Publish an event to Redis"""
        if not self.redis_client:
            return
            
        try:
            event_message = {
                "timestamp": datetime.now().isoformat(),
                "sim_time": self.simulation_time_getter() if self.simulation_time_getter else 0,
                **event_data
            }
            
            message = json.dumps(event_message)
            await self.redis_client.publish(EVENT_CHANNEL, message)
            
        except Exception as e:
            logger.error(f"[RedisClient] Error publishing event: {e}")
    
    def add_recent_event(self, event_data: Dict[str, Any]):
        """Add an event to the recent events buffer"""
        # Add to recent events buffer
        self.recent_events.append(event_data)
        
        # Keep only the last max_recent_events
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events = self.recent_events[-self.max_recent_events:]
    
    async def handle_command_message(self, message_data: Dict[str, Any]):
        """Handle incoming command from Redis"""
        try:
            command = message_data.get("command", "").lower()
            request_id = message_data.get("request_id")
            
            logger.info(f"[RedisClient] Received command: {command}")
            
            response = {"success": False, "message": "Unknown command"}
            
            if command in self.command_handlers:
                # Call the registered handler
                handler = self.command_handlers[command]
                response = await handler(message_data)
            else:
                response = {"success": False, "message": f"No handler for command: {command}"}
            
            # Send response back
            if request_id:
                response_data = {
                    "type": "command_response",
                    "request_id": request_id,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
                response_channel = f"simulation:response:{request_id}"
                message = json.dumps(response_data)
                await self.redis_client.publish(response_channel, message)
                
                # Set expiration for response channel
                await self.redis_client.expire(response_channel, 60)  # 1 minute TTL
                
        except Exception as e:
            logger.error(f"[RedisClient] Error handling command: {e}")
    
    async def command_listener_task(self):
        """Main task to listen for commands from Redis"""
        if not self.pubsub:
            logger.error("[RedisClient] PubSub not initialized")
            return
            
        self.running = True
        logger.info("[RedisClient] Started command listener")
        
        try:
            while self.running:
                try:
                    message = await self.pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            await self.handle_command_message(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"[RedisClient] Invalid JSON in command: {e}")
                        except Exception as e:
                            logger.error(f"[RedisClient] Error processing command: {e}")
                            
                except asyncio.TimeoutError:
                    continue  # Normal timeout, keep listening
                    
        except asyncio.CancelledError:
            logger.info("[RedisClient] Command listener cancelled")
        except Exception as e:
            logger.error(f"[RedisClient] Command listener error: {e}")
        finally:
            self.running = False
    
    async def periodic_state_publisher(self, interval: float = 2.0):
        """Periodically publish state updates"""
        logger.info(f"[RedisClient] Started periodic state publisher (interval: {interval}s)")
        
        last_state_hash = None
        publish_count = 0
        
        try:
            while self.running:
                # Always try to publish - let's see what's happening
                if self.state:
                    current_state_str = json.dumps(self.state, sort_keys=True)
                    current_state_hash = hash(current_state_str)
                    
                    if current_state_hash != last_state_hash:
                        await self.publish_state_update()
                        last_state_hash = current_state_hash
                        publish_count += 1
                        logger.info(f"[RedisClient] State published #{publish_count} (state changed)")
                    else:
                        logger.info("[RedisClient] State unchanged, skipping publish")
                else:
                    logger.warning("[RedisClient] No state to publish")
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.info("[RedisClient] State publisher cancelled")
        except Exception as e:
            logger.error(f"[RedisClient] State publisher error: {e}")

# Global Redis client instance
redis_client = SimulationRedisClient()

async def start_redis_integration(
    state: Dict[str, Any],
    simulation_time_getter: Callable,
    command_handlers: Optional[Dict[str, Callable]] = None
):
    """Start Redis integration with the simulation"""
    
    # Connect to Redis
    if not await redis_client.connect():
        logger.error("[RedisClient] Failed to start Redis integration")
        return None
    
    # Set up state reference
    redis_client.set_state_reference(state, simulation_time_getter)
    
    # Register command handlers
    if command_handlers:
        for command, handler in command_handlers.items():
            redis_client.register_command_handler(command, handler)
    
    # Start background tasks
    command_task = asyncio.create_task(redis_client.command_listener_task())
    state_task = asyncio.create_task(redis_client.periodic_state_publisher())
    
    logger.info("[RedisClient] Redis integration started")
    
    return [command_task, state_task]

async def stop_redis_integration():
    """Stop Redis integration"""
    await redis_client.disconnect()

# Convenience functions for publishing
async def publish_state_update():
    """Publish state update to Redis"""
    await redis_client.publish_state_update()

async def publish_event(event_data: Dict[str, Any]):
    """Publish event to Redis"""
    await redis_client.publish_event(event_data)

def get_redis_client() -> Optional[SimulationRedisClient]:
    """Get the global Redis client instance"""
    return redis_client