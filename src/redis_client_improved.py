"""
Improved Redis client with connection pooling, error handling, and retry logic.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
import redis.asyncio as redis
from datetime import datetime
import time
from enum import Enum

from .config import SIMULACRA_KEY, ACTIVE_SIMULACRA_IDS_KEY

# Redis configuration
REDIS_HOST = "127.0.0.1" 
REDIS_PORT = 6379
REDIS_DB = 0

# Connection pool settings
MAX_CONNECTIONS = 10
MIN_CONNECTIONS = 2
CONNECTION_TIMEOUT = 30  # seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0  # seconds
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 30  # seconds

# Channel names
STATE_CHANNEL = "simulation:state"
COMMAND_CHANNEL = "simulation:commands"
EVENT_CHANNEL = "simulation:events"

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker tripped
    HALF_OPEN = "half_open"  # Testing if service recovered


class RedisCircuitBreaker:
    """Circuit breaker for Redis operations"""
    
    def __init__(self, threshold: int = CIRCUIT_BREAKER_THRESHOLD, 
                 timeout: int = CIRCUIT_BREAKER_TIMEOUT):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        
    def can_execute(self) -> bool:
        """Check if operation should be allowed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("[RedisCircuitBreaker] Transitioning to HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        """Record successful operation"""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("[RedisCircuitBreaker] Transitioning to CLOSED")
    
    def on_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"[RedisCircuitBreaker] Circuit breaker OPEN after {self.failure_count} failures")


class ImprovedRedisClient:
    """Enhanced Redis client with connection pooling and error handling"""
    
    def __init__(self):
        self.connection_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.state: Optional[Dict[str, Any]] = None
        self.simulation_time_getter: Optional[Callable] = None
        self.command_handlers: Dict[str, Callable] = {}
        self.running = False
        self.recent_events: List[Dict] = []
        self.max_recent_events = 50
        self.circuit_breaker = RedisCircuitBreaker()
        self._connection_healthy = False
        
    async def connect(self) -> bool:
        """Connect to Redis with connection pooling"""
        try:
            # Create connection pool
            self.connection_pool = redis.ConnectionPool(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                max_connections=MAX_CONNECTIONS,
                socket_connect_timeout=CONNECTION_TIMEOUT,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                retry_on_error=[redis.ConnectionError, redis.TimeoutError]
            )
            
            # Create Redis client with connection pool
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                socket_connect_timeout=CONNECTION_TIMEOUT,
                retry_on_timeout=True
            )
            
            # Test connection with retry
            for attempt in range(RETRY_ATTEMPTS):
                try:
                    await self.redis_client.ping()
                    self._connection_healthy = True
                    break
                except Exception as e:
                    logger.warning(f"[RedisClient] Connection attempt {attempt + 1} failed: {e}")
                    if attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
                    else:
                        raise
            
            logger.info(f"[RedisClient] Connected to Redis at {REDIS_HOST}:{REDIS_PORT} with connection pool")
            
            # Set up pub/sub for commands
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe(COMMAND_CHANNEL)
            
            self.circuit_breaker.on_success()
            return True
            
        except Exception as e:
            logger.error(f"[RedisClient] Failed to connect to Redis: {e}")
            self._connection_healthy = False
            self.circuit_breaker.on_failure()
            return False
    
    async def disconnect(self):
        """Disconnect from Redis and cleanup"""
        self.running = False
        
        try:
            if self.pubsub:
                await self.pubsub.unsubscribe(COMMAND_CHANNEL)
                await self.pubsub.close()
                
            if self.redis_client:
                await self.redis_client.close()
                
            if self.connection_pool:
                await self.connection_pool.disconnect()
                
        except Exception as e:
            logger.error(f"[RedisClient] Error during disconnect: {e}")
        
        self._connection_healthy = False
        logger.info("[RedisClient] Disconnected from Redis")
    
    async def _execute_with_retry(self, operation: Callable, *args, **kwargs) -> Optional[Any]:
        """Execute Redis operation with retry logic and circuit breaker"""
        if not self.circuit_breaker.can_execute():
            logger.debug("[RedisClient] Circuit breaker is OPEN, skipping operation")
            return None
            
        if not self.redis_client:
            logger.error("[RedisClient] No Redis client available")
            return None
        
        for attempt in range(RETRY_ATTEMPTS):
            try:
                result = await operation(*args, **kwargs)
                self.circuit_breaker.on_success()
                self._connection_healthy = True
                return result
                
            except (redis.ConnectionError, redis.TimeoutError) as e:
                self._connection_healthy = False
                logger.warning(f"[RedisClient] Operation failed (attempt {attempt + 1}): {e}")
                
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
                    # Try to reconnect
                    try:
                        await self.redis_client.ping()
                        self._connection_healthy = True
                    except:
                        logger.debug("[RedisClient] Reconnection ping failed")
                else:
                    self.circuit_breaker.on_failure()
                    logger.error(f"[RedisClient] Operation failed after {RETRY_ATTEMPTS} attempts")
                    return None
                    
            except Exception as e:
                logger.error(f"[RedisClient] Unexpected error in Redis operation: {e}")
                self.circuit_breaker.on_failure()
                return None
        
        return None
    
    def set_state_reference(self, state: Dict[str, Any], simulation_time_getter: Callable):
        """Set references to simulation state and time getter"""
        self.state = state
        self.simulation_time_getter = simulation_time_getter
    
    def register_command_handler(self, command: str, handler: Callable):
        """Register a command handler function"""
        self.command_handlers[command] = handler
        logger.info(f"[RedisClient] Registered handler for command: {command}")
    
    async def publish_state_update(self):
        """Publish current simulation state to Redis with error handling"""
        if not self.state or not self.simulation_time_getter:
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
                "connection_healthy": self._connection_healthy,
                "timestamp": datetime.now().isoformat()
            }
            
            message = json.dumps(state_data)
            
            # Use retry mechanism for publish
            await self._execute_with_retry(
                self.redis_client.publish,
                STATE_CHANNEL,
                message
            )
            
        except Exception as e:
            logger.error(f"[RedisClient] Error preparing state update: {e}")
    
    async def publish_event(self, event_data: Dict[str, Any]):
        """Publish an event to Redis with error handling"""
        try:
            event_message = {
                "timestamp": datetime.now().isoformat(),
                "sim_time": self.simulation_time_getter() if self.simulation_time_getter else 0,
                **event_data
            }
            
            message = json.dumps(event_message)
            
            # Use retry mechanism for publish
            await self._execute_with_retry(
                self.redis_client.publish,
                EVENT_CHANNEL,
                message
            )
            
        except Exception as e:
            logger.error(f"[RedisClient] Error preparing event: {e}")
    
    def add_recent_event(self, event_data: Dict[str, Any]):
        """Add an event to the recent events buffer"""
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
                handler = self.command_handlers[command]
                try:
                    response = await handler(message_data)
                except Exception as e:
                    logger.error(f"[RedisClient] Error in command handler: {e}")
                    response = {"success": False, "message": f"Handler error: {str(e)}"}
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
                
                # Use retry mechanism for response
                result = await self._execute_with_retry(
                    self.redis_client.publish,
                    response_channel,
                    message
                )
                
                if result:
                    # Set expiration for response channel
                    await self._execute_with_retry(
                        self.redis_client.expire,
                        response_channel,
                        60  # 1 minute TTL
                    )
                
        except Exception as e:
            logger.error(f"[RedisClient] Error handling command: {e}")
    
    async def command_listener_task(self):
        """Main task to listen for commands from Redis with reconnection"""
        if not self.pubsub:
            logger.error("[RedisClient] PubSub not initialized")
            return
            
        self.running = True
        logger.info("[RedisClient] Started command listener")
        
        reconnect_delay = 1.0
        max_reconnect_delay = 30.0
        
        while self.running:
            try:
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self.handle_command_message(data)
                        reconnect_delay = 1.0  # Reset reconnect delay on success
                    except json.JSONDecodeError as e:
                        logger.error(f"[RedisClient] Invalid JSON in command: {e}")
                    except Exception as e:
                        logger.error(f"[RedisClient] Error processing command: {e}")
                        
            except asyncio.TimeoutError:
                continue  # Normal timeout, keep listening
                
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"[RedisClient] Connection lost in command listener: {e}")
                self._connection_healthy = False
                
                # Try to reconnect with exponential backoff
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                
                try:
                    logger.info("[RedisClient] Attempting to reconnect...")
                    await self.connect()
                    if self._connection_healthy:
                        logger.info("[RedisClient] Reconnected successfully")
                        reconnect_delay = 1.0
                except Exception as reconnect_error:
                    logger.error(f"[RedisClient] Reconnection failed: {reconnect_error}")
                    
            except asyncio.CancelledError:
                logger.info("[RedisClient] Command listener cancelled")
                break
                
            except Exception as e:
                logger.error(f"[RedisClient] Unexpected error in command listener: {e}")
                await asyncio.sleep(1.0)
        
        logger.info("[RedisClient] Command listener stopped")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if not self.connection_pool:
            return {"error": "No connection pool"}
            
        return {
            "max_connections": self.connection_pool.max_connections,
            "created_connections": len(self.connection_pool._available_connections) + len(self.connection_pool._in_use_connections),
            "available_connections": len(self.connection_pool._available_connections),
            "in_use_connections": len(self.connection_pool._in_use_connections),
            "healthy": self._connection_healthy,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "failure_count": self.circuit_breaker.failure_count
        }


# Global client instance
improved_redis_client = ImprovedRedisClient()


async def start_redis_integration(
    state: Dict[str, Any],
    simulation_time_getter: Callable,
    command_handlers: Optional[Dict[str, Callable]] = None
) -> Optional[List[asyncio.Task]]:
    """Start Redis integration with improved client"""
    
    logger.info("[RedisClient] Starting improved Redis integration...")
    
    # Connect to Redis
    if not await improved_redis_client.connect():
        logger.error("[RedisClient] Failed to start improved Redis integration")
        return None
    
    # Set up state reference
    improved_redis_client.set_state_reference(state, simulation_time_getter)
    
    # Register command handlers
    if command_handlers:
        for command, handler in command_handlers.items():
            improved_redis_client.register_command_handler(command, handler)
    
    # Start background tasks
    tasks = []
    
    # Command listener task
    command_task = asyncio.create_task(improved_redis_client.command_listener_task())
    tasks.append(command_task)
    
    # State publisher task (every 0.5 seconds)
    async def state_publisher_task():
        while improved_redis_client.running:
            try:
                await improved_redis_client.publish_state_update()
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[RedisClient] Error in state publisher: {e}")
                await asyncio.sleep(1.0)
    
    publisher_task = asyncio.create_task(state_publisher_task())
    tasks.append(publisher_task)
    
    logger.info(f"[RedisClient] Improved Redis integration started with {len(tasks)} tasks")
    return tasks


async def stop_redis_integration():
    """Stop Redis integration"""
    logger.info("[RedisClient] Stopping improved Redis integration...")
    await improved_redis_client.disconnect()