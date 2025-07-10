# src/simulation_sync_patch.py - Integration patch for synchronized components

"""
This module contains the updated imports and initialization code to integrate
the new synchronization components into the main simulation.

To apply this patch:
1. Update the imports in simulation_async.py
2. Replace the queue initialization code
3. Update the perception manager initialization
4. Replace the simulacra agent task function
"""

# ===== UPDATED IMPORTS (add to simulation_async.py imports) =====

from .state_manager import StateManager, CircuitBreaker
from .queue_manager import SequencedEventBus, EventProcessor, QueueHealthMonitor
from .perception_manager import SynchronizedPerceptionManager

# ===== GLOBAL VARIABLES (replace in simulation_async.py) =====

# Replace the old asyncio.Queue instances with SequencedEventBus
event_bus = None  # Will be initialized as SequencedEventBus
narration_queue = None  # Will be initialized as SequencedEventBus

# Add new synchronized components
state_manager_global: Optional[StateManager] = None
circuit_breaker_global: Optional[CircuitBreaker] = None
event_processor_global: Optional[EventProcessor] = None
queue_health_monitor_global: Optional[QueueHealthMonitor] = None

# ===== INITIALIZATION FUNCTION (add to simulation_async.py) =====

async def initialize_synchronized_components(logger_instance: logging.Logger):
    """
    Initialize the synchronized components for the simulation.
    This should be called early in run_simulation().
    """
    global event_bus, narration_queue, state_manager_global, circuit_breaker_global
    global event_processor_global, queue_health_monitor_global, perception_manager_global
    
    logger_instance.info("[SyncInit] Initializing synchronized components...")
    
    # Initialize enhanced event buses
    event_bus = SequencedEventBus("EventBus", maxsize=100)
    narration_queue = SequencedEventBus("NarrationQueue", maxsize=50)
    
    # Initialize state manager
    state_manager_global = StateManager(state, logger_instance)
    
    # Initialize circuit breaker
    circuit_breaker_global = CircuitBreaker(max_repetitions=3, window_size=5)
    
    # Initialize event processor
    event_processor_global = EventProcessor(logger_instance)
    
    # Initialize queue health monitor
    queue_health_monitor_global = QueueHealthMonitor(check_interval=10.0)
    queue_health_monitor_global.register_queue("event_bus", event_bus)
    queue_health_monitor_global.register_queue("narration_queue", narration_queue)
    
    # Update perception manager to use synchronized version
    perception_manager_global = SynchronizedPerceptionManager(state, state_manager_global)
    
    logger_instance.info("[SyncInit] Synchronized components initialized successfully")
    
    # Start queue health monitoring
    asyncio.create_task(queue_health_monitor_global.start_monitoring())

# ===== UPDATED SIMULACRA AGENT TASK (replace in simulation_async.py) =====

async def simulacra_agent_task_llm_synchronized(agent_id: str, instance_uuid: str, logger_instance: logging.Logger):
    """
    Enhanced simulacra agent task with proper synchronization and circuit breaking.
    """
    global state_manager_global, circuit_breaker_global, event_bus, perception_manager_global
    
    logger_instance.info(f"[{agent_id}] Synchronized agent task started")
    
    # Validate that synchronized components are available
    if not all([state_manager_global, circuit_breaker_global, event_bus, perception_manager_global]):
        logger_instance.error(f"[{agent_id}] Synchronized components not initialized!")
        return
    
    try:
        # Initialize agent-specific state
        if agent_id not in simulacra_agents_map or agent_id not in simulacra_runners_map:
            logger_instance.error(f"[{agent_id}] Agent or runner not found in maps")
            return
        
        agent_llm = simulacra_agents_map[agent_id]
        agent_runner = simulacra_runners_map[agent_id]
        
        # Initialize agent status safely
        await state_manager_global.safe_status_transition(
            agent_id, "unknown", "idle", 
            additional_updates=[
                (f"{SIMULACRA_KEY}.{agent_id}.current_action_description", "initializing"),
                (f"{SIMULACRA_KEY}.{agent_id}.last_interjection_sim_time", 0.0),
                (f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None)
            ]
        )
        
        # Main agent loop
        while True:
            current_sim_time = state.get("world_time", 0.0)
            
            # Check if we should stop
            if current_sim_time >= float(os.getenv("MAX_SIMULATION_TIME", "9996000.0")):
                break
            
            # Get current status safely
            async with state_manager_global._lock:
                current_status = get_nested(state, SIMULACRA_KEY, agent_id, "status")
                
            if current_status == "idle":
                # Agent is ready to act
                await _handle_agent_action_cycle(agent_id, agent_llm, agent_runner, logger_instance)
            
            elif current_status == "busy":
                # Agent is busy, check for self-reflection opportunity
                await _check_self_reflection(agent_id, logger_instance)
            
            else:
                logger_instance.debug(f"[{agent_id}] Status: {current_status}, waiting...")
            
            # Controlled polling interval
            await asyncio.sleep(AGENT_BUSY_POLL_INTERVAL_REAL_SECONDS)
    
    except Exception as e:
        logger_instance.error(f"[{agent_id}] Agent task failed: {e}", exc_info=True)
        # Try to clean up agent state
        await state_manager_global.safe_status_transition(agent_id, "busy", "idle")

async def _handle_agent_action_cycle(agent_id: str, agent_llm: LlmAgent, agent_runner: Runner, 
                                   logger_instance: logging.Logger):
    """
    Handle a complete agent action cycle with synchronization.
    """
    global state_manager_global, circuit_breaker_global, event_bus, perception_manager_global
    
    try:
        # Step 1: Transition to thinking state atomically
        thinking_transition = await state_manager_global.safe_status_transition(
            agent_id, "idle", "thinking"
        )
        
        if not thinking_transition:
            logger_instance.warning(f"[{agent_id}] Failed to transition to thinking state")
            return
        
        # Step 2: Build perception context with synchronization
        fresh_percepts = await perception_manager_global.get_fresh_percepts(agent_id)
        
        # Step 3: Build full agent context
        full_context = await _build_synchronized_agent_context(agent_id, fresh_percepts, logger_instance)
        
        # Step 4: Call LLM for intent
        intent_response = await _call_agent_llm_safely(agent_llm, agent_runner, full_context, logger_instance)
        
        if not intent_response:
            # Failed to get response, reset to idle
            await state_manager_global.safe_status_transition(agent_id, "thinking", "idle")
            return
        
        # Step 5: Validate intent with circuit breaker
        action_type = intent_response.get("action_type", "think")
        
        if circuit_breaker_global.add_action(agent_id, action_type):
            logger_instance.warning(f"[{agent_id}] Circuit breaker triggered for action '{action_type}'")
            # Force a different action
            intent_response = {
                "action_type": "think",
                "target_id": None,
                "details": "Taking a moment to reconsider my approach"
            }
            circuit_breaker_global.reset_agent(agent_id)
        
        # Step 6: Submit intent with synchronization
        intent_event = {
            "event_type": "intent_declared",
            "actor_id": agent_id,
            "data": intent_response,
            "trigger_sim_time": state.get("world_time", 0.0)
        }
        
        success = await event_bus.put_event(intent_event, task_name=f"simulacra_{agent_id}")
        
        if success:
            # Successfully submitted intent
            logger_instance.info(f"[{agent_id}] Intent submitted: {action_type}")
        else:
            # Failed to submit, reset to idle
            logger_instance.error(f"[{agent_id}] Failed to submit intent")
            await state_manager_global.safe_status_transition(agent_id, "thinking", "idle")
    
    except Exception as e:
        logger_instance.error(f"[{agent_id}] Error in action cycle: {e}", exc_info=True)
        await state_manager_global.safe_status_transition(agent_id, "thinking", "idle")

async def _build_synchronized_agent_context(agent_id: str, fresh_percepts: Dict[str, Any], 
                                          logger_instance: logging.Logger) -> str:
    """
    Build agent context with synchronized state access.
    """
    async with state_manager_global._lock:
        # Get agent data safely
        agent_data = get_nested(state, SIMULACRA_KEY, agent_id, default={})
        
        # Build context string (simplified version)
        context_parts = []
        
        # Add persona information
        persona_details = agent_data.get("persona_details", {})
        if persona_details:
            context_parts.append(f"You are {persona_details.get('Name', agent_id)}")
            context_parts.append(f"Background: {persona_details.get('Background', 'Unknown background')}")
        
        # Add current location and perception
        current_location = fresh_percepts.get("current_location_id", "unknown")
        location_desc = fresh_percepts.get("location_description", "You are in an undefined space.")
        context_parts.append(f"Current location: {current_location}")
        context_parts.append(f"You see: {location_desc}")
        
        # Add visible simulacra
        visible_sims = fresh_percepts.get("visible_simulacra", [])
        if visible_sims:
            sim_descriptions = []
            for sim in visible_sims:
                sim_desc = f"{sim['name']} (status: {sim['status']})"
                if sim.get('current_action'):
                    sim_desc += f" - {sim['current_action']}"
                sim_descriptions.append(sim_desc)
            context_parts.append(f"Other people here: {', '.join(sim_descriptions)}")
        
        # Add last observation
        last_obs = agent_data.get("last_observation", "")
        if last_obs:
            context_parts.append(f"Last observation: {last_obs}")
        
        # Add world time
        world_time = state.get("world_time", 0.0)
        context_parts.append(f"Current simulation time: {world_time:.1f}s")
        
        context_parts.append("What do you want to do next? Respond with valid JSON containing action_type, target_id, and details.")
        
        return "\n\n".join(context_parts)

async def _call_agent_llm_safely(agent_llm: LlmAgent, agent_runner: Runner, context: str, 
                                logger_instance: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Call agent LLM with proper error handling and timeout.
    """
    try:
        # Implement timeout for LLM calls
        response = await asyncio.wait_for(
            agent_runner.run(agent_llm, context),
            timeout=30.0  # 30 second timeout
        )
        
        if hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)
        
        # Parse JSON response
        intent_data = parse_json_output_last(response_text)
        
        if intent_data and isinstance(intent_data, dict):
            return intent_data
        else:
            logger_instance.warning(f"[Agent] Invalid JSON response: {response_text[:200]}")
            return None
    
    except asyncio.TimeoutError:
        logger_instance.error(f"[Agent] LLM call timed out")
        return None
    except Exception as e:
        logger_instance.error(f"[Agent] LLM call failed: {e}", exc_info=True)
        return None

async def _check_self_reflection(agent_id: str, logger_instance: logging.Logger):
    """
    Check if busy agent should engage in self-reflection.
    """
    # Implementation would check timing and probability for self-reflection
    # This is a placeholder for the existing self-reflection logic
    pass

# ===== UPDATED WORLD ENGINE TASK (replace in simulation_async.py) =====

async def world_engine_task_llm_synchronized(instance_uuid: str, logger_instance: logging.Logger):
    """
    Enhanced world engine task with proper synchronization.
    """
    global state_manager_global, event_bus, narration_queue, world_engine_agent, world_engine_runner
    
    logger_instance.info("[WorldEngine] Synchronized task started")
    
    while True:
        try:
            # Get intent event with timeout
            intent_event = await event_bus.get_event(timeout=5.0, task_name="world_engine")
            
            if not intent_event:
                await asyncio.sleep(0.1)
                continue
            
            event_type = intent_event.get("event_type")
            if event_type != "intent_declared":
                logger_instance.warning(f"[WorldEngine] Unexpected event type: {event_type}")
                continue
            
            actor_id = intent_event.get("actor_id")
            intent_data = intent_event.get("data", {})
            
            # Process intent with synchronization
            await _process_intent_synchronized(actor_id, intent_data, logger_instance)
        
        except Exception as e:
            logger_instance.error(f"[WorldEngine] Task error: {e}", exc_info=True)
            await asyncio.sleep(1.0)

async def _process_intent_synchronized(actor_id: str, intent_data: Dict[str, Any], 
                                     logger_instance: logging.Logger):
    """
    Process agent intent with proper state synchronization.
    """
    global state_manager_global, narration_queue, world_engine_agent, world_engine_runner
    
    action_type = intent_data.get("action_type", "unknown")
    
    # Transition agent to busy state atomically
    busy_transition = await state_manager_global.safe_status_transition(
        actor_id, "thinking", "busy",
        additional_updates=[
            (f"{SIMULACRA_KEY}.{actor_id}.current_action_description", f"performing {action_type}"),
            (f"{SIMULACRA_KEY}.{actor_id}.current_action_start_time", state.get("world_time", 0.0))
        ]
    )
    
    if not busy_transition:
        logger_instance.warning(f"[WorldEngine] Failed to transition {actor_id} to busy state")
        return
    
    try:
        # Build world engine context
        context = await _build_world_engine_context(actor_id, intent_data, logger_instance)
        
        # Call world engine LLM
        response = await asyncio.wait_for(
            world_engine_runner.run(world_engine_agent, context),
            timeout=30.0
        )
        
        # Process response and update state
        await _apply_world_engine_response(actor_id, intent_data, response, logger_instance)
    
    except Exception as e:
        logger_instance.error(f"[WorldEngine] Error processing intent for {actor_id}: {e}", exc_info=True)
        # Reset agent to idle on error
        await state_manager_global.safe_status_transition(actor_id, "busy", "idle")

async def _build_world_engine_context(actor_id: str, intent_data: Dict[str, Any], 
                                    logger_instance: logging.Logger) -> str:
    """
    Build context for world engine with synchronized state access.
    """
    # This would contain the logic to build world engine prompts
    # Similar to existing implementation but with synchronized state access
    return f"Process action for {actor_id}: {intent_data}"

async def _apply_world_engine_response(actor_id: str, intent_data: Dict[str, Any], 
                                     response: Any, logger_instance: logging.Logger):
    """
    Apply world engine response to state with synchronization.
    """
    # This would contain the logic to parse world engine response and apply changes
    # Similar to existing implementation but using state_manager for atomic updates
    pass