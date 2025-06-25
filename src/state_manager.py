# src/state_manager.py - Atomic State Management with Synchronization

import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple, Optional
from .simulation_utils import _update_state_value


class StateManager:
    """
    Thread-safe state manager for atomic operations and synchronization.
    Prevents race conditions in multi-agent scenarios.
    """
    
    def __init__(self, state: Dict[str, Any], logger: logging.Logger):
        self.state = state
        self.logger = logger
        self._lock = asyncio.Lock()
        self._version = 0
        self._last_update_time = time.time()
        
    async def atomic_update(self, updates: List[Tuple[str, Any]], operation_name: str = "unknown") -> bool:
        """
        Perform multiple state updates atomically.
        All updates succeed or all fail.
        """
        async with self._lock:
            try:
                self.logger.debug(f"[StateManager] Starting atomic operation: {operation_name}")
                
                # Store backup of current values for rollback
                backup = {}
                for key_path, _ in updates:
                    current_value = self._get_nested_value(key_path)
                    backup[key_path] = current_value
                
                # Apply all updates
                success_count = 0
                for key_path, value in updates:
                    if _update_state_value(self.state, key_path, value, self.logger):
                        success_count += 1
                    else:
                        self.logger.error(f"[StateManager] Failed to update {key_path} in {operation_name}")
                        # Rollback all changes
                        for rollback_path, rollback_value in backup.items():
                            if rollback_value is not None:
                                _update_state_value(self.state, rollback_path, rollback_value, self.logger)
                        return False
                
                # Update version and timestamp
                self._version += 1
                self._last_update_time = time.time()
                self.state["state_version"] = self._version
                self.state["last_state_update"] = self._last_update_time
                
                self.logger.info(f"[StateManager] Atomic operation '{operation_name}' completed successfully ({success_count} updates)")
                return True
                
            except Exception as e:
                self.logger.error(f"[StateManager] Atomic operation '{operation_name}' failed: {e}", exc_info=True)
                return False
    
    async def safe_status_transition(self, agent_id: str, from_status: str, to_status: str, 
                                   additional_updates: Optional[List[Tuple[str, Any]]] = None) -> bool:
        """
        Safely transition agent status with atomic operation.
        Only succeeds if current status matches expected from_status.
        """
        async with self._lock:
            current_status = self._get_nested_value(f"simulacra_profiles.{agent_id}.status")
            
            if current_status != from_status:
                self.logger.warning(f"[StateManager] Status transition failed for {agent_id}: "
                                  f"expected '{from_status}' but found '{current_status}'")
                return False
            
            # Prepare updates
            updates = [(f"simulacra_profiles.{agent_id}.status", to_status)]
            if additional_updates:
                updates.extend(additional_updates)
            
            return await self.atomic_update(updates, f"status_transition_{agent_id}_{from_status}_to_{to_status}")
    
    async def safe_location_change(self, agent_id: str, new_location_id: str, 
                                 location_details: Optional[str] = None) -> bool:
        """
        Safely change agent location with proper previous location tracking.
        """
        async with self._lock:
            current_location = self._get_nested_value(f"simulacra_profiles.{agent_id}.current_location")
            
            updates = [
                (f"simulacra_profiles.{agent_id}.current_location", new_location_id),
                (f"simulacra_profiles.{agent_id}.previous_location_id", current_location)
            ]
            
            if location_details:
                updates.append((f"simulacra_profiles.{agent_id}.location_details", location_details))
            
            return await self.atomic_update(updates, f"location_change_{agent_id}")
    
    async def safe_conversation_start(self, speaker_id: str, listener_id: str, 
                                    message_content: str) -> bool:
        """
        Atomically start a conversation between two agents.
        Sets both agents to busy and delivers speech interrupt.
        """
        updates = [
            (f"simulacra_profiles.{speaker_id}.status", "busy"),
            (f"simulacra_profiles.{listener_id}.status", "busy"),
            (f"simulacra_profiles.{listener_id}.last_observation", f"Speech from {speaker_id}: {message_content}"),
            (f"simulacra_profiles.{listener_id}.current_interrupt_probability", None)
        ]
        
        return await self.atomic_update(updates, f"conversation_start_{speaker_id}_to_{listener_id}")
    
    def _get_nested_value(self, key_path: str) -> Any:
        """Get nested value using dot notation."""
        try:
            keys = key_path.split('.')
            value = self.state
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except Exception:
            return None
    
    def get_state_version(self) -> int:
        """Get current state version for conflict detection."""
        return self._version
    
    def get_last_update_time(self) -> float:
        """Get timestamp of last state update."""
        return self._last_update_time
    
    async def with_lock(self, coro):
        """Execute coroutine with state lock held."""
        async with self._lock:
            return await coro


class CircuitBreaker:
    """
    Prevents agents from getting stuck in repetitive action loops.
    """
    
    def __init__(self, max_repetitions: int = 3, window_size: int = 5):
        self.max_repetitions = max_repetitions
        self.window_size = window_size
        self.action_history = {}  # agent_id -> List[action_type]
        
    def add_action(self, agent_id: str, action_type: str) -> bool:
        """
        Add an action to history and check if circuit should break.
        Returns True if action should be blocked (circuit broken).
        """
        if agent_id not in self.action_history:
            self.action_history[agent_id] = []
        
        history = self.action_history[agent_id]
        history.append(action_type)
        
        # Keep only recent actions
        if len(history) > self.window_size:
            history.pop(0)
        
        # Check for repetitive pattern
        if len(history) >= self.max_repetitions:
            recent_actions = history[-self.max_repetitions:]
            if len(set(recent_actions)) == 1:  # All same action
                logging.getLogger(__name__).warning(
                    f"[CircuitBreaker] Agent {agent_id} stuck in loop with action '{action_type}'. "
                    f"Recent history: {history}"
                )
                return True
        
        return False
    
    def reset_agent(self, agent_id: str):
        """Reset action history for an agent."""
        if agent_id in self.action_history:
            del self.action_history[agent_id]