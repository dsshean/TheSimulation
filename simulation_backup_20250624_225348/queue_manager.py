# src/queue_manager.py - Enhanced Queue Management with Proper Synchronization

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, List
from collections import deque


class SequencedEventBus:
    """
    Event bus with proper sequencing and guaranteed ordering.
    Prevents event processing issues and race conditions.
    """
    
    def __init__(self, name: str, maxsize: int = 0):
        self.name = name
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.logger = logging.getLogger(f"EventBus.{name}")
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._stall_detection = {
            "last_size": 0,
            "stall_count": 0,
            "last_check_time": time.time()
        }
        
    async def put_event(self, event: Dict[str, Any], timeout: float = 5.0, task_name: str = "Unknown") -> bool:
        """
        Put event with sequence number and proper error handling.
        """
        async with self._lock:
            # Add sequence number for ordering
            event["sequence"] = self._sequence
            event["timestamp"] = time.time()
            self._sequence += 1
        
        try:
            self.logger.debug(f"[{self.name}] <- {task_name}: {json.dumps(event, indent=2)}")
            await asyncio.wait_for(self.queue.put(event), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            self.logger.error(f"[{self.name}] PUT TIMEOUT after {timeout}s from {task_name}")
            return False
        except Exception as e:
            self.logger.error(f"[{self.name}] PUT ERROR from {task_name}: {e}", exc_info=True)
            return False
    
    async def get_event(self, timeout: float = 5.0, task_name: str = "Unknown") -> Optional[Dict[str, Any]]:
        """
        Get event with proper task_done handling and error recovery.
        """
        try:
            event = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            self.logger.debug(f"[{self.name}] -> {task_name}: {json.dumps(event, indent=2)}")
            
            # Always call task_done after successful get
            try:
                self.queue.task_done()
            except ValueError as e:
                self.logger.warning(f"[{self.name}] task_done() called on empty queue: {e}")
            
            return event
        except asyncio.TimeoutError:
            self.logger.debug(f"[{self.name}] GET TIMEOUT after {timeout}s for {task_name}")
            return None
        except Exception as e:
            self.logger.error(f"[{self.name}] GET ERROR for {task_name}: {e}", exc_info=True)
            return None
    
    def qsize(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check queue health and detect stalls.
        """
        current_size = self.qsize()
        current_time = time.time()
        
        # Detect stalls (same size for multiple checks)
        if current_size == self._stall_detection["last_size"] and current_size > 0:
            self._stall_detection["stall_count"] += 1
        else:
            self._stall_detection["stall_count"] = 0
        
        self._stall_detection["last_size"] = current_size
        self._stall_detection["last_check_time"] = current_time
        
        is_stalled = self._stall_detection["stall_count"] >= 3  # 3 consecutive same sizes
        is_large = current_size > 10  # Arbitrarily large queue
        
        if is_stalled:
            self.logger.warning(f"[{self.name}] QUEUE STALLED: size={current_size}, stall_count={self._stall_detection['stall_count']}")
        
        if is_large:
            self.logger.warning(f"[{self.name}] LARGE QUEUE: size={current_size}")
        
        return {
            "name": self.name,
            "size": current_size,
            "is_stalled": is_stalled,
            "is_large": is_large,
            "stall_count": self._stall_detection["stall_count"],
            "unfinished_tasks": self.queue._unfinished_tasks
        }


class EventProcessor:
    """
    Processes events in chronological order with proper synchronization.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._pending_events = []
        self._lock = asyncio.Lock()
    
    async def add_event(self, event: Dict[str, Any]):
        """Add event to pending list in chronological order."""
        async with self._lock:
            # Insert in chronological order by trigger_sim_time
            trigger_time = event.get("trigger_sim_time", float('inf'))
            insert_pos = 0
            
            for i, existing_event in enumerate(self._pending_events):
                existing_time = existing_event.get("trigger_sim_time", float('inf'))
                if trigger_time < existing_time:
                    insert_pos = i
                    break
                insert_pos = i + 1
            
            self._pending_events.insert(insert_pos, event)
            self.logger.debug(f"[EventProcessor] Added event at position {insert_pos}, trigger_time={trigger_time}")
    
    async def process_due_events(self, current_sim_time: float) -> List[Dict[str, Any]]:
        """
        Process all events that are due and return them in chronological order.
        """
        processed_events = []
        
        async with self._lock:
            # Find all events that are due
            due_events = []
            remaining_events = []
            
            for event in self._pending_events:
                trigger_time = event.get("trigger_sim_time", float('inf'))
                if trigger_time <= current_sim_time:
                    due_events.append(event)
                else:
                    remaining_events.append(event)
            
            self._pending_events = remaining_events
            
            # Sort due events by sequence number to maintain order
            due_events.sort(key=lambda x: x.get("sequence", 0))
            
            self.logger.debug(f"[EventProcessor] Processing {len(due_events)} due events at sim_time={current_sim_time}")
            
            return due_events
    
    async def get_pending_count(self) -> int:
        """Get count of pending events."""
        async with self._lock:
            return len(self._pending_events)


class QueueHealthMonitor:
    """
    Monitors queue health and detects potential deadlocks.
    """
    
    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self.queues = {}
        self.logger = logging.getLogger("QueueHealthMonitor")
        self._running = False
    
    def register_queue(self, name: str, queue_bus: SequencedEventBus):
        """Register a queue for monitoring."""
        self.queues[name] = queue_bus
    
    async def start_monitoring(self):
        """Start the health monitoring task."""
        self._running = True
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_all_queues()
            except Exception as e:
                self.logger.error(f"Error in queue health monitoring: {e}", exc_info=True)
    
    async def _check_all_queues(self):
        """Check health of all registered queues."""
        for name, queue_bus in self.queues.items():
            health = queue_bus.check_health()
            
            if health["is_stalled"]:
                self.logger.error(f"Queue {name} appears to be stalled! Health: {health}")
            elif health["is_large"]:
                self.logger.warning(f"Queue {name} is getting large: {health}")
            else:
                self.logger.debug(f"Queue {name} health OK: size={health['size']}")
    
    def stop_monitoring(self):
        """Stop the health monitoring."""
        self._running = False