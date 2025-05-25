# src/core_tasks.py - Core asynchronous tasks for the simulation (ADK-independent ones)

import asyncio
import logging
import time  # For time_manager_task
from typing import Any, Dict # Removed List, Optional as they are no longer used

# Import from our new modules
# Removed many constants as their tasks are now agents
from .config import MAX_SIMULATION_TIME, UPDATE_INTERVAL
# loop_utils and simulation_utils are still used by time_manager_task (via generate_table)
# from .loop_utils import get_nested # Not directly used by time_manager_task anymore
from .simulation_utils import generate_table


async def time_manager_task(
    current_state: Dict[str, Any],
    event_bus_qsize_func, # Function to get event_bus.qsize()
    narration_qsize_func, # Function to get narration_queue.qsize()
    live_display: Any, # Rich Live object
    logger_instance: logging.Logger
):
    """Updates display periodically."""
    logger_instance.info("[TimeManager] Task started.")

    try:
        while current_state.get("world_time", 0.0) < MAX_SIMULATION_TIME:
            # The ADK's TimeAdvancementPhase is now solely responsible for advancing
            # current_state["world_time"] (via synchronization from adk_session.state).
            # This task now only needs to update the display periodically.
            # The `current_state` dictionary is updated by `sync_state_from_adk_session` in simulation_async.py
            # before this task's loop iteration effectively sees it (due to await asyncio.sleep).

            live_display.update(generate_table(current_state, event_bus_qsize_func(), narration_qsize_func()))
            await asyncio.sleep(UPDATE_INTERVAL)

    except asyncio.CancelledError:
        logger_instance.info("[TimeManager] Task cancelled.")
    except Exception as e:
        logger_instance.exception(f"[TimeManager] Error in main loop: {e}")
    finally:
        logger_instance.info(f"[TimeManager] Loop finished at sim time {current_state.get('world_time', 0.0):.1f}")

