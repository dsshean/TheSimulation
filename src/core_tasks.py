# src/core_tasks.py - Core asynchronous tasks for the simulation (ADK-independent ones)

import asyncio
import logging
import time # For time_manager_task
from typing import Any, Dict, Optional, List

# Import from our new modules
from .config import (
    MAX_SIMULATION_TIME, SIMULATION_SPEED_FACTOR, UPDATE_INTERVAL, MAX_MEMORY_LOG_ENTRIES,
    WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS, MAX_WORLD_FEED_ITEMS, USER_ID,
    WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY
)
# simulation_utils will be called by tasks in simulation_async.py or here, passing state
from .simulation_utils import _update_state_value, generate_table, generate_simulated_world_feed_content
from .loop_utils import get_nested

async def time_manager_task(
    current_state: Dict[str, Any],
    event_bus_qsize_func, # Function to get event_bus.qsize()
    narration_qsize_func, # Function to get narration_queue.qsize()
    live_display: Any, # Rich Live object
    logger_instance: logging.Logger
):
    """Advances time, applies completed action effects, and updates display."""
    logger_instance.info("[TimeManager] Task started.")
    last_real_time = time.monotonic()

    try:
        while current_state.get("world_time", 0.0) < MAX_SIMULATION_TIME:
            current_real_time = time.monotonic()
            real_delta_time = current_real_time - last_real_time
            last_real_time = current_real_time
            sim_delta_time = real_delta_time * SIMULATION_SPEED_FACTOR
            current_sim_time_val = current_state.setdefault("world_time", 0.0)
            new_sim_time = current_sim_time_val + sim_delta_time
            current_state["world_time"] = new_sim_time

            for agent_id, agent_state_data in list(current_state.get("simulacra", {}).items()):
                if agent_state_data.get("status") == "busy":
                    action_end_time = agent_state_data.get("current_action_end_time", -1.0)
                    if action_end_time <= new_sim_time:
                        logger_instance.info(f"[TimeManager] Applying completed action effects for {agent_id} at time {new_sim_time:.1f} (due at {action_end_time:.1f}).")
                        pending_results = agent_state_data.get("pending_results", {})
                        if pending_results:
                            memory_log_updated = False
                            for key_path, value in list(pending_results.items()):
                                success = _update_state_value(current_state, key_path, value, logger_instance)
                                if success and key_path == f"simulacra.{agent_id}.memory_log":
                                    memory_log_updated = True
                            _update_state_value(current_state, f"simulacra.{agent_id}.pending_results", {}, logger_instance)
                            if memory_log_updated:
                                current_mem_log = get_nested(current_state, "simulacra", agent_id, "memory_log", default=[])
                                if isinstance(current_mem_log, list) and len(current_mem_log) > MAX_MEMORY_LOG_ENTRIES:
                                    _update_state_value(current_state, f"simulacra.{agent_id}.memory_log", current_mem_log[-MAX_MEMORY_LOG_ENTRIES:], logger_instance)
                                    logger_instance.debug(f"[TimeManager] Pruned memory log for {agent_id} to {MAX_MEMORY_LOG_ENTRIES} entries.")
                        else:
                            logger_instance.debug(f"[TimeManager] No pending results found for completed action of {agent_id}.")
                        _update_state_value(current_state, f"simulacra.{agent_id}.status", "idle", logger_instance)
                        logger_instance.info(f"[TimeManager] Set {agent_id} status to idle.")
            
            live_display.update(generate_table(current_state, event_bus_qsize_func(), narration_qsize_func()))
            await asyncio.sleep(UPDATE_INTERVAL)

    except asyncio.CancelledError:
        logger_instance.info("[TimeManager] Task cancelled.")
    except Exception as e:
        logger_instance.exception(f"[TimeManager] Error in main loop: {e}")
    finally:
        logger_instance.info(f"[TimeManager] Loop finished at sim time {current_state.get('world_time', 0.0):.1f}")


async def interaction_dispatcher_task(
    current_state: Dict[str, Any],
    event_bus_instance: asyncio.Queue,
    logger_instance: logging.Logger
):
    """Listens for intents and classifies them before sending to World Engine."""
    logger_instance.info("[InteractionDispatcher] Task started.")
    while True:
        intent_event = None
        try:
            intent_event = await event_bus_instance.get()
            if get_nested(intent_event, "type") != "intent_declared":
                logger_instance.debug(f"[InteractionDispatcher] Ignoring event type: {get_nested(intent_event, 'type')}")
                event_bus_instance.task_done()
                continue

            actor_id = get_nested(intent_event, "actor_id")
            intent = get_nested(intent_event, "intent")
            if not actor_id or not intent:
                logger_instance.warning(f"[InteractionDispatcher] Received invalid intent event: {intent_event}")
                event_bus_instance.task_done()
                continue

            target_id = intent.get("target_id")
            action_type = intent.get("action_type")
            interaction_class = "environment"

            if target_id:
                if target_id in get_nested(current_state, "simulacra", default={}):
                    interaction_class = "entity"
                elif target_id in get_nested(current_state, "objects", default={}) and get_nested(current_state, "objects", target_id, "interactive", default=False):
                     interaction_class = "entity"

            logger_instance.info(f"[InteractionDispatcher] Intent from {actor_id} ({action_type} on {target_id or 'N/A'}) classified as '{interaction_class}'.")
            await event_bus_instance.put({"type": "resolve_action_request", "actor_id": actor_id, "intent": intent, "interaction_class": interaction_class})
            event_bus_instance.task_done()
            await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger_instance.info("[InteractionDispatcher] Task cancelled.")
            if intent_event and event_bus_instance._unfinished_tasks > 0:
                try: event_bus_instance.task_done()
                except ValueError: pass
            break
        except Exception as e:
            logger_instance.exception(f"[InteractionDispatcher] Error processing event: {e}")
            if intent_event and event_bus_instance._unfinished_tasks > 0:
                try: event_bus_instance.task_done()
                except ValueError: pass

async def world_info_gatherer_task(
    current_state: Dict[str, Any],
    world_mood: str,
    # These ADK components are passed because generate_simulated_world_feed_content uses them
    search_agent_runner_instance: Optional[Any], # Runner
    search_agent_session_id_for_search: Optional[str],
    logger_instance: logging.Logger
):
    """Periodically fetches/generates world information."""
    logger_instance.info("[WorldInfoGatherer] Task started.")
    await asyncio.sleep(10)

    while True:
        try:
            current_sim_time_val = current_state.get("world_time", 0.0)
            location_info_parts = [
                get_nested(current_state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'city', default="Unknown City"),
                get_nested(current_state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'state', default="Unknown State"),
                get_nested(current_state, WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, 'country', default="Unknown Country")
            ]
            location_context_str = ", ".join(filter(None, location_info_parts)) or "an unspecified location"
            logger_instance.info(f"[WorldInfoGatherer] Updating world feeds at sim_time {current_sim_time_val:.1f} for location: {location_context_str}")
            
            _update_state_value(current_state, 'world_feeds.last_update_sim_time', current_sim_time_val, logger_instance)
            
            weather_data = await generate_simulated_world_feed_content(
                current_sim_state=current_state, category="weather", simulation_time=current_sim_time_val,
                location_context=location_context_str, world_mood=world_mood,
                global_search_agent_runner=search_agent_runner_instance, # Passed through
                search_agent_session_id=search_agent_session_id_for_search, # Passed through
                user_id_for_search=USER_ID, # From config
                logger_instance=logger_instance
            )
            _update_state_value(current_state, 'world_feeds.weather', weather_data, logger_instance)

            news_categories = ["world_news", "regional_news", "local_news", "pop_culture"]
            for news_cat in news_categories:
                news_item = await generate_simulated_world_feed_content(
                    current_sim_state=current_state, category=news_cat, simulation_time=current_sim_time_val,
                    location_context=location_context_str, world_mood=world_mood,
                    global_search_agent_runner=search_agent_runner_instance, # Passed through
                    search_agent_session_id=search_agent_session_id_for_search, # Passed through
                    user_id_for_search=USER_ID, # From config
                    logger_instance=logger_instance
                )
                feed_key = "news_updates" if "news" in news_cat else "pop_culture_updates"
                current_feed = get_nested(current_state, 'world_feeds', feed_key, default=[])
                current_feed.insert(0, news_item)
                _update_state_value(current_state, f'world_feeds.{feed_key}', current_feed[-MAX_WORLD_FEED_ITEMS:], logger_instance)

            logger_instance.info(f"[WorldInfoGatherer] World feeds updated. Next check in {WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS} sim_seconds.")
            next_run_sim_time = current_sim_time_val + WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS
            while current_state.get("world_time", 0.0) < next_run_sim_time:
                await asyncio.sleep(UPDATE_INTERVAL * 5) # From config
        except asyncio.CancelledError:
            logger_instance.info("[WorldInfoGatherer] Task cancelled.")
            break
        except Exception as e:
            logger_instance.exception(f"[WorldInfoGatherer] Error: {e}")
            await asyncio.sleep(60)
