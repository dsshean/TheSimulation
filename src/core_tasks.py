# src/core_tasks.py - Core asynchronous tasks for the simulation (ADK-independent ones)

import asyncio
import logging
import time # For time_manager_task
from typing import Any, Dict, Optional, List

# Import from our new modules
from .config import (
    MAX_SIMULATION_TIME, SIMULATION_SPEED_FACTOR, UPDATE_INTERVAL, MAX_MEMORY_LOG_ENTRIES,
    WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS, MAX_WORLD_FEED_ITEMS, USER_ID, SIMULACRA_KEY,
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

            for agent_id, agent_state_data in list(get_nested(current_state, SIMULACRA_KEY, default={}).items()):
                if agent_state_data.get("status") == "busy":
                    action_end_time = agent_state_data.get("current_action_end_time", -1.0)
                    if action_end_time <= new_sim_time:
                        logger_instance.info(f"[TimeManager] Applying completed action effects for {agent_id} at time {new_sim_time:.1f} (due at {action_end_time:.1f}).")
                        pending_results = agent_state_data.get("pending_results", {}) # Get a copy or ensure it's mutable if needed
                        if pending_results:
                            for key_path, value in list(pending_results.items()):
                                success = _update_state_value(current_state, key_path, value, logger_instance)
                                # Check if the memory_log for this specific agent was updated
                                if success and key_path == f"{SIMULACRA_KEY}.{agent_id}.memory_log":
                                    # memory_log_updated = True # No longer needed
                                    current_mem_log = get_nested(current_state, SIMULACRA_KEY, agent_id, "memory_log", default=[])
                                    if isinstance(current_mem_log, list) and len(current_mem_log) > MAX_MEMORY_LOG_ENTRIES:
                                        _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.memory_log", current_mem_log[-MAX_MEMORY_LOG_ENTRIES:], logger_instance)
                                        logger_instance.debug(f"[TimeManager] Pruned memory log for {agent_id} to {MAX_MEMORY_LOG_ENTRIES} entries.")
                            _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.pending_results", {}, logger_instance)
                        else:
                            logger_instance.debug(f"[TimeManager] No pending results found for completed action of {agent_id}.")
                        # Clear interrupt probability and set status to idle for agent-specific action completions
                        _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None, logger_instance) # Clear probability
                        _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger_instance)
                        logger_instance.info(f"[TimeManager] Set {agent_id} status to idle.")

            # --- BEGIN ADDITION: Process General Scheduled Events from pending_simulation_events ---
            processed_event_indices = []
            pending_events_list = current_state.get("pending_simulation_events", [])

            for i, event_data in enumerate(pending_events_list):
                if event_data.get("trigger_sim_time", float('inf')) <= new_sim_time:
                    event_type = event_data.get("event_type")
                    logger_instance.info(f"[TimeManager] Processing scheduled event: '{event_type}' at sim_time {new_sim_time:.2f} (due at {event_data.get('trigger_sim_time'):.2f})")

                    if event_type == "simulacra_speech_received_as_interrupt":
                        target_agent_id = event_data.get("target_agent_id")
                        message_details = event_data.get("details", {})
                        speech_content = message_details.get("message_content", "Someone spoke to you.")
                        speaker_name = message_details.get("speaker_name", "Someone") # Fallback

                        # Check if the target agent exists and is a simulacra
                        if target_agent_id and target_agent_id in get_nested(current_state, SIMULACRA_KEY, default={}):
                            logger_instance.info(
                                f"[TimeManager] Applying 'simulacra_speech_received_as_interrupt' to {target_agent_id} "
                                f"from {speaker_name}."
                            )

                            updates_for_interrupt = {
                                f"{SIMULACRA_KEY}.{target_agent_id}.last_observation": speech_content,
                                f"{SIMULACRA_KEY}.{target_agent_id}.status": "idle",
                                # Make them act very soon by setting their action end time to now + tiny delay
                                f"{SIMULACRA_KEY}.{target_agent_id}.current_action_end_time": new_sim_time + 0.01,
                                f"{SIMULACRA_KEY}.{target_agent_id}.pending_results": {}, # Clear any pending results from a potentially interrupted action
                                f"{SIMULACRA_KEY}.{target_agent_id}.current_action_description": f"Interrupted by {speaker_name} saying something.",
                                f"{SIMULACRA_KEY}.{target_agent_id}.current_interrupt_probability": None, # Reset interrupt probability
                            }
                            for key_path, value in updates_for_interrupt.items():
                                _update_state_value(current_state, key_path, value, logger_instance)

                            logger_instance.info(
                                f"[TimeManager] Agent {target_agent_id} processed speech interrupt. New observation set. Status set to idle."
                            )
                            # Optional: Trigger narration for the *interrupted agent* here if desired (conceptual)
                        else:
                            logger_instance.warning(f"[TimeManager] Target agent '{target_agent_id}' for speech interrupt not found or invalid.")

                    # Add other event_type handlers here if needed in the future
                    # elif event_type == "another_event_type":
                    #    ...

                    processed_event_indices.append(i) # Mark event for removal

            # Remove processed events (iterate in reverse to avoid index issues during pop)
            for i in sorted(processed_event_indices, reverse=True):
                pending_events_list.pop(i)
            # --- END ADDITION ---
            
            live_display.update(generate_table(current_state, event_bus_qsize_func(), narration_qsize_func()))
            await asyncio.sleep(UPDATE_INTERVAL)

    except asyncio.CancelledError:
        logger_instance.info("[TimeManager] Task cancelled.")
    except Exception as e:
        logger_instance.exception(f"[TimeManager] Error in main loop: {e}")
    finally:
        logger_instance.info(f"[TimeManager] Loop finished at sim time {current_state.get('world_time', 0.0):.1f}")


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

            # Parallelize fetching for different news categories
            news_categories = ["world_news", "regional_news", "local_news", "pop_culture"]
            news_tasks = [
                generate_simulated_world_feed_content(
                    current_sim_state=current_state, category=news_cat, simulation_time=current_sim_time_val,
                    location_context=location_context_str, world_mood=world_mood,
                    global_search_agent_runner=search_agent_runner_instance, # Passed through
                    search_agent_session_id=search_agent_session_id_for_search, # Passed through
                    user_id_for_search=USER_ID, # From config
                    logger_instance=logger_instance
                ) for news_cat in news_categories
            ]
            
            news_results = await asyncio.gather(*news_tasks, return_exceptions=True)

            for i, result_item in enumerate(news_results):
                news_cat = news_categories[i]
                news_item: Dict[str, Any] # Ensure news_item is defined for type hinting

                if isinstance(result_item, Exception):
                    logger_instance.error(f"[WorldInfoGatherer] Error fetching {news_cat}: {result_item}")
                    news_item = {"error": f"Failed to fetch {news_cat}", "raw_response": str(result_item), "timestamp": current_sim_time_val, "source_category": news_cat}
                elif isinstance(result_item, dict): # Ensure it's a dict
                    news_item = result_item
                else: # Should not happen if generate_simulated_world_feed_content returns a dict
                    logger_instance.error(f"[WorldInfoGatherer] Unexpected result type for {news_cat}: {type(result_item)}")
                    news_item = {"error": f"Unexpected result type for {news_cat}", "timestamp": current_sim_time_val, "source_category": news_cat}

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
