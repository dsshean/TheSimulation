# src/core_tasks.py - Core asynchronous tasks for the simulation (ADK-independent ones)

import asyncio
import json
import logging
import random # Added for dynamic_interruption_task
import re # Added for narrative_image_generation_task
import time # For time_manager_task
import os # Added for narrative_image_generation_task
from io import BytesIO # Added for narrative_image_generation_task
from typing import Any, Dict, Optional, List
import websockets

import google.generativeai as genai # Added for dynamic_interruption_task
from google import genai as genai_image # Added for narrative_image_generation_task
from PIL import Image # Added for narrative_image_generation_task
from atproto import Client as BlueskyClient, models as atproto_models # Added for narrative_image_generation_task

# Import from our new modules
from .config import (
    MAX_SIMULATION_TIME, SIMULATION_SPEED_FACTOR, UPDATE_INTERVAL, MAX_MEMORY_LOG_ENTRIES,
    WORLD_INFO_GATHERER_INTERVAL_SIM_SECONDS, MAX_WORLD_FEED_ITEMS, USER_ID, SIMULACRA_KEY,
    WORLD_TEMPLATE_DETAILS_KEY, LOCATION_KEY, ACTIVE_SIMULACRA_IDS_KEY, CURRENT_LOCATION_KEY, # Added CURRENT_LOCATION_KEY
    INTERJECTION_COOLDOWN_SIM_SECONDS, MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK,
    DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS, DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION,
    DYNAMIC_INTERRUPTION_MAX_PROB_CAP, DYNAMIC_INTERRUPTION_MIN_PROB, DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS, MODEL_NAME,
    ENABLE_NARRATIVE_IMAGE_GENERATION, IMAGE_GENERATION_INTERVAL_REAL_SECONDS, IMAGE_GENERATION_MODEL_NAME, # For image task
    IMAGE_GENERATION_OUTPUT_DIR, ENABLE_BLUESKY_POSTING, BLUESKY_HANDLE, BLUESKY_APP_PASSWORD, # For image task
    SOCIAL_POST_TEXT_LIMIT, SOCIAL_POST_HASHTAGS, WORLD_STATE_KEY, LOCATION_DETAILS_KEY,  # For image task
    ENABLE_WEB_VISUALIZATION, VISUALIZATION_WEBSOCKET_PORT  # For web visualization
)
# simulation_utils will be called by tasks in simulation_async.py or here, passing state
from .simulation_utils import (_update_state_value, generate_table, generate_simulated_world_feed_content,
                               _log_event, get_random_style_combination, get_time_string_for_prompt) # Added image utils
from .loop_utils import get_nested


def _create_event_notification_message(event_type: str, event_details: Dict[str, Any], 
                                     event_location: Optional[str], current_location: str) -> str:
    """
    Create a natural language message for any scheduled future event.
    This makes the generic event system work for any event type.
    """
    # Extract common details
    details = event_details or {}
    
    # Create base message based on event type
    if event_type == "elevator_arrival":
        destination = details.get("destination_floor", event_location)
        return f"The elevator arrived and transported you to {destination or 'your destination'}."
    
    elif event_type == "vehicle_arrival":
        vehicle_type = details.get("vehicle_type", "vehicle")
        return f"The {vehicle_type} arrived and took you to {event_location or 'your destination'}."
    
    elif event_type == "teleport_complete":
        return f"You were teleported to {event_location or 'a new location'}."
    
    elif event_type == "delivery_arrival":
        item = details.get("item", "package")
        return f"A {item} was delivered to you."
    
    elif event_type == "appointment_reminder":
        appointment = details.get("appointment", "appointment")
        return f"Reminder: You have a {appointment} scheduled."
    
    elif event_type == "timer_complete":
        task = details.get("task", "task")
        return f"Timer finished: Your {task} is complete."
    
    elif event_type == "weather_change":
        weather = details.get("weather", "weather")
        return f"The weather changed to {weather}."
    
    elif event_type == "device_notification":
        device = details.get("device", "device")
        message = details.get("message", "notification")
        return f"Your {device} sent a notification: {message}"
    
    elif event_type == "npc_arrival":
        npc_name = details.get("npc_name", "someone")
        return f"{npc_name} arrived at your location."
    
    elif event_type == "door_unlock":
        door = details.get("door", "door")
        return f"The {door} unlocked automatically."
    
    elif event_type == "alarm_trigger":
        alarm_type = details.get("alarm_type", "alarm")
        return f"An {alarm_type} is going off."
    
    else:
        # Generic fallback for unknown event types
        if event_location and event_location != current_location:
            return f"A {event_type} event occurred and brought you to {event_location}."
        else:
            action_description = details.get("action_description", f"a {event_type} event occurred")
            return f"Something happened: {action_description}."

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
                        # Get location BEFORE any results are applied for this tick
                        old_location_for_previous_id_tracking = agent_state_data.get(CURRENT_LOCATION_KEY)
                        logger_instance.info(f"[TimeManager] Applying completed action effects for {agent_id} at time {new_sim_time:.1f} (due at {action_end_time:.1f}).")
                        
                        # --- MODIFICATION: Apply action_completion_results first ---
                        completion_results = agent_state_data.get("action_completion_results", {})
                        if completion_results:
                            logger_instance.debug(f"[TimeManager] Applying action_completion_results for {agent_id}: {completion_results}")
                            for key_path_comp, value_comp in list(completion_results.items()):
                                logger_instance.info(f"[TimeManager] DEBUG: Applying completion result: {key_path_comp} = {value_comp}")
                                # If this result is updating the agent's current_location,
                                # set their previous_location_id first.
                                if key_path_comp == f"{SIMULACRA_KEY}.{agent_id}.{CURRENT_LOCATION_KEY}" and \
                                   old_location_for_previous_id_tracking and \
                                   old_location_for_previous_id_tracking != value_comp:
                                    logger_instance.info(f"[TimeManager] DEBUG: Location change detected! Previous: {old_location_for_previous_id_tracking}, New: {value_comp}")
                                    _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.previous_location_id", old_location_for_previous_id_tracking, logger_instance)
                                _update_state_value(current_state, key_path_comp, value_comp, logger_instance)
                            _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.action_completion_results", {}, logger_instance) # Clear after applying
                        
                        pending_results = agent_state_data.get("pending_results", {})
                        if pending_results: # Apply any remaining immediate pending_results
                            logger_instance.info(f"[TimeManager] DEBUG: Applying pending_results for {agent_id}: {pending_results}")
                            for key_path, value in list(pending_results.items()):
                                logger_instance.info(f"[TimeManager] DEBUG: Applying pending result: {key_path} = {value}")
                                # If this result is updating the agent's current_location,
                                # set their previous_location_id first.
                                # This check is now more critical if location wasn't in completion_results
                                if key_path == f"{SIMULACRA_KEY}.{agent_id}.{CURRENT_LOCATION_KEY}" and \
                                   old_location_for_previous_id_tracking and \
                                   old_location_for_previous_id_tracking != value:
                                    logger_instance.info(f"[TimeManager] DEBUG: Location change detected in pending! Previous: {old_location_for_previous_id_tracking}, New: {value}")
                                    _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.previous_location_id", old_location_for_previous_id_tracking, logger_instance)
                                success = _update_state_value(current_state, key_path, value, logger_instance)
                                # Check if the memory_log for this specific agent was updated
                                if success and key_path == f"{SIMULACRA_KEY}.{agent_id}.memory_log":
                                    # memory_log_updated = True # No longer needed
                                    current_mem_log = get_nested(current_state, SIMULACRA_KEY, agent_id, "memory_log", default=[])
                                    if isinstance(current_mem_log, list) and len(current_mem_log) > MAX_MEMORY_LOG_ENTRIES:
                                        _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.memory_log", current_mem_log[-MAX_MEMORY_LOG_ENTRIES:], logger_instance)
                                        logger_instance.debug(f"[TimeManager] Pruned memory log for {agent_id} to {MAX_MEMORY_LOG_ENTRIES} entries.")
                            _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.pending_results", {}, logger_instance)
                        # else: # This log can be noisy if pending_results is often empty after completion_results
                            # logger_instance.debug(f"[TimeManager] No pending results found for completed action of {agent_id}.")
                        # Clear interrupt probability and set status to idle for agent-specific action completions
                        _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.current_interrupt_probability", None, logger_instance) # Clear probability
                        _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.status", "idle", logger_instance) # Agent is now idle and ready for next turn
                        # Reset action interrupt tracking for next action
                        _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.action_interrupted_flag", False, logger_instance)
                        _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id}.current_action_id", None, logger_instance)
                        logger_instance.info(f"[TimeManager] Action for {agent_id} completed. Set status to idle.")

            await process_pending_simulation_events(current_state, logger_instance)
            
            # Process ready narration events and flush overdue ones
            try:
                from .simulation_async import add_narration_event_with_ordering, flush_overdue_narration_events
                if add_narration_event_with_ordering is not None and flush_overdue_narration_events is not None:
                    # Trigger processing of any ready events (with empty event)
                    await add_narration_event_with_ordering({}, task_name="TimeManager_Trigger")
                    
                    # Flush any overdue events periodically
                    await flush_overdue_narration_events()
            except Exception as e:
                logger_instance.debug(f"[TimeManager] Error processing narration events: {e}")
            
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

async def dynamic_interruption_task(
    current_state: Dict[str, Any],
    world_mood: str, # world_mood_global
    logger_instance: logging.Logger,
    event_logger_instance: Optional[logging.Logger] # For _log_event
):
    """
    Periodically checks busy simulacra and probabilistically interrupts them
    with a direct narrative observation. (Moved from simulation_async.py)
    """
    logger_instance.info("[DynamicInterruptionTask] Task started.")
    await asyncio.sleep(random.uniform(5.0, 15.0))

    while True:
        await asyncio.sleep(DYNAMIC_INTERRUPTION_CHECK_REAL_SECONDS)

        current_sim_time_dit = current_state.get("world_time", 0.0)
        active_sim_ids_dit = list(current_state.get(ACTIVE_SIMULACRA_IDS_KEY, []))

        for agent_id_to_check in active_sim_ids_dit:
            agent_state_to_check = get_nested(current_state, SIMULACRA_KEY, agent_id_to_check, default={})
            if not agent_state_to_check or agent_state_to_check.get("status") != "busy":
                # If agent is not busy, their interrupt probability should be None.
                # Only update if it's not already None to avoid redundant state updates.
                if get_nested(agent_state_to_check, "current_interrupt_probability") is not None:
                    _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_interrupt_probability", None, logger_instance)
                continue

            agent_name_to_check = get_nested(agent_state_to_check, "persona_details", "Name", default=agent_id_to_check)
            last_interruption_time = agent_state_to_check.get("last_interjection_sim_time", 0.0)
            cooldown_passed = (current_sim_time_dit - last_interruption_time) >= INTERJECTION_COOLDOWN_SIM_SECONDS

            if not cooldown_passed:
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_interrupt_probability", None, logger_instance)
                continue

            remaining_duration = agent_state_to_check.get("current_action_end_time", 0.0) - current_sim_time_dit
            if remaining_duration < MIN_DURATION_FOR_DYNAMIC_INTERRUPTION_CHECK:
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_interrupt_probability", None, logger_instance)
                continue

            # Check if this action has already been interrupted
            action_interrupted_flag = agent_state_to_check.get("action_interrupted_flag", False)
            if action_interrupted_flag:
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_interrupt_probability", None, logger_instance)
                continue

            interrupt_probability = 0.0
            if DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS > 0:
                duration_factor = remaining_duration / DYNAMIC_INTERRUPTION_TARGET_DURATION_SECONDS
                scaled_prob = duration_factor * DYNAMIC_INTERRUPTION_PROB_AT_TARGET_DURATION
                interrupt_probability = min(DYNAMIC_INTERRUPTION_MAX_PROB_CAP, max(DYNAMIC_INTERRUPTION_MIN_PROB, scaled_prob))
            else:
                interrupt_probability = min(DYNAMIC_INTERRUPTION_MAX_PROB_CAP, DYNAMIC_INTERRUPTION_MIN_PROB)
            _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_interrupt_probability", interrupt_probability, logger_instance)

            if random.random() < interrupt_probability:
                logger_instance.info(f"[DynamicInterruptionTask] Triggering dynamic interruption for {agent_name_to_check} (Prob: {interrupt_probability:.3f}, RemDur: {remaining_duration:.1f}s).")
                
                agent_action_desc_dit = agent_state_to_check.get("current_action_description", "their current activity")
                interruption_context_for_llm_prompt: str

                is_listening_action = (
                    "listening to" in agent_action_desc_dit.lower() or
                    ("waiting for" in agent_action_desc_dit.lower() and "to speak" in agent_action_desc_dit.lower()) or
                    ("paying attention to" in agent_action_desc_dit.lower() and "as they speak" in agent_action_desc_dit.lower()) or
                    "listened attentively to" in agent_action_desc_dit.lower() # From WE outcome
                )

                if is_listening_action:
                    interruption_context_for_llm_prompt = f"""Agent {agent_name_to_check} is currently listening to someone speak.
Their stated activity is: "{agent_action_desc_dit}".
The general world mood is: "{world_mood}".
They suddenly have a thought or an urge to interject or respond before the speaker finishes.
Describe this internal realization or the minor event that triggers their desire to speak, from {agent_name_to_check}'s perspective.
Example: "While listening, an important counter-point flashes into your mind, and you feel a sudden need to voice it."
Example: "As the speaker pauses for a micro-second, you see an opening and decide to jump in."
Output ONLY the narrative sentence(s) describing this urge/trigger for you ({agent_name_to_check}) to speak."""
                else:
                    interruption_context_for_llm_prompt = f"""Agent {agent_name_to_check} is currently busy with: "{agent_action_desc_dit}".
The general world mood is: "{world_mood}".
An unexpected minor interruption occurs, breaking {agent_name_to_check}'s concentration or prompting a change of thought.
Describe this interruption in one or two engaging narrative sentences from an observational perspective, suitable for {agent_name_to_check} to perceive.
Example: "Suddenly, a loud crash from the kitchen shatters the quiet, making you jump."
Output ONLY the narrative sentence(s)."""

                interruption_text = f"A minor unexpected event occurs, breaking your concentration." # Default
                try:
                    interrupt_llm = genai.GenerativeModel(MODEL_NAME) # Assumes genai is configured
                    response = await interrupt_llm.generate_content_async(interruption_context_for_llm_prompt)
                    if response.text: interruption_text = response.text.strip()
                except Exception as e_interrupt_text:
                    logger_instance.error(f"[DynamicInterruptionTask] Failed to generate LLM text for interruption, using default. Error: {e_interrupt_text}")

                logger_instance.info(f"[DynamicInterruptionTask] Interrupting {agent_name_to_check} with: {interruption_text}")
                
                action_desc_after_interrupt = "Interrupted by a dynamic event."
                if is_listening_action and \
                   ("interject" in interruption_text.lower() or "speak" in interruption_text.lower() or "respond" in interruption_text.lower() or "voice it" in interruption_text.lower() or "jump in" in interruption_text.lower()):
                    action_desc_after_interrupt = "Had a thought while listening; decided to interject."

                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.status", "idle", logger_instance)
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.last_observation", interruption_text, logger_instance)
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.pending_results", {}, logger_instance)
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_action_end_time", current_sim_time_dit, logger_instance)
                # Reset action interrupt tracking after dynamic interruption
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.action_interrupted_flag", False, logger_instance)
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_action_id", None, logger_instance)
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_action_description", action_desc_after_interrupt, logger_instance)
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.last_interjection_sim_time", current_sim_time_dit, logger_instance)
                _update_state_value(current_state, f"{SIMULACRA_KEY}.{agent_id_to_check}.current_interrupt_probability", None, logger_instance)
                _log_event(current_sim_time_dit, "DynamicInterruptionTask", "agent_interrupted", {"agent_id": agent_id_to_check, "interruption_text": interruption_text}, logger_instance, event_logger_instance)
                break # Only interrupt one agent per check cycle for now
        await asyncio.sleep(0.1) # Brief yield

async def narrative_image_generation_task(
    current_state: Dict[str, Any],
    world_mood: str, # world_mood_global from simulation_async
    logger_instance: logging.Logger,
    event_logger_instance: Optional[logging.Logger] # For _log_event
):
    """
    Periodically generates an image based on the latest narrative log entry.
    Logs the filename to the event logger. (Moved from simulation_async.py)
    """
    if not ENABLE_NARRATIVE_IMAGE_GENERATION:
        logger_instance.info("[NarrativeImageGenerator] Task is disabled by configuration.")
        try:
            while True: await asyncio.sleep(3600)
        except asyncio.CancelledError:
            logger_instance.info("[NarrativeImageGenerator] Idling task cancelled.")
            raise
        return

    logger_instance.info(f"[NarrativeImageGenerator] Task started. Interval: {IMAGE_GENERATION_INTERVAL_REAL_SECONDS}s. Model: {IMAGE_GENERATION_MODEL_NAME}. Output: {IMAGE_GENERATION_OUTPUT_DIR}")

    img_gen_client = genai_image.Client() # Initialize image generation client
    bluesky_api_client: Optional[BlueskyClient] = None
    if ENABLE_BLUESKY_POSTING:
        if BLUESKY_HANDLE and BLUESKY_APP_PASSWORD:
            bluesky_api_client = BlueskyClient()
        else:
            logger_instance.warning("[NarrativeImageGenerator] Bluesky posting enabled, but handle or app password missing. Posting will be skipped.")

    await asyncio.sleep(random.uniform(5.0, 10.0)) # Initial delay

    while True:
        await asyncio.sleep(IMAGE_GENERATION_INTERVAL_REAL_SECONDS)

        if not current_state or not current_state.get("narrative_log"):
            logger_instance.debug("[NarrativeImageGenerator] No narrative log. Skipping.")
            continue

        narrative_log_entries = current_state.get("narrative_log", [])
        if not narrative_log_entries:
            logger_instance.debug("[NarrativeImageGenerator] Narrative log empty. Skipping.")
            continue

        latest_narrative_full = narrative_log_entries[-1]
        original_narrative_prompt_text = re.sub(r'^\[T\d+\.\d+\]\s*', '', latest_narrative_full).strip()
        if not original_narrative_prompt_text:
            logger_instance.debug("[NarrativeImageGenerator] Latest narrative entry empty after strip. Skipping.")
            continue

        current_sim_time_for_filename = current_state.get("world_time", 0.0)
        actor_name_in_narrative = "the observer"
        match = re.match(r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)", original_narrative_prompt_text)
        if match and match.group(1) not in ["As", "The", "A", "An", "It", "He", "She", "They", "Then", "Suddenly", "During", "While"]:
            actor_name_in_narrative = match.group(1)

        time_string_for_image_prompt = get_time_string_for_prompt(current_state, sim_elapsed_time_seconds=current_sim_time_for_filename)
        weather_condition_for_image_prompt = get_nested(current_state, 'world_feeds', 'weather', 'condition', default='Weather unknown.')
        current_world_mood_ig = get_nested(current_state, WORLD_TEMPLATE_DETAILS_KEY, 'mood', default=world_mood)

        refined_narrative_for_image = original_narrative_prompt_text
        try:
            refinement_llm = genai.GenerativeModel(MODEL_NAME) # Standard text model for refinement
            prompt_for_refinement = f"""You are an expert at transforming narrative text into concise, visually descriptive prompts ideal for an image generation model. Your goal is to focus on a single, clear subject, potentially with a naturally blurred background.
Original Narrative Context: "{original_narrative_prompt_text}"
Current Time: "{time_string_for_image_prompt}"
Current Weather: "{weather_condition_for_image_prompt}"
World Mood: "{current_world_mood_ig}"
Instructions for Refinement:
1.  Identify a single, compelling visual element or a very brief, static moment from the 'Original Narrative Context'.
2.  Describe this single subject clearly and vividly. Use descriptive language for the subject and its relationship to any implied background.
3.  If appropriate, suggest a composition that would naturally lead to a blurred background (e.g., "A close-up of...", "A detailed shot of...", "A lone figure with the background softly blurred...").
4.  Keep the refined description concise (preferably 1-2 sentences).
5.  The refined description should be purely visual and directly usable as an image prompt.
6.  Do NOT include any instructions for the image generation model itself (like "Generate an image of..."). Just provide the refined descriptive text.
Refined Visual Description:"""
            logger_instance.info(f"[NarrativeImageGenerator] Refining narrative for image: '{original_narrative_prompt_text[:100]}...'")
            response_refinement = await refinement_llm.generate_content_async(prompt_for_refinement)
            if response_refinement.text:
                refined_narrative_for_image = response_refinement.text.strip()
                logger_instance.info(f"[NarrativeImageGenerator] Refined narrative: '{refined_narrative_for_image}'")
        except Exception as e_refine:
            logger_instance.error(f"[NarrativeImageGenerator] Error refining narrative: {e_refine}. Using original.", exc_info=True)

        random_style_for_image = get_random_style_combination(logger_instance=logger_instance, num_general=0, num_lighting=1, num_color=1, num_technique=1, num_composition=1, num_atmosphere=1)
        prompt_for_image_gen = f"""Generate a high-quality, visually appealing, **photo-realistic** photograph of a scene or subject directly related to the following narrative context, as if captured by {actor_name_in_narrative}.
Narrative Context: "{refined_narrative_for_image}"
Style: "{random_style_for_image}"
Instructions for the Image:
The image should feature:
-   Time of Day: Reflect the lighting and atmosphere typical of "{time_string_for_image_prompt}".
-   Weather: Depict the conditions described by "{weather_condition_for_image_prompt}".
-   Season: Infer the season from the date in the time string and depict it.
-   A clear subject directly related to the Narrative Context.
-   Lighting, composition, and focus that give it the aesthetic of a professional, high-engagement social media photograph.
-   Details that align with the World Mood: "{current_world_mood_ig}".
-   A composition that is balanced and aesthetically pleasing, with a strong emphasis on a clear, well-defined subject within a natural or slightly blurred background.
Style: Modern, editorial-quality, photo-realistic photograph, authentic textures, natural colors. Aspect ratio: 4:5 (portrait) or 1:1 (square).
Crucial Exclusions: No digital overlays, UI elements, watermarks, logos. The actor ({actor_name_in_narrative}) MUST NOT be visible.
Generate this image."""
        logger_instance.info(f"[NarrativeImageGenerator] Requesting image (T{current_sim_time_for_filename:.1f}): \"{refined_narrative_for_image}\"")

        try:
            response = await asyncio.to_thread(
                img_gen_client.models.generate_images, model=IMAGE_GENERATION_MODEL_NAME,
                prompt=prompt_for_image_gen,
                config=genai_image.types.GenerateImagesConfig(number_of_images=1)
            )
            image_generated_successfully = False
            saved_image_path_for_social_post: Optional[str] = None

            for gen_img in response.generated_images:
                try:
                    pil_image = Image.open(BytesIO(gen_img.image.image_bytes))
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    sim_time_str_file = f"T{current_sim_time_for_filename:.0f}"
                    image_filename = f"narrative_{sim_time_str_file}_{timestamp_str}.png"
                    image_path = os.path.join(IMAGE_GENERATION_OUTPUT_DIR, image_filename)
                    pil_image.save(image_path)
                    logger_instance.info(f"[NarrativeImageGenerator] Saved image: {image_path}")
                    image_generated_successfully = True
                    saved_image_path_for_social_post = image_path
                    _log_event(current_sim_time_for_filename, "ImageGenerator", "image_generation",
                               {"image_filename": image_filename, "prompt_snippet": refined_narrative_for_image},
                               logger_instance, event_logger_instance)
                    break
                except Exception as e_img_proc:
                    logger_instance.error(f"[NarrativeImageGenerator] Error processing/saving image: {e_img_proc}", exc_info=True)

            if ENABLE_BLUESKY_POSTING and bluesky_api_client and image_generated_successfully and saved_image_path_for_social_post:
                logger_instance.info(f"[NarrativeImageGenerator] Attempting Bluesky post for: {saved_image_path_for_social_post}")
                try:
                    if not bluesky_api_client.me: # Check if login is needed
                        bluesky_api_client.login(BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)

                    image_bytes_for_upload = None
                    with open(saved_image_path_for_social_post, 'rb') as f_img_bs:
                        image_bytes_for_upload = f_img_bs.read()
                    # Note: Bluesky image size limit (around 976KB) might require compression logic here if images are large.
                    # This example assumes images are generally within limits or compression is handled by the image model.

                    alt_text_bs = f"Image from simulation at T{current_sim_time_for_filename:.0f}s: {refined_narrative_for_image}"[:1000] # Bluesky alt text limit
                    post_text_raw_bs = alt_text_bs
                    effective_text_limit_bs = max(0, SOCIAL_POST_TEXT_LIMIT - (len(SOCIAL_POST_HASHTAGS) + 5)) # Approx space for hashtags
                    post_text_bs = post_text_raw_bs[:effective_text_limit_bs]
                    if len(post_text_raw_bs) > effective_text_limit_bs:
                        post_text_bs = post_text_bs.rsplit(' ', 1)[0] + '...' if ' ' in post_text_bs else post_text_bs + '...'
                    final_post_content_bs = f"{post_text_bs}\n\n{SOCIAL_POST_HASHTAGS}".strip()

                    upload_blob_response = bluesky_api_client.com.atproto.repo.upload_blob(image_bytes_for_upload)
                    embed_image = atproto_models.AppBskyEmbedImages.Image(alt=alt_text_bs, image=upload_blob_response.blob)
                    embed_main = atproto_models.AppBskyEmbedImages.Main(images=[embed_image])

                    bluesky_api_client.com.atproto.repo.create_record(
                        atproto_models.ComAtprotoRepoCreateRecord.Data(
                            repo=bluesky_api_client.me.did, collection=atproto_models.ids.AppBskyFeedPost,
                            record=atproto_models.AppBskyFeedPost.Main(
                                created_at=bluesky_api_client.get_current_time_iso(), text=final_post_content_bs, embed=embed_main
                            )
                        )
                    )
                    logger_instance.info(f"[NarrativeImageGenerator] Successfully posted to Bluesky: '{final_post_content_bs[:50]}...'")
                except Exception as e_bsky:
                    logger_instance.error(f"[NarrativeImageGenerator] Error posting to Bluesky: {e_bsky}", exc_info=True)

        except Exception as e_gen:
            logger_instance.error(f"[NarrativeImageGenerator] Error during image generation API call: {e_gen}", exc_info=True)


async def process_pending_simulation_events(current_state: Dict[str, Any], logger_instance: logging.Logger):
    """
    Process and deliver pending simulation events, including speech interrupts.
    Removes processed events from the pending list.
    """
    current_sim_time = current_state.get("world_time", 0.0)
    pending_events_list = current_state.get("pending_simulation_events", [])
    
    # New list to hold events that are NOT processed in this cycle (i.e., future events)
    remaining_pending_events = []
    events_were_processed_in_this_call = False # Flag to see if any event was processed

    if not pending_events_list:
        return

    # Sort by trigger time to ensure chronological processing if multiple are due in the same tick
    pending_events_list.sort(key=lambda x: x.get("trigger_sim_time", float('inf')))

    for event_data in list(pending_events_list): # Iterate over a copy
        if event_data.get("trigger_sim_time", float('inf')) <= current_sim_time:
            events_were_processed_in_this_call = True # Mark that we are processing an event
            event_type = event_data.get("event_type")
            target_agent_id = event_data.get("target_agent_id", "N/A")
            source_actor_id = event_data.get("source_actor_id", "N/A") 
            event_trigger_time = event_data.get("trigger_sim_time", "N/A")

            # FIX: Pre-format the event_trigger_time
            formatted_event_trigger_time = f"{event_trigger_time:.2f}" if isinstance(event_trigger_time, float) else str(event_trigger_time)

            logger_instance.info(
                f"[ProcessEvents] Processing scheduled event: '{event_type}' for target '{target_agent_id}' "
                f"from source '{source_actor_id}' at sim_time {current_sim_time:.2f} "
                f"(event trigger: {formatted_event_trigger_time})" # Use the pre-formatted string
            )

            if event_type == "simulacra_speech_received_as_interrupt":
                # Target can be Simulacra or NPC
                message_details = event_data.get("details", {})
                speech_content = message_details.get("message_content", "Someone spoke to you.")
                speaker_name = message_details.get("speaker_name", "Someone")
                # source_actor_id is already defined above

                # Check if target is a Simulacra
                if target_agent_id and target_agent_id in get_nested(current_state, SIMULACRA_KEY, default={}):
                    logger_instance.info(f"[ProcessEvents] Applying speech interrupt to Simulacra {target_agent_id} from {speaker_name} ({source_actor_id})")

                    updates_for_interrupt = {
                        f"{SIMULACRA_KEY}.{target_agent_id}.last_observation": speech_content,
                        f"{SIMULACRA_KEY}.{target_agent_id}.status": "idle", # Allows agent to react
                        f"{SIMULACRA_KEY}.{target_agent_id}.current_action_end_time": current_sim_time + 0.01, # Minimal time
                        f"{SIMULACRA_KEY}.{target_agent_id}.pending_results": {},
                        f"{SIMULACRA_KEY}.{target_agent_id}.current_action_description": f"Interrupted by {speaker_name} ({source_actor_id}) saying something.",
                        f"{SIMULACRA_KEY}.{target_agent_id}.current_interrupt_probability": None,
                    }
                    for key_path, value in updates_for_interrupt.items():
                        _update_state_value(current_state, key_path, value, logger_instance)
                    logger_instance.info(f"[ProcessEvents] Simulacra {target_agent_id} processed speech interrupt from {source_actor_id}.")

                else: # Check if target is an NPC
                    npc_found_in_any_location = False
                    location_details_map = get_nested(current_state, WORLD_STATE_KEY, LOCATION_DETAILS_KEY, default={})
                    if isinstance(location_details_map, dict):
                        for loc_id, loc_data in location_details_map.items():
                            if isinstance(loc_data, dict):
                                ephemeral_npcs_list = loc_data.get("ephemeral_npcs", [])
                                if isinstance(ephemeral_npcs_list, list):
                                    for npc_state in ephemeral_npcs_list:
                                        if isinstance(npc_state, dict) and npc_state.get("id") == target_agent_id:
                                            npc_state["last_heard"] = speech_content 
                                            npc_state["status"] = "idle" 
                                            logger_instance.info(f"[ProcessEvents] Delivered speech from {source_actor_id} to NPC {target_agent_id} in location {loc_id}")
                                            npc_found_in_any_location = True
                                            break
                            if npc_found_in_any_location: break
                    if not npc_found_in_any_location:
                        logger_instance.warning(f"[ProcessEvents] Speech target '{target_agent_id}' for speech from '{source_actor_id}' not found as Simulacra or active NPC.")
            
            elif event_type == "delayed_narration":
                narration_data = event_data.get("narration_data")
                source_actor_for_narration = event_data.get("source_actor_id", "Unknown")
                if narration_data:
                    from .simulation_async import add_narration_event_with_ordering # Use temporal ordering system
                    
                    if add_narration_event_with_ordering is not None:
                        logger_instance.info(f"[ProcessEvents] Queuing delayed narration for original actor {narration_data.get('actor_id')}, event sourced by {source_actor_for_narration}")
                        await add_narration_event_with_ordering(narration_data, task_name=f"ProcessEvents_DelayedNarration_{source_actor_for_narration}")
                    else:
                        logger_instance.error("[ProcessEvents] add_narration_event_with_ordering is None/not available. Cannot queue delayed narration.")
                else:
                    logger_instance.warning(f"[ProcessEvents] Delayed narration event from source {source_actor_for_narration} missing narration_data.")
            
            else:
                # GENERIC HANDLER: Process any scheduled future event type
                logger_instance.info(f"[ProcessEvents] Processing generic scheduled event: '{event_type}' for target '{target_agent_id}'")
                
                # Extract event details
                event_details = event_data.get("details", {})
                event_location = event_data.get("location_id")
                
                # Check if target is a valid simulacra
                if target_agent_id and target_agent_id in get_nested(current_state, SIMULACRA_KEY, default={}):
                    agent_state = get_nested(current_state, SIMULACRA_KEY, target_agent_id, default={})
                    current_location = agent_state.get(CURRENT_LOCATION_KEY, "unknown")
                    
                    # Create generic event notification message
                    event_message = _create_event_notification_message(event_type, event_details, event_location, current_location)
                    
                    # Generic updates that work for most event types
                    updates_for_event = {
                        f"{SIMULACRA_KEY}.{target_agent_id}.status": "idle",
                        f"{SIMULACRA_KEY}.{target_agent_id}.last_observation": event_message,
                        f"{SIMULACRA_KEY}.{target_agent_id}.current_action_end_time": current_sim_time + 0.01,
                        f"{SIMULACRA_KEY}.{target_agent_id}.pending_results": {},
                        f"{SIMULACRA_KEY}.{target_agent_id}.current_action_description": f"Responding to {event_type} event",
                        f"{SIMULACRA_KEY}.{target_agent_id}.current_interrupt_probability": None,
                        # Reset action interrupt tracking for scheduled events
                        f"{SIMULACRA_KEY}.{target_agent_id}.action_interrupted_flag": False,
                        f"{SIMULACRA_KEY}.{target_agent_id}.current_action_id": None,
                    }
                    
                    # Handle location-changing events (like elevators, teleporters, vehicles, etc.)
                    if event_location and event_location != current_location:
                        updates_for_event[f"{SIMULACRA_KEY}.{target_agent_id}.{CURRENT_LOCATION_KEY}"] = event_location
                        updates_for_event[f"{SIMULACRA_KEY}.{target_agent_id}.previous_location_id"] = current_location
                        logger_instance.info(f"[ProcessEvents] Event '{event_type}' moved {target_agent_id} from {current_location} to {event_location}")
                    
                    # Apply all updates
                    for key_path, value in updates_for_event.items():
                        _update_state_value(current_state, key_path, value, logger_instance)
                    
                    logger_instance.info(f"[ProcessEvents] Successfully processed '{event_type}' event for {target_agent_id}")
                    
                else:
                    logger_instance.warning(f"[ProcessEvents] Event target '{target_agent_id}' for '{event_type}' not found as active Simulacra")
            
            # Add other event_type handling here if needed for other custom events
            
        else: # Event's trigger time is in the future
            remaining_pending_events.append(event_data)

    # Update the state's list with only the events that are still truly pending
    if events_were_processed_in_this_call or len(pending_events_list) != len(remaining_pending_events):
        current_state["pending_simulation_events"] = remaining_pending_events
        logger_instance.debug(f"[ProcessEvents] Updated pending_simulation_events. Original size: {len(pending_events_list)}, New size: {len(remaining_pending_events)}")


# --- WebSocket Visualization Server ---

async def visualization_websocket_task(state: Dict[str, Any], logger_instance: logging.Logger):
    """
    WebSocket server that streams real-time simulation state to web visualizer.
    Replaces the pygame visualization with a modern web-based interface.
    """
    # Use a list to make it mutable from nested functions
    connected_clients = []
    
    async def handle_client(websocket, path):
        """Handle individual WebSocket client connections"""
        connected_clients.append(websocket)
        logger_instance.info(f"[WebSocketViz] Client connected from {websocket.remote_address}")
        
        try:
            # Send initial state
            viz_data = _prepare_visualization_data(state)
            await websocket.send(json.dumps(viz_data))
            logger_instance.info(f"[WebSocketViz] Sent initial data to client")
            
            # Keep connection alive and handle any client messages
            async for message in websocket:
                try:
                    # Handle client requests (e.g., zoom to agent, get details)
                    client_request = json.loads(message)
                    response = _handle_client_request(client_request, state)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    logger_instance.warning(f"[WebSocketViz] Invalid JSON from client: {message}")
                except Exception as e:
                    logger_instance.error(f"[WebSocketViz] Error handling client message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger_instance.info(f"[WebSocketViz] Client disconnected")
        except Exception as e:
            logger_instance.error(f"[WebSocketViz] Client handler error: {e}")
        finally:
            if websocket in connected_clients:
                connected_clients.remove(websocket)
    
    async def broadcast_updates():
        """Continuously broadcast state updates to all connected clients"""
        last_world_time = 0
        
        while True:
            try:
                current_world_time = state.get('world_time', 0)
                
                # Only broadcast if simulation time has advanced and we have clients
                if current_world_time != last_world_time and len(connected_clients) > 0:
                    viz_data = _prepare_visualization_data(state)
                    logger_instance.debug(f"[WebSocketViz] Broadcasting update: time={current_world_time:.1f}, agents={len(viz_data.get('simulacra', {}))}, locations={len(viz_data.get('locations', {}))}")
                    
                    # Broadcast to all connected clients
                    if connected_clients:
                        disconnected_clients = []
                        for client in connected_clients[:]:  # Create a copy to iterate safely
                            try:
                                await client.send(json.dumps(viz_data))
                            except websockets.exceptions.ConnectionClosed:
                                disconnected_clients.append(client)
                            except Exception as e:
                                logger_instance.warning(f"[WebSocketViz] Error sending to client: {e}")
                                disconnected_clients.append(client)
                        
                        # Clean up disconnected clients
                        for client in disconnected_clients:
                            if client in connected_clients:
                                connected_clients.remove(client)
                        
                        if disconnected_clients:
                            logger_instance.debug(f"[WebSocketViz] Cleaned up {len(disconnected_clients)} disconnected clients")
                            logger_instance.info(f"[WebSocketViz] Active clients: {len(connected_clients)}")
                    
                    last_world_time = current_world_time
                
                await asyncio.sleep(0.5)  # Update rate: 2Hz for smooth visualization
                
            except Exception as e:
                logger_instance.error(f"[WebSocketViz] Broadcast error: {e}")
                await asyncio.sleep(1)
    
    try:
        logger_instance.info(f"[WebSocketViz] Starting WebSocket server on localhost:{VISUALIZATION_WEBSOCKET_PORT}")
        
        # Start WebSocket server
        server = await websockets.serve(handle_client, "localhost", VISUALIZATION_WEBSOCKET_PORT)
        logger_instance.info(f"[WebSocketViz] WebSocket server started on port {VISUALIZATION_WEBSOCKET_PORT} - visualization available at http://localhost:8080")
        
        # Start broadcast task
        broadcast_task = asyncio.create_task(broadcast_updates())
        
        # Keep server running
        await asyncio.gather(
            server.wait_closed(),
            broadcast_task
        )
        
    except Exception as e:
        logger_instance.error(f"[WebSocketViz] Server error: {e}")


def _prepare_visualization_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare simulation state data for web visualization"""
    
    # Extract simulacra data
    simulacra_data = {}
    simulacra_profiles = state.get('simulacra_profiles', {})
    for agent_id, agent_data in simulacra_profiles.items():
        simulacra_data[agent_id] = {
            'id': agent_id,
            'name': agent_data.get('persona_details', {}).get('Name', agent_id),
            'current_location': agent_data.get('current_location'),
            'status': agent_data.get('status', 'unknown'),
            'current_action': agent_data.get('current_action_description', 'No current action'),
            'last_observation': agent_data.get('last_observation', 'No recent observations'),
            'goal': agent_data.get('goal', 'No current goal')[:100] + '...' if len(agent_data.get('goal', '')) > 100 else agent_data.get('goal', 'No current goal'),
            'age': agent_data.get('persona_details', {}).get('Age'),
            'occupation': agent_data.get('persona_details', {}).get('Occupation')
        }
    
    # Extract location data
    locations_data = {}
    location_details = state.get('current_world_state', {}).get('location_details', {})
    for loc_id, loc_data in location_details.items():
        # Count objects and NPCs
        ephemeral_objects = loc_data.get('ephemeral_objects', [])
        ephemeral_npcs = loc_data.get('ephemeral_npcs', [])
        connected_locations = loc_data.get('connected_locations', [])
        
        locations_data[loc_id] = {
            'id': loc_id,
            'name': loc_data.get('name', loc_id),
            'description': loc_data.get('description', 'No description'),
            'object_count': len(ephemeral_objects),
            'npc_count': len(ephemeral_npcs),
            'connections': [conn.get('to_location_id_hint') for conn in connected_locations if isinstance(conn, dict)],
            'objects': [{'id': obj.get('id'), 'name': obj.get('name'), 'description': obj.get('description')} 
                       for obj in ephemeral_objects if isinstance(obj, dict)][:10]  # Limit for performance
        }
    
    # Extract global objects
    objects_data = {}
    global_objects = state.get('objects', {})
    
    # Handle both dictionary and list formats
    if isinstance(global_objects, dict):
        for obj_id, obj_data in global_objects.items():
            objects_data[obj_id] = {
                'id': obj_id,
                'properties': obj_data.get('properties', {}),
                'name': obj_data.get('name', obj_id)
            }
    elif isinstance(global_objects, list):
        for obj_data in global_objects:
            if isinstance(obj_data, dict) and 'id' in obj_data:
                obj_id = obj_data['id']
                objects_data[obj_id] = {
                    'id': obj_id,
                    'properties': obj_data.get('properties', {}),
                    'name': obj_data.get('name', obj_id)
                }
    
    # Extract narrative timeline (last 10 entries)
    narrative_log = state.get('narrative_log', [])
    recent_narrative = narrative_log[-10:] if len(narrative_log) > 10 else narrative_log
    
    return {
        'type': 'simulation_state',
        'timestamp': time.time(),
        'world_time': state.get('world_time', 0),
        'simulacra': simulacra_data,
        'locations': locations_data, 
        'objects': objects_data,
        'recent_narrative': recent_narrative,
        'world_feeds': state.get('world_feeds', {}),
        'active_simulacra_count': len(simulacra_data),
        'total_locations': len(locations_data),
        'total_objects': len(objects_data)
    }


def _handle_client_request(request: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle specific client requests for detailed information"""
    
    request_type = request.get('type')
    
    if request_type == 'get_agent_details':
        agent_id = request.get('agent_id')
        if agent_id and agent_id in state.get('simulacra_profiles', {}):
            agent_data = state['simulacra_profiles'][agent_id]
            return {
                'type': 'agent_details',
                'agent_id': agent_id,
                'details': {
                    'full_persona': agent_data.get('persona_details', {}),
                    'monologue_history': agent_data.get('monologue_history', [])[-5:],  # Last 5 thoughts
                    'memory_log': agent_data.get('memory_log', []),
                    'current_location_details': agent_data.get('location_details', ''),
                    'home_location': agent_data.get('home_location'),
                    'action_end_time': agent_data.get('current_action_end_time'),
                    'last_interjection_time': agent_data.get('last_interjection_sim_time')
                }
            }
    
    elif request_type == 'get_location_details':
        location_id = request.get('location_id')
        location_details = state.get('current_world_state', {}).get('location_details', {})
        if location_id and location_id in location_details:
            loc_data = location_details[location_id]
            return {
                'type': 'location_details',
                'location_id': location_id,
                'details': loc_data
            }
    
    return {'type': 'error', 'message': f'Unknown request type: {request_type}'}
