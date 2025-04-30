# src/tools/world_state_tools.py

import json
import logging
from datetime import datetime, timedelta, timezone # Added timezone
from typing import Dict, Any, Optional, List # Added List

from google.adk.tools import ToolContext
from GoogleNews import GoogleNews
from rich.console import Console

# Assuming these imports exist from previous steps
try:
    # Attempt to import Client and IMPERIAL from the specified path
    from src.tools.python_weather.client import Client
    from src.tools.python_weather.constants import IMPERIAL
    # Initialize logger here after potential import error
    logger = logging.getLogger(__name__)
except ImportError:
    # Initialize logger first if import fails, then log warning
    logger = logging.getLogger(__name__)
    logger.warning("python-weather library not found or import failed. Weather functionality disabled.")
    Client = None # Set Client to None if import fails

WORLD_STATE_KEY = "current_world_state"
SCHEDULED_EVENTS_KEY = "scheduled_events" # Key for the list of events
SIMULACRA_PROFILES_KEY = "simulacra_profiles"
PROFILE_STATUS_KEY = "status"
STATUS_INTERACTION_STATUS_KEY = "interaction_status"
STATUS_INTERACTION_PARTNER_KEY = "interaction_partner_id"
STATUS_INTERACTION_MEDIUM_KEY = "interaction_medium"
STATUS_LAST_INTERACTION_SNIPPET_KEY = "last_interaction_snippet"
LOCATION_DETAILS_KEY = "location_details" # Key within world_state holding location info dicts

googlenews = GoogleNews()
console = Console()
# Logger is initialized above after handling potential import error
DEFAULT_TIME_INCREMENT_SECONDS = 60 * 5 # 5 minutes

# --- Tool: Process Scheduled Events ---
def process_scheduled_events(tool_context: ToolContext) -> None:
    """
    Checks the scheduled_events list in the state. Processes any events
    whose arrival_turn matches the current_turn by generating state updates
    for the recipient and updating the scheduled_events list.
    Signals updates via tool_context.state_delta.
    """
    console.print("[dim purple]--- Tool (WorldState): Processing Scheduled Events ---[/dim purple]")
    state = tool_context.state
    state_delta_updates = {}
    processed_event_count = 0
    warnings = []

    try:
        world_state = state.get(WORLD_STATE_KEY, {})
        current_turn = world_state.get("current_turn")
        scheduled_events: List[Dict[str, Any]] = state.get(SCHEDULED_EVENTS_KEY, [])

        if current_turn is None:
            logger.error("Cannot process scheduled events: 'current_turn' not found in world state.")
            warnings.append("Current turn number missing, cannot process events.")
            # Don't modify state_delta if we can't process
            return

        if not isinstance(scheduled_events, list):
            logger.error(f"'{SCHEDULED_EVENTS_KEY}' is not a list in state. Cannot process events.")
            warnings.append(f"'{SCHEDULED_EVENTS_KEY}' invalid, cannot process events.")
            # Reset the key to an empty list in the delta
            state_delta_updates[SCHEDULED_EVENTS_KEY] = []
            tool_context.state_delta = state_delta_updates
            return

        logger.info(f"Processing scheduled events for turn {current_turn}. Found {len(scheduled_events)} events.")

        remaining_events = []
        for event_data in scheduled_events:
            if not isinstance(event_data, dict):
                logger.warning(f"Skipping invalid item in scheduled_events: {event_data}")
                remaining_events.append(event_data) # Keep invalid items for now? Or discard? Let's keep.
                continue

            arrival_turn = event_data.get("arrival_turn")
            event_type = event_data.get("type")
            recipient_id = event_data.get("recipient_id")

            if arrival_turn == current_turn:
                logger.info(f"Processing event due this turn: {event_data}")
                processed_event_count += 1

                # --- Process 'message_arrival' event type ---
                if event_type == "message_arrival" and recipient_id:
                    sender_id = event_data.get("sender_id")
                    content = event_data.get("content")
                    medium = event_data.get("medium")

                    # Construct the state key for the recipient's status
                    recipient_status_key = f"{SIMULACRA_PROFILES_KEY}.{recipient_id}.{PROFILE_STATUS_KEY}"

                    # Prepare the status update dictionary
                    status_update = {
                        STATUS_INTERACTION_STATUS_KEY: "receiving_message", # Or 'receiving_call'/'receiving_text' based on medium? Let's use generic for now.
                        STATUS_INTERACTION_PARTNER_KEY: sender_id,
                        STATUS_INTERACTION_MEDIUM_KEY: medium,
                        STATUS_LAST_INTERACTION_SNIPPET_KEY: f"Received {medium} from {sender_id}: '{content}'"
                    }
                    # Add this update to the main delta dictionary
                    # Note: This assumes the state service can handle nested updates like this.
                    # If not, the agent would need to read the whole status, update it, and put the whole dict back.
                    # Let's assume nested updates work for now via dot notation in the key.
                    # We need to merge this with potentially existing status updates, not overwrite the whole status dict.
                    # A safer approach for the tool is to just return the intended changes.
                    # Let's refine: the delta should specify the *specific fields* to change.
                    state_delta_updates[f"{recipient_status_key}.{STATUS_INTERACTION_STATUS_KEY}"] = "receiving_message"
                    state_delta_updates[f"{recipient_status_key}.{STATUS_INTERACTION_PARTNER_KEY}"] = sender_id
                    state_delta_updates[f"{recipient_status_key}.{STATUS_INTERACTION_MEDIUM_KEY}"] = medium
                    state_delta_updates[f"{recipient_status_key}.{STATUS_LAST_INTERACTION_SNIPPET_KEY}"] = f"Received {medium} from {sender_id}: '{content}'"

                    logger.info(f"Prepared status update for recipient {recipient_id} due to message arrival.")

                # --- Add handlers for other event types here if needed ---
                # elif event_type == "other_event":
                #    ...

                else:
                    logger.warning(f"Unknown or incomplete event type '{event_type}' for recipient '{recipient_id}'. Keeping event.")
                    remaining_events.append(event_data) # Keep unprocessed event

            else:
                # Event is not for this turn, keep it
                remaining_events.append(event_data)

        # --- Update the scheduled events list in the delta ---
        # Only add the update if the list actually changed
        if len(remaining_events) != len(scheduled_events):
            state_delta_updates[SCHEDULED_EVENTS_KEY] = remaining_events
            logger.info(f"Prepared update for '{SCHEDULED_EVENTS_KEY}' list (removed {processed_event_count} events).")
        else:
            logger.info("No change needed to scheduled events list.")

        # --- Assign the collected updates to state_delta ---
        if state_delta_updates:
            # Check if state_delta already exists and merge if necessary
            if hasattr(tool_context, 'state_delta') and isinstance(tool_context.state_delta, dict):
                 tool_context.state_delta.update(state_delta_updates)
            else:
                 tool_context.state_delta = state_delta_updates
            logger.info(f"Signaled state updates via state_delta: {list(state_delta_updates.keys())}")
        else:
            logger.info("No state updates generated from scheduled events.")

    except Exception as e:
        logger.exception(f"Error processing scheduled events: {e}")
        warnings.append(f"Error during event processing: {e}")
        # Avoid modifying state_delta on error? Or signal error? Let's avoid for now.

    finally:
        if warnings:
             # Optionally add warnings to state_delta if needed by agent
             # state_delta_updates["world_state_warnings"] = warnings
             pass
        console.print(f"[dim purple]--- Tool (WorldState): Finished Processing Events ({processed_event_count} processed) ---[/dim purple]")
        # Return None as state is updated via context
        return None


# --- Existing Tools (get_setting_details, get_weather, get_news, update_and_get_world_state) ---

def get_setting_details(location: str, tool_context: ToolContext) -> None:
    """
    Provides descriptive details about the specified location as a dictionary
    and signals the update via tool_context.state_delta under 'location_details'.
    Uses external services/functions for weather and news.
    Uses CURRENT REAL-WORLD time for context.
    """
    console.print(f"[dim green]--- Tool (WorldState): Getting Setting Details for [i]{location}[/i] ---[/dim green]")
    details_dict = {} # Initialize dictionary to hold details
    state_delta_updates = {} # Initialize delta for this tool

    try:
        # Step 1: Get base description
        base_description = f"A placeholder description for {location}." # Placeholder
        details_dict["description"] = base_description

        # Step 2: Get Weather & News (Check if Client is available)
        weather_info = "Weather data unavailable."
        if Client:
            weather_info = get_weather(location)
        else:
            logger.warning("Weather client not available, skipping weather fetch.")

        local_news_info = get_news(f"local news in {location}")
        regional_news_info = get_news(f"regional news near {location}")
        world_news_info = get_news("world news")

        details_dict["weather"] = weather_info
        details_dict["local_news"] = local_news_info
        details_dict["regional_news"] = regional_news_info
        details_dict["world_news"] = world_news_info

        # Step 3: Add Real Time
        real_time_now_str = datetime.now(timezone.utc).isoformat() # Use UTC
        details_dict["current_real_time"] = real_time_now_str
        details_dict["location_name"] = location

        # Prepare the state delta payload for this tool
        # We want to update the specific location's details within the main location_details dict
        location_details_key = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{location}" # Nested key
        state_delta_updates[location_details_key] = details_dict

        console.print("[dim green]--- Tool (WorldState): Setting details dictionary generated ---[/dim green]")

    except Exception as e:
        logger.error(f"Error in get_setting_details for {location}: {e}", exc_info=True)
        console.print(f"[bold red]Error generating setting details for {location}: {e}[/bold red]")
        # Signal error state update via state_delta
        location_details_key = f"{WORLD_STATE_KEY}.{LOCATION_DETAILS_KEY}.{location}"
        state_delta_updates[location_details_key] = {"error": f"Failed to retrieve details: {e}"}

    finally:
        # Assign the collected updates (success or error) to state_delta
        if state_delta_updates:
            # Check if state_delta already exists and merge if necessary
            if hasattr(tool_context, 'state_delta') and isinstance(tool_context.state_delta, dict):
                 tool_context.state_delta.update(state_delta_updates)
            else:
                 tool_context.state_delta = state_delta_updates
            logger.info(f"Signaled state_delta update for location details: {location}")
        # Return None as state is updated via context
        return None

def get_weather(location: str) -> str:
    """
    Fetches the current weather, daily, and hourly forecasts for the specified location.
    Filters daily forecasts to include only the current date.
    Requires python-weather library.
    """
    if not Client: # Check if Client was imported successfully
        return "Weather client unavailable."
    try:
        # Use the Client with imperial units (Fahrenheit)
        # Consider making the client creation more robust or async if needed
        with Client(unit=IMPERIAL) as client:
            forecast = client.get(location)
            current_temp = forecast.temperature
            description = forecast.description
            humidity = forecast.humidity
            precipitation = forecast.precipitation
            current_date = datetime.now(timezone.utc).date() # Use timezone aware date

            daily_forecasts = [
                f"Date: {daily.date}, Max: {daily.highest_temperature}°F, Min: {daily.lowest_temperature}°F, Sunrise: {daily.sunrise}, Sunset: {daily.sunset}"
                for daily in forecast.daily_forecasts if daily.date == current_date
            ]
            hourly_forecasts = [
                f"Time: {hourly.time}, Temp: {hourly.temperature}°F, Desc: {hourly.description}"
                for daily in forecast.daily_forecasts if daily.date == current_date
                for hourly in daily.hourly_forecasts
            ]

            daily_summary = "\n".join(daily_forecasts) if daily_forecasts else "No daily forecast for today."
            hourly_summary = "\n".join(hourly_forecasts[:5]) if hourly_forecasts else "No hourly forecast available." # Limit hourly

            return (
                f"Current: {description}, Temp: {current_temp}°F, Humidity: {humidity}%, Precip: {precipitation}%\n"
                f"Today's Forecast:\n{daily_summary}\n"
                f"Hourly:\n{hourly_summary}"
            )
    except Exception as e:
        logger.error(f"Error fetching weather for {location}: {e}")
        return f"Error fetching weather for {location}."


def get_news(query: str) -> str:
    """
    Fetches news articles for the specified query using the GoogleNews library.
    Fetches news from the last day.
    """
    try:
        googlenews.clear()
        googlenews.set_lang('en')
        googlenews.set_period('1d') # Last 1 day
        googlenews.search(query)
        results = googlenews.result(sort=True) # Sort by date might be useful

        if results:
            # Limit description length
            news_items = []
            for result in results[:3]: # Top 3
                title = result.get('title', 'No Title')
                desc = result.get('desc', 'No Description')
                desc_short = desc[:100] + ('...' if len(desc) > 100 else '')
                news_items.append(f"{title} - {desc_short}")
            return " | ".join(news_items)
        return "No relevant news found."
    except Exception as e:
        logger.error(f"Error fetching news for query '{query}': {e}")
        return "News data unavailable."


def update_and_get_world_state(tool_context: ToolContext) -> None:
    """
    Updates the world state time and signals the update via tool_context.state_delta.
    """
    console.print("[dim blue]--- Tool (WorldState): Updating world time... ---[/dim blue]")
    state = tool_context.state
    state_delta_updates = {} # Initialize delta for this tool

    try:
        world_state = state.get(WORLD_STATE_KEY, {})
        current_time_str = world_state.get("world_time", None)
        new_time_str = None

        if current_time_str:
            try:
                # Ensure parsing handles potential 'Z' for UTC
                if current_time_str.endswith('Z'):
                    current_time_dt = datetime.fromisoformat(current_time_str[:-1] + '+00:00')
                else:
                    current_time_dt = datetime.fromisoformat(current_time_str)
                    if current_time_dt.tzinfo is None: # Assume UTC if naive
                         current_time_dt = current_time_dt.replace(tzinfo=timezone.utc)

                time_increment = timedelta(seconds=DEFAULT_TIME_INCREMENT_SECONDS)
                new_time_dt = current_time_dt + time_increment
                new_time_str = new_time_dt.isoformat() # Keep ISO format with offset

                # Prepare the update for the delta using nested key
                time_key = f"{WORLD_STATE_KEY}.world_time"
                state_delta_updates[time_key] = new_time_str
                logger.info(f"Calculated time advancement from {current_time_str} to {new_time_str}")

            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing/advancing time tick '{current_time_str}': {e}. Time not advanced.")
            except Exception as e:
                 logger.exception(f"Unexpected error during time advancement for '{current_time_str}': {e}")
        else:
            logger.warning("World time not found in state. Cannot advance time.")

    except Exception as e:
        logger.exception(f"Error preparing world state update: {e}")

    finally:
        # Assign the collected updates to state_delta
        if state_delta_updates:
            # Check if state_delta already exists and merge if necessary
            if hasattr(tool_context, 'state_delta') and isinstance(tool_context.state_delta, dict):
                 tool_context.state_delta.update(state_delta_updates)
            else:
                 tool_context.state_delta = state_delta_updates
            logger.info(f"Signaled world state update via state_delta for keys: {list(state_delta_updates.keys())}")
        else:
            logger.info("No world state updates generated by this tool.")

        console.print("[dim blue]--- Tool (WorldState): Finished signaling world state update. ---[/dim blue]")
        # Return None as state is updated via context
        return None
