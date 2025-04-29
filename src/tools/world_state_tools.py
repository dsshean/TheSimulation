# src/tools/world_state_tools.py (Executor Tool)

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from google.adk.tools import ToolContext
from GoogleNews import GoogleNews
from rich.console import Console

from src.generation.llm_service import LLMService
from src.tools.python_weather.client import Client  # Import the Client class
from src.tools.python_weather.constants import IMPERIAL

googlenews = GoogleNews()
console = Console()
logger = logging.getLogger(__name__)
DEFAULT_TIME_INCREMENT_SECONDS = 60 * 5 # 5 minutes
WORLD_STATE_KEY = "current_world_state"
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent"
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
ACTION_VALIDATION_KEY_FORMAT = "simulacra_{}_validation_result" # Ensure this matches simulation_loop.py

def get_setting_details(location: str, tool_context: ToolContext) -> None:
    """
    Provides descriptive details about the specified location as a dictionary
    and signals the update via tool_context.state_delta under 'location_details'.
    Uses external services/functions for weather and news.
    Uses CURRENT REAL-WORLD time for context.
    """
    console.print(f"[dim green]--- Tool: World Engine providing setting details for [i]{location}[/i] ---[/dim green]")
    details_dict = {} # Initialize dictionary to hold details

    try:
        # Step 1: Get base description
        base_description = f"A placeholder description for {location}." # Placeholder
        details_dict["description"] = base_description

        # Step 2: Get Weather & News
        weather_info = get_weather(location)
        local_news_info = get_news(f"local news in {location}")
        regional_news_info = get_news(f"regional news near {location}")
        world_news_info = get_news("world news")

        details_dict["weather"] = weather_info
        details_dict["local_news"] = local_news_info
        details_dict["regional_news"] = regional_news_info
        details_dict["world_news"] = world_news_info

        # Step 3: Add Real Time
        real_time_now_str = datetime.now().isoformat()
        details_dict["current_real_time"] = real_time_now_str
        details_dict["location_name"] = location

        # --- REMOVE Direct State Modification ---

        console.print("[dim green]--- Tool: Setting details dictionary generated ---[/dim green]")

        # --- MODIFIED: Assign payload to tool_context.state_delta ---
        state_update_payload = {
            "location_details": {
                location: details_dict # The dictionary containing all details
            }
        }
        # Assign the dictionary to the state_delta attribute of the context
        # tool_context.state_delta = state_update_payload
        tool_context.state_delta = 'test'
        logger.info(f"Assigned payload to tool_context.state_delta for location: {location}")
        # --- END MODIFICATION ---

        # <<< Return None (or a status string) instead of the state payload >>>
        return None

    except Exception as e:
        logger.error(f"Error in get_setting_details for {location}: {e}", exc_info=True)
        console.print(f"[bold red]Error generating setting details for {location}: {e}[/bold red]")
        # Signal error state update via state_delta
        tool_context.state_delta = {
            "location_details": {
                location: {"error": f"Failed to retrieve details: {e}"}
            }
        }
        # <<< Return None (or an error status string) >>>
        return None

def get_weather(location: str) -> str:
    """
    Fetches the current weather, daily, and hourly forecasts for the specified location.
    Filters daily forecasts to include only the current date.
    """
    try:
        # Use the Client with imperial units (Fahrenheit)
        with Client(unit=IMPERIAL) as client:
            # Fetch the weather for the specified location
            forecast = client.get(location)

            # Extract the current weather details
            current_temp = forecast.temperature  # Current temperature
            description = forecast.description  # Current weather description
            humidity = forecast.humidity  # Current humidity
            precipitation = forecast.precipitation  # Current precipitation

            # Get the current date
            current_date = datetime.now().date()

            # Filter daily forecasts for the current date
            daily_forecasts = [
                f"Date: {daily.date}, Max Temp: {daily.highest_temperature}°F, Min Temp: {daily.lowest_temperature}°F, sunrise: {daily.sunrise} sunset: {daily.sunset}"
                for daily in forecast.daily_forecasts
                if daily.date == current_date
            ]

            # Extract hourly forecasts for the current date
            hourly_forecasts = [
                f"Time: {hourly.time}, Temp: {hourly.temperature}°F, Description: {hourly.description}"
                for daily in forecast.daily_forecasts
                if daily.date == current_date
                for hourly in daily.hourly_forecasts
            ]

            # Format the output
            daily_summary = "\n".join(daily_forecasts)
            hourly_summary = "\n".join(hourly_forecasts[:5])  # Limit to the first 5 hours for brevity

            return (
                f"Current Weather: {description}, Temperature: {current_temp}°F, Humidity: {humidity}%, Precipitation: {precipitation}%\n\n"
                f"Daily Forecast:\n{daily_summary}\n\n"
                f"Hourly Forecast:\n{hourly_summary}"
            )
    except Exception as e:
        return f"Error fetching weather for {location}: {e}"

def get_news(query: str) -> str:
    """
    Fetches news articles for the specified query using the GoogleNews library.
    Dynamically fetches news from today and the previous day.
    """
    try:
        # Calculate today's and yesterday's dates
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        googlenews.clear()
        # Set up GoogleNews instance
        googlenews.set_lang('en')  # Set language to English
        # googlenews.set_time_range(yesterday, today)  # Set the date range dynamically
        googlenews.set_period('1d')
        googlenews.search(query)  # Perform the search with the query

        # Fetch the results
        results = googlenews.result()  # Get the search results
        if results:
            # Combine the top 3 results into a single string
            return " | ".join(f"{result['title']} - {result['desc']}" for result in results[:3])
        return "No relevant information found."
    except Exception as e:
        console.print(f"[bold red]Error fetching data for query '{query}':[/bold red] {e}")
        return "Data unavailable."

# --- Tool: Update World State ---
def update_and_get_world_state(tool_context: ToolContext) -> None:
    """
    Updates the world state (time, potentially other dynamics) and signals
    the update via tool_context.state_delta.
    """
    console.print("[dim blue]--- Tool (WorldState): Updating world state... ---[/dim blue]")
    state = tool_context.state # Read current state

    # --- Time Advancement ---
    world_state = state.get(WORLD_STATE_KEY, {}) # Get a copy or default
    current_time_str = world_state.get("world_time", None)
    new_time_str = current_time_str # Default to original if parsing fails
    time_advanced = False

    if current_time_str:
        try:
            current_time_dt = datetime.fromisoformat(current_time_str)
            time_increment = timedelta(seconds=DEFAULT_TIME_INCREMENT_SECONDS)
            new_time_dt = current_time_dt + time_increment
            new_time_str = new_time_dt.isoformat()
            # Prepare the update for the delta, don't modify world_state directly here yet
            time_advanced = True
            logger.info(f"Calculated time advancement from {current_time_str} to {new_time_str}")
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing/advancing time tick '{current_time_str}': {e}. Time not advanced.")
        except Exception as e:
             logger.exception(f"Unexpected error during time advancement for '{current_time_str}': {e}")
    else:
        logger.warning("World time not found in state. Cannot advance time.")

    # --- Prepare the state delta ---
    # Start with the existing world state to preserve other keys
    updated_world_state_payload = world_state.copy()
    if time_advanced:
        updated_world_state_payload["world_time"] = new_time_str

    # --- Other State Updates (Optional) ---
    # updated_world_state_payload["weather"] = "Slightly cloudy" # Example

    # --- Assign to state_delta ---
    # The key is WORLD_STATE_KEY, the value is the entire updated world state dictionary
    tool_context.state_delta = {
        WORLD_STATE_KEY: updated_world_state_payload
    }
    logger.info(f"Signaled world state update via state_delta for key '{WORLD_STATE_KEY}'.")
    # ---

    # --- REMOVE Direct State Modification ---
    # state[WORLD_STATE_KEY] = world_state
    # ---

    console.print("[dim blue]--- Tool (WorldState): Finished signaling world state update. ---[/dim blue]")
    # <<< Return None (or a status string) instead of the state dict >>>
    return None