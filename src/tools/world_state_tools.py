# src/tools/world_state_tools.py (Executor Tool)

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from google.adk.tools.tool_context import ToolContext
from GoogleNews import GoogleNews
from rich.console import Console

from src.generation.llm_service import LLMService
from src.tools.python_weather.client import Client  # Import the Client class
from src.tools.python_weather.constants import IMPERIAL

WORLD_STATE_KEY = "current_world_state" # Make sure this line exists and is spelled correctly
SIMULACRA_INTENT_KEY_FORMAT = "simulacra_{}_intent"
SIMULACRA_LOCATION_KEY_FORMAT = "simulacra_{}_location"
SIMULACRA_STATUS_KEY_FORMAT = "simulacra_{}_status"
ACTIVE_SIMULACRA_IDS_KEY = "active_simulacra_ids"
ACTION_VALIDATION_KEY_FORMAT = "simulacra_{}_validation_result" # Ensure this matches simulation_loop.py

googlenews = GoogleNews()
console = Console()
logger = logging.getLogger(__name__)

DEFAULT_TIME_INCREMENT_SECONDS = 60 * 5 # 5 minutes

def get_setting_details(location: str, tool_context: ToolContext) -> str:
    """
    Provides descriptive details about the specified location based on world state.
    Uses the LLMService to generate location details synchronously.
    Fetches real-time weather and news. Uses CURRENT REAL-WORLD time for context.
    """
    console.print(f"[dim green]--- Tool: World Engine providing setting details for [i]{location}[/i] ---[/dim green]")

    # Step 1: Get a base description of the location using LLMService
    try:
        llm_service = LLMService()
        prompt = f"Provide a brief (4 to 5 sentences max) description of the location '{location}'. Include historical, cultural, or notable features."
        base_description = llm_service.generate_content_text(
            prompt=prompt
        )

    except Exception as e:
        console.print(f"[bold red]Error querying LLMService for location details:[/bold red] {e}")
        base_description = f"You are at {location}. There's nothing particularly notable right now."

    # Step 2: Fetch high-level details using tools
    weather = get_weather(f"{location}")
    local_news = get_news(f"local news in {location}")
    regional_news = get_news(f"regional news near {location}")
    world_news = get_news("world news")

    # Step 3: Combine all details into a full description
    # --- MODIFIED: Use current real-world time ---
    real_time_now_str = datetime.now().isoformat()
    # current_time = tool_context.state.get("world_time", "an unknown time") # Old way
    # ---
    full_description = (
        f"Location: {location}. Current Real Time: {real_time_now_str}. " # Use real time
        f"Description: {base_description} "
        f"Weather: {weather}. "
        f"Local News: {local_news}. "
        f"Regional News: {regional_news}. "
        f"World News: {world_news}."
    )

    # Ensure "location_details" exists in the state
    if "location_details" not in tool_context.state:
        tool_context.state["location_details"] = {}

    # Store the generated description in the session state
    tool_context.state["location_details"][location] = full_description
    console.print("[dim green]--- Tool: Setting details generated ---[/dim green]")

    # Return the description as well (optional, but can be useful)
    return full_description # Changed to return the string

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
            current_date = date.today()

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

def update_and_get_world_state(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Updates the world state (time, potentially other dynamics) and returns the complete current state.
    """
    console.print("[dim blue]--- Tool (WorldState): Updating and getting full world state... ---[/dim blue]")
    state = tool_context.state

    # --- Time Advancement ---
    # Uses WORLD_STATE_KEY defined above
    world_state = state.get(WORLD_STATE_KEY, {})
    current_time_str = world_state.get("world_time", None)
    new_time_str = current_time_str # Default to original if parsing fails

    if current_time_str:
        try:
            current_time_dt = datetime.fromisoformat(current_time_str)
            time_increment = timedelta(seconds=DEFAULT_TIME_INCREMENT_SECONDS)
            new_time_dt = current_time_dt + time_increment
            new_time_str = new_time_dt.isoformat()
            world_state["world_time"] = new_time_str
            logger.info(f"Advanced world time from {current_time_str} to {new_time_str}")
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing/advancing time tick '{current_time_str}': {e}. Returning original.")
        except Exception as e:
             logger.exception(f"Unexpected error during time advancement for '{current_time_str}': {e}")
    else:
        logger.warning("World time not found in state. Cannot advance time.")

    # --- Other State Updates (Optional) ---
    # world_state["weather"] = "Slightly cloudy" # Example

    # Uses WORLD_STATE_KEY defined above
    state[WORLD_STATE_KEY] = world_state
    console.print("[dim blue]--- Tool (WorldState): Finished updating world state. ---[/dim blue]")
    return world_state