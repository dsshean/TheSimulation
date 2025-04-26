# src/tools/world_engine_tools.py
from google.adk.tools.tool_context import ToolContext
from rich.console import Console
import requests
from GoogleNews import GoogleNews
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from src.tools.python_weather.client import Client  # Import the Client class
from src.tools.python_weather.constants import IMPERIAL
from src.generation.llm_service import LLMService
# Load environment variables from .env file

googlenews = GoogleNews()
console = Console()

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

def process_movement(origin: str, destination: str, tool_context: ToolContext) -> dict:
    """Calculates travel time, updates world time, and updates the Simulacra's location in the state. Returns a result dictionary."""
    console.print(f"[dim green]--- Tool: World Engine processing move from [i]{origin}[/i] to [i]{destination}[/i] ---[/dim green]")
    # !! Placeholder Logic !!
    travel_time_minutes = 30
    current_time_str = tool_context.state.get("world_time", "Day 1, 09:00")
    new_time_str = f"{current_time_str} (+{travel_time_minutes}m)" # Simple placeholder

    tool_context.state["world_time"] = new_time_str
    tool_context.state["simulacra_location"] = destination

    result = {
        "status": "success",
        "duration": travel_time_minutes,
        "new_location": destination,
        "new_time": new_time_str,
        "message": f"Travel from {origin} to {destination} took {travel_time_minutes} minutes. You arrived at {destination} at {new_time_str}."
    }
    tool_context.state["last_world_engine_update"] = result # Store result in state
    console.print(f"[dim green]--- Tool: World Engine updated state: Time=[b]{new_time_str}[/b], Location=[b]{destination}[/b] ---[/dim green]")
    return result

def advance_time(minutes: int, tool_context: ToolContext) -> str:
    """Advances the world clock by a specified number of minutes (e.g., for waiting). Called ONLY by Narration."""
    console.print(f"[dim green]--- Tool: World Engine advancing time by {minutes} minutes ---[/dim green]")
    current_time_str = tool_context.state.get("world_time", "Day 1, 09:00")
    # !! Placeholder Logic !!
    new_time_str = f"{current_time_str} (+{minutes}m)"
    tool_context.state["world_time"] = new_time_str
    result_msg = f"Time advanced by {minutes} minutes. Current time is {new_time_str}."
    tool_context.state["last_world_engine_update"] = {"status": "success", "message": result_msg} # Store result
    console.print(f"[dim green]--- Tool: World Engine updated state: Time=[b]{new_time_str}[/b] ---[/dim green]")
    return result_msg