# src/tools/world_state_tools.py (Simplified with Orchestrator Tool & Explicit Save)
import json
import datetime
from pathlib import Path
import requests # Assuming needed for news/weather eventually
from GoogleNews import GoogleNews
from dotenv import load_dotenv
from rich.console import Console
from src.tools.python_weather.client import Client as WeatherClient # Alias to avoid name clash
from src.tools.python_weather.constants import IMPERIAL
from google.adk.tools.tool_context import ToolContext # Import ToolContext

# Instantiate Console & Clients
console = Console()
googlenews = GoogleNews()

# --- Helper Function for Config Reading ---
def _read_world_config(config_file_path: str) -> dict:
    """Internal helper to read the world configuration."""
    console.print(f"[dim blue]Helper: Reading world config from: {config_file_path}[/dim blue]")
    try:
        config_path = Path(config_file_path)
        if not config_path.is_file():
            console.print(f"[bold red]Helper: Config file not found at {config_path.resolve()}[/bold red]")
            return {"status": "error", "error_message": f"Config file not found: {config_file_path}"}
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        console.print("[dim green]Helper: Successfully read world config.[/dim green]")
        config_data['status'] = 'success'
        return config_data
    except Exception as e:
        console.print(f"[bold red]Helper: Failed to read world config file {config_file_path}: {e}[/bold red]")
        return {"status": "error", "error_message": f"Could not read config file: {e}"}

# --- Helper Function for Weather ---
def _get_current_weather(location: str) -> dict:
    """Internal helper to fetch weather."""
    console.print(f"[dim blue]Helper: Fetching weather for {location}...[/dim blue]")
    try:
        with WeatherClient(unit=IMPERIAL) as client:
            forecast = client.get(location)
            current_date = datetime.date.today()
            daily_forecasts_details = [
                f"Date: {daily.date}, Max Temp: {daily.highest_temperature}°F, Min Temp: {daily.lowest_temperature}°F, Sunrise: {daily.sunrise}, Sunset: {daily.sunset}"
                for daily in forecast.daily_forecasts if daily.date == current_date
            ]
            hourly_forecasts_details = [
                f"Time: {hourly.time}, Temp: {hourly.temperature}°F, Description: {hourly.description}"
                for daily in forecast.daily_forecasts if daily.date == current_date
                for hourly in daily.hourly_forecasts
            ]
            daily_summary = "\n".join(daily_forecasts_details) if daily_forecasts_details else "No daily forecast available for today."
            hourly_summary = "\n".join(hourly_forecasts_details[:5]) if hourly_forecasts_details else "No hourly forecast available for today."

            weather_data = {
                 "current": {
                     "temperature_fahrenheit": forecast.temperature,
                     "description": forecast.description,
                     "humidity_percent": forecast.humidity,
                     "precipitation_percent": forecast.precipitation
                 },
                 "daily_summary": daily_summary,
                 "hourly_forecast_summary": hourly_summary,
                 "timestamp": datetime.datetime.now().isoformat()
             }
            console.print(f"[dim green]Helper: Weather fetched successfully for {location}.[/dim green]")
            return {"status": "success", "weather": weather_data}
    except Exception as e:
        console.print(f"[bold red]Helper: Error fetching weather for {location}: {e}[/bold red]")
        return {"status": "error", "error_message": f"Error fetching weather: {e}"}

# --- Helper Function for News ---
def _get_current_news(query: str) -> dict:
    """Internal helper to fetch news."""
    console.print(f"[dim blue]Helper: Fetching news for query: '{query}'...[/dim blue]")
    try:
        googlenews.clear()
        googlenews.set_lang('en')
        googlenews.set_period('1d')
        googlenews.search(query)
        results = googlenews.result()
        if results:
            headlines = " | ".join(f"{result['title']} - {result.get('desc', 'No description')}" for result in results[:3])
            console.print(f"[dim green]Helper: News fetched successfully for '{query}'.[/dim green]")
            return {"status": "success", "headlines": headlines, "timestamp": datetime.datetime.now().isoformat()}
        else:
             console.print(f"[dim yellow]Helper: No relevant news found for '{query}' in the last day.[/dim yellow]")
             return {"status": "success", "headlines": "No relevant news found for the last day.", "timestamp": datetime.datetime.now().isoformat()}
    except Exception as e:
        console.print(f"[bold red]Helper: Error fetching news for query '{query}': {e}[/bold red]")
        # Check for specific rate limit error
        if "HTTP Error 429" in str(e):
             return {"status": "error", "error_message": "News API rate limit exceeded. Try again later."}
        return {"status": "error", "error_message": f"News data unavailable: {e}"}

# --- Helper Function for Time ---
def _get_current_time(location: str) -> dict:
    """Internal helper to get current time (placeholder)."""
    console.print(f"[dim blue]Helper: Getting current time for {location}...[/dim blue]")
    # !!! Replace with actual time zone logic !!!
    try:
        now = datetime.datetime.now() # Fallback to local time
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        timezone_str = str(now.astimezone().tzinfo)
        console.print(f"[dim green]Helper: Current time retrieved for {location}.[/dim green]")
        return {
            "status": "success",
            "time": current_time_str,
            "timezone": timezone_str,
            "timestamp": now.isoformat()
        }
    except Exception as e:
        console.print(f"[bold red]Helper: Error getting time for {location}: {e}[/bold red]")
        return {"status": "error", "error_message": f"Could not determine time: {e}"}

# --- MAIN ORCHESTRATOR TOOL ---
def get_full_world_state(tool_context: ToolContext) -> dict:
    """
    Orchestrates the gathering of all world state information.
    Reads config, checks type, calls helper functions for time/weather/news if needed,
    saves the final state to session state, and returns the compiled dictionary.
    """
    console.print("[bold cyan]Tool: get_full_world_state starting...[/bold cyan]")
    final_world_state = {} # Initialize empty dictionary

    try:
        # 1. Get config file path from state (set in main.py)
        config_file_path = tool_context.state.get("world_config_path", "world_config.json") # Default if not set
        console.print(f"Tool: Using config path from state: {config_file_path}")

        # 2. Load Configuration using helper
        config_data = _read_world_config(config_file_path)
        if config_data.get("status") == "error":
            console.print("[bold red]Tool: Error reading config. Saving error state.[/bold red]")
            final_world_state = config_data # Use error dict as the state
            # --- EXPLICIT STATE SAVE ---
            tool_context.state['current_world_state'] = final_world_state
            console.print("[bold red]Tool: Saved error state to 'current_world_state'.[/bold red]")
            # --- END EXPLICIT SAVE ---
            return final_world_state # Return error

        # 3. Initialize Final Output (already done above)

        # 4. Copy Base Info
        final_world_state["world_type"] = config_data.get("world_type", "unknown")
        final_world_state["sub_genre"] = config_data.get("sub_genre", "unknown")
        final_world_state["description"] = config_data.get("description", "N/A")
        final_world_state["rules"] = config_data.get("rules", {})
        final_world_state["location"] = config_data.get("location", {})

        # 5. Gather Realtime Data (IF Condition Met)
        if final_world_state["world_type"] == 'real' and final_world_state["sub_genre"] == 'realtime':
            console.print("[cyan]Tool: Realtime world detected. Gathering time, weather, news...[/cyan]")
            city = final_world_state.get("location", {}).get("city", "Unknown")
            state = final_world_state.get("location", {}).get("state", "Unknown")
            location_string = f"{city}, {state}" if city != "Unknown" else "Default Location"
            news_query = f"local news in {location_string}"

            # Call helpers and add results to final state
            final_world_state["current_time"] = _get_current_time(location_string)
            final_world_state["current_weather"] = _get_current_weather(location_string)
            final_world_state["current_news"] = _get_current_news(news_query)
        else:
            console.print("[cyan]Tool: Not a realtime world. Skipping time/weather/news gathering.[/cyan]")

        final_world_state["status"] = "success" # Add overall status

    except Exception as e:
         # Catch any unexpected error during orchestration
         console.print(f"[bold red]Tool: UNEXPECTED error in get_full_world_state: {e}[/bold red]")
         final_world_state = {"status": "error", "error_message": f"Unexpected error in state tool: {e}"}

    # --- EXPLICIT STATE SAVE (Success or Catch Block) ---
    tool_context.state['current_world_state'] = final_world_state
    console.print("[bold green]Tool: Saved final state to 'current_world_state'.[/bold green]")
    # --- END EXPLICIT SAVE ---

    # 6. Final Output Generation
    console.print("[bold green]Tool: get_full_world_state returning final dictionary.[/bold green]")
    return final_world_state

