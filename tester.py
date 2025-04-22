from src.tools.python_weather.client import Client
from src.tools.python_weather.constants import IMPERIAL

def get_weather(location: str) -> str:
    """
    Fetches the current weather, daily, and hourly forecasts for the specified location
    using the updated python_weather client.
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

            # Extract daily forecasts
            daily_forecasts = [
                f"Date: {daily.date}, Max Temp: {daily.highest_temperature}°F, Min Temp: {daily.lowest_temperature}°F, sunrise: {daily.sunrise} sunset: {daily.sunset}"
                for daily in forecast.daily_forecasts
            ]

            # Extract hourly forecasts
            hourly_forecasts = [
                f"Time: {hourly.time}, Temp: {hourly.temperature}°F, Description: {hourly.description}"
                for daily in forecast.daily_forecasts
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


# Example call
print(get_weather("Asheville, North Carolina"))