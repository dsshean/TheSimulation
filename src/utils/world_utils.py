# src/utils/world_utils.py

def get_day_phase(hour: int) -> str:
    """
    Determines the general phase of the day based on the hour (0-23).

    Args:
        hour: The hour of the day (0-23).

    Returns:
        A string representing the day phase (e.g., "Morning", "Night").
    """
    if not isinstance(hour, int) or not (0 <= hour <= 23):
        # Handle invalid input, though typically hour comes from datetime
        return "Unknown Phase"

    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Midday" # Or "Afternoon" - adjust based on preference
    elif 17 <= hour < 21:
        return "Evening"
    elif 21 <= hour < 24: # Until midnight
        return "Night"
    elif 0 <= hour < 5: # From midnight until early morning
        return "Late Night" # Or just "Night"
    else: # Should not happen with valid 0-23 input
        return "Unknown Phase"