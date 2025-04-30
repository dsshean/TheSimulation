import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytz
from google.generativeai import types
from rich.console import Console
from timezonefinder import TimezoneFinder

logger = logging.getLogger(__name__)

def print_event_details(
    event: Any, # Use Any temporarily to bypass type errors
    phase_name: str,
    console: Console,
    logger: logging.Logger,
    max_content_length: int = 5000,
    max_response_length: int = 5000
):
    logger.debug(f"[{phase_name}] Received event of type: {type(event)}")

    agent_id = getattr(event, 'author', 'UnknownAuthor')
    is_final = getattr(event, 'is_final_response', lambda: False)() # Call if it's a method
    actions = getattr(event, 'actions', None)
    content = getattr(event, 'content', None)

    logger.debug(f"{phase_name} Event ({agent_id}): Final={is_final}, Actions={actions}, Content={str(content)[:max_content_length]}...")

    if content and hasattr(content, 'parts') and content.parts:
        # Assuming content.parts structure is somewhat stable (like google.generativeai.types.Content)
        part = content.parts[0]
        if hasattr(part, 'function_call') and part.function_call:
            tool_call = part.function_call
            # Ensure args is dictionary-like before converting
            args_dict = getattr(tool_call, 'args', {})
            try:
                args_display = dict(args_dict)
            except (TypeError, ValueError):
                args_display = str(args_dict) # Fallback to string representation
            console.print(f"[dim blue]  {phase_name} ({agent_id}) -> Tool Call: {getattr(tool_call, 'name', 'UnknownTool')} with args: {args_display}[/dim blue]")
        elif hasattr(part, 'function_response') and part.function_response:
            tool_response = part.function_response
            response_content = getattr(tool_response, 'response', {})
            try:
                response_str = json.dumps(dict(response_content))
            except (TypeError, ValueError):
                 response_str = str(response_content) # Fallback to string
            response_display = response_str[:max_response_length] + ('...' if len(response_str) > max_response_length else '')
            console.print(f"[dim green]  {phase_name} ({agent_id}) <- Tool Response: {getattr(tool_response, 'name', 'UnknownTool')} -> {response_display}[/dim green]")
        elif is_final and hasattr(part, 'text'):
            text_content = getattr(part, 'text', '')
            console.print(f"[dim cyan]  {phase_name} ({agent_id}) Final Output: {text_content if text_content else '[No text output]'}[/dim cyan]")
        elif hasattr(part, 'text'):
            text_content = getattr(part, 'text', '')
            console.print(f"[dim cyan]  {phase_name} ({agent_id}) Text Output: {text_content if text_content else '[No text output]'}[/dim cyan]")
    elif is_final:
         logger.debug(f"{phase_name} ({agent_id}) Final event with no standard parts. Event: {event}")

def parse_json_output_last(
    raw_text: Optional[str],
    phase_name: str,
    agent_name: str,
    console: Console,
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON from an agent's text output, robustly handling
    markdown code fences and preceding/trailing text.
    Prioritizes the LAST JSON block found.
    Returns the parsed dictionary or None on failure.
    """
    if not raw_text:
        console.print(f"[yellow]{phase_name}: {agent_name} returned empty final response.[/yellow]")
        return None

    text_to_parse = raw_text.strip()
    logger.debug(f"[{phase_name}/{agent_name}] Raw output: {repr(text_to_parse)}")

    json_string_to_parse = None

    # --- More Robust JSON Extraction - Prioritize LAST block ---
    # 1. Look for ```json ... ``` blocks
    json_matches = re.findall(r"```json\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
    if json_matches:
        # Take the last match found
        json_string_to_parse = json_matches[-1].strip()
        logger.debug(f"[{phase_name}/{agent_name}] Extracted LAST JSON block using ```json regex.")
    else:
        # 2. If no ```json block, look for ``` ... ``` blocks
        code_matches = re.findall(r"```\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
        if code_matches:
            # Take the last match found
            json_string_to_parse = code_matches[-1].strip()
            logger.debug(f"[{phase_name}/{agent_name}] Extracted LAST JSON block using ``` regex.")
        else:
            # 3. If no markdown block, find the LAST '{' and try parsing from there
            # This is less reliable but better than taking the first one
            last_brace = text_to_parse.rfind('{')
            if last_brace != -1:
                # Attempt to find the matching closing brace (simple heuristic)
                # Find the last '}' after the last '{'
                last_close_brace = text_to_parse.rfind('}', last_brace)
                if last_close_brace != -1:
                    potential_json = text_to_parse[last_brace : last_close_brace + 1]
                    json_string_to_parse = potential_json
                    logger.debug(f"[{phase_name}/{agent_name}] No markdown found, trying from last '{{' to last '}}'.")
                else:
                    # Fallback: try parsing from last '{' to the end
                    potential_json = text_to_parse[last_brace:]
                    json_string_to_parse = potential_json
                    logger.debug(f"[{phase_name}/{agent_name}] No markdown found, trying from last '{{' to end (fallback).")
            else:
                # 4. If no '{' found at all
                logger.error(f"{phase_name}: Could not find JSON object start '{{' in output from {agent_name}.")
                console.print(f"[bold red]{phase_name}: Error parsing {agent_name} JSON output (no '{{' found).[/bold red]")
                console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
                return None
    # --- End Robust Extraction ---

    if json_string_to_parse is None:
         # This case should be rare if step 3 or 4 found something, but added for safety
         logger.error(f"{phase_name}: Could not extract any potential JSON string from {agent_name}.")
         console.print(f"[bold red]{phase_name}: Error extracting potential JSON from {agent_name} output.[/bold red]")
         console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
         return None

    logger.debug(f"[{phase_name}/{agent_name}] Attempting to parse: {repr(json_string_to_parse)}")
    try:
        # Attempt to repair common issues like trailing commas (requires external library or more complex logic)
        # For now, just try standard parsing
        parsed_data = json.loads(json_string_to_parse)
        if not isinstance(parsed_data, dict):
            raise ValueError(f"Parsed JSON is not a dictionary (type: {type(parsed_data)}).")
        logger.info(f"[{phase_name}/{agent_name}] Successfully parsed JSON.")
        return parsed_data
    except (json.JSONDecodeError, ValueError) as json_error:
        logger.exception(f"{phase_name}: Failed to parse {agent_name} output as JSON: {json_error}")
        console.print(f"[bold red]{phase_name}: Error parsing {agent_name} JSON output.[/bold red]")
        console.print(f"[dim]Attempted to parse:[/dim]\n[yellow]{json_string_to_parse}[/yellow]")
        console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
        return None

def parse_json_output(
    raw_text: Optional[str],
    phase_name: str,
    agent_name: str,
    console: Console,
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON from an agent's text output, robustly handling
    markdown code fences and preceding/trailing text.
    Returns the parsed dictionary or None on failure.
    """
    if not raw_text:
        console.print(f"[yellow]{phase_name}: {agent_name} returned empty final response.[/yellow]")
        return None

    text_to_parse = raw_text.strip()
    logger.debug(f"[{phase_name}/{agent_name}] Raw output: {repr(text_to_parse)}")

    # --- More Robust JSON Extraction ---
    # 1. Look for ```json ... ``` block first
    match = re.search(r"```json\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
    if match:
        json_string_to_parse = match.group(1).strip()
        logger.debug(f"[{phase_name}/{agent_name}] Extracted JSON block using ```json regex.")
    else:
        # 2. If no ```json block, look for ``` ... ``` block
        match = re.search(r"```\s*(\{.*?\})\s*```", text_to_parse, re.DOTALL)
        if match:
            json_string_to_parse = match.group(1).strip()
            logger.debug(f"[{phase_name}/{agent_name}] Extracted JSON block using ``` regex.")
        else:
            # 3. If no markdown block, find the first '{' and try parsing from there
            start_brace = text_to_parse.find('{')
            if start_brace != -1:
                # Attempt to find the matching closing brace (simple heuristic)
                # This might fail for complex nested structures if there's trailing text
                potential_json = text_to_parse[start_brace:]
                # Try to find a balanced closing brace - this is tricky without full parsing
                # A simpler approach: just try parsing the substring from the first brace
                json_string_to_parse = potential_json
                logger.debug(f"[{phase_name}/{agent_name}] No markdown found, trying from first '{{'.")
            else:
                # 4. If no '{' found at all
                logger.error(f"{phase_name}: Could not find JSON object start '{{' in output from {agent_name}.")
                console.print(f"[bold red]{phase_name}: Error parsing {agent_name} JSON output (no '{{' found).[/bold red]")
                console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
                return None
    # --- End Robust Extraction ---

    logger.debug(f"[{phase_name}/{agent_name}] Attempting to parse: {repr(json_string_to_parse)}")
    try:
        parsed_data = json.loads(json_string_to_parse)
        if not isinstance(parsed_data, dict):
            raise ValueError(f"Parsed JSON is not a dictionary (type: {type(parsed_data)}).")
        # console.print(f"[dim]  {phase_name} ({agent_name}) Final Output (Parsed JSON): {len(parsed_data)} results received.[/dim]") # Less verbose success
        logger.info(f"[{phase_name}/{agent_name}] Successfully parsed JSON.")
        return parsed_data
    except (json.JSONDecodeError, ValueError) as json_error:
        logger.exception(f"{phase_name}: Failed to parse {agent_name} output as JSON: {json_error}")
        console.print(f"[bold red]{phase_name}: Error parsing {agent_name} JSON output.[/bold red]")
        console.print(f"[dim]Attempted to parse:[/dim]\n[yellow]{json_string_to_parse}[/yellow]")
        console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
        return None

def format_iso_timestamp(iso_str: Optional[str]) -> str:
    """Formats an ISO timestamp string into 'YYYY-MM-DD h:MM AM/PM'."""
    if not iso_str:
        return "Unknown Time"
    try:
        dt_obj = datetime.fromisoformat(iso_str)
        return dt_obj.strftime("%Y-%m-%d %I:%M %p")
    except (ValueError, TypeError):
        logging.warning(f"Could not parse timestamp for formatting: {iso_str}")
        return iso_str
    
def format_localized_timestamp(
    iso_str: Optional[str],
    timezone_str: str = 'UTC', # Default to UTC if no specific timezone provided
    display_format: str = "%Y-%m-%d %I:%M:%S %p %Z" # Example: 2023-10-27 03:45:10 PM EDT
) -> str:
    """
    Formats an ISO timestamp string into a human-readable, localized time.

    Args:
        iso_str: The ISO timestamp string (assumed to be UTC or timezone-aware).
        timezone_str: The target IANA timezone name (e.g., 'America/New_York').
        display_format: The strftime format string for the output.

    Returns:
        The formatted, localized time string or an error message.
    """
    if not iso_str:
        return "Unknown Time"

    try:
        # 1. Parse the ISO string into a timezone-aware datetime object
        # Handle 'Z' for UTC explicitly if present
        if iso_str.endswith('Z'):
            utc_dt = datetime.fromisoformat(iso_str[:-1] + '+00:00')
        else:
            # Assume it's already timezone-aware or implicitly UTC if offset is missing
            # (datetime.fromisoformat handles offsets like +00:00)
            utc_dt = datetime.fromisoformat(iso_str)
            # If the parsed object is naive, assume it's UTC
            if utc_dt.tzinfo is None:
                utc_dt = utc_dt.replace(tzinfo=timezone.utc)


        # 2. Get the target timezone object
        try:
            target_tz = pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone '{timezone_str}'. Falling back to UTC.")
            target_tz = pytz.utc
            timezone_str = 'UTC' # Update for display

        # 3. Convert the datetime object to the target timezone
        local_dt = utc_dt.astimezone(target_tz)

        # 4. Format the localized datetime object
        return local_dt.strftime(display_format)

    except ValueError:
        logger.warning(f"Could not parse ISO timestamp: {iso_str}")
        return f"Invalid Time ({iso_str})"
    except Exception as e:
        logger.error(f"Error formatting localized time for '{iso_str}' with tz '{timezone_str}': {e}")
        return f"Time Error ({iso_str})"
    
def get_timezone_from_location(
    location_dict: Optional[Dict[str, Any]],
    tf_instance: TimezoneFinder # Pass the instance
) -> Optional[str]:
    """
    Attempts to find the IANA timezone from location details using coordinates.

    Args:
        location_dict: Dictionary containing location details (must have 'coordinates').
        tf_instance: An initialized TimezoneFinder instance.

    Returns:
        The IANA timezone string or None if not found/error.
    """
    if not location_dict:
        logger.debug("get_timezone_from_location: location_dict is None.")
        return None

    coords = location_dict.get("coordinates")
    if coords and coords.get("latitude") is not None and coords.get("longitude") is not None:
        lat = coords["latitude"]
        lon = coords["longitude"]
        try:
            # Use the passed instance
            found_tz = tf_instance.timezone_at(lng=lon, lat=lat)
            if found_tz:
                logger.debug(f"Found timezone '{found_tz}' from coordinates ({lat}, {lon})")
                return found_tz
            else:
                logger.warning(f"timezonefinder returned None for coordinates ({lat}, {lon}).")
                return None
        except Exception as tz_err:
            logger.error(f"Error using timezonefinder for coords ({lat}, {lon}): {tz_err}")
            return None
    else:
        city = location_dict.get("city", "N/A") # Use city for logging if available
        logger.debug(f"Coordinates missing or incomplete for location '{city}'. Cannot determine timezone.")
        return None