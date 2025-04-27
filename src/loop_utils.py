import json
import logging
from rich.console import Console
from google.generativeai import types
# --- ADDED: Import datetime ---
from datetime import datetime
from typing import Optional, Any, Dict
# ---


def print_event_details(
    event: Any, # Use Any temporarily to bypass type errors
    phase_name: str,
    console: Console,
    logger: logging.Logger,
    max_content_length: int = 5000,
    max_response_length: int = 5000
):
    """Prints details of an agent completion event to the console."""
    # --- ADDED: Log the actual type to help identify the correct import ---
    logger.debug(f"[{phase_name}] Received event of type: {type(event)}")
    # ---

    # Access attributes using getattr for safety until the correct type is known
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
            # Ensure response content is dictionary-like before converting
            try:
                response_str = json.dumps(dict(response_content))
            except (TypeError, ValueError):
                 response_str = str(response_content) # Fallback to string
            response_display = response_str[:max_response_length] + ('...' if len(response_str) > max_response_length else '')
            console.print(f"[dim green]  {phase_name} ({agent_id}) <- Tool Response: {getattr(tool_response, 'name', 'UnknownTool')} -> {response_display}[/dim green]")
        elif is_final and hasattr(part, 'text'):
            text_content = getattr(part, 'text', '')
            # parse_json_output(text_content, phase_name, agent_id, console, logger)
            console.print(f"[dim cyan]  {phase_name} ({agent_id}) Final Output: {text_content if text_content else '[No text output]'}[/dim cyan]")
    elif is_final:
         # Handle cases where there might be a final response without typical content parts
         logger.debug(f"{phase_name} ({agent_id}) Final event with no standard parts. Event: {event}")


def parse_json_output(
    raw_text: Optional[str],
    phase_name: str,
    agent_name: str,
    console: Console,
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Attempts to parse JSON from an agent's final text output, stripping markdown.
    Returns the parsed dictionary or None on failure.
    """
    # Ensure this part is robust as before
    if not raw_text:
        console.print(f"[yellow]{phase_name}: {agent_name} returned empty final response.[/yellow]")
        return None

    json_string_to_parse = raw_text.strip()

    # Strip markdown fences more robustly
    if json_string_to_parse.startswith("```"):
        # Find the first newline after ```
        first_newline = json_string_to_parse.find('\n')
        if first_newline != -1:
            # Check if the line is ```json or similar
            lang_spec = json_string_to_parse[:first_newline].strip()
            if lang_spec.startswith("```"): # e.g., ```json, ```
                json_string_to_parse = json_string_to_parse[first_newline + 1:].strip()

    if json_string_to_parse.endswith("```"):
        json_string_to_parse = json_string_to_parse[:-3].strip()

    try:
        parsed_data = json.loads(json_string_to_parse)
        if not isinstance(parsed_data, dict):
            raise ValueError(f"Parsed JSON is not a dictionary (type: {type(parsed_data)}).")
        console.print(f"[dim]  {phase_name} ({agent_name}) Final Output (Parsed JSON): {len(parsed_data)} results received.[/dim]")
        return parsed_data
    except (json.JSONDecodeError, ValueError) as json_error:
        logger.exception(f"{phase_name}: Failed to parse {agent_name} output as JSON: {json_error}")
        console.print(f"[bold red]{phase_name}: Error parsing {agent_name} JSON output.[/bold red]")
        console.print(f"[dim]Attempted to parse:[/dim]\n[yellow]{json_string_to_parse}[/yellow]")
        console.print(f"[dim]Original Raw Output:[/dim]\n[grey50]{raw_text}[/grey50]")
        return None

# --- ADDED: Timestamp Formatting Function ---
def format_iso_timestamp(iso_str: Optional[str]) -> str:
    """Formats an ISO timestamp string into 'YYYY-MM-DD h:MM AM/PM'."""
    if not iso_str:
        return "Unknown Time"
    try:
        # Attempt to parse the ISO string
        dt_obj = datetime.fromisoformat(iso_str)
        # Format: Year-Month-Day Hour(12-hour, zero-padded):Minute AM/PM
        return dt_obj.strftime("%Y-%m-%d %I:%M %p")
    except (ValueError, TypeError):
        # If parsing fails, return the original string
        logging.warning(f"Could not parse timestamp for formatting: {iso_str}")
        return iso_str