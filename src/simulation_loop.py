# src/simulation_loop.py
import asyncio
import traceback
from google.genai import types as genai_types
from src.config import settings # Import constants
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.pretty import pretty_repr

# Instantiate Console for rich output
console = Console(width=120) # Adjust width as needed

# Define colors for agents (add more as needed)
AGENT_COLORS = {
    "narration_agent": "yellow",
    "simulacra": "blue",
    "world_engine": "green",
    "npc_agent": "magenta",
    "user": "bold white", # For the trigger
    "system": "dim white" # For internal/system events if needed
}

def get_agent_color(author):
    return AGENT_COLORS.get(author, "white") # Default to white

def generate_npc_response(received_message: str, npc_name: str, npc_role: str) -> dict:
    """
    Generates a response for the NPC based on the received message and its role.
    :param received_message: The message received by the NPC.
    :param npc_name: The name of the NPC.
    :param npc_role: The role or identity of the NPC (e.g., person, animal, object).
    :return: A dictionary containing the generated response.
    """
    response = f"{npc_name} ({npc_role}) says: 'I received your message: {received_message}'"
    return {"result": response}

async def run_simulation_turn(turn_number: int, runner, session_service):
    """Runs a single turn of the simulation with enhanced Rich logging."""
    if not runner:
        console.print("[bold red]Runner not initialized. Cannot run simulation turn.[/bold red]")
        return

    console.print(Rule(f"Starting Simulation Turn {turn_number}", style="bold cyan"))
    user_input_text = f"Continue simulation (Turn {turn_number})."
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=user_input_text)])

    final_response_text = "Simulation turn did not produce a final narrative."
    console.print(f"[bold {get_agent_color('user')}]>>> User Trigger:[/bold {get_agent_color('user')}] {user_input_text}")

    try:
        event_count = 0
        async for event in runner.run_async(user_id=settings.USER_ID, session_id=settings.SESSION_ID, new_message=content):
            event_count += 1
            # --- Add initial safety check for content ---
            if not event.content:
                 console.print(Rule(f"Processing Event {event_count} (No Content)", style="dim white", align="left"))
                 author_color = get_agent_color(event.author)
                 console.print(f"  Author: [bold {author_color}]{event.author}[/bold {author_color}], Type: [i]{type(event).__name__}[/i], Final Step for Author: {event.is_final_response()}")
                 # Log actions if they exist even without content
                 if event.actions:
                      if event.actions.state_delta: console.print(f"    State Delta: [dim]{pretty_repr(event.actions.state_delta)}[/]")
                      if event.actions.artifact_delta: console.print(f"    Artifact Delta: [dim]{pretty_repr(event.actions.artifact_delta)}[/]")
                      if event.actions.transfer_to_agent: console.print(f"    Transfer Action: [dim]To {event.actions.transfer_to_agent}[/]")
                      if event.actions.escalate: console.print(f"    Escalate Action: [dim]True[/]")
                 console.print(f"  (Skipping detailed content/tool logging as event.content is None)")
                 continue # Skip the rest of the loop for this content-less event
            # --- End initial safety check ---

            # If we reach here, event.content is guaranteed to be not None
            author_color = get_agent_color(event.author)
            console.print(Rule(f"Processing Event {event_count}", style="dim blue", align="left"))
            console.print(f"  Author: [bold {author_color}]{event.author}[/bold {author_color}], Type: [i]{type(event).__name__}[/i], Final Step for Author: {event.is_final_response()}")


            # Log Content Parts (safe now because we checked event.content above)
            if event.content.parts:
                content_str_parts = []
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        content_str_parts.append(f"Text: {part.text[:150]}...")
                    elif hasattr(part, 'function_call') and part.function_call:
                        content_str_parts.append(f"FunctionCall Part: {part.function_call.name}")
                    elif hasattr(part, 'function_response') and part.function_response:
                        content_str_parts.append(f"FunctionResponse Part: {part.function_response.name}")
                if content_str_parts:
                    console.print(f"    Content Parts: [italic cyan][{'; '.join(content_str_parts)}][/italic cyan]")
            else:
                 console.print("    Content: [italic]Present but has no parts.[/italic]")


            # Log Function Calls/Responses (safe now because get_ methods handle None content)
            function_calls = event.get_function_calls()
            function_responses = event.get_function_responses()
            if function_calls: console.print(f"    Function Calls: [bright_blue]{pretty_repr(function_calls)}[/bright_blue]")
            if function_responses: console.print(f"    Function Responses: [bright_green]{pretty_repr(function_responses)}[/bright_green]")

            # --- Agent Specific Logging (Mostly safe as they check func_calls/responses) ---
            log_prefix = f"  [bold {author_color}]>> {event.author.upper()}:[/bold {author_color}]"
            # (Agent-specific logging logic remains the same as it relies on checks above)
            # ... (rest of your agent-specific logging) ...
            if event.author == 'narration_agent':
                if function_calls:
                    for call in function_calls:
                        # ... (logging for narration calls) ...
                        pass # Placeholder for brevity
                if function_responses:
                     for resp in function_responses:
                         console.print(f"{log_prefix} Received result for OWN tool '[i]{resp.name}[/i]'.")


            elif event.author == 'simulacra':
                 if function_calls:
                    for call in function_calls:
                        console.print(f"{log_prefix} Calling tool '[i]{call.name}[/i]' (Args: {call.args})...")
                 if function_responses:
                     for resp in function_responses:
                         console.print(f"{log_prefix} Received result for tool '[i]{resp.name}[/i]'.")
                 # Removed the is_final_response check here as it's handled below

            elif event.author == 'world_engine':
                 if function_calls: # Added check for calls if WE calls tools itself
                     for call in function_calls:
                         console.print(f"{log_prefix} Calling tool '[i]{call.name}[/i]' (Args: {call.args})...")
                 if function_responses:
                     for resp in function_responses:
                          console.print(f"{log_prefix} Finished processing tool '[i]{resp.name}[/i]'. Storing result. Snippet: [italic]{str(resp.response)[:100]}...[/italic]")
                 # Removed the is_final_response check here as it's handled below


            elif event.author == 'npc_agent':
                 if function_calls: # Added check for calls if NPC calls tools itself
                     for call in function_calls:
                          console.print(f"{log_prefix} Calling tool '[i]{call.name}[/i]' (Args: {call.args})...")
                 if function_responses:
                     for resp in function_responses:
                          console.print(f"{log_prefix} Finished processing tool '[i]{resp.name}[/i]'. Storing result. Snippet: [italic]{str(resp.response)[:100]}...[/italic]")
                 # Removed the is_final_response check here as it's handled below
            # --- End of Descriptive Logging ---

            # Check for the final response for the *entire turn* (Narrator only)
            # This block is now safe because we check event.content earlier
            if event.is_final_response() and event.author == 'narration_agent':
                console.print(f"  [bold yellow]>> NARRATOR:[/bold yellow] Preparing final narrative for the turn.")
                # We already checked event.content is not None if we reached here
                if event.content.parts:
                     # Check the first part for text
                     first_part = event.content.parts[0]
                     if hasattr(first_part, 'text') and first_part.text:
                         final_response_text = first_part.text
                     else:
                         final_response_text = f"[Narrator finished turn, content type not plain text or only action]"
                elif event.error_message:
                    final_response_text = f"[bold red]Agent Error:[/bold red] {event.error_message}"
                else:
                    # Event content exists but has no parts and no error
                    final_response_text = f"[Narrator finished turn without parts or error]"
                break # Got the final response event for this turn

        if final_response_text == "Simulation turn did not produce a final narrative.":
             console.print("[bold red]WARNING:[/bold red] Loop finished but no final response from narrator detected.")

        # Print final narrative in a panel
        console.print(Panel(final_response_text, title=f"Narrator (Turn {turn_number})", title_align="left", border_style="bold green", expand=False))

        # Inspect final state for this turn
        # (State inspection logic remains the same)
        # ...

    except Exception as e:
        console.print(f"[bold red]\n--- ERROR during simulation turn {turn_number} ---[/bold red]")
        console.print_exception(show_locals=True)

async def main(runner, session_service):
    """Main async function to run multiple simulation turns."""
    if not runner:
        console.print("[bold red]Simulation cannot start - Runner was not initialized.[/bold red]")
        return

    num_turns = 3 # Number of turns to simulate
    for i in range(1, num_turns + 1):
        await run_simulation_turn(i, runner, session_service)
        # Add a small delay if needed, e.g., await asyncio.sleep(0.5)

    console.print(Rule("Simulation Ended", style="bold magenta"))