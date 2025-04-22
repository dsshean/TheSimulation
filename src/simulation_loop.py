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
    "user": "bold white" # For the trigger
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
            author_color = get_agent_color(event.author)

            # *** ENHANCED DESCRIPTIVE LOGGING with Rich ***
            console.print(Rule(f"Processing Event {event_count}", style="dim blue", align="left"))
            console.print(f"  Author: [bold {author_color}]{event.author}[/bold {author_color}], Type: [i]{type(event).__name__}[/i], Final Step for Author: {event.is_final_response()}")

            # Log Content Parts
            if event.content and event.content.parts:
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

            # Log Function Calls/Responses using methods
            function_calls = event.get_function_calls()
            function_responses = event.get_function_responses()
            # Use pretty_repr for better formatting of complex objects
            if function_calls: console.print(f"    Function Calls: [bright_blue]{pretty_repr(function_calls)}[/bright_blue]")
            if function_responses: console.print(f"    Function Responses: [bright_green]{pretty_repr(function_responses)}[/bright_green]")

            # --- Add More Descriptive Logging Based on Events ---
            log_prefix = f"  [bold {author_color}]>> {event.author.upper()}:[/bold {author_color}]"

            if event.author == 'narration_agent':
                if function_calls:
                    for call in function_calls:
                        if call.name == 'transfer_to_agent':
                            console.print(f"{log_prefix} Delegating control to agent '[bold]{call.args.get('agent_name')}[/bold]'...")
                        elif call.name == 'process_movement':
                             console.print(f"{log_prefix} Instructing [bold green]World Engine[/] to use '[i]process_movement[/i]' (Args: {call.args})...")
                        elif call.name == 'generate_npc_response':
                             console.print(f"{log_prefix} Instructing [bold magenta]NPC Agent[/] to use '[i]generate_npc_response[/i]' (Args: {call.args})...")
                        elif call.name == 'get_setting_details':
                             console.print(f"{log_prefix} Instructing [bold green]World Engine[/] to use '[i]get_setting_details[/i]' (Args: {call.args})...")
                        elif call.name == 'get_current_simulation_state_summary':
                            console.print(f"{log_prefix} Calling OWN tool '[i]{call.name}[/i]'...")
                        elif call.name == 'set_simulacra_daily_goal':
                            console.print(f"{log_prefix} Calling OWN tool '[i]{call.name}[/i]'...")
                if function_responses:
                     for resp in function_responses:
                         # Log result from Narrator's own tool call
                         console.print(f"{log_prefix} Received result for OWN tool '[i]{resp.name}[/i]'.")


            elif event.author == 'simulacra':
                 if function_calls:
                    for call in function_calls:
                        console.print(f"{log_prefix} Calling tool '[i]{call.name}[/i]' (Args: {call.args})...")
                 if function_responses:
                     for resp in function_responses:
                         console.print(f"{log_prefix} Received result for tool '[i]{resp.name}[/i]'.")
                 if event.is_final_response():
                     console.print(f"{log_prefix} Finished its step (declared intent via tool call).")

            elif event.author == 'world_engine':
                 if function_responses:
                     for resp in function_responses:
                          # Log result from World Engine's tool call (requested by Narrator)
                          console.print(f"{log_prefix} Finished processing tool '[i]{resp.name}[/i]'. Storing result. Snippet: [italic]{str(resp.response)[:100]}...[/italic]")
                 if event.is_final_response():
                      # This indicates the World Engine agent has finished its step in the sequence
                      console.print(f"{log_prefix} Finished its step.")


            elif event.author == 'npc_agent':
                 if function_responses:
                     for resp in function_responses:
                          # Log result from NPC Agent's tool call (requested by Narrator)
                          console.print(f"{log_prefix} Finished processing tool '[i]{resp.name}[/i]'. Storing result. Snippet: [italic]{str(resp.response)[:100]}...[/italic]")
                 if event.is_final_response():
                      # This indicates the NPC Agent agent has finished its step in the sequence
                      console.print(f"{log_prefix} Finished its step.")
            # --- End of Descriptive Logging ---

            # Check for the final response for the *entire turn*
            if event.is_final_response() and event.author == 'narration_agent':
                console.print(f"  [bold yellow]>> NARRATOR:[/bold yellow] Preparing final narrative for the turn.")
                if event.content and event.content.parts:
                     if hasattr(event.content.parts[0], 'text') and event.content.parts[0].text:
                         final_response_text = event.content.parts[0].text
                     else:
                         final_response_text = f"[Narrator finished turn, content type not plain text or only action]"
                elif event.error_message:
                    final_response_text = f"[bold red]Agent Error:[/bold red] {event.error_message}"
                else:
                    final_response_text = f"[Narrator finished turn without explicit content or error]"
                break # Got the final response event for this turn

        if final_response_text == "Simulation turn did not produce a final narrative.":
             console.print("[bold red]WARNING:[/bold red] Loop finished but no final response from narrator detected.")

        # Print final narrative in a panel
        console.print(Panel(final_response_text, title=f"Narrator (Turn {turn_number})", title_align="left", border_style="bold green", expand=False))

        # Inspect final state for this turn
        current_session = session_service.get_session(app_name=settings.APP_NAME, user_id=settings.USER_ID, session_id=settings.SESSION_ID)
        if current_session:
            console.print(Rule("End of Turn State", style="dim"))
            # Use pretty_repr for better state dictionary formatting
            console.print(f"  Time: {current_session.state.get('world_time')}")
            console.print(f"  Location: {current_session.state.get('simulacra_location')}")
            console.print(f"  Goal: {current_session.state.get('simulacra_goal')}")
            # console.print(pretty_repr(current_session.state)) # Uncomment for full state view
        else:
            console.print("[bold red]Could not retrieve session state after turn.[/bold red]")

    except Exception as e:
        console.print(f"[bold red]\n--- ERROR during simulation turn {turn_number} ---[/bold red]")
        # Print exception using Rich for better formatting
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