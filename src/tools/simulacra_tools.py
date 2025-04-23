# src/tools/simulacra_tools.py
from google.adk.tools.tool_context import ToolContext
from rich.console import Console
# --- Add LLMService import ---
from src.generation.llm_service import LLMService # Assuming this is your service for LLM calls
# --- End Add ---


console = Console()

# --- NEW REFLECTION TOOL ---
def generate_internal_monologue(
    current_goal: str,
    current_location: str,
    current_time: str,
    setting_description: str,
    tool_context: ToolContext
) -> str:
    """
    Generates a brief internal monologue based on the Simulacra's current context.
    Saves the monologue to the 'last_simulacra_monologue' state key.
    """
    console.print("[dim blue]--- Tool: Simulacra generating internal monologue ---[/dim blue]")
    monologue = "Error generating monologue." # Default message
    try:
        llm_service = LLMService() # Instantiate your LLM service
        prompt = (
            f"You are the Simulacra. Your current context is:\n"
            f"- Location: {current_location}\n"
            f"- Time: {current_time}\n"
            f"- Goal: {current_goal}\n"
            f"- Setting Description: {setting_description}\n\n"
            f"Based *only* on this context, write a very brief (2-4 sentences) internal monologue reflecting your current thoughts or feelings related to your goal or situation."
        )
        generated_text = llm_service.generate_content_text(prompt=prompt) # Use your service method

        if generated_text:
            monologue = generated_text.strip()
            tool_context.state['last_simulacra_monologue'] = monologue # Save to state
            console.print(f"[dim blue]--- Tool: Monologue generated: '[italic]{monologue}[/italic]' ---[/dim blue]")
        else:
            console.print("[yellow]--- Tool: Monologue generation returned empty result. ---[/yellow]")
            tool_context.state['last_simulacra_monologue'] = "Could not generate a thought."

    except Exception as e:
        console.print(f"[bold red]--- Tool Error (generate_internal_monologue): {e} ---[/bold red]")
        tool_context.state['last_simulacra_monologue'] = f"Error reflecting: {e}" # Save error state

    return monologue # Return the generated text
# --- END NEW REFLECTION TOOL ---


def attempt_move_to(destination: str, tool_context: ToolContext) -> dict: # <<< Changed return type to dict
    """
    Declares the Simulacra's intent to move. Writes intent to state AND returns the details.
    """
    current_location = tool_context.state.get("simulacra_location", "Unknown")
    console.print(f"[dim blue]--- Tool: Simulacra intends to move from [i]{current_location}[/i] to [i]{destination}[/i] ---[/dim blue]")
    action_details = {
        "action": "move",
        "destination": destination,
        "origin": current_location
    }
    tool_context.state["last_simulacra_action"] = action_details # Still save to state (good practice)
    # --- Return the dictionary directly ---
    return action_details


def attempt_talk_to(npc_name: str, message: str, tool_context: ToolContext) -> dict: # <<< Changed return type to dict
    """
    Declares the Simulacra's intent to talk. Writes intent to state AND returns the details.
    """
    current_location = tool_context.state.get("simulacra_location", "Unknown")
    console.print(f"[dim blue]--- Tool: Simulacra intends to talk to [i]{npc_name}[/i] at [i]{current_location}[/i], saying: '[italic]{message}[/italic]' ---[/dim blue]")
    action_details = {
        "action": "talk",
        "npc": npc_name,
        "message": message,
        "location": current_location
    }
    tool_context.state["last_simulacra_action"] = action_details # Still save to state
    # --- Return the dictionary directly ---
    return action_details


def check_self_status(tool_context: ToolContext) -> dict:
    """Allows the Simulacra to check its own status (inventory, goal, location) when deciding its action."""
    console.print("[dim blue]--- Tool: Simulacra checking self status ---[/dim blue]")
    status = {
        "location": tool_context.state.get("simulacra_location", "Unknown"),
        "goal": tool_context.state.get("simulacra_goal", "None set"),
        "status": tool_context.state.get("simulacra_status", {}),
    }
    tool_context.state["last_simulacra_status_check"] = status # Store result in state
    console.print(f"[dim blue]--- Tool: Status check result: {status} ---[/dim blue]")
    return status