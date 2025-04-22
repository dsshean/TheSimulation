# src/tools/simulacra_tools.py
from google.adk.tools.tool_context import ToolContext
from rich.console import Console

console = Console()

def attempt_move_to(destination: str, tool_context: ToolContext) -> str:
    """Declares the Simulacra's intent to move to a new location. Writes intent to 'last_simulacra_action' state."""
    current_location = tool_context.state.get("simulacra_location", "Unknown")
    console.print(f"[dim blue]--- Tool: Simulacra intends to move from [i]{current_location}[/i] to [i]{destination}[/i] ---[/dim blue]")
    tool_context.state["last_simulacra_action"] = {"action": "move", "destination": destination, "origin": current_location}
    result_msg = f"Intent registered: Move from {current_location} to {destination}."
    return result_msg

def attempt_talk_to(npc_name: str, message: str, tool_context: ToolContext) -> str:
    """Declares the Simulacra's intent to talk to an NPC. Writes intent to 'last_simulacra_action' state."""
    current_location = tool_context.state.get("simulacra_location", "Unknown")
    console.print(f"[dim blue]--- Tool: Simulacra intends to talk to [i]{npc_name}[/i] at [i]{current_location}[/i], saying: '[italic]{message}[/italic]' ---[/dim blue]")
    tool_context.state["last_simulacra_action"] = {"action": "talk", "npc": npc_name, "message": message, "location": current_location}
    result_msg = f"Intent registered: Talk to {npc_name} with message '{message}'."
    return result_msg

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