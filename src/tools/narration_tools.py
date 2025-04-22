# src/tools/narration_tools.py
from google.adk.tools.tool_context import ToolContext
from rich.console import Console

console = Console()

def set_simulacra_daily_goal(goal: str, tool_context: ToolContext) -> str:
    """Sets or updates the Simulacra's main goal for the current simulation cycle (e.g., day)."""
    console.print(f"[dim cyan]--- Tool: Setting Simulacra goal to: [italic]{goal}[/italic] ---[/dim cyan]")
    tool_context.state["simulacra_goal"] = goal
    result_msg = f"Simulacra goal updated to: {goal}"
    tool_context.state["last_goal_update"] = result_msg # Store result in state
    return result_msg

def get_current_simulation_state_summary(tool_context: ToolContext) -> dict:
    """Retrieves key current state information like time, simulacra location, and goal for the Narrator."""
    console.print("[dim cyan]--- Tool: Retrieving current simulation state summary ---[/dim cyan]")
    summary = {
        "time": tool_context.state.get("world_time", "Unknown"),
        "simulacra_location": tool_context.state.get("simulacra_location", "Unknown"),
        "simulacra_goal": tool_context.state.get("simulacra_goal", "None set"),
        "last_narration": tool_context.state.get("last_narration", "None")
    }
    tool_context.state["last_state_summary"] = summary # Store result in state
    console.print(f"[dim cyan]--- Tool: Summary retrieved: {summary} ---[/dim cyan]")
    return summary