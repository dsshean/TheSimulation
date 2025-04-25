# src/agents/world_state_agent.py (Updater & Executor Role)
import logging
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from src.config import settings
# Ensure world_state_tools has both update_and_get_world_state AND execute_physical_actions_batch defined
from src.tools import world_state_tools
from src.prompts import world_state_instructions
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

try:
    world_state_agent = Agent(
        name="world_state_agent",
        model=settings.MODEL_GEMINI_PRO,
        description=(
            "Manages the overall simulation state. In Phase 1, it updates world time and dynamics. "
            "In Phase 4b, it executes approved physical actions (like movement) based on a provided batch, modifying the state." # Added execution role
        ),
        instruction=world_state_instructions.WORLD_STATE_INSTRUCTION, # Ensure instructions cover both roles if needed, or rely on trigger message context
        tools=[
            # Tool for Phase 1
            FunctionTool(world_state_tools.update_and_get_world_state),
            # --- ADDED BACK: Tool for Phase 4b ---
            FunctionTool(world_state_tools.execute_physical_actions_batch),
            # --- END ADDED BACK ---
        ],
        # Output key might capture confirmation/summary depending on phase
        output_key="world_state_agent_confirmation",
    )
    console.print(f"Agent '[bold blue]{world_state_agent.name}[/bold blue]' defined (State Updater & Executor Role).") # Updated Role Desc

except Exception as e:
    console.print(f"[bold red]Error creating world_state_agent:[/bold red] {e}")
    console.print_exception(show_locals=True)
    world_state_agent = None