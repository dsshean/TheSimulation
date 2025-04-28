import logging
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from src.config import settings
from src.tools import world_state_tools, world_engine_tools # Need both tool modules
from src.prompts import world_state_instructions # Phase 1 instructions
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

try:
    world_state_agent = Agent(
        name="world_state_agent",
        model=settings.MODEL_GEMINI_PRO,
        description=(
            "Manages the start-of-turn world state update. In Phase 1, it fetches "
            "real-world details (weather, news) and advances the simulation time." # Updated description
        ),
        instruction=world_state_instructions.WORLD_STATE_INSTRUCTION, # Phase 1 instructions
        tools=[
            FunctionTool(world_state_tools.get_setting_details), # Moved from world_engine_agent
            FunctionTool(world_state_tools.update_and_get_world_state),
        ],
        output_key="world_state_agent_confirmation", # Captures the confirmation text
    )
    console.print(f"Agent '[bold blue]{world_state_agent.name}[/bold blue]' defined (Phase 1 State Updater Role).") # Updated Role Desc

except Exception as e:
    console.print(f"[bold red]Error creating world_state_agent:[/bold red] {e}")
    console.print_exception(show_locals=True)
    world_state_agent = None