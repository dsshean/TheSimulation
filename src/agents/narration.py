# src/agents/narration.py (Simple Narrator Definition)

from google.adk.agents import Agent # Alias for LlmAgent
from src.config import settings
# Import the NEW tools module
from src.tools import narration_tools
# Import the NEW instructions module
from src.prompts import narration_instructions
from rich.console import Console
import logging

console = Console()
logger = logging.getLogger(__name__)

# --- Define the Narrator Agent Instance ---
# No factory needed now as it has no external agent dependencies
narration_agent = None
try:
    narration_agent = Agent(
        name="NarrationAgent", # Simpler name might be fine now
        model=settings.MODEL_GEMINI_PRO, # Needs good language generation
        description="Describes the events of the simulation turn based on the final state and interaction results.",
        instruction=narration_instructions.NARRATION_AGENT_INSTRUCTION,
        tools=[
            # Only needs the tool to get the final context
            narration_tools.get_narration_context,
        ],
        output_key="last_narration" # Keep saving the output
    )
    console.print(f"Agent '[bold yellow]{narration_agent.name}[/bold yellow]' defined (Narrator Role).")
    logger.info(f"NarrationAgent initialized with tool: {narration_agent.tools[0].__name__}")

except Exception as e:
    console.print(f"[bold red]Error creating NarrationAgent:[/bold red] {e}")
    console.print_exception(show_locals=True)