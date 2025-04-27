from google.adk.agents import Agent # Alias for LlmAgent
from src.config import settings
from src.tools import narration_tools
from src.prompts import narration_instructions
from rich.console import Console
import logging

console = Console()
logger = logging.getLogger(__name__)

narration_agent = None
try:
    narration_agent = Agent(
        name="NarrationAgent",
        model=settings.MODEL_GEMINI_PRO,
        description="Describes the events of the simulation turn based on the final state and interaction results, outputting structured JSON.", # Updated description slightly
        instruction=narration_instructions.NARRATION_AGENT_INSTRUCTION,
        tools=[ # <<< KEEP THIS
            narration_tools.get_narration_context,
        ],
    )
    console.print(f"Agent '[bold yellow]{narration_agent.name}[/bold yellow]' defined (Narrator Role).")
    logger.info(f"NarrationAgent initialized with tool: {narration_agent.tools[0].__name__}") # Log tool only

except Exception as e:
    console.print(f"[bold red]Error creating NarrationAgent:[/bold red] {e}")
    console.print_exception(show_locals=True)
    narration_agent = None # Ensure it's None on error